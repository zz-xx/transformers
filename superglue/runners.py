import collections as col
import logging
import numpy as np
from tqdm import tqdm, trange
from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .evaluate import compute_task_metrics
from shared.runners import warmup_linear
from glue.runners import TrainEpochState

logger = logging.getLogger(__name__)


@dataclass
class DatasetWithMetadata:
    dataset: TensorDataset
    metadata: Dict


@dataclass
class RunnerParameters:
    max_seq_length: int
    local_rank: int
    n_gpu: int
    fp16: bool
    learning_rate: float
    gradient_accumulation_steps: int
    t_total: int
    warmup_proportion: float
    num_train_epochs: float
    train_batch_size: int
    eval_batch_size: int


def full_batch_to_dataset(full_batch):
    dataset_ls = []
    others_dict = {}
    descriptors = []
    for i, (k, v) in enumerate(full_batch.asdict().items()):
        if isinstance(v, torch.Tensor):
            descriptors.append(("dataset", k, len(dataset_ls)))
            dataset_ls.append(v)
        elif v is None:
            descriptors.append(("none", k, None))
        else:
            descriptors.append(("other", k, None))
            others_dict[k] = v
    return DatasetWithMetadata(
        dataset=TensorDataset(*dataset_ls),
        metadata={
            "descriptors": descriptors,
            "other": others_dict,
        }
    )


def convert_examples_to_dataset(examples, tokenizer, max_seq_len, task):
    data_rows = [
        example.tokenize(tokenizer).featurize(tokenizer, max_seq_len)
        for example in examples
    ]
    """
    for i in range(2):
        print(examples[i])
        for i, tokens in enumerate(data_rows[i].get_tokens()):
            print(i, tokens)
        print("======")
    """
    full_batch = task.Batch.from_data_rows(data_rows)
    dataset_with_metadata = full_batch_to_dataset(full_batch)
    return dataset_with_metadata


class HybridLoader:
    def __init__(self, dataloader, metadata, task):
        self.dataloader = dataloader
        self.metadata = metadata
        self.task = task

    def __iter__(self):
        batch_size = self.dataloader.batch_size
        for i, batch in enumerate(self.dataloader):
            batch_dict = {}
            for descriptor, k, pos in self.metadata["descriptors"]:
                if descriptor == "dataset":
                    batch_dict[k] = batch[pos]
                elif descriptor == "none":
                    batch_dict[k] = None
                elif descriptor == "other":
                    batch_dict[k] = self.metadata["other"][k][i * batch_size: (i+1) * batch_size]
                else:
                    raise KeyError(descriptor)
            yield self.task.Batch(**batch_dict)

    def __len__(self):
        return len(self.dataloader)


class SuperglueTaskRunner:
    def __init__(self, task, model, optimizer, tokenizer, device, rparams):
        self.task = task
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.rparams = rparams

    def run_train(self, train_examples, verbose=True):
        if verbose:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", self.rparams.train_batch_size)
            logger.info("  Num steps = %d", self.rparams.t_total)
        train_dataloader = self.get_train_dataloader(train_examples)

        for _ in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            self.run_train_epoch(train_dataloader)

    def run_train_val(self, train_examples, val_examples):
        epoch_result_dict = col.OrderedDict()
        for i in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            train_dataloader = self.get_train_dataloader(train_examples)
            self.run_train_epoch(train_dataloader)
            epoch_result = self.run_val(val_examples)
            del epoch_result["logits"]
            epoch_result["metrics"] = epoch_result["metrics"].asdict()
            epoch_result_dict[i] = epoch_result
        return epoch_result_dict

    def run_train_epoch(self, train_dataloader):
        for _ in self.run_train_epoch_context(train_dataloader):
            pass

    def run_train_epoch_context(self, train_dataloader):
        self.model.train()
        train_epoch_state = TrainEpochState()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.run_train_step(
                step=step,
                batch=batch,
                train_epoch_state=train_epoch_state,
            )
            yield step, batch, train_epoch_state

    def run_train_step(self, step, batch, train_epoch_state):
        batch = batch.to(self.device)
        loss = self.model.forward_batch(batch)
        if self.rparams.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if self.rparams.gradient_accumulation_steps > 1:
            loss = loss / self.rparams.gradient_accumulation_steps
        if self.rparams.fp16:
            self.optimizer.backward(loss)
        else:
            loss.backward()

        train_epoch_state.tr_loss += loss.item()
        train_epoch_state.nb_tr_examples += len(batch)
        train_epoch_state.nb_tr_steps += 1
        if (step + 1) % self.rparams.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = self.rparams.learning_rate * warmup_linear(
                train_epoch_state.global_step / self.rparams.t_total, self.rparams.warmup_proportion)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_this_step
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_epoch_state.global_step += 1

    def run_val(self, val_examples):
        self.model.eval()
        val_dataloader = self.get_eval_dataloader(val_examples)
        total_eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_logits = []
        for step, batch in enumerate(tqdm(val_dataloader, desc="Evaluating (Val)")):
            batch = batch.to(self.device)

            with torch.no_grad():
                tmp_eval_loss = self.model.forward_batch(batch)
                logits = self.model.forward_batch_hide_label(batch)

            logits = logits.detach().cpu().numpy()
            total_eval_loss += tmp_eval_loss.mean().item()

            nb_eval_examples += len(batch)
            nb_eval_steps += 1
            all_logits.append(logits)
        eval_loss = total_eval_loss / nb_eval_steps
        all_logits = np.concatenate(all_logits, axis=0)

        return {
            "logits": all_logits,
            "loss": eval_loss,
            "metrics": compute_task_metrics(self.task, all_logits, val_examples),
        }

    def run_test(self, test_examples):
        test_dataloader = self.get_eval_dataloader(test_examples)
        self.model.eval()
        all_logits = []
        for step, batch in enumerate(tqdm(test_dataloader, desc="Predictions (Test)")):
            batch = batch.to(self.device)
            with torch.no_grad():
                logits = self.model.forward_batch_hide_label(batch)
            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)

        all_logits = np.concatenate(all_logits, axis=0)
        return all_logits

    def get_train_dataloader(self, train_examples):
        dataset_with_metadata = convert_examples_to_dataset(
            examples=train_examples,
            max_seq_len=self.rparams.max_seq_length,
            tokenizer=self.tokenizer,
            task=self.task,
        )
        train_sampler = get_sampler(
            dataset=dataset_with_metadata.dataset,
            local_rank=self.rparams.local_rank,
        )
        train_dataloader = DataLoader(
            dataset=dataset_with_metadata.dataset,
            sampler=train_sampler,
            batch_size=self.rparams.train_batch_size,
        )
        return HybridLoader(
            dataloader=train_dataloader,
            metadata=dataset_with_metadata.metadata,
            task=self.task,
        )

    def get_eval_dataloader(self, eval_examples):
        dataset_with_metadata = convert_examples_to_dataset(
            examples=eval_examples,
            max_seq_len=self.rparams.max_seq_length,
            tokenizer=self.tokenizer,
            task=self.task,
        )
        eval_sampler = SequentialSampler(dataset_with_metadata.dataset)
        eval_dataloader = DataLoader(
            dataset=dataset_with_metadata.dataset,
            sampler=eval_sampler,
            batch_size=self.rparams.eval_batch_size,
        )
        return HybridLoader(
            dataloader=eval_dataloader,
            metadata=dataset_with_metadata.metadata,
            task=self.task,
        )


def get_sampler(dataset, local_rank):
    if local_rank == -1:
        return RandomSampler(dataset)
    else:
        return DistributedSampler(dataset)
