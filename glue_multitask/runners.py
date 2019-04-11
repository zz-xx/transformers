import logging
import random
from tqdm import trange
import glue.runners as glue_runners

logger = logging.getLogger(__name__)


class RunnerParameters:
    def __init__(self, max_seq_length, local_rank, n_gpu, fp16,
                 learning_rate, gradient_accumulation_steps,
                 task_lr_scaler_dict, task_sampling_mode,
                 t_total, warmup_proportion,
                 num_train_epochs, train_batch_size, eval_batch_size):
        self.max_seq_length = max_seq_length
        self.local_rank = local_rank
        self.n_gpu = n_gpu
        self.fp16 = fp16
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.task_lr_scaler_dict = task_lr_scaler_dict
        self.task_sampling_mode = task_sampling_mode
        self.t_total = t_total
        self.warmup_proportion = warmup_proportion
        self.num_train_epochs = num_train_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    @classmethod
    def format_task_lr_scaler_dict(cls, args_task_lr_scaler_ls, task_ls):
        if args_task_lr_scaler_ls is None:
            task_lr_scaler_ls = [1] * len(task_ls)
        else:
            task_lr_scaler_ls = list(map(float, args_task_lr_scaler_ls.split(",")))
            assert len(task_ls) == len(task_lr_scaler_ls)
        return {
            task.name: task_lr_scaler
            for task, task_lr_scaler
            in zip(task_ls, task_lr_scaler_ls)
        }


class GlueMultitaskRunner:
    def __init__(self, model, optimizer, tokenizer, task_ls, device, rparams):
        self.model = model
        self.rparams = rparams
        self.runner_dict = {}
        for task in task_ls:
            self.runner_dict[task.name] = glue_runners.GlueTaskRunner(
                model=model[task.name],
                optimizer=optimizer,
                tokenizer=tokenizer,
                label_list=task.get_labels(),
                device=device,
                rparams=self.create_task_rparams(),
            )

    def run_train(self, train_examples_dict, verbose=True):
        if verbose:
            logger.info("***** Running training *****")
            for task_name, train_examples in train_examples_dict.items():
                logger.info("  Num examples[%s] = %d", task_name, len(train_examples))
            logger.info("  Batch size = %d", self.rparams.train_batch_size)
            logger.info("  Num steps = %d", self.rparams.t_total)
        train_dataloader_dict = self.get_train_dataloader_dict(train_examples_dict, verbose=True)

        for _ in trange(int(self.rparams.num_train_epochs), desc="Epoch"):
            self.run_train_epoch(train_dataloader_dict)

    def run_train_epoch(self, train_dataloader_dict):
        for _ in self.run_train_epoch_context(train_dataloader_dict):
            pass

    def run_train_epoch_context(self, train_dataloader_dict):
        self.model.train()
        train_epoch_state_dict = {
            task_name: glue_runners.TrainEpochState()
            for task_name in train_dataloader_dict
        }
        for step, (task_name, batch) in enumerate(self.sample_trainloader_dict(train_dataloader_dict)):
            train_epoch_state = train_epoch_state_dict[task_name]
            self.runner_dict[task_name].run_train_step(
                step=step,
                batch=batch,
                train_epoch_state=train_epoch_state,
            )
            print(task_name, train_epoch_state)
            yield step, batch, train_epoch_state

    def run_val(self, val_examples_dict, verbose=True):
        results_dict = {}
        for task_name, val_examples in val_examples_dict.items():
            print(task_name)
            results_dict[task_name] = self.runner_dict[task_name].run_val(
                val_examples=val_examples,
                task_name=task_name,
                verbose=verbose,
            )
        return results_dict

    def run_test(self, test_examples_dict, verbose=True):
        results_dict = {}
        for task_name, test_examples in test_examples_dict.items():
            print(task_name)
            results_dict[task_name] = self.runner_dict[task_name].run_test(
                test_examples=test_examples,
                verbose=verbose,
            )
        return results_dict

    def sample_trainloader_dict(self, train_dataloader_dict):
        if self.rparams.task_sampling_mode == "basic":
            sampled_task_ls = [
                task_name
                for task_name, train_dataloader in train_dataloader_dict.items()
                for _ in range(len(train_dataloader))
            ]
            random.shuffle(sampled_task_ls)
        if self.rparams.task_sampling_mode == "minumum":
            minimum = min(len(train_dataloader) for train_dataloader in train_dataloader_dict.items())
            sampled_task_ls = list(train_dataloader_dict) * minimum
            random.shuffle(sampled_task_ls)
        if self.rparams.task_sampling_mode == "maximum":
            maximum = max(len(train_dataloader) for train_dataloader in train_dataloader_dict.items())
            sampled_task_ls = list(train_dataloader_dict) * maximum
            random.shuffle(sampled_task_ls)
        else:
            raise KeyError(self.rparams.task_sampling_mode)

        iter_dict = {
            task_name: iter(train_dataloader)
            for task_name, train_dataloader in train_dataloader_dict.items()
        }
        allow_repeat_iter_list = ["maximum"]
        for task_name in sampled_task_ls:
            try:
                yield task_name, next(iter_dict[task_name])
            except StopIteration:
                if self.rparams.task_sampling_mode not in allow_repeat_iter_list:
                    raise
                print(f"Reloading {task_name} dataloader")
                iter_dict[task_name] = iter(train_dataloader_dict[task_name])
                yield task_name, next(iter_dict[task_name])

    def get_train_dataloader_dict(self, train_examples_dict, verbose):
        train_dataloader_dict = {}
        for task_name, train_examples in train_examples_dict.items():
            train_dataloader = self.runner_dict[task_name].get_train_dataloader(train_examples, verbose=False)
            if verbose:
                print(f"{task_name}: {len(train_dataloader)} batches")
            train_dataloader_dict[task_name] = train_dataloader
        return train_dataloader_dict

    def create_task_rparams(self):
        return glue_runners.RunnerParameters(
            max_seq_length=self.rparams.max_seq_length,
            local_rank=self.rparams.local_rank,
            n_gpu=self.rparams.n_gpu,
            fp16=self.rparams.fp16,
            learning_rate=self.rparams.learning_rate,
            gradient_accumulation_steps=self.rparams.gradient_accumulation_steps,
            t_total=self.rparams.t_total,
            warmup_proportion=self.rparams.warmup_proportion,
            num_train_epochs=self.rparams.num_train_epochs,
            train_batch_size=self.rparams.train_batch_size,
            eval_batch_size=self.rparams.eval_batch_size,
        )
