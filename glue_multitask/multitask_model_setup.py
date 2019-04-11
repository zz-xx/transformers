import torch.nn as nn

from glue.tasks import get_task

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from shared.model_setup import stage_model
from glue.model_setup import load_bert, create_from_pretrained, load_bert_adapter
from pytorch_pretrained_bert.modeling import sync_bert
from pytorch_pretrained_bert.utils import random_sample


def get_task_ls(task_name_ls, data_dir):
    return [get_task(task_name, data_dir) for task_name in task_name_ls.split(",")]


def get_train_examples_dict(task_ls, train_examples_number_ls):
    train_examples_dict = {}
    for task, train_examples_number in zip(task_ls, train_examples_number_ls):
        train_examples = task.get_train_examples()
        if train_examples_number is not None:
            train_examples = random_sample(train_examples, train_examples_number)
        train_examples_dict[task.name] = train_examples
    return train_examples_dict


def get_val_examples_dict(task_ls):
    return {
        task.name: task.get_dev_examples()
        for task in task_ls
    }


def get_test_examples_dict(task_ls):
    return {
        task.name: task.get_test_examples()
        for task in task_ls
    }


def get_train_examples_number_ls(args_train_examples_number, num_tasks):
    if args_train_examples_number is None:
        return [None] * num_tasks
    train_examples_number_ls = list(map(int, args_train_examples_number.split(",")))
    if len(train_examples_number_ls) == 1:
        return [train_examples_number_ls[0]] * num_tasks
    else:
        return train_examples_number_ls


def create_model(task_ls, bert_model_name, bert_load_mode, bert_load_args,
                 all_state,
                 device, n_gpu, fp16, local_rank,
                 bert_config_json_path=None):
    model_dict = {}
    for task in task_ls:
        if bert_load_mode == "from_pretrained":
            assert bert_load_args is None
            assert all_state is None
            assert bert_config_json_path is None
            cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(local_rank)
            model = create_from_pretrained(
                task_type=task.processor.TASK_TYPE,
                bert_model_name=bert_model_name,
                cache_dir=cache_dir,
                num_labels=len(task.processor.get_labels()),
            )
        elif bert_load_mode in ["model_only", "state_model_only", "state_all", "state_full_model",
                                "full_model_only"]:
            assert bert_load_args is None
            model = load_bert(
                task_type=task.processor.TASK_TYPE,
                bert_model_name=bert_model_name,
                bert_load_mode=bert_load_mode,
                all_state=all_state,
                num_labels=len(task.processor.get_labels()),
                bert_config_json_path=bert_config_json_path,
            )
        elif bert_load_mode in ["state_adapter"]:
            model = load_bert_adapter(
                task_type=task.processor.TASK_TYPE,
                bert_model_name=bert_model_name,
                bert_load_mode=bert_load_mode,
                bert_load_args=bert_load_args,
                all_state=all_state,
                num_labels=len(task.processor.get_labels()),
                bert_config_json_path=bert_config_json_path,
            )
        else:
            raise KeyError(bert_load_mode)
        model = stage_model(model, fp16=fp16, device=device, local_rank=local_rank, n_gpu=n_gpu)
        model_dict[task.name] = model

    return MultiTaskGlueModel(model_dict)


class MultiTaskGlueModel(nn.Module):
    def __init__(self, model_dict):
        super().__init__()
        self.model_dict = nn.ModuleDict(model_dict)
        sync_bert(list(model_dict.values()))

    def forward(self, *input):
        raise NotImplementedError("Call individual model")

    def __getitem__(self, item):
        # item = task_name
        return self.model_dict[item]
