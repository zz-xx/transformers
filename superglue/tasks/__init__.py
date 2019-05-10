import os

from .commitmentbank import CommitmentBankTask
from .copa import CopaTask
from .multirc import MultiRCTask
from .rte import RteTask
from .wic import WiCTask
from .wsc import WSCTask


TASK_DICT = {
    "cb": CommitmentBankTask,
    "copa": CopaTask,
    "mrc": MultiRCTask,
    "rte": RteTask,
    "wic": WiCTask,
    "wsc": WSCTask,
}

DEFAULT_FOLDER_NAMES = {
    "cb": "CB",
    "copa": "COPA",
    "mrc": "MultiRC",
    "rte": "RTE",
    "wic": "WiC",
    "wsc": "WSC",
}


def get_task(task_name, data_dir):
    task_name = task_name.lower()
    task_class = TASK_DICT[task_name]
    if data_dir is None:
        data_dir = os.path.join(os.environ["SUPERGLUE_DIR"], DEFAULT_FOLDER_NAMES[task_name])
    return task_class(task_name, data_dir)
