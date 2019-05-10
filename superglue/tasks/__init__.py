from .commitmentbank import CommitmentBankTask
from .copa import CopaTask
from .multirc import MultiRCTask
from .rte import RteTask
from .wic import WicTask
from .wsc import WSCTask


TASK_DICT = {
    "cb": CommitmentBankTask,
    "copa": CopaTask,
    "mrc": MultiRCTask,
    "rte": RteTask,
    "wic": WicTask,
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
