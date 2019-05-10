from .commitmentbank import CommitmentBankTask
from .rte import RteTask


PROCESSORS = {
    "cb": CommitmentBankTask,
    "rte": RteTask,
}

DEFAULT_FOLDER_NAMES = {
    "cb": "CB",
    "rte": "RTE",
}
