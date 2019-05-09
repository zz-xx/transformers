import numpy as np
import os
import pandas as pd
import tqdm

# TODO: Fix horizontal import
from glue.tasks import DataProcessor, Task, TaskType
from moretasks.core.gap_core import InputExample


class GapProcessor(DataProcessor):
    """Processor for the GAP data set"""

    SPAN_KEYS = ["Pronoun", "A", "B"]
    LABEL_DICT = {
        (True, False): "a",
        (False, True): "b",
        (False, False): "neither",
    }
    TASK_TYPE = TaskType.SPAN_CHOICE

    DEFAULT_VERBOSE = True

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_train_examples(self, data_dir, verbose=DEFAULT_VERBOSE):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "gap-development.tsv"), "train", verbose)

    def get_dev_examples(self, data_dir, verbose=DEFAULT_VERBOSE):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "gap-validation.tsv"), "dev", verbose)

    def get_test_examples(self, data_dir, verbose=DEFAULT_VERBOSE):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, "gap-test.tsv"), "test", verbose)

    def get_labels(self):
        """See base class."""
        return ["a", "b", "neither"]

    def _create_examples(self, path, set_type, verbose=True):
        """Creates examples for the training and dev sets."""
        df = pd.read_csv(path, sep="\t")
        examples = []
        iterable = df.iterrows() if not verbose else tqdm.tqdm(df.iterrows(), total=len(df))
        for i, datum in iterable:
            datum = self.process_datum(
                datum=datum.to_dict(),
                tokenizer=self.tokenizer,
                check=True,
            )
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                datum["label"] = None
            examples.append(InputExample(
                guid=guid,
                text=datum["Text"],
                span_pronoun=datum["Pronoun"],
                span_a=datum["A"],
                span_b=datum["B"],
                label=datum["label"],
            ))
        return examples

    @classmethod
    def process_datum(cls, datum, tokenizer, check=True):
        new_datum = {}
        text = new_datum["Text"] = datum["Text"]
        bert_tokens = tokenizer.tokenize(text)
        bert_postsum = np.cumsum([
            len(token.replace("##", ""))
            for token in bert_tokens
        ])
        for span_key in cls.SPAN_KEYS:
            before = text[:datum[f"{span_key}-offset"]]
            chars_before = len(before.replace(" ", ""))
            span_chars = len(datum[span_key].replace(" ", ""))
            if chars_before == 0:
                start_idx = 0
            else:
                start_idx = np.argmax(bert_postsum == chars_before) + 1
            end_idx = np.argmax(bert_postsum == chars_before + span_chars) + 1  # inclusive
            new_datum[span_key] = [start_idx, end_idx]  # json compatibility

            if check:
                bert_chars_str = "".join(bert_tokens[start_idx:end_idx]).replace("##", "")
                span_chars_str = datum[span_key].replace(" ", "")
                assert bert_chars_str.lower() == span_chars_str.lower()
                assert bert_postsum[-1] == len(text.replace(" ", ""))
        new_datum["label"] = cls.LABEL_DICT[datum["A-coref"], datum["B-coref"]]
        return new_datum


def get_task(task_name, tokenizer, data_dir):
    return Task(
        name=task_name,
        processor=GapProcessor(tokenizer),
        data_dir=data_dir,
    )
