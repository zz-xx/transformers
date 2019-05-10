import os

import torch
from dataclasses import dataclass
from typing import List

from .shared import read_json_lines, Task
from shared.core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap
from shared.constants import CLS, SEP
from pytorch_pretrained_bert.utils import truncate_sequences


@dataclass
class Example(BaseExample):
    guid: str
    sent1: str
    sent2: str
    word: str
    sent1_idx: int
    sent2_idx: int
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            sent1=tokenizer.tokenize(self.sent1),
            sent2=tokenizer.tokenize(self.sent2),
            word=tokenizer.tokenize(self.word),  # might be more than one token
            sent1_idx=self.sent1_idx,
            sent2_idx=self.sent2_idx,
            label_id=WicTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    sent1: List
    sent2: List
    word: List
    sent1_idx: int
    sent2_idx: int
    label_id: int

    def featurize(self, tokenizer, max_seq_length, label_map):
        sent1, sent2 = truncate_sequences(
            tokens_ls=[self.sent1, self.sent2],
            max_length=max_seq_length - len(self.word) - 4,
        )
        tokens = [CLS] + self.word + [SEP] + sent1 + [SEP] + sent2 + [SEP]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # Don't have a choice here -- just leave words as part of sent1
        segment_ids = (
            [0] * (len(sent1) + len(self.word) + 3)
            + [1] * (len(sent2) + 1)
        )
        input_mask = [1] * len(input_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return DataRow(
            guid=self.guid,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            sent1_span=None,
            sent2_span=None,
            label_id=self.label_id,
            tokens=tokens,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: list
    input_mask: list
    segment_ids: list
    sent1_span: list
    sent2_span: list
    label_id: int
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    segment_ids: torch.Tensor
    label_ids: torch.Tensor
    tokens: list

    @classmethod
    def from_data_rows(cls, data_row_ls):
        return Batch(
            input_ids=torch.tensor([f.input_ids for f in data_row_ls], dtype=torch.long),
            input_mask=torch.tensor([f.input_mask for f in data_row_ls], dtype=torch.long),
            segment_ids=torch.tensor([f.segment_ids for f in data_row_ls], dtype=torch.long),
            label_ids=torch.tensor([f.label_id for f in data_row_ls], dtype=torch.long),
            tokens=[f.tokens for f in data_row_ls],
        )


class WicTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    LABELS = ["false", "true"]
    LABEL_BIMAP = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._get_examples(set_type="train", file_name="train.jsonl")

    def get_dev_examples(self):
        return self._get_examples(set_type="dev", file_name="val.jsonl")

    def get_test_examples(self):
        return self._get_examples(set_type="test", file_name="test.jsonl")

    def _get_examples(self, set_type, file_name):
        return self._create_examples(read_json_lines(os.path.join(self.data_dir, file_name)), set_type)

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for line in lines:
            examples.append(Example(
                guid="%s-%s" % (set_type, line["idx"]),
                input_premise=line["premise"],
                input_hypothesis=line["hypothesis"],
                label=line["label"] if set_type != "test" else "contradiction",
            ))
        return examples
