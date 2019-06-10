import os
import pandas as pd

import torch
from dataclasses import dataclass
from typing import List

from .shared import Task
from shared.core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap
from shared.constants import CLS, SEP
from pytorch_pretrained_bert.utils import truncate_sequences, pad_to_max_seq_length


@dataclass
class Example(BaseExample):
    guid: str
    input_text: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            input_tokens=tokenizer.tokenize(self.input_text),
            label_id=YelpPolarityTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    input_tokens: List
    label_id: int

    def featurize(self, tokenizer, max_seq_length):
        input_tokens, = truncate_sequences(
            tokens_ls=[self.input_tokens],
            max_length=max_seq_length - 3,
        )
        tokens = [CLS] + input_tokens + [SEP]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(input_tokens) + 2)
        input_mask = [1] * len(input_ids)

        return DataRow(
            guid=self.guid,
            input_ids=pad_to_max_seq_length(input_ids, max_seq_length),
            input_mask=pad_to_max_seq_length(input_mask, max_seq_length),
            segment_ids=pad_to_max_seq_length(segment_ids, max_seq_length),
            label_id=self.label_id,
            tokens=tokens,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: list
    input_mask: list
    segment_ids: list
    label_id: int
    tokens: list

    def get_tokens(self):
        return [self.tokens]


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


class YelpPolarityTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    LABELS = [1, 2]
    LABEL_BIMAP = labels_to_bimap(LABELS)

    def get_train_examples(self):
        df = pd.read_csv(
            os.path.join(self.data_dir, "train.csv"),
            header=None,
            names=["label", "text"],
        )
        examples = [
            Example(
                guid=f"train-{i}",
                input_text=row["text"],
                label=row["label"],
            )
            for i, row in df.iterrows()
        ]
        return examples

    def get_dev_examples(self):
        raise NotImplementedError()

    def get_test_examples(self):
        raise NotImplementedError()
