import os

import torch
from dataclasses import dataclass
from typing import List

from .shared import read_tsv, Task
from shared.core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap
from shared.constants import CLS, SEP
from pytorch_pretrained_bert.utils import truncate_sequences


@dataclass
class Example(BaseExample):
    guid: str
    input_premise: str
    input_hypothesis: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            input_premise=tokenizer.tokenize(self.input_premise),
            input_hypothesis=tokenizer.tokenize(self.input_hypothesis),
            label_id=RteTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    input_premise: List
    input_hypothesis: List
    label_id: int

    def featurize(self, tokenizer, max_seq_length, label_map):
        input_premise, input_hypothesis = truncate_sequences(
            tokens_ls=[self.input_premise, self.input_hypothesis],
            max_length=max_seq_length - 3,
        )
        tokens = [CLS] + input_premise + [SEP] + input_hypothesis + [SEP]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(input_premise) + 2) + [1] * (len(input_hypothesis) + 1)
        input_mask = [1] * len(input_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        return DataRow(
            guid=self.guid,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
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


class RteTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    LABELS = ["entailment", "not_entailment"]
    LABEL_BIMAP = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._get_examples(set_type="train", file_name="train.tsv")

    def get_dev_examples(self):
        return self._get_examples(set_type="dev", file_name="dev.tsv")

    def get_test_examples(self):
        return self._get_examples(set_type="test", file_name="test.tsv")

    def _get_examples(self, set_type, file_name):
        return self._create_examples(read_tsv(os.path.join(self.data_dir, file_name)), set_type)

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            premise = line[1]
            hypothesis = line[2]
            if set_type == "test":
                label = "entailment"
            else:
                label = line[-1]
            examples.append(Example(
                guid=guid,
                input_premise=premise,
                input_hypothesis=hypothesis,
                label=label,
            ))
        return examples
