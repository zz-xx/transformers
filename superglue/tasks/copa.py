import os

import torch
from dataclasses import dataclass
from typing import List

from .shared import read_json_lines, Task
from shared.core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin
from shared.constants import CLS, SEP
from pytorch_pretrained_bert.utils import truncate_sequences, pad_to_max_seq_length


@dataclass
class Example(BaseExample):
    guid: str
    input_premise: str
    input_choice1: str
    input_choice2: str
    question: str
    label_id: int

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            input_premise=tokenizer.tokenize(self.input_premise),
            input_choice1=tokenizer.tokenize(self.input_choice1),
            input_choice2=tokenizer.tokenize(self.input_choice2),
            # Safe assumption that question is a single word
            question=tokenizer.tokenize(self.question)[0],
            label_id=self.label_id,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    input_premise: List
    input_choice1: List
    input_choice2: List
    question: str  # Safe assumption that question is a single word
    label_id: int

    def featurize(self, tokenizer, max_seq_length):
        input_premise, input_choice1 = truncate_sequences(
            tokens_ls=[self.input_premise, self.input_choice1],
            max_length=max_seq_length - 5,
        )
        input_premise, input_choice2 = truncate_sequences(
            tokens_ls=[self.input_premise, self.input_choice2],
            max_length=max_seq_length - 5,
        )
        tokens1 = [CLS] + [self.question] + [SEP] + input_premise + [SEP] + input_choice1 + [SEP]
        tokens2 = [CLS] + [self.question] + [SEP] + input_premise + [SEP] + input_choice2 + [SEP]

        input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)
        segment_ids1 = [0] * (len(input_premise) + 4) + [1] * (len(input_choice1) + 1)
        input_mask1 = [1] * len(input_ids1)

        input_ids2 = tokenizer.convert_tokens_to_ids(tokens2)
        segment_ids2 = [0] * (len(input_premise) + 4) + [1] * (len(input_choice2) + 1)
        input_mask2 = [1] * len(input_ids2)

        return DataRow(
            guid=self.guid,
            input_ids1=pad_to_max_seq_length(input_ids1, max_seq_length),
            input_mask1=pad_to_max_seq_length(input_mask1, max_seq_length),
            segment_ids1=pad_to_max_seq_length(segment_ids1, max_seq_length),
            input_ids2=pad_to_max_seq_length(input_ids2, max_seq_length),
            input_mask2=pad_to_max_seq_length(input_mask2, max_seq_length),
            segment_ids2=pad_to_max_seq_length(segment_ids2, max_seq_length),
            label_id=self.label_id,
            tokens1=tokens1,
            tokens2=tokens2,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids1: list
    input_mask1: list
    segment_ids1: list
    input_ids2: list
    input_mask2: list
    segment_ids2: list
    label_id: int
    tokens1: list
    tokens2: list


@dataclass
class Batch(BatchMixin):
    input_ids1: torch.Tensor
    input_mask1: torch.Tensor
    segment_ids1: torch.Tensor
    input_ids2: torch.Tensor
    input_mask2: torch.Tensor
    segment_ids2: torch.Tensor
    label_ids: torch.Tensor
    tokens1: list
    tokens2: list

    @classmethod
    def from_data_rows(cls, data_row_ls):
        return Batch(
            input_ids1=torch.tensor([f.input_ids1 for f in data_row_ls], dtype=torch.long),
            input_mask1=torch.tensor([f.input_mask1 for f in data_row_ls], dtype=torch.long),
            segment_ids1=torch.tensor([f.segment_ids1 for f in data_row_ls], dtype=torch.long),
            input_ids2=torch.tensor([f.input_ids2 for f in data_row_ls], dtype=torch.long),
            input_mask2=torch.tensor([f.input_mask2 for f in data_row_ls], dtype=torch.long),
            segment_ids2=torch.tensor([f.segment_ids2 for f in data_row_ls], dtype=torch.long),
            label_ids=torch.tensor([f.label_id for f in data_row_ls], dtype=torch.long),
            tokens1=[f.tokens1 for f in data_row_ls],
            tokens2=[f.tokens2 for f in data_row_ls],
        )


class CopaTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    LABELS = [0, 1]

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
                input_choice1=line["choice1"],
                input_choice2=line["choice2"],
                question=line["question"],
                label_id=line["label"] if set_type != "test" else cls.LABELS[-1],
            ))
        return examples
