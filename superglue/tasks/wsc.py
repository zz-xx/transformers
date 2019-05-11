import os

import torch
from dataclasses import dataclass
from typing import List

from .shared import read_json_lines, Task
from shared.core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap
from shared.constants import CLS, SEP
from shared.utils import convert_char_span_for_bert_tokens
from pytorch_pretrained_bert.utils import truncate_sequences, pad_to_max_seq_length


@dataclass
class Example(BaseExample):
    guid: str
    text: str
    span1_idx: int
    span2_idx: int
    span1_text: str
    span2_text: str
    label: str

    def tokenize(self, tokenizer):
        text_tokens = self.text.split()
        bert_tokens = tokenizer.tokenize(self.text)
        if self.span1_idx == 0:
            span1_start_idx = 0
        else:
            span1_start_idx = len(" ".join(text_tokens[:self.span1_idx]) + " ")
        if self.span2_idx == 0:
            span2_start_idx = 0
        else:
            span2_start_idx = len(" ".join(text_tokens[:self.span2_idx]) + " ")
        span1_span = convert_char_span_for_bert_tokens(
            text=self.text,
            bert_tokens=bert_tokens,
            span_ls=[[span1_start_idx, self.span1_text]],
            check=False,
        )[0]
        span2_span = convert_char_span_for_bert_tokens(
            text=self.text,
            bert_tokens=bert_tokens,
            span_ls=[[span2_start_idx, self.span2_text]],
            check=False,
        )[0]
        return TokenizedExample(
            guid=self.guid,
            tokens=bert_tokens,
            span1_span=span1_span,
            span2_span=span2_span,
            span1_text=self.span1_text,
            span2_text=self.span2_text,
            label_id=WSCTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    tokens: List
    span1_span: List
    span2_span: List
    span1_text: str
    span2_text: str
    label_id: int

    def featurize(self, tokenizer, max_seq_length):
        tokens = truncate_sequences(
            tokens_ls=[self.tokens],
            max_length=max_seq_length - 2,
        )[0]
        tokens = [CLS] + tokens + [SEP]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # Don't have a choice here -- just leave words as part of sent1
        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(input_ids)
        span1_span = [
            self.span1_span[0] + 1,
            self.span1_span[1] + 1 - 1,
        ]
        span2_span = [
            self.span2_span[0] + 1,
            self.span2_span[1] + 1 - 1,
        ]

        return DataRow(
            guid=self.guid,
            input_ids=pad_to_max_seq_length(input_ids, max_seq_length),
            input_mask=pad_to_max_seq_length(input_mask, max_seq_length),
            segment_ids=pad_to_max_seq_length(segment_ids, max_seq_length),
            span1_span=span1_span,
            span2_span=span2_span,
            label_id=self.label_id,
            tokens=tokens,
            span1_text=self.span1_text,
            span2_text=self.span2_text,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: list
    input_mask: list
    segment_ids: list
    span1_span: List
    span2_span: List
    label_id: int
    tokens: list
    span1_text: str
    span2_text: str

    def get_tokens(self):
        return [self.tokens]


@dataclass
class Batch(BatchMixin):
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    segment_ids: torch.Tensor
    span1_span: torch.Tensor
    span2_span: torch.Tensor
    label_ids: torch.Tensor
    tokens: list
    span1_text: list
    span2_text: list

    @classmethod
    def from_data_rows(cls, data_row_ls):
        return Batch(
            input_ids=torch.tensor([f.input_ids for f in data_row_ls], dtype=torch.long),
            input_mask=torch.tensor([f.input_mask for f in data_row_ls], dtype=torch.long),
            segment_ids=torch.tensor([f.segment_ids for f in data_row_ls], dtype=torch.long),
            span1_span=torch.tensor([f.span1_span for f in data_row_ls], dtype=torch.long),
            span2_span=torch.tensor([f.span2_span for f in data_row_ls], dtype=torch.long),
            label_ids=torch.tensor([f.label_id for f in data_row_ls], dtype=torch.long),
            tokens=[f.tokens for f in data_row_ls],
            span1_text=[f.span1_text for f in data_row_ls],
            span2_text=[f.span2_text for f in data_row_ls],
        )


class WSCTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    LABELS = [False, True]
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
                text=line["text"],
                span1_idx=line["target"]["span1_index"],
                span2_idx=line["target"]["span2_index"],
                span1_text=line["target"]["span1_text"],
                span2_text=line["target"]["span2_text"],
                label=line["label"] if set_type != "test" else cls.LABELS[-1],
            ))
        return examples
