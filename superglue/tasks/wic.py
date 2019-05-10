import os

import torch
from dataclasses import dataclass
from typing import List

from .shared import read_json_lines, Task
from shared.core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap
from shared.constants import CLS, SEP
from shared.utils import convert_word_idx_for_bert_tokens
from pytorch_pretrained_bert.utils import truncate_sequences, pad_to_max_seq_length


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
        sent1_tokens = tokenizer.tokenize(self.sent1)
        sent2_tokens = tokenizer.tokenize(self.sent2)
        sent1_span = convert_word_idx_for_bert_tokens(
            text=self.sent1,
            bert_tokens=sent1_tokens,
            word_idx_ls=[self.sent1_idx],
            check=False,
        )[0]
        sent2_span = convert_word_idx_for_bert_tokens(
            text=self.sent2,
            bert_tokens=sent2_tokens,
            word_idx_ls=[self.sent2_idx],
            check=False,
        )[0]
        return TokenizedExample(
            guid=self.guid,
            sent1_tokens=tokenizer.tokenize(self.sent1),
            sent2_tokens=tokenizer.tokenize(self.sent2),
            word=tokenizer.tokenize(self.word),  # might be more than one token
            sent1_span=sent1_span,
            sent2_span=sent2_span,
            label_id=WiCTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    sent1_tokens: List
    sent2_tokens: List
    word: List
    sent1_span: List
    sent2_span: List
    label_id: int

    def featurize(self, tokenizer, max_seq_length):
        sent1_tokens, sent2_tokens = truncate_sequences(
            tokens_ls=[self.sent1_tokens, self.sent2_tokens],
            max_length=max_seq_length - len(self.word) - 4,
        )
        tokens = [CLS] + self.word + [SEP] + sent1_tokens + [SEP] + sent2_tokens + [SEP]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # Don't have a choice here -- just leave words as part of sent1
        segment_ids = (
            [0] * (len(sent1_tokens) + len(self.word) + 3)
            + [1] * (len(sent2_tokens) + 1)
        )
        input_mask = [1] * len(input_ids)
        sent1_span = [
            self.sent1_span[0] + 2 + len(self.word),
            self.sent1_span[1] + 2 + len(self.word),
        ]
        sent2_span = [
            self.sent2_span[0] + 3 + len(self.word) + len(sent1_tokens),
            self.sent2_span[1] + 3 + len(self.word) + len(sent1_tokens),
        ]

        return DataRow(
            guid=self.guid,
            input_ids=pad_to_max_seq_length(input_ids, max_seq_length),
            input_mask=pad_to_max_seq_length(input_mask, max_seq_length),
            segment_ids=pad_to_max_seq_length(segment_ids, max_seq_length),
            sent1_span=sent1_span,
            sent2_span=sent2_span,
            label_id=self.label_id,
            tokens=tokens,
            word=self.word,
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
    word: List


@dataclass
class Batch(BatchMixin):
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    segment_ids: torch.Tensor
    sent1_span: torch.Tensor
    sent2_span: torch.Tensor
    label_ids: torch.Tensor
    tokens: list
    word: list

    @classmethod
    def from_data_rows(cls, data_row_ls):
        return Batch(
            input_ids=torch.tensor([f.input_ids for f in data_row_ls], dtype=torch.long),
            input_mask=torch.tensor([f.input_mask for f in data_row_ls], dtype=torch.long),
            segment_ids=torch.tensor([f.segment_ids for f in data_row_ls], dtype=torch.long),
            sent1_span=torch.tensor([f.sent1_span for f in data_row_ls], dtype=torch.long),
            sent2_span=torch.tensor([f.sent2_span for f in data_row_ls], dtype=torch.long),
            label_ids=torch.tensor([f.label_id for f in data_row_ls], dtype=torch.long),
            tokens=[f.tokens for f in data_row_ls],
            word=[f.word for f in data_row_ls],
        )


class WiCTask(Task):
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
                sent1=line["sentence1"],
                sent2=line["sentence2"],
                word=line["word"],
                sent1_idx=int(line["sentence1_idx"]),
                sent2_idx=int(line["sentence2_idx"]),
                label=line["label"] if set_type != "test" else cls.LABELS[-1],
            ))
        return examples
