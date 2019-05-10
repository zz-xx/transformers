import bs4
import os

import torch
from dataclasses import dataclass
from typing import List

from .shared import read_json_lines, Task
from shared.core import BaseExample, BaseTokenizedExample, BaseDataRow, BatchMixin, labels_to_bimap
from shared.constants import CLS, SEP
from pytorch_pretrained_bert.utils import truncate_sequences, pad_to_max_seq_length


@dataclass
class Example(BaseExample):
    guid: str
    paragraph: str
    question: str
    answer: str
    label: str

    def tokenize(self, tokenizer):
        return TokenizedExample(
            guid=self.guid,
            paragraph=tokenizer.tokenize(self.paragraph),
            question=tokenizer.tokenize(self.question),
            answer=tokenizer.tokenize(self.answer),
            label_id=MultiRCTask.LABEL_BIMAP.a[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    paragraph: List
    question: List
    answer: List
    label_id: int

    def featurize(self, tokenizer, max_seq_length):
        paragraph = truncate_sequences(
            tokens_ls=[self.paragraph],
            max_length=max_seq_length - 4 - len(self.question) - len(self.answer),
        )[0]
        tokens = [CLS] + paragraph + [SEP] + self.question + [SEP] + self.answer + [SEP]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = (
            [0] * (len(paragraph) + 2)
            + [1] * (len(self.question) + len(self.answer) + 2)
        )
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


class MultiRCTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    LABELS = [True, False]
    LABEL_BIMAP = labels_to_bimap(LABELS)

    def __init__(self, name, data_dir, filter_sentences=True):
        super().__init__(name=name, data_dir=data_dir)
        self.name = name
        self.data_dir = data_dir
        self.filter_sentences = filter_sentences

    def get_train_examples(self):
        return self._get_examples(set_type="train", file_name="train.jsonl")

    def get_dev_examples(self):
        return self._get_examples(set_type="dev", file_name="val.jsonl")

    def get_test_examples(self):
        return self._get_examples(set_type="test", file_name="test.jsonl")

    def _get_examples(self, set_type, file_name):
        return self._create_examples(read_json_lines(os.path.join(self.data_dir, file_name)), set_type)

    def _create_examples(self, lines, set_type):
        examples = []
        for line in lines:
            soup = bs4.BeautifulSoup(line["paragraph"]["text"], features="lxml")
            sentence_ls = []
            for i, elem in enumerate(soup.html.body.contents):
                if isinstance(elem, bs4.element.NavigableString):
                    sentence_ls.append(str(elem).strip())

            for question_dict in line["paragraph"]["questions"]:
                question = question_dict["question"]
                if self.filter_sentences:
                    paragraph = " ".join(
                        sentence
                        for i, sentence in enumerate(sentence_ls, start=1)
                        if i in question_dict["sentences_used"]
                    )
                else:
                    paragraph = " ".join(sentence_ls)
                for answer_dict in question_dict["answers"]:
                    answer = answer_dict["text"]
                    examples.append(Example(
                        guid="%s-%s" % (set_type, line["idx"]),
                        paragraph=paragraph,
                        question=question,
                        answer=answer,
                        label=answer_dict["isAnswer"] if set_type != "test" else self.LABELS[-1],
                    ))
        return examples
