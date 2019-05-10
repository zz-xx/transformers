from typing import NamedTuple

import torch


class IDS:
    UNK = 100
    CLS = 101
    SEP = 102
    MASK = 103


class ExtendedDataClassMixin:

    def asdict(self):
        return {
            k: getattr(self, k)
            for k in self.__dataclass_fields__
        }

    def new(self, **new_kwargs):
        kwargs = {
            k: v
            for k, v in self.asdict()
        }
        for k, v in new_kwargs.items():
            kwargs[k] = v
        return self.__class__(**kwargs)


class BatchMixin(NamedTuple):
    def to(self, device):
        return self.__class__({
            k: self._val_to_device(v, device)
            for k, v in self._asdict()
        })

    @classmethod
    def _val_to_device(cls, v, device):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        else:
            return v

    def __len__(self):
        return len(getattr(self, self._fields[0]))


class BaseExample(ExtendedDataClassMixin):
    def tokenize(self, tokenizer):
        raise NotImplementedError


class BaseTokenizedExample(ExtendedDataClassMixin):
    def featurize(self, tokenizer, max_seq_length, label_map):
        raise NotImplementedError


class BaseDataRow(ExtendedDataClassMixin):
    pass


class BaseBatch(BatchMixin):
    @classmethod
    def from_data_rows(cls, data_row_ls):
        raise NotImplementedError


class BiMap:
    def __init__(self, a, b):
        self.a = {}
        self.b = {}
        for i, j in zip(a, b):
            self.a[i] = j
            self.b[j] = i
        assert len(self.a) == len(self.b) == len(a) == len(b)


def labels_to_bimap(labels):
    return BiMap(a=labels, b=list(range(len(labels))))
