from typing import List

import torch
from torch.utils.data import TensorDataset

from shared.core import DataObject


class ExampleInput(DataObject):
    pass


class ExampleOutput(DataObject):
    pass


class Example(DataObject):
    guid: str
    inputs: ExampleInput
    outputs: ExampleOutput = None


class TokenizedInputs(DataObject):
    pass


class TokenizedOutputs(DataObject):
    pass


class TokenizedExample(DataObject):
    guid: str
    tokenized_inputs: TokenizedInputs
    tokenized_outputs: TokenizedOutputs = None


class SingleInputTextFeature:
    input_ids: List
    input_mask: List
    segment_ids: List


class InputFeature:
    pass


class OutputFeature:
    pass


class Features(DataObject):
    guid: str
    input_features: InputFeature
    output_features: OutputFeature


class Batch(DataObject):
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    segment_ids: torch.Tensor
    label_ids: torch.Tensor
    tokens: List

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, key):
        return self.new(
            input_ids=self.input_ids[key],
            input_mask=self.input_mask[key],
            segment_ids=self.segment_ids[key],
            label_ids=self.label_ids[key],
            tokens=self.tokens[key],
        )




def tokenize_example(example, tokenizer):
    tokens_a = tokenizer.tokenize(example.text_a)
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
    else:
        tokens_b = example.text_b
    return TokenizedExample(
        guid=example.guid,
        tokens_a=tokens_a,
        tokens_b=tokens_b,
        label=example.label,
    )


def convert_example_to_feature(example, tokenizer, max_seq_length, label_map):
    if isinstance(example, InputExample):
        example = tokenize_example(example, tokenizer)

    tokens_a, tokens_b = example.tokens_a, example.tokens_b
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]


    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if is_null_label_map(label_map):
        label_id = example.label
    else:
        label_id = label_map[example.label]
    return InputFeatures(
        guid=example.guid,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        tokens=tokens,
    )


def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer, verbose=True):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for i, example in enumerate(examples):
        feature_instance = convert_example_to_feature(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            label_map=label_map,
        )
        if verbose and i < 3:
            print(str(feature_instance))
        features.append(feature_instance)
    return features


def convert_to_dataset(features, label_mode):
    full_batch = features_to_data(features, label_mode=label_mode)
    if full_batch.label_ids is None:
        dataset = TensorDataset(full_batch.input_ids, full_batch.input_mask,
                                full_batch.segment_ids)
    else:
        dataset = TensorDataset(full_batch.input_ids, full_batch.input_mask,
                                full_batch.segment_ids, full_batch.label_ids)
    return dataset, full_batch.tokens


def features_to_data(features, label_mode):
    if label_mode == LabelModes.CLASSIFICATION:
        label_type = torch.long
    elif label_mode == LabelModes.REGRESSION:
        label_type = torch.float
    else:
        raise KeyError(label_mode)
    return Batch(
        input_ids=torch.tensor([f.input_ids for f in features], dtype=torch.long),
        input_mask=torch.tensor([f.input_mask for f in features], dtype=torch.long),
        segment_ids=torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        label_ids=torch.tensor([f.label_id for f in features], dtype=label_type),
        tokens=[f.tokens for f in features],
    )


class HybridLoader:
    def __init__(self, dataloader, tokens):
        self.dataloader = dataloader
        self.tokens = tokens

    def __iter__(self):
        batch_size = self.dataloader.batch_size
        for i, batch in enumerate(self.dataloader):
            if len(batch) == 4:
                input_ids, input_mask, segment_ids, label_ids = batch
            elif len(batch) == 3:
                input_ids, input_mask, segment_ids = batch
                label_ids = None
            else:
                raise RuntimeError()
            batch_tokens = self.tokens[i * batch_size: (i+1) * batch_size]
            yield Batch(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                tokens=batch_tokens,
            )

    def __len__(self):
        return len(self.dataloader)
