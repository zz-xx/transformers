import logging

import torch
from torch.utils.data import TensorDataset

from ..core.gap_core import InputExample, TokenizedExample, InputFeatures, Batch

logger = logging.getLogger(__name__)


def tokenize_example(example, tokenizer):
    tokens = tokenizer.tokenize(example.text)
    return TokenizedExample(
        guid=example.guid,
        tokens=tokens,
        span_pronoun=example.span_pronoun,
        span_a=example.span_a,
        span_b=example.span_b,
        label=example.label,
    )


def convert_example_to_feature(example, tokenizer, max_seq_length, label_map):
    if isinstance(example, InputExample):
        example = tokenize_example(example, tokenizer)

    tokens = ["[CLS]"] + example.tokens + ["[SEP]"]
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]
    segment_ids = [0] * len(tokens)

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

    label_id = label_map[example.label]

    return InputFeatures(
        guid=example.guid,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        span_pronoun=example.span_pronoun,
        span_a=example.span_a,
        span_b=example.span_b,
        label_id=label_id,
        tokens=tokens,
    )


def convert_examples_to_features(examples, label_map, max_seq_length, tokenizer, verbose=True):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        feature_instance = convert_example_to_feature(
            example=example,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            label_map=label_map,
        )
        if verbose and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in feature_instance.tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in feature_instance.input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in feature_instance.input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in feature_instance.segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, feature_instance.label_id))

        features.append(feature_instance)
    return features


def convert_to_dataset(features):
    full_batch = features_to_data(features)
    if full_batch.label_ids is None:
        dataset = TensorDataset(full_batch.input_ids, full_batch.input_mask,
                                full_batch.segment_ids,
                                full_batch.span_pronoun, full_batch.span_a,
                                full_batch.span_b)
    else:
        dataset = TensorDataset(full_batch.input_ids, full_batch.input_mask,
                                full_batch.segment_ids,
                                full_batch.span_pronoun, full_batch.span_a,
                                full_batch.span_b,
                                full_batch.label_ids)
    return dataset, full_batch.tokens


def features_to_data(features):
    return Batch(
        input_ids=torch.tensor([f.input_ids for f in features], dtype=torch.long),
        input_mask=torch.tensor([f.input_mask for f in features], dtype=torch.long),
        segment_ids=torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        span_pronoun=torch.tensor([f.segment_ids for f in features], dtype=torch.long),
        span_a=torch.tensor([f.span_a for f in features], dtype=torch.long),
        span_b=torch.tensor([f.span_b for f in features], dtype=torch.long),
        label_ids=torch.tensor([f.label_id for f in features], dtype=torch.long),
        tokens=[f.tokens for f in features],
    )


class HybridLoader:
    def __init__(self, dataloader, tokens):
        self.dataloader = dataloader
        self.tokens = tokens

    def __iter__(self):
        batch_size = self.dataloader.batch_size
        for i, batch in enumerate(self.dataloader):
            if len(batch) == 7:
                input_ids, input_mask, segment_ids, \
                    span_pronoun, span_a, span_b, label_ids = batch
            elif len(batch) == 6:
                input_ids, input_mask, segment_ids, \
                    span_pronoun, span_a, span_b = batch
                label_ids = None
            else:
                raise RuntimeError()
            batch_tokens = self.tokens[i * batch_size: (i+1) * batch_size]
            yield Batch(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                span_pronoun=span_pronoun,
                span_a=span_a,
                span_b=span_b,
                label_ids=label_ids,
                tokens=batch_tokens,
            )

    def __len__(self):
        return len(self.dataloader)
