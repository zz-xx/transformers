import argparse
import logging
import os
import pickle

import numpy as np
import tqdm

import torch
from glue.model_setup import load_bert
from glue.tasks import get_task
from pytorch_pretrained_bert.tokenization import BertTokenizer
from shared import model_setup as shared_model_setup

BERT_MODEL_NAME = "bert-large-uncased"
# conf_file = "/Users/thibault/.pytorch_pretrained_bert/"
# os.environ["BERT_ALL_DIR"] = conf_file



def convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer
):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[: (max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
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

        label_id = label_map[example.label]

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_args(*in_args):
    parser = argparse.ArgumentParser()

    # === Required parameters === #
    parser.add_argument(
        "--save_dir",
        required=True,
        type=str,
        help="The directory to save the pickled embeddings",
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        help="The directory where the data is saved",
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Device to compute embeddings"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The task to compute the embeddings for",
    )
    parser.add_argument(
        "--bert_load_path",
        type=str,
        required=True,
        help="The path where the stilts like model is saved",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size to use for conversion",
    )
    args = parser.parse_args(*in_args)
    return args


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def compute_embeddings(model, features, batch_size, device):
    n_examples = len(features)
    total = n_examples // batch_size + 1
    all_emb = []
    for i in tqdm.tqdm(range(0, n_examples, batch_size), total=total):
        feat = features[i : i + batch_size]
        input_ids = torch.tensor([f.input_ids for f in feat], dtype=torch.long).to(device=device)
        input_mask = torch.tensor(
            [f.input_mask for f in feat], dtype=torch.long
        ).to(device=device)
        segment_ids = torch.tensor(
            [f.segment_ids for f in feat], dtype=torch.long
        ).to(device=device)
        _, res = model(input_ids, segment_ids, input_mask, False)
        all_emb.append(res.detach().cpu().numpy())
        del _, res
    all_emb = np.concatenate(all_emb)
    assert len(all_emb) == n_examples
    return all_emb


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    args = get_args()
    task = get_task(args.task_name, args.data_dir)
    tokenizer = BertTokenizer.from_pretrained(
        BERT_MODEL_NAME, do_lower_case=True
    )
    all_state = shared_model_setup.load_overall_state(
        args.bert_load_path, relaxed=True
    )
    logger.info("Initializing model")
    model = load_bert(
        "EXTRACTION", BERT_MODEL_NAME, "state_model_only", all_state, 1
    ).to(args.device)
    model.eval()
    logger.info("Getting task %s", task.name)
    processor = task.processor
    label_list = processor.get_labels()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    splits = [
        ("train", task.get_train_examples),
        ("dev", task.get_dev_examples),
        ("test", task.get_test_examples),
    ]
    for split_name, get_examples_for_split in splits:
        logger.info("Processing %s", split_name)
        examples = get_examples_for_split()
        features = convert_examples_to_features(
            examples, label_list, 128, tokenizer
        )

        emb = compute_embeddings(model, features, args.batch_size, args.device)
        labels = np.array([f.label for f in examples])
        with open(os.path.join(args.save_dir, f"{split_name}.pkl"), "wb") as outfile:
            pickle.dump([emb, labels], outfile)


if __name__ == "__main__":
    main()
