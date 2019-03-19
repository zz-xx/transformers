import copy

import numpy as np
import spacy
import tqdm

import torch
from language_modeling.runners import (InputExample, convert_example_to_features,
                                       features_to_data, tokenize_example)

nlp = spacy.load("en_core_web_sm")

BLACKLIST = [
    "he",
    "she",
    "it",
    "they",
    "who",
    "her",
    "we",
    "them",
    "him",
    "his",
    "their",
    "hers",
    "his",
    "theirs",
    "i",
    "me",
    "you",
    "us",
]
MASK = "[MASK]"
max_sequence_length = 128


def get_pos(sent):
    return [token.pos_ for token in nlp(sent)]


def is_noun(pos):
    return pos in ["PRON", "PROPN", "NOUN"]


def get_pos_dict(sent):
    return {token.text.lower(): token.pos_ for token in nlp(sent)}


def filter_pos_dict(pos_dict):
    return {
        noun: pos
        for noun, pos in pos_dict.items()
        if is_noun(pos) and noun not in BLACKLIST
    }


def get_token_groups(tokenizer, words):
    return [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in words]


def get_pos_dict_and_token_groups(tokenizer, text):
    pos_dict = filter_pos_dict(get_pos_dict(text))
    tok_groups = get_token_groups(tokenizer, pos_dict.keys())
    return pos_dict, tok_groups


def get_length_info(tok_a, tok_b):
    sent_a_lengths = set(len(a_i) for a_i in tok_a)
    sent_b_lengths = set(len(b_i) for b_i in tok_b)
    sent_a_one_length = len(sent_a_lengths) == 1
    sent_b_one_length = len(sent_b_lengths) == 1
    same_length = sent_a_lengths == sent_b_lengths
    return sent_a_one_length, sent_b_one_length, same_length


def mask_predict_one_example(tokenizer, model, row, device):
    alt_ls = []
    pred_result = True
    example = InputExample(
        guid=0, text_a=row["sentence1"], text_b=row["sentence2"], is_next=True
    )

    tokens_a_pos_dict, tokens_a_token_groups = get_pos_dict_and_token_groups(
        tokenizer, example.text_a
    )
    tokens_b_pos_dict, tokens_b_token_groups = get_pos_dict_and_token_groups(
        tokenizer, example.text_b
    )

    tokenized_example = tokenize_example(example, tokenizer)

    b_ids = tokenizer.convert_tokens_to_ids(tokenized_example.tokens_b)

    sent_a_one_length, sent_b_one_length, same_length = get_length_info(
        tokens_a_token_groups, tokens_b_token_groups
    )
    #         if sent_a_one_length and sent_b_one_length and same_length:
    #             pred_ls.append(False)
    #             all_alt_ls.append(alt_ls)
    #             continue

    for i, tok_id in enumerate(b_ids):
        if not any(tok_id == tok_group[0] for tok_group in tokens_b_token_groups):
            continue
        else:
            to_mask = max(
                len(tok_group)
                for tok_group in tokens_b_token_groups
                if tok_id == tok_group[0]
            )

        tokenized_example_changed = copy.deepcopy(tokenized_example)
        for idx in range(i, i + to_mask):
            tokenized_example_changed.tokens_b[idx] = MASK

        features = convert_example_to_features(
            tokenized_example_changed, tokenizer, max_sequence_length, select_prob=0.0
        )

        batch = features_to_data([features]).to(device)
        with torch.no_grad():
            result = model(batch.input_ids, batch.segment_ids, batch.input_mask)

        masked_indices = np.arange(batch.input_ids.shape[1])[
            batch.input_ids[0].cpu().numpy() == 103
        ]
        first_masked_index = masked_indices[0]

        preds_first_token = result[0][0, first_masked_index, :].cpu().numpy()
        possible_first_tokens = [tok_id]
        # Possible first tokens for b:
        # Tokens in a that are not in b once masked (no repeats)
        # Though this does not seem to be beneficial so disabling for now
        # We're already overpredicting True...
        remaining_b_tok_ids = tokenizer.convert_tokens_to_ids(
            tokenized_example_changed.tokens_b
        )
        possible_first_tokens += [
            tok_id_a[0]
            for tok_id_a, tok_a in zip(tokens_a_token_groups, tokens_a_pos_dict)
            if tok_a not in tokenized_example_changed.tokens_b and tok_a[0] != tok_id
        ]  # This handles weird cases with plurals...

        kept_preds_first_token = preds_first_token[possible_first_tokens]

        most_predicted = np.argmax(kept_preds_first_token)
        # If different than 0, than we're predicting something else....
        actually_predicted = tokenizer.ids_to_tokens[
            possible_first_tokens[most_predicted]
        ]
        replaced = tokenizer.ids_to_tokens[tok_id]
        alt_ls.append(
            example.text_b.lower().replace(replaced, actually_predicted.upper())
        )

        if most_predicted:
            pred_result = False

    return (
        pred_result,
        alt_ls,
        tokens_a_pos_dict,
        tokens_a_token_groups,
        tokens_b_pos_dict,
        tokens_b_token_groups,
    )


def masking_predictor(df, tokenizer, model, device):
    all_pos_dicts_sent1 = []
    all_pos_dicts_sent2 = []
    all_tok_groups_sent1 = []
    all_tok_groups_sent2 = []
    pred_ls = []
    all_alt_ls = []
    for _, row in tqdm.tqdm_notebook(df.iterrows(), total=len(df)):
        pred, alt_ls, pos_dict_a, tok_group_a, pos_dict_b, tok_group_b = mask_predict_one_example(
            tokenizer, model, row, device
        )
        all_pos_dicts_sent1.append(pos_dict_a)
        all_tok_groups_sent1.append(tok_group_a)
        all_pos_dicts_sent2.append(pos_dict_b)
        all_tok_groups_sent2.append(tok_group_b)
        pred_ls.append(pred)
        all_alt_ls.append(alt_ls)

    pred_arr = np.array(pred_ls)
    return (
        pred_arr,
        all_alt_ls,
        all_pos_dicts_sent1,
        all_pos_dicts_sent2,
        all_tok_groups_sent1,
        all_tok_groups_sent2,
    )


def get_noun_chunks(sentence):
    doc = nlp(sentence)
    nc = list(set(filter(lambda n: n.text.lower() not in BLACKLIST, doc.noun_chunks)))
    return nc


def get_masked_examples(sent1, begin, end, tokenizer):
    """
    Generate the masked examples
    """
    base_example = InputExample(guid=0, text_a=sent1, text_b=begin + end, is_next=True)
    tokenized_example = tokenize_example(base_example, tokenizer)

    begin_tok = tokenizer.tokenize(begin)
    end_tok = tokenizer.tokenize(end)
    masked_examples = []
    right = tokenizer.convert_tokens_to_ids(end_tok)
    for i in range(len(end_tok)):
        masked_end_tok = end_tok.copy()
        masked_end_tok[i] = MASK
        tokenized_example_changed = copy.deepcopy(tokenized_example)
        tokenized_example_changed.tokens_a = (
            tokenized_example_changed.tokens_a + begin_tok + masked_end_tok
        )
        tokenized_example_changed.tokens_b = ["glue"]
        #         tokenized_example_changed.tokens_b = begin_tok + masked_end_tok
        masked_examples.append(tokenized_example_changed)
    return masked_examples, right


def get_mean_predictions(model, masked_examples, right):
    """
    Once we have a sentence, see how likely the end is when removing words one by one
    """
    features = [
        convert_example_to_features(ex, tokenizer, max_sequence_length, select_prob=0.0)
        for ex in masked_examples
    ]
    batch = features_to_data(features).to(device)
    with torch.no_grad():
        result = model(batch.input_ids, batch.segment_ids, batch.input_mask)
    ids = np.arange(batch.input_ids.shape[1])
    probs = []
    for i, right_idx in enumerate(right):
        masked_idx = ids[batch.input_ids[i].cpu().numpy() == 103]
        assert len(masked_idx) == 1
        masked_idx = masked_idx[0]
        pred_token = result[0][0, masked_idx, :].cpu().numpy()
        prob = np.exp(pred_token[right_idx]) / np.exp(pred_token).sum()
        probs.append(prob)
    probs = np.array(probs)
    return probs.mean()


def filling_predictor(df, tokenizer, model):
    """
    The approach here is closer to `A simple method for commonsense reasoning`
    e.g:
        text_a = "the yellow duck liked the fish because it was beautiful"
        text_b = "the fish was beautiful"
    We produce:
        text_b_alt_1 = mean("the fish was [...]" -> beautiful ; "the fish [...] beautiful" -> was)
        text_b_alt_2 = mean("the yellow duck was [...]" -> beautiful ; "the yellow duck [...] beautiful" -> was)
    If it agrees with initial sentence, then keep that one, otherwise discard.
    The nice thing is we can operate at the noun chunk level
    """
    raise NotImplementedError
    text_a = "the trophy didn't fit in the bag because it was too big. "
    masked_examples, right = get_masked_examples(
        text_a, "the trophy", "was too big", tokenizer
    )
    print(get_mean_predictions(model, masked_examples, right))
    masked_examples, right = get_masked_examples(
        text_a, "the bag", "was too big", tokenizer
    )
    print(get_mean_predictions(model, masked_examples, right))
