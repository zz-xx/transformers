import numpy as np


def to_cpu(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            new_state_dict[k] = to_cpu(v)
        elif isinstance(v, int):
            new_state_dict[k] = v
        elif isinstance(v, list):
            # May need to change this in the future
            assert k == "param_groups"
            new_state_dict[k] = v
        else:
            new_state_dict[k] = v.cpu()
    return new_state_dict


def only_one_of(ls):
    return count_bool(ls) == 1


def at_most_one_of(ls):
    return count_bool(ls) <= 1


def count_bool(ls):
    return sum([1 if elem else 0 for elem in ls])


def truncate_seq_pair(tokens_a, tokens_b, max_length):
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


def truncate_sequences(tokens_ls, max_length):
    lengths = [len(tokens) for tokens in tokens_ls]
    total_length = sum(lengths)
    if total_length < max_length:
        return tokens_ls
    divided_length = total_length // len(tokens_ls)
    rem = total_length - divided_length
    target_lengths = [
        divided_length + (1 if i < rem else 0)
        for i in range(len(tokens_ls))
    ]
    return [
        tokens[:target_length]
        for tokens, target_length in zip(tokens_ls, target_lengths)
    ]


def pad_to_max_seq_length(ls, max_seq_length, pad_idx=0, check=True):
    padding = [pad_idx] * (max_seq_length - len(ls))
    result = ls + padding

    if check:
        assert len(result) == max_seq_length
    return result


def random_sample(ls, size, replace=True):
    indices = np.random.choice(range(len(ls)), size=size, replace=replace)
    return [ls[i] for i in indices]
