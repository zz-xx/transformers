import numpy as np
import pandas as pd
import tqdm

from glue.wnli_utils import get_noun_chunks


def generate_alternates(
    sent1, sent2, exclude_existing=False, exclude_trivial=False, replace_root_mode="no"
):
    """
    exclude_existing: if the replacement is already in the sentence, do not repeat
    exclude_trivial: if the root word for sent2 is not in sent1, exclude it. This has
    the side effect of removing reformulations and synonyms, which is not ideal...
    replace_root: "yes", "no", "both"
    """
    sent2s = []
    replacements = get_noun_chunks(sent1)
    noun_chunks = get_noun_chunks(sent2)
    if exclude_trivial:
        noun_chunks = list(
            filter(lambda n: n.root.text.lower() in sent1.lower(), noun_chunks)
        )
    for nc in noun_chunks:
        for repl in replacements:
            if (
                repl.root.text.lower() in nc.text.lower()
                or nc.root.text.lower() in repl.text.lower()
            ):
                continue
            if exclude_existing and (
                repl.text.lower() in sent2.lower()
                or repl.root.text.lower() in sent2.lower()
            ):
                continue
            if replace_root_mode == "yes":
                sent2s.append(sent2.replace(nc.root.text, repl.root.text, 1).lower())
            elif replace_root_mode == "no":
                sent2s.append(sent2.replace(nc.text, repl.text, 1).lower())
            elif replace_root_mode == "both":
                sent2s.append(sent2.replace(nc.text, repl.text, 1).lower())
                sent2s.append(sent2.replace(nc.root.text, repl.root.text, 1).lower())
            else:
                raise NotImplementedError
    sent2s = [s for s in set(sent2s) if s != sent2.lower()]
    return sent2s


def generate_dataset(
    sent1s, sent2s, reg_size=None, config_alternates=None, skip_empty=False
):
    """
    Given true sentences sent1s and sent2s, constructs a dataset of True and False examples
    If reg_size, make sure there's always reg_size alternatives, either by removing some alternates or
    by adding random sentences
    config_alternates a dict that can contain:
        exclude_existing, exclude_trivial, replace_root_mode
        exclude_existing=False, exclude_trivial=False, replace_root_mode="both",
    """
    # Avoid mutable default arguments.
    if config_alternates is None:
        config_alternates = {}
    all_sent1 = []
    all_sent2 = []
    label = []
    fake_created = []
    weird_sentences = [
        "Glue Glue Glue",
        "Leaderboard Leaderboard Leaderboard",
        "Meuh Meuh Meuh",
        "Beh Beh Beh",
        "LOL LOL LOL",
        "DOIU DOIU DOIU",
        "RRR RRR RRR",
    ]
    cnt = 0
    for sent1, sent2 in tqdm.tqdm_notebook(
        zip(sent1s, sent2s), total=len(sent1s), desc="Generating dataset..."
    ):
        sent2s_false = generate_alternates(sent1, sent2, **config_alternates)
        if skip_empty and not sent2s_false:
            continue
        all_sent1.append(sent1)
        all_sent2.append(sent2)
        label.append(1)

        n_alternates = len(sent2s_false)
        if not n_alternates:
            cnt += 1
            sent2s_false = ["Glue Glue Glue"]
            n_alternates = 1
        if reg_size is not None:
            if n_alternates == (reg_size - 1):
                pass  # We got the right number!
            elif n_alternates > (reg_size - 1):
                sent2s_false = list(
                    np.random.choice(sent2s_false, reg_size - 1, replace=False)
                )
            elif n_alternates < (reg_size - 1):
                sent2s_false = sent2s_false + list(
                    np.random.choice(
                        weird_sentences, reg_size - 1 - len(sent2s_false), replace=False
                    )
                )
                # Keep replace = False to ensure we have the right number...
            n_alternates = reg_size - 1

        fake_created.append(n_alternates)
        all_sent1.extend([sent1] * n_alternates)
        all_sent2.extend(sent2s_false)
        label.extend([0] * n_alternates)
    print("Number of sentence 1s:", len(fake_created))
    print("Mean number of fakes:", sum(fake_created) / len(fake_created))
    print("Counter of glu", cnt)
    return all_sent1, all_sent2, label, fake_created


def construct_sent1(df):
    sent1s = []
    for sents in zip(
        df.InputSentence1, df.InputSentence2, df.InputSentence3, df.InputSentence4
    ):
        sent1s.append(" ".join(sents))
    return sent1s


def construct_sent2(df):
    sent2s = []
    for ans1, ans2, right in zip(
        df.RandomFifthSentenceQuiz1, df.RandomFifthSentenceQuiz2, df.AnswerRightEnding
    ):
        if right == 1:
            sent2s.append(ans1)
        else:
            sent2s.append(ans2)
    return sent2s


def generate_split(df, save_path):
    df["sent1"] = construct_sent1(df)
    df["sent2"] = construct_sent2(df)
    all_s1, all_s2, labels, _ = generate_dataset(df["sent1"], df["sent2"])
    data = pd.DataFrame({"sentence1": all_s1, "sentence2": all_s2, "label": labels})
    # Shuffle rows to avoid weird training artefacts
    data = data.sample(frac=1)
    print("Data shape", data.shape)
    print("Saving to ", save_path)
    data.to_csv(save_path, sep="\t", index=False)
    return data


def generate_train_for_csv(s1s, s2s, num_choices):
    """
    In training every first sentence (0th) is true
    """
    data = get_data_from_sents(s1s, s2s, num_choices)
    data["label"] = 0
    return data


def generate_val_for_csv(s1s, s2s, labels, num_choices):
    """
    Labels has len s1s // num_choices
    """
    assert len(labels) == len(s1s) // num_choices
    data = get_data_from_sents(s1s, s2s, num_choices)
    data["label"] = labels
    return data


def get_data_from_sents(s1s, s2s, num_choices):
    rows = []
    for i, (s1, s2) in enumerate(zip(s1s, s2s)):
        if i % num_choices == 0:
            rows.append([s1, s2])
        else:
            rows[-1].append(s2)
    names = ["start"] + [f"cont_{i}" for i in range(num_choices)]
    return pd.DataFrame(rows, columns=names)
