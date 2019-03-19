import pandas as pd
import tqdm

from glue.wnli_utils import get_noun_chunks


def generate_alternates(sent1, sent2, exclude_existing=True, exclude_trivial=True):
    """
    exclude_existing: if the replacement is already in the sentence, do not repeat
    exclude_trivial: if the root word for sent2 is not in sent1, exclude it. This has
    the side effect of removing reformulations and synonyms, which is not ideal...
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
                repl.root.text.lower() not in nc.text.lower()
                and nc.root.text.lower() not in repl.text.lower()
            ):
                if exclude_existing:
                    if repl.text.lower() not in sent2.lower():
                        sent2s.append(sent2.replace(nc.text, repl.text))
                else:
                    sent2s.append(sent2.replace(nc.text, repl.text))
    return sent2s


def generate_dataset(sent1s, sent2s):
    """
    Given true sentences sent1s and sent2s, constructs a dataset of True and False examples
    """
    all_sent1 = []
    all_sent2 = []
    label = []
    fake_created = []
    for sent1, sent2 in tqdm.tqdm_notebook(zip(sent1s, sent2s), total=len(sent1s), desc="Generating dataset"):
        all_sent1.append(sent1)
        all_sent2.append(sent2)
        label.append(1)
        sent2s_false = generate_alternates(sent1, sent2)
        n_alternates = len(sent2s_false)
        fake_created.append(n_alternates)
        all_sent1.extend([sent1] * n_alternates)
        all_sent2.extend(sent2s_false)
        label.extend([0] * n_alternates)
    print(f"Created {len(fake_created)} examples")
    print(
        f"Created an average of: {sum(fake_created) / len(fake_created)} fake examples"
    )
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
