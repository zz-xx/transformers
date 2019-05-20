import argparse
import json
import numpy as np
import pandas as pd

import superglue.tasks as sgtasks


def write_jsonl(data, path):
    with open(path, "w") as f:
        f.write("\n".join([json.dumps(row) for row in data]))


def load_csv(path):
    return pd.read_csv(path, header=None)


def get_argmax(df):
    return np.argmax(df.values, 1)


def format_output(preds, labels):
    labels_dict = dict(enumerate(labels))
    return [
        {"idx": i, "label": labels_dict[k]}
        for i, k in enumerate(preds)
    ]


def format_preds(task_name, input_path, output_path):
    if task_name in ("cb", "copa", "rte", "wic", "wsc"):
        task = sgtasks.TASK_DICT[task_name]
        write_jsonl(
            format_output(get_argmax(load_csv(input_path)), task.LABELS),
            output_path,
        )
    elif task_name == "diagnostic":
        write_jsonl([
            {"idx": i, "label": "entailment"}
            for i in range(1104)
        ], output_path)
    else:
        raise KeyError(task_name)


def main():
    parser = argparse.ArgumentParser(description='superglue')
    parser.add_argument('task-name', type=str, required=True)
    parser.add_argument('input_path', required=True)
    parser.add_argument('output_path', required=True)
    args = parser.parse_args()
    format_preds(
        task_name=args.task_name,
        input_path=args.input_path,
        output_path=args.output_path,
    )
