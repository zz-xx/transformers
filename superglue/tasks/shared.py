import csv
import json


class Task:
    Example = None
    DataRow = None
    Batch = None

    def __init__(self, name, data_dir):
        self.name = name
        self.data_dir = data_dir

    def get_train_examples(self):
        raise NotImplementedError

    def get_dev_examples(self):
        raise NotImplementedError

    def get_test_examples(self):
        raise NotImplementedError


def read_tsv(input_file, quotechar=None):
    with open(input_file, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


def read_json_lines(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines
