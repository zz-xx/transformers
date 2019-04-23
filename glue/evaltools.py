import json
import pandas as pd


MAJOR_METRICS_KEY_DICT = {
    "cola": "mcc",
    "sst": "acc",
    "mrpc": "acc_and_f1",
    "stsb": "corr",
    "qqp": "acc_and_f1",
    "mnli": "acc",
    "qnli": "acc",
    "rte": "acc",
}


class MetricsModes:
    MAJOR = "major"
    ALL = "all"


def get_major_metric(eval_dict, task_name):
    return eval_dict["metrics"][MAJOR_METRICS_KEY_DICT[task_name]]


def get_all_metrics(eval_dict):
    return eval_dict["metrics"]


def get_metrics(task_eval_dict, task_name, mode=MetricsModes.MAJOR):
    if mode == MetricsModes.MAJOR:
        return get_major_metric(task_eval_dict, task_name)
    elif mode == MetricsModes.ALL:
        return get_all_metrics(task_eval_dict)
    else:
        raise KeyError(mode)


def read_metrics_from_single_task(path, task_name, mode=MetricsModes.MAJOR):
    eval_dict = read_json(path)
    return get_metrics(
        task_eval_dict=eval_dict,
        task_name=task_name,
        mode=mode,
    )


def read_metrics_from_multitask(path, mode=MetricsModes.MAJOR):
    eval_dict = read_json(path)
    result = {}
    for task_name, task_eval_dict in eval_dict.items():
        result[task_name] = get_metrics(
            task_eval_dict=task_eval_dict,
            task_name=task_name,
            mode=mode,
        )
    return result


def read_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())


def non_numbers(s):
    return ''.join([i for i in s if not i.isdigit()])


def identity(x):
    return x


def group_by(ls, key_func):
    result = {}
    for elem in ls:
        key = key_func(elem)
        if key not in result:
            result[key] = []
        result[key].append(elem)
    return result


def dict_map(dictionary, key_func=identity, val_func=identity):
    result = {}
    for k, v in dictionary.items():
        result[key_func(k)] = val_func(v)
    assert len(result) == len(dictionary)
    return result


def avg_dict(ls):
    if not ls:
        raise RuntimeError()
    return pd.DataFrame(ls).mean().to_dict()


def max_dict(ls):
    if not ls:
        raise RuntimeError()
    return pd.DataFrame(ls).max().to_dict()


def collate_results(data_rows, agg_func):
    grouped = group_by(data_rows, lambda _: _["task_name"])
    collated_results = {
        task_name: agg_func([elem["metrics"] for elem in task_group])
        for task_name, task_group in grouped.items()
    }
    return collated_results


def dict_get_apply(dictionary, key, func, default):
    return dict_get_apply(dictionary, [key], func, default)


def dict_chain_get_apply(dictionary, key_ls, func, default):
    try:
        curr = dictionary
        for key in key_ls:
            curr = curr[key]
        return func(curr)
    except KeyError:
        return default


def format_collated_results(collated_results, mult_factor=100,
                            qnli_version="v2", omit_averages=True):
    def mult_str(x):
        return str(mult_factor * x)
    dcga = dict_chain_get_apply
    ls = []
    ls.append(dcga(collated_results, ["cola", "mcc"], mult_str, ""))
    ls.append(dcga(collated_results, ["sst", "acc"], mult_str, ""))
    if omit_averages:
        ls += [""] * 4
    else:
        ls.append(dcga(collated_results, ["mrpc", "acc_and_f1"], mult_str, ""))
        ls.append(dcga(collated_results, ["stsb", "corr"], mult_str, ""))
        ls.append(dcga(collated_results, ["qqp", "acc_and_f1"], mult_str, ""))
        ls.append(dcga(collated_results, ["mnli", "acc"], mult_str, ""))
    if qnli_version == "v1":
        ls.append(dcga(collated_results, ["qnli", "acc"], mult_str, ""))
    else:
        ls.append("")
    ls.append(dcga(collated_results, ["rte", "acc"], mult_str, ""))
    ls.append("")
    if qnli_version == "v2":
        ls.append(dcga(collated_results, ["qnli", "acc"], mult_str, ""))
    else:
        ls.append("")
    ls.append("")
    ls.append(dcga(collated_results, ["qqp", "f1"], mult_str, ""))
    ls.append(dcga(collated_results, ["qqp", "acc"], mult_str, ""))
    ls.append("")
    ls.append(dcga(collated_results, ["mrpc", "f1"], mult_str, ""))
    ls.append(dcga(collated_results, ["mrpc", "acc"], mult_str, ""))
    ls.append("")
    ls.append(dcga(collated_results, ["stsb", "pearson"], mult_str, ""))
    ls.append(dcga(collated_results, ["stsb", "spearmanr"], mult_str, ""))
    ls.append("")
    ls.append(dcga(collated_results, ["mnli", "acc"], mult_str, ""))
    ls.append(dcga(collated_results, ["mnli", "mm_acc"], mult_str, ""))
    return ",".join(ls)

