from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


def create_model(task_type, bert_model_name, bert_load_mode, bert_load_args,
                 all_state,
                 num_labels, device, n_gpu, fp16, local_rank,
                 bert_config_json_path=None):
    if bert_load_mode == "from_pretrained":
        assert bert_load_args is None
        assert all_state is None
        assert bert_config_json_path is None
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(local_rank)
        model = create_from_pretrained(
            task_type=task_type,
            bert_model_name=bert_model_name,
            cache_dir=cache_dir,
            num_labels=num_labels,
        )
    elif bert_load_mode in ["model_only", "state_model_only", "state_all", "state_full_model",
                            "full_model_only"]:
        assert bert_load_args is None
        model = load_bert(
            task_type=task_type,
            bert_model_name=bert_model_name,
            bert_load_mode=bert_load_mode,
            all_state=all_state,
            num_labels=num_labels,
            bert_config_json_path=bert_config_json_path,
        )
    elif bert_load_mode in ["state_adapter"]:
        model = load_bert_adapter(
            task_type=task_type,
            bert_model_name=bert_model_name,
            bert_load_mode=bert_load_mode,
            bert_load_args=bert_load_args,
            all_state=all_state,
            num_labels=num_labels,
            bert_config_json_path=bert_config_json_path,
        )
    else:
        raise KeyError(bert_load_mode)
    model = stage_model(model, fp16=fp16, device=device, local_rank=local_rank, n_gpu=n_gpu)
    return model


def create_from_pretrained(task_type, bert_model_name, cache_dir, num_labels):
    if task_type == TaskType.CLASSIFICATION:
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=bert_model_name,
            cache_dir=cache_dir,
            num_labels=num_labels,
        )
    elif task_type == TaskType.REGRESSION:
        assert num_labels == 1
        model = BertForSequenceRegression.from_pretrained(
            pretrained_model_name_or_path=bert_model_name,
            cache_dir=cache_dir,
        )
    else:
        raise KeyError(task_type)
    return model


def load_bert(task_type, bert_model_name, bert_load_mode, all_state, num_labels,
              bert_config_json_path=None):
    if bert_config_json_path is None:
        bert_config_json_path = os.path.join(get_bert_config_path(bert_model_name), "bert_config.json")
    if bert_load_mode in ("model_only", "full_model_only"):
        state_dict = all_state
    elif bert_load_mode in ["state_model_only", "state_all", "state_full_model"]:
        state_dict = all_state["model"]
    else:
        raise KeyError(bert_load_mode)

    if task_type == TaskType.CLASSIFICATION:
        if bert_load_mode in ("state_full_model", "full_model_only"):
            model = BertForSequenceClassification.from_state_dict_full(
                config_file=bert_config_json_path,
                state_dict=state_dict,
                num_labels=num_labels,
            )
        else:
            model = BertForSequenceClassification.from_state_dict(
                config_file=bert_config_json_path,
                state_dict=state_dict,
                num_labels=num_labels,
            )
    elif task_type == TaskType.REGRESSION:
        assert num_labels == 1
        if bert_load_mode in ("state_full_model", "full_model_only"):
            model = BertForSequenceRegression.from_state_dict_full(
                config_file=bert_config_json_path,
                state_dict=state_dict,
            )
        else:
            model = BertForSequenceRegression.from_state_dict(
                config_file=bert_config_json_path,
                state_dict=state_dict,
            )
    else:
        raise KeyError(task_type)
    return model

