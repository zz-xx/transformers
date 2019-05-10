import os

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from .modeling import map_task_to_model_class
from shared.model_setup import stage_model, get_bert_config_path


def create_model(task, bert_model_name, bert_load_mode, bert_load_args,
                 all_state, device, n_gpu, fp16, local_rank,
                 bert_config_json_path=None):
    model_class = map_task_to_model_class(task)
    if bert_load_mode == "from_pretrained":
        assert bert_load_args is None
        assert all_state is None
        assert bert_config_json_path is None
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(local_rank)
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=bert_model_name,
            cache_dir=cache_dir,
        )
    elif bert_load_mode in ["model_only", "state_model_only", "state_all", "state_full_model",
                            "full_model_only"]:
        assert bert_load_args is None
        model = load_bert(
            model_class=model_class,
            bert_model_name=bert_model_name,
            bert_load_mode=bert_load_mode,
            all_state=all_state,
            bert_config_json_path=bert_config_json_path,
        )
    else:
        raise KeyError(bert_load_mode)
    model = stage_model(model, fp16=fp16, device=device, local_rank=local_rank, n_gpu=n_gpu)
    return model


def load_bert(model_class, bert_model_name, bert_load_mode, all_state,
              bert_config_json_path=None):
    if bert_config_json_path is None:
        bert_config_json_path = os.path.join(get_bert_config_path(bert_model_name), "bert_config.json")
    if bert_load_mode in ("model_only", "full_model_only"):
        state_dict = all_state
    elif bert_load_mode in ["state_model_only", "state_all", "state_full_model"]:
        state_dict = all_state["model"]
    else:
        raise KeyError(bert_load_mode)

    if bert_load_mode in ("state_full_model", "full_model_only"):
        model = model_class.from_state_dict_full(
            config_file=bert_config_json_path,
            state_dict=state_dict,
        )
    else:
        model = model_class.from_state_dict(
            config_file=bert_config_json_path,
            state_dict=state_dict,
        )
    return model


