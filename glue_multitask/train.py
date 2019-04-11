import argparse
import json
import os
import pandas as pd

import logging

from glue.tasks import MnliMismatchedProcessor
from glue_multitask.runners import GlueMultitaskRunner, RunnerParameters
from glue import model_setup as glue_model_setup
from glue_multitask import multitask_model_setup as multitask_model_setup
from shared import model_setup as shared_model_setup
from pytorch_pretrained_bert.utils import at_most_one_of
import shared.initialization as initialization
import shared.log_info as log_info

# todo: cleanup imports


def get_args(*in_args):
    parser = argparse.ArgumentParser()

    # === Required parameters === #
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name_ls",
                        default=None,
                        type=str,
                        required=True,
                        help="List of task names, comma-separated")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # === Model parameters === #
    parser.add_argument("--bert_load_path", default=None, type=str)
    parser.add_argument("--bert_load_mode", default="from_pretrained", type=str,
                        help="from_pretrained, model_only, state_model_only, state_all")
    parser.add_argument("--bert_load_args", default=None, type=str)
    parser.add_argument("--bert_config_json_path", default=None, type=str)
    parser.add_argument("--bert_vocab_path", default=None, type=str)
    parser.add_argument("--bert_save_mode", default="all", type=str)

    # === Other parameters === #
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_val",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_val_history",
                        action='store_true',
                        help="")
    parser.add_argument("--train_examples_number", type=str, default=None,
                        help="int, or comma-separated list of ints")
    parser.add_argument("--train_save_every", type=int, default=None)
    parser.add_argument("--train_save_every_epoch", action="store_true")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=-1,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. "
                             "Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--print-trainable-params', action="store_true")
    parser.add_argument('--not-verbose', action="store_true")
    parser.add_argument('--force-overwrite', action="store_true")

    parser.add_argument('--task_lr_scaler_ls', type=str, default=None)
    parser.add_argument('--task_sampling_mode', type=str, default='basic')

    args = parser.parse_args(*in_args)
    return args


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = get_args()
    log_info.print_args(args)

    device, n_gpu = initialization.init_cuda_from_args(args, logger=logger)
    initialization.init_seed(args, n_gpu=n_gpu, logger=logger)
    initialization.init_train_batch_size(args)
    initialization.init_output_dir(args)
    initialization.save_args(args)

    task_ls = multitask_model_setup.get_task_ls(
        task_name_ls=args.task_name_ls,
        data_dir=args.data_dir,
    )
    tokenizer = shared_model_setup.create_tokenizer(
        bert_model_name=args.bert_model,
        bert_load_mode=args.bert_load_mode,
        do_lower_case=args.do_lower_case,
        bert_vocab_path=args.bert_vocab_path,
    )
    all_state = shared_model_setup.load_overall_state(args.bert_load_path, relaxed=True)
    model = multitask_model_setup.create_model(
        task_ls=task_ls,
        bert_model_name=args.bert_model,
        bert_load_mode=args.bert_load_mode,
        bert_load_args=args.bert_load_args,
        all_state=all_state,
        device=device,
        n_gpu=n_gpu,
        fp16=args.fp16,
        local_rank=args.local_rank,
        bert_config_json_path=args.bert_config_json_path,
    )
    if args.do_train:
        if args.print_trainable_params:
            log_info.print_trainable_params(model)
        train_examples_number_ls = multitask_model_setup.get_train_examples_number_ls(
            args_train_examples_number=args.train_examples_number,
            num_tasks=len(task_ls),
        )
        train_examples_dict = multitask_model_setup.get_train_examples_dict(
            task_ls=task_ls,
            train_examples_number_ls=train_examples_number_ls,
        )
        t_total = shared_model_setup.get_opt_train_steps(
            num_train_examples=sum(len(train_examples) for train_examples in train_examples_dict.values()),
            args=args,
        )
        optimizer = shared_model_setup.create_optimizer(
            model=model,
            learning_rate=args.learning_rate,
            t_total=t_total,
            loss_scale=args.loss_scale,
            fp16=args.fp16,
            warmup_proportion=args.warmup_proportion,
            state_dict=all_state["optimizer"] if args.bert_load_mode == "state_all" else None,
        )
    else:
        train_examples_dict = None
        t_total = 0
        optimizer = None

    runner = GlueMultitaskRunner(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        task_ls=task_ls,
        device=device,
        rparams=RunnerParameters(
            max_seq_length=args.max_seq_length,
            local_rank=args.local_rank, n_gpu=n_gpu, fp16=args.fp16,
            learning_rate=args.learning_rate, gradient_accumulation_steps=args.gradient_accumulation_steps,
            task_lr_scaler_dict=RunnerParameters.format_task_lr_scaler_dict(args.task_lr_scaler_ls, task_ls),
            task_sampling_mode=args.task_sampling_mode,
            t_total=t_total, warmup_proportion=args.warmup_proportion,
            num_train_epochs=args.num_train_epochs,
            train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size,
        )
    )

    if args.do_train:
        assert at_most_one_of([args.do_val_history,
                               args.train_save_every,
                               args.train_save_every_epoch])
        if args.do_val_history:
            raise NotImplementedError()
        elif args.train_save_every:
            raise NotImplementedError()
        elif args.train_save_every_epoch:
            raise NotImplementedError()
        else:
            runner.run_train(train_examples_dict)

    if args.do_save:
        # Save a trained model
        glue_model_setup.save_bert(
            model=model, optimizer=optimizer, args=args,
            save_path=os.path.join(args.output_dir, "all_state.p"),
            save_mode=args.bert_save_mode,
        )

    if args.do_val:
        val_examples_dict = multitask_model_setup.get_val_examples_dict(task_ls)
        all_results = runner.run_val(val_examples_dict, verbose=False)
        val_metrics = {}
        for task_name, task_results in all_results.items():
            df = pd.DataFrame(task_results["logits"])
            df.to_csv(os.path.join(args.output_dir, f"{task_name}_val_preds.csv"), header=False, index=False)
            val_metrics[task_name] = {"loss": task_results["loss"], "metrics": task_results["metrics"]}

        # HACK for MNLI-mismatched
        task_dict = {task.name: task for task in task_ls}
        if "mnli" in task_dict:
            task = task_dict["mnli"]
            mm_val_examples = MnliMismatchedProcessor().get_dev_examples(task.data_dir)
            mm_results = runner.runner_dict["mnli"].run_val(
                mm_val_examples, task_name=task.name, verbose=False)
            df = pd.DataFrame(all_results["mnli"]["logits"])
            df.to_csv(os.path.join(args.output_dir, "mnli_mm_val_preds.csv"), header=False, index=False)
            for k, v in mm_results["metrics"].items():
                val_metrics["mnli"]["metrics"]["mm-"+k] = v

        metrics_str = json.dumps(val_metrics, indent=2)
        print(metrics_str)
        with open(os.path.join(args.output_dir, f"val_metrics.json"), "w") as f:
            f.write(metrics_str)

    if args.do_test:
        test_examples_dict = multitask_model_setup.get_test_examples_dict(task_ls)
        all_logits = runner.run_test(test_examples_dict, verbose=False)
        for task_name, task_logits in all_logits.items():
            df = pd.DataFrame(task_logits)
            df.to_csv(os.path.join(args.output_dir, f"{task_name}_test_preds.csv"), header=False, index=False)

        # HACK for MNLI-mismatched
        task_dict = {task.name: task for task in task_ls}
        if "mnli" in task_dict:
            test_examples = MnliMismatchedProcessor().get_test_examples(task_dict["mnli"].data_dir)
            logits = runner.runner_dict["mnli"](test_examples)
            df = pd.DataFrame(logits)
            df.to_csv(os.path.join(args.output_dir, "mnli_mm_test_preds.csv"), header=False, index=False)


if __name__ == "__main__":
    main()
