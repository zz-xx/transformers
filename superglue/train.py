import argparses
import json
import os
import pandas as pd

import logging

from superglue.tasks import get_task
from glue.runners import GlueTaskRunner, RunnerParameters
from glue import model_setup as glue_model_setup
from shared import model_setup as shared_model_setup
from pytorch_pretrained_bert.utils import at_most_one_of, random_sample
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
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
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
    parser.add_argument("--train_examples_number", type=int, default=None)
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
    task = get_task(args.task_name, args.data_dir)
