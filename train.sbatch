#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.
#SBATCH --output=/scratch/tjf324/slurm/bert%j.out
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=30000
#SBATCH --time=24:00:00



PATH=$HOME/anaconda3/envs/bert/bin:$HOME/anaconda3/bin:$PATH
PYTHONPATH=/scratch/tjf324/pytorch-pretrained-BERT/:$PYTHONPATH

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - $JIANT_OVERRIDES >> ~/jiant_machine_assignments.txt

source activate bert
# Run.
cd /scratch/tjf324/pytorch-pretrained-BERT/


export TASK_A=roc
export TASK_B=wnli
export OUTPUT_PATH_A=cloze
export OUTPUT_PATH_B=cloze__wnli
export PYTORCH_PRETRAINED_BERT_CACHE=/scratch/tjf324/models/bert/
export GLUE_DIR=/scratch/tjf324/data/glue_auto_dl/
python glue/train.py \
    --task_name $TASK_A \
    --do_train --do_val \
    --do_save \
    --do_lower_case \
    --bert_model bert-large-uncased \
    --bert_load_mode from_pretrained \
    --bert_save_mode model_all \
    --train_batch_size 24 \
    --learning_rate 2e-5 \
    --output_dir $OUTPUT_PATH_A

echo DONE
