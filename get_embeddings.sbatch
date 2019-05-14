#!/bin/bash

# Generic job script for all experiments on NYU CILVR machines.
#SBATCH --output=/scratch/tjf324/slurm/bert%j.out
#SBATCH --gres=gpu:p40:1
#SBATCH --mem=30000
#SBATCH --time=12:00:00



PATH=$HOME/anaconda3/envs/bert/bin:$HOME/anaconda3/bin:$PATH
PYTHONPATH=/scratch/tjf324/pytorch-pretrained-BERT/:$PYTHONPATH

# Log what we're running and where.
echo $SLURM_JOBID - `hostname` - $JIANT_OVERRIDES >> ~/jiant_machine_assignments.txt

source activate bert
# Run.
cd /scratch/tjf324/pytorch-pretrained-BERT/
export NT=cola
export TASK=COLA

for i in {1..5}
do 
    echo $i
    export BERT_ALL_DIR=/scratch/tjf324/models/bert/
    export OUTPUT_PATH=embeddings/1k/${NT}${i}/
    export PYTORCH_PRETRAINED_BERT_CACHE=/scratch/tjf324/models/bert/
    export MODEL_PATH=/scratch/tjf324/models/stilts/1k/${NT}${i}/all_state.p
    export GLUE_DIR=/scratch/tjf324/data/glue_auto_dl/
    python get_embeddings.py \
        --task_name $TASK \
        --data_dir $GLUE_DIR/$TASK \
        --bert_load_path $MODEL_PATH \
        --batch_size 32 \
        --save_dir $OUTPUT_PATH
done

echo DONE
