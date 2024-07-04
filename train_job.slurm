#!/bin/bash

#SBATCH --partition=hopper-prod
#SBATCH --gpus=8
#SBATCH -o training_logs/%x_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --qos=normal
#SBATCH --array=0-3

source /admin/home/andres_marafioti/.bashrc

cd /fsx/andi/florence2-finetuning/
source .venv/bin/activate

# Define datasets and LoRA configurations
datasets=("docvqa" "docvqa" "vqainstruct" "vqainstruct")
lora_flags=(0 1 0 1)

# Determine the configuration based on the SLURM_ARRAY_TASK_ID
dataset=${datasets[$SLURM_ARRAY_TASK_ID]}
use_lora=${lora_flags[$SLURM_ARRAY_TASK_ID]}
batch_size=6
job_name="${dataset}_no_lora"
if [ $use_lora -eq 1 ]; then
    batch_size=10
    lora_arg="--use-lora"
    job_name="${dataset}_lora"
else
    lora_arg=""
fi

# Run the Python script with the appropriate parameters
python distributed_train.py --dataset $dataset --batch-size $batch_size --epochs 10 --lr 1e-6 --eval-steps 1000 --max-val-item-count 1000 $lora_arg
