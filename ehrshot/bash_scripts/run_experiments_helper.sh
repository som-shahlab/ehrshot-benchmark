#!/bin/bash
#SBATCH --job-name=ehrshot
#SBATCH --output=logs/ehrshot_%A.out
#SBATCH --time=1-00:00:00
#SBATCH --mem=80G

# CPU
# --partition=compute
# --cpus-per-task=20

# GPU resources: gpu, nvidia_a100_80gb_pcie:1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

# Disable Python output buffering
export PYTHONUNBUFFERED=1

# Check if instruction file at position 9 is given, then set instructions_file_arg
if [ -z "$9" ]
then
    instructions_file_arg=""
else
    instructions_file_arg="--instructions_file $9"
fi

python  /home/sthe14/ehrshot-benchmark/ehrshot/run_experiments.py \
    --base_dir $1 \
    --experiment_folder $2 \
    --path_to_database $3 \
    --path_to_labels_dir $4 \
    --path_to_split_csv $5 \
    --num_threads $6 \
    --text_encoder $7 \
    --serialization_strategy $8 \
    $instructions_file_arg
