#!/bin/bash
#SBATCH --job-name=ehrshot
#SBATCH --output=logs/ehrshot_%A.log
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=40
# --qos=long_job 

# DGX resources: 
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:8
#SBATCH --mem=800G

# PGPU resources: 
# --partition=pgpu
# --gres=gpu:nvidia_a100-sxm4-80gb:4
# --mem=480G

# GPU resources: 
# --partition=gpu
# --gres=gpu:nvidia_a100_80gb_pcie:1
# --mem=256GB

# CPU
# --partition=compute

# Disable Python output buffering
export PYTHONUNBUFFERED=1

# Check if instruction file at last position, then set instructions_file_arg
if [ -z "${11}" ]
then
    instructions_file_arg=""
else
    instructions_file_arg="--task_to_instructions ${11}"
fi

# TODO: For multi-gpu setup
# CUDA_VISIBLE_DEVICES=0 
python /home/sthe14/ehrshot-benchmark/ehrshot/run_experiments.py \
    --base_dir $1 \
    --experiment_folder $2 \
    --path_to_database $3 \
    --path_to_labels_dir $4 \
    --path_to_split_csv $5 \
    --num_threads $6 \
    --text_encoder $7 \
    --serialization_strategy $8 \
    --excluded_ontologies $9 \
    --num_aggregated ${10} \
    $instructions_file_arg
