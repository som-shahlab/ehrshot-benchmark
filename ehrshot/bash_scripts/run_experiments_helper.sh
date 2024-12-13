#!/bin/bash
#SBATCH --job-name=ehrshot
#SBATCH --output=logs/ehrshot_%A.log
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --qos=long_job 

# DGX resources: 
# --partition=pgpu
# --gres=gpu:nvidia_a100-sxm4-80gb:8
# --mem=800G

# PGPU resources: 
# --partition=pgpu
# --gres=gpu:nvidia_a100-sxm4-80gb:4
# --mem=400G

# GPU resources: 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --mem=80G

# CPU
# --partition=compute

# Disable Python output buffering
export PYTHONUNBUFFERED=1

# Check if instruction file at position 10 is given, then set instructions_file_arg
if [ -z "${11}" ]
then
    instructions_file_arg=""
else
    instructions_file_arg="--task_to_instructions ${14}"
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
    --unique_events ${10} \
    --numeric_values ${11} \
    --num_aggregated ${12} \
    --add_parent_concepts ${13} \
    $instructions_file_arg
