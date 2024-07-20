#!/bin/bash
#SBATCH --job-name=7__eval_helper_gpu
#SBATCH --output=logs/7__eval_helper_gpu_%A.out
#SBATCH --error=logs/7__eval_helper_gpu_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu,nigam-v100,nigam-a100
#SBATCH --mem=200G
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --exclude=secure-gpu-1,secure-gpu-2

python3 ../7_eval_finetune.py \
    --path_to_database $1 \
    --path_to_labels_dir $2 \
    --path_to_features_dir $3 \
    --path_to_split_csv $4 \
    --path_to_output_dir $5 \
    --labeling_function $6 \
    --shot_strat $7 \
    --num_threads 3 \
    --heads finetune_layers=1,finetune_layers=2,finetune_full,finetune_frozen