#!/bin/bash
#SBATCH --job-name=7__eval_helper
#SBATCH --output=logs/7__eval_helper_%A.out
#SBATCH --error=logs/7__eval_helper_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=compute
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#--exclude=secure-gpu-3,secure-gpu-4,secure-gpu-5,secure-gpu-6,secure-gpu-7

python3 ../7_eval.py \
    --path_to_database $1 \
    --path_to_labels_dir $2 \
    --path_to_features_dir $3 \
    --path_to_split_csv $4 \
    --path_to_output_dir $5 \
    --labeling_function $6 \
    --shot_strat $7 \
    --num_threads $8