#!/bin/bash
#SBATCH --job-name=7__eval_helper
#SBATCH --output=logs/7__eval_helper_%A.out
#SBATCH --error=logs/7__eval_helper_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=nigam-a100
#SBATCH --mem=400G
#SBATCH --cpus-per-task=25
#SBATCH --exclude=secure-gpu-1,secure-gpu-2,secure-gpu-3

python3 ../7_eval.py \
    --path_to_database $1 \
    --path_to_labels_dir $2 \
    --path_to_features_dir $3 \
    --path_to_output_dir $4 \
    --labeling_function $5 \
    --shot_strat $6 \
    --num_threads $7


#SBATCH --partition=nigam-a100
#SBATCH --mem=300G
#SBATCH --cpus-per-task=50