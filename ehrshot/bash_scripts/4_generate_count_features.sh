#!/bin/bash
#SBATCH --job-name=4_generate_count_features
#SBATCH --output=logs/4_generate_count_features_%A.out
#SBATCH --error=logs/4_generate_count_features_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10

python3 4_generate_count_features.py \
    --path_to_database ../EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_and_feats_dir ../EHRSHOT_ASSETS/benchmark \
    --num_threads 10