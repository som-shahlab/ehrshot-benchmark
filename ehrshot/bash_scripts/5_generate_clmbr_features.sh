#!/bin/bash
#SBATCH --job-name=5_generate_clmbr_features
#SBATCH --output=logs/5_generate_clmbr_features_%A.out
#SBATCH --error=logs/5_generate_clmbr_features_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10

python3 5_generate_clmbr_features.py \
    --path_to_database ../EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_and_feats_dir ../EHRSHOT_ASSETS/benchmark \
    --path_to_output_dir # TODO \
    --model clmbr

python3 5_generate_clmbr_features.py \
    --path_to_database ../EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_and_feats_dir ../EHRSHOT_ASSETS/benchmark \
    --path_to_output_dir # TODO \
    --model motor