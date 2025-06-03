#!/bin/bash
#SBATCH --job-name=5_generate_clmbr_features
#SBATCH --output=logs/5_generate_clmbr_features_%A.out
#SBATCH --error=logs/5_generate_clmbr_features_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=nigam-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=10
#SBATCH --exclude=secure-gpu-1,secure-gpu-2

# Time to run: 20 mins

# Usage: sbatch 5_generate_clmbr_features.sh

BASE_EHRSHOT_DIR="../../EHRSHOT_ASSETS"

path_to_database="${BASE_EHRSHOT_DIR}/femr/extract"
path_to_labels_dir="${BASE_EHRSHOT_DIR}/benchmark"
path_to_features_dir="${BASE_EHRSHOT_DIR}/features"

python3 ../5_generate_clmbr_features.py \
    --path_to_database $path_to_database \
    --path_to_labels_dir $path_to_labels_dir \
    --path_to_features_dir $path_to_features_dir \
    --path_to_models_dir ../../EHRSHOT_ASSETS/models \
    --model clmbr  \
    --is_force_refresh