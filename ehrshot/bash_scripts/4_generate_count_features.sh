#!/bin/bash
#SBATCH --job-name=4_generate_count_features
#SBATCH --output=logs/4_generate_count_features_%A.out
#SBATCH --error=logs/4_generate_count_features_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# Time to run: 10 mins

# Usage: sbatch 4_generate_count_features.sh

BASE_EHRSHOT_DIR="../../EHRSHOT_ASSETS"

path_to_database="${BASE_EHRSHOT_DIR}/femr/extract"
path_to_labels_dir="${BASE_EHRSHOT_DIR}/benchmark"
path_to_features_dir="${BASE_EHRSHOT_DIR}/features"

mkdir -p $path_to_features_dir

python3 ../4_generate_count_features.py \
    --path_to_database $path_to_database \
    --path_to_labels_dir $path_to_labels_dir \
    --path_to_features_dir $path_to_features_dir \
    --num_threads 20 \
    --is_force_refresh