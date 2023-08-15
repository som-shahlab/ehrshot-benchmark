#!/bin/bash
#SBATCH --job-name=4_generate_count_features
#SBATCH --output=logs/4_generate_count_features_%A.out
#SBATCH --error=logs/4_generate_count_features_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# Time to run: 10 mins

mkdir -p ../../EHRSHOT_ASSETS/custom_features

python3 ../4_generate_count_features.py \
    --path_to_database ../../EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir ../../EHRSHOT_ASSETS/custom_benchmark \
    --path_to_features_dir ../../EHRSHOT_ASSETS/custom_features \
    --num_threads 20 \
    --is_force_refresh