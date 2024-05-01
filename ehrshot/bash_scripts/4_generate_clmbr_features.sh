#!/bin/bash
#SBATCH --job-name=5_generate_clmbr_features
#SBATCH --output=logs/5_generate_clmbr_features_%A.out
#SBATCH --error=logs/5_generate_clmbr_features_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# Time to run: 20 mins

python3 ../4_generate_clmbr_features.py \
    --model clmbr \
    --is_force_refresh
    # --path_to_database ../../EHRSHOT_ASSETS/database_no_visit_merge \
    # --path_to_labels_dir ../../EHRSHOT_ASSETS/labels_no_visit_merge \
    # --path_to_features_dir ../../EHRSHOT_ASSETS/features_no_visit_merge \