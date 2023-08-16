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

python3 ../5_generate_clmbr_features.py \
    --path_to_database ../../EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir ../../EHRSHOT_ASSETS/custom_benchmark \
    --path_to_features_dir ../../EHRSHOT_ASSETS/custom_features \
    --path_to_models_dir ../../EHRSHOT_ASSETS/models \
    --model clmbr 

# Time to run: XXXX mins

# python3 ../5_generate_clmbr_features.py \
#     --path_to_database ../../EHRSHOT_ASSETS/femr/extract \
    # --path_to_labels_dir ../../EHRSHOT_ASSETS/custom_benchmark \
    # --path_to_features_dir ../../EHRSHOT_ASSETS/custom_features \
#     --path_to_models_dir ../../EHRSHOT_ASSETS/models \
#     --model motor \
#     --is_force_refresh