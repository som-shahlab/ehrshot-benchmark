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

# Usage: 
#   EHRSHOT: sbatch 5_generate_clmbr_features.sh --ehrshot
#   MIMIC-IV: sbatch 5_generate_clmbr_features.sh --mimic4
#   EHRSHOT tasks on full STARR-OMOP: sbatch 5_generate_clmbr_features.sh --starr

if [[ " $* " == *" --mimic4 "* ]]; then
    path_to_database="/share/pi/nigam/datasets/femr_mimic_4_extract"
    path_to_labels_dir="../../EHRSHOT_ASSETS/benchmark_mimic4"
    path_to_features_dir="../../EHRSHOT_ASSETS/features_mimic4"
elif [[ " $* " == *" --starr "* ]]; then
    path_to_database="/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes"
    path_to_labels_dir="../../EHRSHOT_ASSETS/benchmark_starr"
    path_to_features_dir="../../EHRSHOT_ASSETS/features_starr"
else
    path_to_database="../../EHRSHOT_ASSETS/femr/extract"
    path_to_labels_dir="../../EHRSHOT_ASSETS/benchmark"
    path_to_features_dir="../../EHRSHOT_ASSETS/features"
fi

python3 ../5_generate_clmbr_features.py \
    --path_to_database $path_to_database \
    --path_to_labels_dir $path_to_labels_dir \
    --path_to_features_dir $path_to_features_dir \
    --path_to_models_dir ../../EHRSHOT_ASSETS/models \
    --model clmbr  \
    --is_force_refresh