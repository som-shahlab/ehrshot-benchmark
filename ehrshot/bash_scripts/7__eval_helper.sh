#!/bin/bash
#SBATCH --job-name=7__eval_helper
#SBATCH --output=logs/7__eval_helper_%A.out
#SBATCH --error=logs/7__eval_helper_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal,gpu,nigam-v100
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --exclude=secure-gpu-3,secure-gpu-4,secure-gpu-5,secure-gpu-6,secure-gpu-7

python3 ../7_eval.py \
    --path_to_database $1 \
    --path_to_labels_dir $2 \
    --path_to_features_dir $3 \
    --path_to_output_dir $4 \
    --labeling_function $5 \
    --shot_strat $6 \
    --num_threads $7

#SBATCH --partition=nigam-a100
#SBATCH --mem=400G
#SBATCH --cpus-per-task=50


#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# python3 ../7_eval.py \
#     --path_to_database '../../EHRSHOT_ASSETS/femr/extract' \
#     --path_to_labels_dir '../../EHRSHOT_ASSETS/custom_benchmark' \
#     --path_to_features_dir '../../EHRSHOT_ASSETS/custom_hf_features' \
#     --path_to_output_dir /share/pi/nigam/mwornow/outputs \
#     --labeling_function guo_icu \
#     --shot_strat 'all' \
#     --num_threads 1