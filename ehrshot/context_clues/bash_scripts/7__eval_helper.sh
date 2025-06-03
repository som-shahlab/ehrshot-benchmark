#!/bin/bash
#SBATCH --job-name=7__eval_helper
#SBATCH --output=logs/7__eval_helper_%A.out
#SBATCH --error=logs/7__eval_helper_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal,gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --exclude=secure-gpu-1,secure-gpu-2,secure-gpu-15,secure-gpu-16,secure-gpu-17,secure-gpu-18,secure-gpu-19,secure-gpu-20

python3 ../../7_eval_finetune.py \
    --path_to_database $1 \
    --path_to_labels_dir $2 \
    --path_to_features_dir $3 \
    --path_to_split_csv $4 \
    --path_to_output_dir $5 \
    --shot_strat $6 \
    --ks $7 \
    --models $8 \
    --heads lr_lbfgs,rf,gbm \
    --num_threads $9 \
    --labeling_function ${10} \
    --is_force_refresh

# For debugging:
#
# python3 ../../7_eval_finetune.py \
#     --path_to_database '../../EHRSHOT_ASSETS/femr/extract' \
#     --path_to_labels_dir '../../EHRSHOT_ASSETS/benchmark' \
#     --path_to_features_dir '../../EHRSHOT_ASSETS/features' \
#     --path_to_split_csv '../../EHRSHOT_ASSETS/splits/person_id_map.csv' \
#     --path_to_output_dir '../../EHRSHOT_ASSETS/results_test' \
#     --labeling_function guo_los \
#     --shot_strat all \
#     --ks -1 \
#     --models count,clmbr \
#     --heads gbm,lr_lbfgs,rf \
#     --num_threads 5

