#!/bin/bash
#SBATCH --job-name=7__eval_helper_gpu
#SBATCH --output=logs/7__eval_helper_gpu_%A.out
#SBATCH --error=logs/7__eval_helper_gpu_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=nigam-a100,nigam-h100
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:3

# For running multiple labeling functions in parallel per node:
echo "11: ${11}"
echo "12: ${12}"
echo "13: ${13}"

CUDA_VISIBLE_DEVICES=0 && python3 ../../7_eval_finetune.py \
    --path_to_database $1 \
    --path_to_labels_dir $2 \
    --path_to_features_dir $3 \
    --path_to_split_csv $4 \
    --path_to_output_dir $5 \
    --shot_strat $6 \
    --ks $7 \
    --models $8 \
    --path_to_tokenized_timelines ${10} \
    --heads finetune_layers=2,finetune_full,finetune_full-logregfirst \
    --labeling_function ${11} \
    --num_threads 5 &


CUDA_VISIBLE_DEVICES=1 && python3 ../../7_eval_finetune.py \
    --path_to_database $1 \
    --path_to_labels_dir $2 \
    --path_to_features_dir $3 \
    --path_to_split_csv $4 \
    --path_to_output_dir $5 \
    --shot_strat $6 \
    --ks $7 \
    --models $8 \
    --path_to_tokenized_timelines ${10} \
    --heads finetune_layers=2,finetune_full,finetune_frozen-logregfirst \
    --labeling_function ${12} \
    --num_threads 5 &

CUDA_VISIBLE_DEVICES=2 && python3 ../../7_eval_finetune.py \
    --path_to_database $1 \
    --path_to_labels_dir $2 \
    --path_to_features_dir $3 \
    --path_to_split_csv $4 \
    --path_to_output_dir $5 \
    --shot_strat $6 \
    --ks $7 \
    --models $8 \
    --path_to_tokenized_timelines ${10} \
    --heads finetune_layers=2,finetune_full,finetune_frozen-logregfirst \
    --labeling_function ${13} \
    --num_threads 5 &

wait

# For running one job per node:
#
# python3 ../../7_eval_finetune.py \
#     --path_to_database $1 \
#     --path_to_labels_dir $2 \
#     --path_to_features_dir $3 \
#     --path_to_split_csv $4 \
#     --path_to_output_dir $5 \
#     --shot_strat $6 \
#     --ks $7 \
#     --labeling_function $9 \
#     --num_threads 3 \
#     --heads finetune_layers=2,finetune_full,finetune_frozen-logregfirst


# For debugging:
#
# python3 ../../7_eval_finetune.py \
#     --path_to_database '../../EHRSHOT_ASSETS/femr/extract' \
#     --path_to_labels_dir '../../EHRSHOT_ASSETS/benchmark' \
#     --path_to_features_dir '../../EHRSHOT_ASSETS/features' \
#     --path_to_split_csv '../../EHRSHOT_ASSETS/splits/person_id_map.csv' \
#     --path_to_output_dir '../../EHRSHOT_ASSETS/results' \
#     --labeling_function guo_los \
#     --shot_strat all \
#     --ks -1 \
#     --num_threads 1 \
#     --heads finetune_layers=2,finetune_full,finetune_frozen-logregfirst


# For debugging:
#
# python3 ../../7_eval_finetune.py \
#     --path_to_database '../../EHRSHOT_ASSETS/femr/extract' \
#     --path_to_labels_dir '../../EHRSHOT_ASSETS/benchmark' \
#     --path_to_features_dir '../../EHRSHOT_ASSETS/features' \
#     --path_to_split_csv '../../EHRSHOT_ASSETS/splits/person_id_map.csv' \
#     --path_to_output_dir '../../EHRSHOT_ASSETS/results' \
#     --labeling_function guo_los \
#     --shot_strat all \
#     --ks -1 \
#     --num_threads 1 \
#     --heads finetune_frozen-logregfirst

