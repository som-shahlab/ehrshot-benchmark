#!/bin/bash
#SBATCH --job-name=7__eval_helper
#SBATCH --output=logs/7__eval_helper_%A.out
#SBATCH --error=logs/7__eval_helper_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=180G
#SBATCH --cpus-per-task=15

labeling_function=$1
shot_strat=$2
num_threads=$3

python3 ../7_eval.py \
    --path_to_database ../../EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir ../../EHRSHOT_ASSETS/custom_benchmark \
    --path_to_features_dir ../../EHRSHOT_ASSETS/custom_features \
    --path_to_output_dir ../../EHRSHOT_ASSETS/results \
    --labeling_function $labeling_function \
    --shot_strat $shot_strat \
    --num_threads $num_threads