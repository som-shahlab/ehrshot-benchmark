#!/bin/bash
#SBATCH --job-name=9_make_cohort_plots
#SBATCH --output=logs/9_make_cohort_plots_%A.out
#SBATCH --error=logs/9_make_cohort_plots_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32

BASE_EHRSHOT_DIR="../../EHRSHOT_ASSETS"

mkdir -p ${BASE_EHRSHOT_DIR}/cohort_stats

python3 ../9_make_cohort_plots.py \
    --path_to_database ${BASE_EHRSHOT_DIR}/femr/extract \
    --path_to_labels_and_feats_dir ${BASE_EHRSHOT_DIR}/benchmark \
    --path_to_input_dir ${BASE_EHRSHOT_DIR}/data \
    --path_to_splits_dir ${BASE_EHRSHOT_DIR}/splits \
    --path_to_output_dir ${BASE_EHRSHOT_DIR}/cohort_stats \
    --num_threads 32
