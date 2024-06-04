#!/bin/bash
#SBATCH --job-name=9_make_cohort_plots
#SBATCH --output=logs/9_make_cohort_plots_%A.out
#SBATCH --error=logs/9_make_cohort_plots_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32

mkdir -p ../../EHRSHOT_ASSETS/cohort_stats

python3 ../9_make_cohort_plots.py \
    --path_to_database ../../EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_and_feats_dir ../../EHRSHOT_ASSETS/benchmark \
    --path_to_input_dir ../../EHRSHOT_ASSETS/data \
    --path_to_splits_dir ../../EHRSHOT_ASSETS/splits \
    --path_to_output_dir ../../EHRSHOT_ASSETS/cohort_stats \
    --num_threads 32
