#!/bin/bash
#SBATCH --job-name=8_make_figures
#SBATCH --output=logs/8_make_figures_%A.out
#SBATCH --error=logs/8_make_figures_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=150G
#SBATCH --cpus-per-task=20

python3 8_make_figures.py \
    --path_to_labels_and_feats_dir ../../EHRSHOT_ASSETS/custom_benchmark \
    --path_to_output_dir ../../EHRSHOT_ASSETS/figures