#!/bin/bash
#SBATCH --job-name=8_make_figures
#SBATCH --output=logs/8_make_figures_%A.out
#SBATCH --error=logs/8_make_figures_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

mkdir -p ../../EHRSHOT_ASSETS/figures

python3 ../8_make_results_plots.py \
    --path_to_labels_and_feats_dir ../../EHRSHOT_ASSETS/benchmark \
    --path_to_results_dir ../../EHRSHOT_ASSETS/results \
    --path_to_output_dir ../../EHRSHOT_ASSETS/figures \
    --model_heads "[('clmbr', 'lr_lbfgs'), ('count', 'lr_lbfgs'), ('count', 'gbm'), ('count', 'rf')]" \
    --shot_strat all