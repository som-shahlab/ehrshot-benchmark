#!/bin/bash
#SBATCH --job-name=8_make_figures
#SBATCH --output=logs/8_make_figures_%A.out
#SBATCH --error=logs/8_make_figures_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

BASE_EHRSHOT_DIR="../../EHRSHOT_ASSETS"

path_to_labels_dir="${BASE_EHRSHOT_DIR}/benchmark"
path_to_results_dir="${BASE_EHRSHOT_DIR}/results"
path_to_figures_dir="${BASE_EHRSHOT_DIR}/figures"

mkdir -p $path_to_figures_dir

python3 ../8_make_results_plots.py \
    --path_to_labels_and_feats_dir $path_to_labels_dir \
    --path_to_results_dir $path_to_results_dir \
    --path_to_output_dir $path_to_figures_dir \
    --shot_strat all \
    --model_heads "[('clmbr', 'lr_lbfgs'), \
                    ('count', 'lr_lbfgs'), \
                    ('count', 'rf'), \
                    ('count', 'gbm'), \
                ]"