#!/bin/bash
#SBATCH --job-name=8_make_figures
#SBATCH --output=logs/8_make_figures_%A.out
#SBATCH --error=logs/8_make_figures_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

mkdir -p ../../EHRSHOT_ASSETS/figures_hf

python3 ../8_make_results_plots.py \
    --path_to_labels_and_feats_dir ../../EHRSHOT_ASSETS/custom_benchmark \
    --path_to_results_dir ../../EHRSHOT_ASSETS/results_hf \
    --path_to_output_dir ../../EHRSHOT_ASSETS/figures_hf \
    --model_heads "[('clmbr', 'lr_femr'), ('count', 'gbm'), ('gpt2-base-v8_chunk:last_embed:last', 'lr_lbfgs'), ('bert-base-v8_chunk:last_embed:last', 'lr_lbfgs')]" \
    --shot_strat all

# --model_heads "[('clmbr', 'lr_femr'), ('count', 'lr_lbfgs'), ('count', 'gbm'), ('count', 'rf'), ('gpt2-base', 'lr_lbfgs')]" \
    # --model_heads "[('gpt2-base', 'lr_lbfgs'), ('gpt2-base', 'gbm'), ('gpt2-base', 'rf')]" \
    # --model_heads "[('gpt2-base-v8_chunk:last_embed:last', 'lr_lbfgs'), ('gpt2-base-v8_chunk:last_embed:last', 'gbm'), ('gpt2-base-v8_chunk:last_embed:last', 'rf')]" \
