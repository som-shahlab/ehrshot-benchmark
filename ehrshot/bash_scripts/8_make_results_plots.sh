#!/bin/bash
#SBATCH --job-name=8_make_figures
#SBATCH --output=logs/8_make_figures_%A.out
#SBATCH --error=logs/8_make_figures_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

mkdir -p ../../EHRSHOT_ASSETS/figures

# --model_heads "[('clmbr', 'lr_lbfgs'), ('llm', 'lr_lbfgs'), ('agr', 'lr_lbfgs'), ('count', 'lr_lbfgs'), ('count', 'gbm'), ('count', 'rf')]" \
# --path_to_results_dir ../../EHRSHOT_ASSETS/results \

python3 ../8_make_results_plots.py \
    --path_to_labels_and_feats_dir ../../EHRSHOT_ASSETS/benchmark \
    --path_to_results_dir ../../EHRSHOT_ASSETS/experiments/full_run/gteqwen2_7b_instruct_unique_then_list_visits_wo_allconds_w_values_4k_no_labs_single_3_0_full_with_baselines_and_llama \
    --path_to_output_dir ../../EHRSHOT_ASSETS/figures \
    --model_heads "[('count', 'gbm'), ('clmbr', 'lr_lbfgs'), ('llm', 'lr_lbfgs'), ('llm_llama', 'lr_lbfgs')]" \
    --shot_strat all