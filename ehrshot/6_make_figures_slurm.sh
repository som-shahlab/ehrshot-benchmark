#!/bin/bash
#SBATCH --job-name=5_eval
#SBATCH --output=logs/job_%A.out
#SBATCH --error=logs/job_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=nigam-a100
#SBATCH --mem=150G
#SBATCH --cpus-per-task=20


python3 6_make_figures.py \
    --path_to_eval ../EHRSHOT_ASSETS/output \
    --path_to_save ../EHRSHOT_ASSETS/figures