#!/bin/bash
#SBATCH --job-name=8_make_cohort_plots
#SBATCH --output=logs/8_make_cohort_plots_%A.out
#SBATCH --error=logs/8_make_cohort_plots_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32

python3 ../8_make_cohort_plots.py \
    --num_threads 32
