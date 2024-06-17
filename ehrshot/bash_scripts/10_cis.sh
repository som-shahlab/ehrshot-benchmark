#!/bin/bash
#SBATCH --job-name=10_cis
#SBATCH --output=logs/10_cis_%A.out
#SBATCH --error=logs/10_cis_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32

python3 ../10_cis.py \
    --path_to_results_dir ../../EHRSHOT_ASSETS/results