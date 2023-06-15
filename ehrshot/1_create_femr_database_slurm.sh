#!/bin/bash
#SBATCH --job-name=femr
#SBATCH --output=logs/job_%A.out
#SBATCH --error=logs/job_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal,gpu
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10

# Usage:

python3 1_create_femr_database.py \
    --path_to_input ../EHRSHOT_ASSETS/data \
    --path_to_target ../EHRSHOT_ASSETS/femr \
    --athena_download ../EHRSHOT_ASSETS/athena_download \
    --num_threads 10
