#!/bin/bash
#SBATCH --job-name=1_create_femr_database
#SBATCH --output=logs/1_create_femr_database_%A.out
#SBATCH --error=logs/1_create_femr_database_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# Time to run: 10 mins

python3 ../1_create_femr_database.py \
    --path_to_input_dir ../../EHRSHOT_ASSETS/data \
    --path_to_output_dir ../../EHRSHOT_ASSETS/femr \
    --path_to_athena_download ../../EHRSHOT_ASSETS/athena_download \
    --num_threads 20 \
    --is_force_refresh