#!/bin/bash
#SBATCH --job-name=3_consolidate_labels
#SBATCH --output=logs/3_consolidate_labels_%A.out
#SBATCH --error=logs/3_consolidate_labels_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=100G
#SBATCH --cpus-per-task=5

# Time to run: 10 secs

# Usage: sbatch 3_consolidate_labels.sh

BASE_EHRSHOT_DIR="../../EHRSHOT_ASSETS"

path_to_labels_dir="${BASE_EHRSHOT_DIR}/benchmark"

python3 ../3_consolidate_labels.py \
    --path_to_labels_dir $path_to_labels_dir \
    --is_force_refresh