#!/bin/bash
#SBATCH --job-name=3_consolidate_labels
#SBATCH --output=logs/3_consolidate_labels_%A.out
#SBATCH --error=logs/3_consolidate_labels_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=100G
#SBATCH --cpus-per-task=5

# Time to run: 10 secs

python3 ../2_consolidate_labels.py \
    --is_force_refresh
    # --path_to_labels_dir ../../EHRSHOT_ASSETS/labels_no_visit_merge \