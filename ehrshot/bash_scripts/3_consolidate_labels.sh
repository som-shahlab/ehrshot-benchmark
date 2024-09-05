#!/bin/bash
#SBATCH --job-name=3_consolidate_labels
#SBATCH --output=logs/3_consolidate_labels_%A.out
#SBATCH --error=logs/3_consolidate_labels_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=100G
#SBATCH --cpus-per-task=5

# Time to run: 10 secs

# Usage: 
#   EHRSHOT: sbatch 3_consolidate_labels.sh --ehrshot
#   MIMIC-IV: sbatch 3_consolidate_labels.sh --mimic4
#   EHRSHOT tasks on full STARR-OMOP: sbatch 3_consolidate_labels.sh --starr

if [[ " $* " == *" --mimic4 "* ]]; then
    path_to_labels_dir="../../EHRSHOT_ASSETS/mimic4_benchmark"
elif [[ " $* " == *" --starr "* ]]; then
    path_to_labels_dir="../../EHRSHOT_ASSETS/starr_benchmark"
else
    path_to_labels_dir="../../EHRSHOT_ASSETS/ehrshot_benchmark"
fi

python3 ../3_consolidate_labels.py \
<<<<<<< HEAD
    --path_to_labels_dir ../../EHRSHOT_ASSETS/benchmark-full \
=======
    --path_to_labels_dir $path_to_labels_dir \
>>>>>>> eb5e4b624baa61902fe35e6f26272b539acf884f
    --is_force_refresh