#!/bin/bash
#SBATCH --job-name=6__eval_helper
#SBATCH --output=logs/6__eval_helper_%A.out
#SBATCH --error=logs/6__eval_helper_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal,gpu,nigam-v100
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --exclude=secure-gpu-3,secure-gpu-4,secure-gpu-5,secure-gpu-6,secure-gpu-7

python3 ../6_eval.py \
    --labeling_function $1 \
    --shot_strat $2 \
    --num_threads $3
    # --path_to_database ../../EHRSHOT_ASSETS/database_no_visit_merge \
    # --path_to_labels_dir ../../EHRSHOT_ASSETS/labels_no_visit_merge \
    # --path_to_features_dir ../../EHRSHOT_ASSETS/features_no_visit_merge \
    # --path_to_output_dir ../../EHRSHOT_ASSETS/outputs_no_visit_merge

#SBATCH --partition=nigam-a100
#SBATCH --mem=400G
#SBATCH --cpus-per-task=20


#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20