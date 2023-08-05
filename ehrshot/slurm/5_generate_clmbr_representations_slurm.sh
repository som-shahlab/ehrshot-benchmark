#!/bin/bash
#SBATCH --job-name=3_reps
#SBATCH --output=logs/job_%A.out
#SBATCH --error=logs/job_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=nigam-a100
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --exclude=secure-gpu-1,secure-gpu-2,secure-gpu-3

labeling_functions=(
    "guo_los" 
    "guo_readmission"
    "guo_icu"
    "new_hypertension"
    "new_hyperlipidemia"
    "new_pancan"
    "new_celiac"
    "new_lupus"
    "new_acutemi"
    "lab_thrombocytopenia"
    "lab_hyperkalemia"
    "lab_hypoglycemia"
    "lab_hyponatremia"
    "lab_anemia"
    "chexpert"
)

for labeling_function in "${labeling_functions[@]}"
do
    python3 5_generate_clmbr_representations.py \
        --path_to_output_dir ../EHRSHOT_ASSETS/models \
        --path_to_database ../EHRSHOT_ASSETS/femr/extract \
        --path_to_labels_and_feats_dir ../EHRSHOT_ASSETS/benchmark \
        --model clmbr
done
