#!/bin/bash
#SBATCH --job-name=2_generate_labels
#SBATCH --output=logs/2_generate_labels_%A.out
#SBATCH --error=logs/2_generate_labels_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=300G
#SBATCH --cpus-per-task=22

# Time to run: 6 mins

# Usage: sbatch 2_generate_labels.sh

BASE_EHRSHOT_DIR="../../EHRSHOT_ASSETS"

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
    # "chexpert" -- This depends on having access to NOTES, which we can't redistribute. Use the labels in the Redivis data release instead.
)
path_to_database="${BASE_EHRSHOT_DIR}/femr/extract"
path_to_labels_dir="${BASE_EHRSHOT_DIR}/benchmark"

mkdir -p $path_to_labels_dir

for labeling_function in "${labeling_functions[@]}"
do
    python3 ../2_generate_labels.py \
        --path_to_database $path_to_database \
        --path_to_labels_dir $path_to_labels_dir \
        --labeling_function ${labeling_function} \
        --num_threads 20
done