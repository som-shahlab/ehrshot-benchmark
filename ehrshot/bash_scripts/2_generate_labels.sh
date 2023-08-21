#!/bin/bash
#SBATCH --job-name=2_generate_labels
#SBATCH --output=logs/2_generate_labels_%A.out
#SBATCH --error=logs/2_generate_labels_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=22

# Time to run: 6 mins

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

mkdir -p ../../EHRSHOT_ASSETS/custom_benchmark

for labeling_function in "${labeling_functions[@]}"
do
    python3 ../2_generate_labels.py \
        --path_to_database ../../EHRSHOT_ASSETS/femr/extract \
        --path_to_labels_dir ../../EHRSHOT_ASSETS/custom_benchmark \
        --path_to_chexpert_csv ../../EHRSHOT_ASSETS/custom_benchmark/chexpert/chexpert_labeled_radiology_notes.csv \
        --labeling_function ${labeling_function} \
        --num_threads 20
done