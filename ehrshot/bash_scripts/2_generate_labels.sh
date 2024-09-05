#!/bin/bash
#SBATCH --job-name=2_generate_labels
#SBATCH --output=logs/2_generate_labels_%A.out
#SBATCH --error=logs/2_generate_labels_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=300G
#SBATCH --cpus-per-task=22

# Time to run: 6 mins

# Usage: 
#   EHRSHOT: sbatch 2_generate_labels.sh --ehrshot
#   MIMIC-IV: sbatch 2_generate_labels.sh --mimic4
#   EHRSHOT tasks on full STARR-OMOP: sbatch 2_generate_labels.sh --starr

if [[ " $* " == *" --mimic4 "* ]]; then
    labeling_functions=(
        # TODO: Add MIMIC-IV labeling functions here
    )
    path_to_database="/share/pi/nigam/datasets/femr_mimic4_extract"
    path_to_labels_dir="../../EHRSHOT_ASSETS/mimic4_benchmark"
elif [[ " $* " == *" --starr "* ]]; then
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
    )
    path_to_database="/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes"
    path_to_labels_dir="../../EHRSHOT_ASSETS/starr_benchmark"
else
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
    path_to_database="../../EHRSHOT_ASSETS/femr/extract"
    path_to_labels_dir="../../EHRSHOT_ASSETS/ehrshot_benchmark"
fi

mkdir -p $path_to_labels_dir
for labeling_function in "${labeling_functions[@]}"
do
    python3 ../2_generate_labels.py \
        --path_to_database $path_to_database \
        --path_to_labels_dir $path_to_labels_dir \
        --labeling_function ${labeling_function} \
        --num_threads 20
done
