#!/bin/bash
#SBATCH --job-name=7_eval
#SBATCH --output=logs/7_eval_%A.out
#SBATCH --error=logs/7_eval_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# Time to run: XXXX (12 mins per task)

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
shot_strats=("all")

for labeling_function in "${labeling_functions[@]}"; do
    for shot_strat in "${shot_strats[@]}"; do
        python3 ../7_eval.py \
            --path_to_database ../../EHRSHOT_ASSETS/femr/extract \
            --path_to_labels_dir ../../EHRSHOT_ASSETS/custom_benchmark \
            --path_to_features_dir ../../EHRSHOT_ASSETS/custom_features \
            --path_to_output_dir ../../EHRSHOT_ASSETS/results \
            --labeling_function ${labeling_function} \
            --shot_strat ${shot_strat} \
            --num_threads 20
    done
done