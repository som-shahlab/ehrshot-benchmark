#!/bin/bash

# Time to run: ~0.5 hrs per subtask
# ChexPert = 4 hrs total
# CheXPert will be the bottleneck, so total run time should be ~4 hrs

path_to_database='../../EHRSHOT_ASSETS/femr/extract'
path_to_labels_dir='../../EHRSHOT_ASSETS/custom_benchmark'
path_to_features_dir='../../EHRSHOT_ASSETS/custom_features'
path_to_output_dir='../../EHRSHOT_ASSETS/results'

labeling_functions=(
    "chexpert" # CheXpert first b/c slowest
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
shot_strats=("all")
num_threads=10

for labeling_function in "${labeling_functions[@]}"; do
    for shot_strat in "${shot_strats[@]}"; do
        sbatch 7__eval_helper.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_output_dir ${labeling_function} ${shot_strat} $num_threads
    done
done