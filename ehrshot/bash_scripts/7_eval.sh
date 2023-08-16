#!/bin/bash

# Time to run: ~0.5 hrs per subtask
# ChexPert = 14 subtasks; Rest = 1 subtask
# CheXPert will be the bottleneck, so total run time should be ~7 hrs

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
num_threads=15

for labeling_function in "${labeling_functions[@]}"; do
    for shot_strat in "${shot_strats[@]}"; do
        sbatch 7__eval_helper.sh ${labeling_function} ${shot_strat} $num_threads
    done
done

# sbatch version
#
#SBATCH --job-name=7_eval
#SBATCH --output=logs/7_eval_%A.out
#SBATCH --error=logs/7_eval_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=180G
#SBATCH --cpus-per-task=15
#
# Time to run: 15 hrs = 7 hrs (all other tasks) + 8 hrs (ChexPert)
# for labeling_function in "${labeling_functions[@]}"; do
#     for shot_strat in "${shot_strats[@]}"; do
#         python3 ../7_eval.py \
#             --path_to_database ../../EHRSHOT_ASSETS/femr/extract \
#             --path_to_labels_dir ../../EHRSHOT_ASSETS/custom_benchmark \
#             --path_to_features_dir ../../EHRSHOT_ASSETS/custom_features \
#             --path_to_output_dir ../../EHRSHOT_ASSETS/results \
#             --labeling_function ${labeling_function} \
#             --shot_strat ${shot_strat} \
#             --num_threads 20
#     done
# done