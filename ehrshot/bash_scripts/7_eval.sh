#!/bin/bash

# NOTE: To run with slurm, pass the `--is_use_slurm` flag

# Time to run: ~0.5 hrs per subtask
# ChexPert = 4 hrs total
# CheXPert will be the bottleneck, so total run time should be ~4 hrs

# Default paths
path_to_database='../../EHRSHOT_ASSETS/femr/extract'
path_to_labels_dir='../../EHRSHOT_ASSETS/benchmark'
path_to_features_dir='../../EHRSHOT_ASSETS/features'
path_to_output_dir='../../EHRSHOT_ASSETS/results'
path_to_split_csv='../../EHRSHOT_ASSETS/splits/person_id_map.csv'

# Flag for SLURM
is_use_slurm=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --path_to_features_dir) path_to_features_dir="$2"; shift ;;
        --path_to_output_dir) path_to_output_dir="$2"; shift ;;
        --is_use_slurm) is_use_slurm=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

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
    # Labs take long time -- need more GB
    "lab_thrombocytopenia"
    "lab_hyperkalemia"
    "lab_hyponatremia"
    "lab_anemia"
    "lab_hypoglycemia" # will OOM at 200G on `gpu` partition
)
shot_strats=("all")
num_threads=20

for labeling_function in "${labeling_functions[@]}"; do
    for shot_strat in "${shot_strats[@]}"; do
        if [ "$is_use_slurm" = true ]; then
            sbatch 7__eval_helper.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_split_csv $path_to_output_dir ${labeling_function} ${shot_strat} $num_threads
        else
            bash 7__eval_helper.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_split_csv $path_to_output_dir ${labeling_function} ${shot_strat} $num_threads
        fi
    done
done