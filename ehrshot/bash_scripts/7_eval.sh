#!/bin/bash

# NOTE: To run with slurm, pass the `--is_use_slurm` flag

# Time to run: ~0.5 hrs per subtask
# ChexPert = 4 hrs total
# CheXPert will be the bottleneck, so total run time should be ~4 hrs

# Example:
# bash 7_eval.sh mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last --ehrshot --is_use_slurm

BASE_EHRSHOT_DIR="../../EHRSHOT_ASSETS"

labeling_functions=(
    # "chexpert" # CheXpert first b/c slowest
    "guo_los"
    "guo_readmission"
    "guo_icu"
    "new_hypertension"
    "new_hyperlipidemia"
    "new_pancan"
    "new_acutemi"
    "new_celiac"
    "new_lupus"
    "lab_thrombocytopenia"
    "lab_hyperkalemia"
    "lab_hyponatremia"
    "lab_anemia"
    "lab_hypoglycemia"
)
path_to_database="${BASE_EHRSHOT_DIR}/femr/extract"
path_to_labels_dir="${BASE_EHRSHOT_DIR}/benchmark"
path_to_features_dir="${BASE_EHRSHOT_DIR}/features"
path_to_output_dir="${BASE_EHRSHOT_DIR}/results"
path_to_split_csv="${BASE_EHRSHOT_DIR}/splits/person_id_map.csv"

models="clmbr,count"
heads="lr_lbfgs,rf,gbm"
ks="1,2,4,8,16,32,64,128,-1"
shot_strats=("all")
num_threads=20

# CPU-bound jobs
for labeling_function in "${labeling_functions[@]}"; do
    for shot_strat in "${shot_strats[@]}"; do
        if [[ " $* " == *" --is_use_slurm "* ]]; then
            sbatch 7__eval_helper.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_split_csv $path_to_output_dir ${shot_strat} $ks $models $heads $num_threads ${labeling_function}
        else
            bash 7__eval_helper.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_split_csv $path_to_output_dir ${shot_strat} $ks $models $heads $num_threads ${labeling_function}
        fi
    done
done