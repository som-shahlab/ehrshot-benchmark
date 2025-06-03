#!/bin/bash

# NOTE: To run with slurm, pass the `--is_use_slurm` flag

# Time to run: ~0.5 hrs per subtask
# ChexPert = 4 hrs total
# CheXPert will be the bottleneck, so total run time should be ~4 hrs

# Usage: 
#   EHRSHOT: bash 7_eval.sh model1,model2 --ehrshot --is_use_slurm
#   MIMIC-IV: bash 7_eval.sh model1,model2 --mimic4 --is_use_slurm
#   EHRSHOT tasks on full STARR-OMOP: bash 7_eval.sh model1,model2 --starr --is_use_slurm

BASE_EHRSHOT_DIR="../../../EHRSHOT_ASSETS"

if [[ " $* " == *" --mimic4 "* ]]; then
    labeling_functions=(
        "mimic4_los" 
        "mimic4_readmission"
        "mimic4_mortality"
    )
    path_to_database="/share/pi/nigam/datasets/femr_mimic_4_extract"
    path_to_labels_dir="${BASE_EHRSHOT_DIR}/benchmark_mimic4"
    path_to_features_dir="${BASE_EHRSHOT_DIR}/features_mimic4"
    path_to_output_dir="${BASE_EHRSHOT_DIR}/results_mimic4"
    path_to_split_csv="${BASE_EHRSHOT_DIR}/splits_mimic4/person_id_map.csv"
elif [[ " $* " == *" --starr "* ]]; then
    labeling_functions=(
        #"chexpert" # CheXpert first b/c slowest
        "guo_los"
        "guo_readmission"
        "guo_icu"
        "new_hypertension"
        "new_hyperlipidemia"
        "new_pancan"
        "new_acutemi"
        "new_celiac"
        "new_lupus"
        #"lab_thrombocytopenia"
        #"lab_hyperkalemia"
        #"lab_hyponatremia"
        #"lab_anemia"
        #"lab_hypoglycemia" # will OOM at 200G on `gpu` partition
    )
    path_to_database="/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes"
    path_to_labels_dir="${BASE_EHRSHOT_DIR}/benchmark_starr"
    path_to_features_dir="${BASE_EHRSHOT_DIR}/features_starr"
    path_to_output_dir="${BASE_EHRSHOT_DIR}/results_starr"
    path_to_split_csv="${BASE_EHRSHOT_DIR}/splits_starr/person_id_map.csv"
    path_to_tokenized_timelines="${BASE_EHRSHOT_DIR}/tokenized_timelines_starr"
else
    labeling_functions=(
        "chexpert" # CheXpert first b/c slowest
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
        "lab_hypoglycemia" # will OOM at 200G on `gpu` partition
    )
    path_to_database="${BASE_EHRSHOT_DIR}/femr/extract"
    path_to_labels_dir="${BASE_EHRSHOT_DIR}/benchmark"
    path_to_features_dir="${BASE_EHRSHOT_DIR}/features"
    path_to_output_dir="${BASE_EHRSHOT_DIR}/results"
    path_to_split_csv="${BASE_EHRSHOT_DIR}/splits/person_id_map.csv"
    path_to_tokenized_timelines="${BASE_EHRSHOT_DIR}/tokenized_timelines"
fi

models=$1
# ks="32,128,-1"
ks="-1"
shot_strats=("all")
num_threads=20

# GPU-bound jobs (loop in chunks of 3 to fit multiple jobs on same GPU node)
for (( i=0; i<${#labeling_functions[@]}; i+=2 )); do
    chunk=("${labeling_functions[@]:i:2}")
    for shot_strat in "${shot_strats[@]}"; do
        if [[ " $* " == *" --is_use_slurm "* ]]; then
            sbatch 7__eval_helper_gpu.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_split_csv $path_to_output_dir ${shot_strat} $ks $models $num_threads $path_to_tokenized_timelines "${chunk[0]}" "${chunk[1]}"
        else
            bash 7__eval_helper_gpu.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_split_csv $path_to_output_dir ${shot_strat} $ks $models $num_threads $path_to_tokenized_timelines "${chunk[0]}" "${chunk[1]}"
        fi
    done
done
