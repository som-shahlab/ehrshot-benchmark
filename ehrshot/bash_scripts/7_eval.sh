#!/bin/bash

# NOTE: To run with slurm, pass the `--is_use_slurm` flag

# Time to run: ~0.5 hrs per subtask
# ChexPert = 4 hrs total
# CheXPert will be the bottleneck, so total run time should be ~4 hrs

path_to_database='../../EHRSHOT_ASSETS/femr/extract'
path_to_labels_dir='../../EHRSHOT_ASSETS/benchmark'
path_to_features_dir='../../EHRSHOT_ASSETS/features'
path_to_output_dir='../../EHRSHOT_ASSETS/results'
path_to_split_csv='../../EHRSHOT_ASSETS/splits/person_id_map.csv'

labeling_functions=(
    "chexpert" # CheXpert first b/c slowest
    "guo_los"
    "guo_readmission"
    "guo_icu"
    "new_hypertension"
    "new_hyperlipidemia"
    "new_pancan"
    "new_acutemi"
    # "new_celiac" # TODO -- ignore for now b/c noisy
    # "new_lupus" # TODO -- ignore for now b/c noisy
    # Labs take long time -- need more GB
    "lab_thrombocytopenia"
    "lab_hyperkalemia"
    "lab_hyponatremia"
    "lab_anemia"
    "lab_hypoglycemia" # will OOM at 200G on `gpu` partition
)
shot_strats=("all")
num_threads=20

# CPU-bound jobs
for labeling_function in "${labeling_functions[@]}"; do
    for shot_strat in "${shot_strats[@]}"; do
        if [[ " $* " == *" --is_use_slurm "* ]]; then
            sbatch 7__eval_helper.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_split_csv $path_to_output_dir ${shot_strat} $num_threads ${labeling_function}
        else
            bash 7__eval_helper.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_split_csv $path_to_output_dir ${shot_strat} $num_threads ${labeling_function}
        fi
    done
done

# GPU-bound jobs (loop in chunks of 3 to fit multiple jobs on same GPU node)
# for (( i=0; i<${#labeling_functions[@]}; i+=3 )); do
#     chunk=("${labeling_functions[@]:i:3}")
#     for shot_strat in "${shot_strats[@]}"; do
#         if [[ " $* " == *" --is_use_slurm "* ]]; then
#             sbatch 7__eval_helper_gpu.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_split_csv $path_to_output_dir ${shot_strat} $num_threads "${chunk[0]}" "${chunk[1]}" "${chunk[2]}"
#         else
#             bash 7__eval_helper_gpu.sh $path_to_database $path_to_labels_dir $path_to_features_dir $path_to_split_csv $path_to_output_dir ${shot_strat} $num_threads "${chunk[0]}" "${chunk[1]}" "${chunk[2]}"
#         fi
#     done
# done