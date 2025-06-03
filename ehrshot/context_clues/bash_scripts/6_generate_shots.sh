#!/bin/bash
#SBATCH --job-name=6_generate_shots
#SBATCH --output=logs/6_generate_shots_%A.out
#SBATCH --error=logs/6_generate_shots_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# Time to run: 2 mins

# Usage: 
#   EHRSHOT: sbatch 6_generate_shots.sh --ehrshot
#   MIMIC-IV: sbatch 6_generate_shots.sh --mimic4
#   EHRSHOT tasks on full STARR-OMOP: sbatch 6_generate_shots.sh --starr

if [[ " $* " == *" --mimic4 "* ]]; then
    echo "MIMIC4" >&2
    labeling_functions=(
        "mimic4_los" 
        "mimic4_readmission"
        "mimic4_mortality"
    )
    path_to_database="/share/pi/nigam/datasets/femr_mimic_4_extract"
    path_to_labels_dir="../../EHRSHOT_ASSETS/benchmark_mimic4"
    path_to_split_csv="../../EHRSHOT_ASSETS/splits_mimic4/person_id_map.csv"
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
        # "lab_thrombocytopenia"
        # "lab_hyperkalemia"
        # "lab_hypoglycemia"
        # "lab_hyponatremia"
        # "lab_anemia"
        # "chexpert"
    )
    path_to_database="/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes"
    path_to_labels_dir="../../EHRSHOT_ASSETS/benchmark_starr"
    path_to_split_csv="../../EHRSHOT_ASSETS/splits_starr/person_id_map.csv"
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
        "chexpert"
    )
    path_to_database="../../EHRSHOT_ASSETS/femr/extract"
    path_to_labels_dir="../../EHRSHOT_ASSETS/benchmark"
    path_to_split_csv="../../EHRSHOT_ASSETS/splits/person_id_map.csv"
fi

shot_strats=("all")


for labeling_function in "${labeling_functions[@]}"; do
    for shot_strat in "${shot_strats[@]}"; do
    python3 ../6_generate_shots.py \
        --path_to_database $path_to_database \
        --path_to_labels_dir $path_to_labels_dir \
        --path_to_split_csv $path_to_split_csv \
        --labeling_function ${labeling_function} \
        --shot_strat ${shot_strat} \
        --n_replicates 5
    done
done

echo "Done!" >&2



# python3 ../6_generate_shots.py \
#         --path_to_database "/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes" \
#         --path_to_labels_dir "../../EHRSHOT_ASSETS/benchmark_starr" \
#         --path_to_split_csv "../../EHRSHOT_ASSETS/splits_starr/person_id_map.csv" \
#         --labeling_function "guo_los" \
#         --shot_strat "all" \
#         --n_replicates 5