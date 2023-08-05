#!/bin/bash
#SBATCH --job-name=3_reps
#SBATCH --output=logs/job_%A.out
#SBATCH --error=logs/job_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=nigam-a100
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --exclude=secure-gpu-1,secure-gpu-2,secure-gpu-3

labeling_functions=("guo_los" "guo_readmission" "guo_icu" "uden_hypertension" "uden_hyperlipidemia" "uden_pancan" "uden_celiac" "uden_lupus" "uden_acutemi" "thrombocytopenia_lab" "hyperkalemia_lab" "hypoglycemia_lab" "hyponatremia_lab" "anemia_lab" "chexpert")

for labeling_function in "${labeling_functions[@]}"
do
    python3 3c_generate_clmbr_representations.py \
        --path_to_clmbr_data ../EHRSHOT_ASSETS/models/clmbr_model \
        --path_to_database ../EHRSHOT_ASSETS/femr/extract \
        --path_to_labeled_featurized_data ../EHRSHOT_ASSETS/benchmark \
        --path_to_save ../EHRSHOT_ASSETS/clmbr_reps \
        --labeling_function ${labeling_function}
done
