#!/bin/bash
#SBATCH --job-name=5_eval
#SBATCH --output=logs/job_%A.out
#SBATCH --error=logs/job_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=nigam-a100
#SBATCH --mem=150G
#SBATCH --cpus-per-task=20
#SBATCH --exclude=secure-gpu-1,secure-gpu-2,secure-gpu-3

labeling_functions=("guo_los" "guo_readmission" "guo_icu" "uden_hypertension" "uden_hyperlipidemia" "uden_pancan" "uden_celiac" "uden_lupus" "uden_acutemi" "thrombocytopenia_lab" "hyperkalemia_lab" "hypoglycemia_lab" "hyponatremia_lab" "anemia_lab" "chexpert")
shot_strats=("few" "long")

# Iterate over labeling_functions
for labeling_function in "${labeling_functions[@]}"; do
    # Iterate over shot_strats
    for shot_strat in "${shot_strats[@]}"; do
        python3 5_eval.py \
            --path_to_data ../EHRSHOT_ASSETS \
            --labeling_function ${labeling_function} \
            --num_replicates 5 \
            --model_head logistic \
            --is_tune_hyperparams \
            --path_to_save ../EHRSHOT_ASSETS/output \
            --shot_strat ${shot_strat}
    done

done
