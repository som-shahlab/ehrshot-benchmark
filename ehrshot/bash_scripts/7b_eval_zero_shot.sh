#!/bin/bash
#SBATCH --job-name=7b_eval_zero_shot
#SBATCH --output=logs/7b_eval_zero_shot_%A.out
#SBATCH --error=logs/7b_eval_zero_shot_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --exclude=secure-gpu-1,secure-gpu-2,secure-gpu-3,secure-gpu-4,secure-gpu-5,secure-gpu-6,secure-gpu-7

# MODIFY THESE
task="guo_los"
model="mamba-tiny-16384-att--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last"
# END MODIFY

# DO NOT MODIFY BELOW
batch_size=4
generation_ranges=(
    "0 5"
    "5 10"
    "10 15"
    "15 20"
)
for i in "${!generation_ranges[@]}"; do
    read start_idx end_idx <<< "${generation_ranges[$i]}"
    python3 ../7b_eval_zero_shot.py \
            --path_to_database '../../EHRSHOT_ASSETS/femr/extract' \
            --path_to_labels_dir '/share/pi/nigam/users/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark_ehrshot' \
            --path_to_features_dir '/share/pi/nigam/users/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/features_ehrshot' \
            --path_to_split_csv '/share/pi/nigam/users/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/splits_ehrshot/person_id_map.csv' \
            --path_to_tokenized_timelines_dir '/share/pi/nigam/users/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/tokenized_timelines_ehrshot' \
            --path_to_output_dir '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/results_ehrshot_zeroshot' \
            --shot_strat all \
            --labeling_function "${task}" \
            --batch_size "${batch_size}" \
            --device "cuda:${i}" \
            --start_generation_idx "${start_idx}" \
            --end_generation_idx "${end_idx}" \
            --models "${model}" &
done

wait