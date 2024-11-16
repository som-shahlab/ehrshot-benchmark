#!/bin/bash

# GPT
python3 ../7b_eval_zero_shot.py \
    	--path_to_database '../../EHRSHOT_ASSETS/femr/extract' \
        --path_to_labels_dir '/share/pi/nigam/users/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark_ehrshot' \
        --path_to_features_dir '/share/pi/nigam/users/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/features_ehrshot' \
        --path_to_split_csv '/share/pi/nigam/users/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/splits_ehrshot/person_id_map.csv' \
        --path_to_tokenized_timelines_dir '/share/pi/nigam/users/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/tokenized_timelines_ehrshot' \
        --path_to_output_dir '../../EHRSHOT_ASSETS/results_ehrshot_zeroshot' \
        --shot_strat all \
        --labeling_function guo_los \
        --batch_size 128 \
        --models gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last 


# Mamba
python3 ../7b_eval_zero_shot.py \
    	--path_to_database '../../EHRSHOT_ASSETS/femr/extract' \
        --path_to_labels_dir '/share/pi/nigam/users/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark_ehrshot' \
        --path_to_features_dir '/share/pi/nigam/users/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/features_ehrshot' \
        --path_to_split_csv '/share/pi/nigam/users/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/splits_ehrshot/person_id_map.csv' \
        --path_to_tokenized_timelines_dir '/share/pi/nigam/users/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/tokenized_timelines_ehrshot' \
        --path_to_output_dir '../../EHRSHOT_ASSETS/results_ehrshot_zeroshot' \
        --shot_strat all \
        --labeling_function guo_los \
        --batch_size 8 \
        --models mamba-tiny-16384-att--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last