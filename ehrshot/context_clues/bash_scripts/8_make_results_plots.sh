#!/bin/bash
#SBATCH --job-name=8_make_figures
#SBATCH --output=logs/8_make_figures_%A.out
#SBATCH --error=logs/8_make_figures_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

if [[ " $* " == *" --mimic4 "* ]]; then
    path_to_labels_dir="../../EHRSHOT_ASSETS/benchmark_mimic4"
    path_to_results_dir='../../EHRSHOT_ASSETS/results_mimic4'
    path_to_figures_dir="../../EHRSHOT_ASSETS/figures_mimic4"
elif [[ " $* " == *" --starr "* ]]; then
    path_to_labels_dir="../../EHRSHOT_ASSETS/starr_benchmark"
    path_to_results_dir='../../EHRSHOT_ASSETS/starr_results'
    path_to_figures_dir="../../EHRSHOT_ASSETS/starr_figures"
else
    path_to_labels_dir="../../EHRSHOT_ASSETS/benchmark"
    path_to_results_dir='../../EHRSHOT_ASSETS/results_zeroshot'
    path_to_figures_dir="../../EHRSHOT_ASSETS/figures_zeroshot"
fi

mkdir -p $path_to_figures_dir

python3 ../8_make_results_plots.py \
    --path_to_labels_and_feats_dir /share/pi/nigam/$USER/ehrshot-benchmark/ehrshot/bash_scripts/$path_to_labels_dir \
    --path_to_results_dir /share/pi/nigam/$USER/ehrshot-benchmark/ehrshot/bash_scripts/$path_to_results_dir \
    --path_to_output_dir $path_to_figures_dir \
    --shot_strat all \
    --model_heads "[('clmbr', 'lr_lbfgs'), \
                    ('llama-base-512-att--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last', 'zero_shot'), \
                    ('llama-base-4096-att--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last', 'zero_shot'), \
                    ('mamba-tiny-1024-att--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last', 'zero_shot'), \
                    ('mamba-tiny-16384-att--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last', 'zero_shot'), \
                ]"
# Zero-shot
    # --model_heads "[('clmbr', 'lr_lbfgs'), \
    #                 ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('gpt2-base-2048--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('gpt2-base-4096--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #             ]"
# Everything

    # --model_heads "[('clmbr', 'lr_lbfgs'), \
    #                 ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('bert-base-512--clmbr_train-tokens-total_nonPAD-true_val=600002304-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

    #                 ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

    #                 ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

    #                 ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('mamba-tiny-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

    #                 ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('hyena-medium-8192--clmbr_train-tokens-total_nonPAD-true_val=600001472-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

    #                 ('mamba-tiny-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('mamba-tiny-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('mamba-tiny-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('mamba-tiny-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('mamba-tiny-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \

    #                 ('hyena-medium-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'lr_lbfgs'), \
    #                 ('hyena-medium-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=1'), \
    #                 ('hyena-medium-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_layer=2'), \
    #                 ('hyena-medium-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_frozen'), \
    #                 ('hyena-medium-16834--clmbr_train-tokens-total_nonPAD-true_val=600008768-ckpt_val=600000000-persist_chunk:last_embed:last', 'finetune_full'), \
    #             ]"
