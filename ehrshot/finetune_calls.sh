#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/bash_scripts/logs/finetune_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/bash_scripts/logs/finetune_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu,nigam-v100
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=20

# GPT-base-512
python3 finetune_test.py --model gpt-base-512 --task all --model_head finetune_full --k -1 --device cuda:0 &
python3 finetune_test.py --model gpt-base-512 --task all --model_head finetune_frozen --k -1 --device cuda:1 &
python3 finetune_test.py --model gpt-base-512 --task all --model_head finetune_layers=1 --k -1 --device cuda:2 &
python3 finetune_test.py --model gpt-base-512 --task all --model_head finetune_layers=2 --k -1 --device cuda:3 &

wait
# GPT-base-1024
# python3 finetune_test.py --model gpt-base-1024 --task all --model_head finetune_full --k -1 --device cuda:0 &
# python3 finetune_test.py --model gpt-base-1024 --task all --model_head finetune_frozen --k -1 --device cuda:1 &
# python3 finetune_test.py --model gpt-base-1024 --task all --model_head finetune_layers=1 --k -1 --device cuda:2 &
# python3 finetune_test.py --model gpt-base-1024 --task all --model_head finetune_layers=2 --k -1 --device cuda:3 &

# python3 finetune_test.py --model gpt-base-512 --task guo_los --model_head finetune_full --k -1 --device cuda:0
