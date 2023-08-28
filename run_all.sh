#!/bin/bash
#SBATCH --job-name=run_all
#SBATCH --output=logs/run_all%A.out
#SBATCH --error=logs/run_all%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=nigam-a100,gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --exclude=secure-gpu-1,secure-gpu-2,secure-gpu-3

python3 run_all.py