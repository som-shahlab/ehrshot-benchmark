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


python3 run_all.py