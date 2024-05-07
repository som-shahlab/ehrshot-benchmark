#!/bin/bash

cd ehrshot/bash_scripts

# Use our labeling functions [defined here](https://github.com/som-shahlab/femr/blob/few_shot_ehr_benchmark/src/femr/labelers/benchmarks.py) 
# to generate labels for our dataset for our benchmark tasks.
bash 1_generate_labels.sh

# Consolidate all labels together to speed up feature generation process
bash 2_consolidate_labels.sh

# Generate count-based feature representations
bash 3_generate_baseline_features.sh

# Generate CLMBR-T-base feature representations for the patients in our cohort.
# NOTE: This step requires a GPU
bash 4_generate_clmbr_feature.sh

# Generate our k-shots for evaluation. 
# NOTE: We provide the k-shots used in the EHRSHOT paper with our data release, so do not run this script if you want to replicate the paper. 
bash 5_generate_shots.sh

# Train baseline models and generate metrics.
# NOTE: Remove the `--is_use_slurm` flag if you aren't on a SLURM cluster
bash 6_eval.sh --is_use_slurm

# Generate plots from the EHRSHOT paper.
bash 7_make_results_plots.sh

# Generate cohort statistics from the EHRSHOT paper.
bash 8_make_cohort_plots.sh