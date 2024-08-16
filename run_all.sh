#!/bin/bash

cd ehrshot/bash_scripts

# Create FEMR database from raw CSVs
# NOTE: We'll skip this step since we provide the FEMR extract with our data release
# bash 1_create_femr_database.sh

# Use our labeling functions [defined here](https://github.com/som-shahlab/femr/blob/few_shot_ehr_benchmark/src/femr/labelers/benchmarks.py) 
# to generate labels for our dataset for our benchmark tasks.
# NOTE: We'll skip this step since we provide the labels with our data release
# bash 2_generate_labels.h

# Consolidate all labels together to speed up feature generation process
bash 3_consolidate_labels.sh

# Generate count-based feature representations
bash 4_generate_count_features.sh

# Generate CLMBR-T-base feature representations for the patients in our cohort.
# NOTE: This step requires a GPU
bash 5_generate_clmbr_features.sh

# Generate our k-shots for evaluation. 
# NOTE: We provide the k-shots used in the EHRSHOT paper with our data release, so do not run this script if you want to replicate the paper. 
# bash 6_generate_shots.sh

# Train baseline models and generate metrics.
bash 7_eval.sh

# Generate plots
bash 8_make_results_plots.sh

# Generate cohort statistics
bash 9_make_cohort_plots.sh
