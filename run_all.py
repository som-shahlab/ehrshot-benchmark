import os
import sys

sys.path.append("./ehrshot")
sys.path.append("./EHRSHOT_ASSETS")

# # ---------------------------------------Script 1---------------------------------------
"""
This script converts the csv files into femr database. Read more about [femr](https://github.com/som-shahlab/femr). 
The job can be run in a cpu.
"""
database_path = "EHRSHOT_ASSETS/femr"
if not os.path.exists(database_path):
    command = (
        'python3 '
        'ehrshot/1_create_femr_database.py '
        '--path_to_input "EHRSHOT_ASSETS/data" '
        '--path_to_target "EHRSHOT_ASSETS/femr" '
        '--athena_download "EHRSHOT_ASSETS/athena_download" '
        '--num_threads "10"'
    )
    os.system(command)


# ---------------------------------------Script 2---------------------------------------
"""
In this step, we will use our labeling functions defined in [femr](https://github.com/som-shahlab/femr/blob/few_shot_ehr_benchmark/src/femr/labelers/benchmarks.py) 
to generate labels for our dataset for our benchmark tasks. This job can be run on a cpu. 
"""
labeling_functions = [
    "guo_los", "guo_readmission", "guo_icu", "uden_hypertension", "uden_hyperlipidemia", "uden_pancan",
    "uden_celiac", "uden_lupus", "uden_acutemi", "thrombocytopenia_lab", "hyperkalemia_lab",
    "hypoglycemia_lab", "hyponatremia_lab", "anemia_lab", "chexpert"
]

for labeling_function in labeling_functions:
    command = (
        'python3 ehrshot/2_generate_labels_and_features.py '
        '--path_to_database EHRSHOT_ASSETS/femr/extract '
        '--path_to_output_dir EHRSHOT_ASSETS/benchmark '
        '--path_to_chexpert_csv EHRSHOT_ASSETS/benchmark/chexpert/chexpert_labeled_radiology_notes.csv '
        '--labeling_function {} '
        '--is_skip_label '
        '--num_threads 10'
    ).format(labeling_function)
    os.system(command)

# ---------------------------------------Script 3---------------------------------------
"""
In the third script, we generate the clmbr representation for the patients in our cohort for each label. 
Below is an example of how to run it for one label. This step requires gpu processing.
"""
clmbr_reps_dirs = "EHRSHOT_ASSETS/clmbr_reps"
os.makedirs(clmbr_reps_dirs, exist_ok=True)
for labeling_function in labeling_functions:
    task_dir = os.path.join(clmbr_reps_dirs, labeling_function)
    if not os.path.exists(task_dir):
        command = (
            'python3 ehrshot/3_generate_clmbr_representations.py '
            '--path_to_clmbr_data EHRSHOT_ASSETS/models/clmbr_model '
            '--path_to_database EHRSHOT_ASSETS/femr/extract '
            '--path_to_labeled_featurized_data EHRSHOT_ASSETS/benchmark '
            '--path_to_save {} '
            '--labeling_function {}'
        ).format(clmbr_reps_dirs, labeling_function)
        os.system(command)

# ---------------------------------------Script 4---------------------------------------
"""
Now, we will generate our k-shots for evaluation. **Note**: We provide the k-shots we used with our data release. 
Please do not run this script if you want to use the k-shots we used in our paper. If you want to regenrate those k-shots, 
then only run it. This process does not require gpu processing.
"""

# We provide data for few shots, so only run it for long shot.
shot_strats = ["long"]
for labeling_function in labeling_functions:
    for shot_strat in shot_strats:
        if shot_strat == "few":
            num_replicates = 5
        else:
            num_replicates = 1
        command = (
            'python3 ehrshot/4_generate_shot.py '
            '--path_to_data EHRSHOT_ASSETS '
            '--labeling_function {} '
            '--num_replicates {} '
            '--path_to_save EHRSHOT_ASSETS/benchmark '
            '--shot_strat {}'
        ).format(labeling_function, num_replicates, shot_strat)
        os.system(command)

# ---------------------------------------Script 5---------------------------------------
"""
Now, we will train our baseline models and generate the metrics. You can use cpu for this process.
"""
shot_strats = ["few", "long"]
eval_dirs = "EHRSHOT_ASSETS/output"
os.makedirs(eval_dirs, exist_ok=True)
for labeling_function in labeling_functions:
    for shot_strat in shot_strats:
        if shot_strat == "few":
            num_replicates = 5
        else:
            num_replicates = 1
        command = (
            'python3 ehrshot/5_eval.py '
            '--path_to_data EHRSHOT_ASSETS '
            '--labeling_function {} '
            '--num_replicates {} '
            '--model_head logistic '
            '--is_tune_hyperparams '
            '--path_to_save {} '
            '--shot_strat {}'
        ).format(labeling_function, num_replicates, eval_dirs, shot_strat)
        os.system(command)

# ---------------------------------------Script 6---------------------------------------
"""
Finally, we will generate all the plots that we included in our paper. This is a cpu job.
"""
figure_dirs = "EHRSHOT_ASSETS/figures"
os.makedirs(figure_dirs, exist_ok=True)
command = (
    'python3 ehrshot/6_make_figures.py '
    '--path_to_eval EHRSHOT_ASSETS/output '
    '--path_to_save {}'   
).format(figure_dirs)
os.system(command)









