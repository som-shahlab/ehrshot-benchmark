import os
import sys

sys.path.append("./ehrshot")
sys.path.append("./EHRSHOT_ASSETS")

TASKS = [
    "guo_los", "guo_readmission", "guo_icu", "uden_hypertension", "uden_hyperlipidemia", "uden_pancan",
    "uden_celiac", "uden_lupus", "uden_acutemi", "thrombocytopenia_lab", "hyperkalemia_lab",
    "hypoglycemia_lab", "hyponatremia_lab", "anemia_lab"
]

num_threads =  16

# # ---------------------------------------Script 1---------------------------------------
"""
This script converts the CSV files into a FEMR database. 
Read more about [FEMR here](https://github.com/som-shahlab/femr).

Time: ~10 minutes using 10 threads
"""
database_path = "EHRSHOT_ASSETS/femr"
if not os.path.exists(database_path):
    command = (
        'python3 '
        'ehrshot/1_create_femr_database.py '
        '--path_to_input EHRSHOT_ASSETS/data '
        '--path_to_target EHRSHOT_ASSETS/femr '
        '--athena_download EHRSHOT_ASSETS/athena_download '
        '--num_threads ' + str(num_threads)
    )
    os.system(command)


# ---------------------------------------Script 2---------------------------------------
"""
Next, we will use our labeling functions [defined here](https://github.com/som-shahlab/femr/blob/few_shot_ehr_benchmark/src/femr/labelers/benchmarks.py) 
to generate labels for our dataset for our benchmark tasks.
"""
labels_path = 'EHRSHOT_ASSETS/labels'
if not os.path.exists(labels_path):
    for task in TASKS:
        command = (
            'python3 ehrshot/2_generate_labels.py '
            '--path_to_database EHRSHOT_ASSETS/femr/extract '
            '--path_to_output_dir EHRSHOT_ASSETS/labels '
            '--path_to_chexpert_csv EHRSHOT_ASSETS/benchmark/chexpert/chexpert_labeled_radiology_notes.csv '
            '--labeling_function {} '
            '--is_skip_label '
            '--num_threads ' + str(num_threads)
        ).format(task)
        os.system(command)

# ---------------------------------------Script 3---------------------------------------
"""
Next, we generate the feature representations for the patients in our cohort.
This step requires a GPU.
"""

"""
First, we will conslidate the labels before generating features
"""

if not os.path.exists('EHRSHOT_ASSETS/features'):
    os.mkdir('EHRSHOT_ASSETS/features')

    command = (
        'python3 ehrshot/3a_consolidate_labels.py '
        '--path_to_output_dir EHRSHOT_ASSETS/labels '
        '--path_to_features_dir EHRSHOT_ASSETS/features '
    )
    print(command)
    os.system(command)

    """
    Now we will generate count based features
    """

    command = (
        'python3 ehrshot/3b_generate_features.py '
        '--path_to_database EHRSHOT_ASSETS/femr/extract '
        '--path_to_features_dir EHRSHOT_ASSETS/features '
        '--num_threads ' + str(num_threads)
    )
    print(command)
    os.system(command)

    command = (
        'python3 ehrshot/3c_generate_clmbr_representations.py '
        '--path_to_database EHRSHOT_ASSETS/femr/extract '
        '--path_to_clmbr_data EHRSHOT_ASSETS/models/clmbr_model '
        '--path_to_features EHRSHOT_ASSETS/features '
    )
    print(command)
    os.system(command)

# ---------------------------------------Script 4---------------------------------------
"""
Next, we will generate our k-shots for evaluation. 
**Note**: We provide the k-shots we used with our data release. 
Please do not run this script if you want to use the k-shots we used in our paper. 
"""
# We provide data for few shots, so only run it for long shot.

shot_strats = ["long"]
for task in TASKS:
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
        ).format(task, num_replicates, shot_strat)
        os.system(command)

# ---------------------------------------Script 5---------------------------------------
"""
Next, we train our baseline models and generate the metrics.
"""
#shot_strats = ["few", "long"]
shot_strats = ["long"]
eval_dirs = "EHRSHOT_ASSETS/output"
os.makedirs(eval_dirs, exist_ok=True)

for task in TASKS:
    for shot_strat in shot_strats:
        if shot_strat == "few":
            num_replicates = 5
        else:
            num_replicates = 1
        command = (
            'python3 ehrshot/5_eval.py '
            '--path_to_database EHRSHOT_ASSETS/femr/extract '
            f'--path_to_labels EHRSHOT_ASSETS/labels/{task} '
            f'--path_to_features EHRSHOT_ASSETS/features '
            f'--path_to_save EHRSHOT_ASSETS/output/{task} '
            f'--num_threads {num_threads}'
        )
        os.system(command)

# ---------------------------------------Script 6---------------------------------------
"""
Finally, we generate the plots that included in our paper.
"""
figure_dirs = "EHRSHOT_ASSETS/figures"
os.makedirs(figure_dirs, exist_ok=True)

command = (
    'python3 ehrshot/6_make_figures.py '
    '--path_to_eval "EHRSHOT_ASSETS/output" '
    '--path_to_save {}'   
).format(figure_dirs)
os.system(command)









