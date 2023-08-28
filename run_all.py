import os
import sys

sys.path.append("./ehrshot")
sys.path.append("./EHRSHOT_ASSETS")


TASKS=(
    "guo_los" 
    "guo_readmission"
    "guo_icu"
    "new_hypertension"
    "new_hyperlipidemia"
    "new_pancan"
    "new_celiac"
    "new_lupus"
    "new_acutemi"
    "lab_thrombocytopenia"
    "lab_hyperkalemia"
    "lab_hypoglycemia"
    "lab_hyponatremia"
    "lab_anemia"
    "chexpert"
)
num_threads =  16

# # ---------------------------------------Script 1---------------------------------------
"""
Convert the EHRSHOT cohort's CSV files into a FEMR database. 
Read more about [FEMR here](https://github.com/som-shahlab/femr).

Time: XXXXXX
"""
command = (
    'python3 ehrshot/1_create_femr_database.py'
    ' --path_to_input_dir EHRSHOT_ASSETS/data'
    ' --path_to_output_dir EHRSHOT_ASSETS/femr'
    ' --path_to_athena_download EHRSHOT_ASSETS/athena_download'
    f' --num_threads {num_threads}'
    ' --is_force_refresh'
)
print(command)
os.system(command)


# ---------------------------------------Script 2---------------------------------------
"""
Use our labeling functions [defined here](https://github.com/som-shahlab/femr/blob/few_shot_ehr_benchmark/src/femr/labelers/benchmarks.py) 
to generate labels for our dataset for our benchmark tasks.
"""
os.makedirs("EHRSHOT_ASSETS/custom_benchmark", exist_ok=True)
for task in TASKS:
    command = (
        'python3 ehrshot/2_generate_labels.py'
        ' --path_to_database EHRSHOT_ASSETS/femr/extract'
        ' --path_to_labels_dir EHRSHOT_ASSETS/custom_benchmark'
        ' --path_to_chexpert_csv EHRSHOT_ASSETS/benchmark/chexpert/chexpert_labeled_radiology_notes.csv'
        f' --labeling_function {task}'
        f' --num_threads {num_threads}'
    )
    print(command)
    os.system(command)
        
# ---------------------------------------Script 3---------------------------------------
"""
Consolidate all labels together to speed up feature generation process
"""
command = (
    'python3 ehrshot/3_consolidate_labels.py'
    ' --path_to_labels_dir EHRSHOT_ASSETS/custom_benchmark'
    ' --is_force_refresh'
)
print(command)
os.system(command)


# ---------------------------------------Script 4---------------------------------------
"""
Generate count-based feature representations
"""

os.makedirs("EHRSHOT_ASSETS/custom_features", exist_ok=True)
command = (
    'python3 ehrshot/4_generate_count_features.py'
    '--path_to_database EHRSHOT_ASSETS/femr/extract'
    '--path_to_labels_dir EHRSHOT_ASSETS/custom_benchmark'
    '--path_to_features_dir EHRSHOT_ASSETS/custom_features'
    f' --num_threads {num_threads}'
    f' --is_force_refresh'
)
print(command)
os.system(command)

# ---------------------------------------Script 5---------------------------------------
"""
Generate CLMBR-T-base feature representations for the patients in our cohort.
This step requires a GPU.
"""
    
command = (
    'python3 ehrshot/5_generate_clmbr_features.py'
    ' --path_to_database EHRSHOT_ASSETS/femr/extract'
    ' --path_to_labels_dir EHRSHOT_ASSETS/models/custom_benchmark'
    ' --path_to_features_dir EHRSHOT_ASSETS/models/custom_features'
    ' --path_to_models_dir EHRSHOT_ASSETS/models'
)
print(command)
os.system(command)

# ---------------------------------------Script 6---------------------------------------
"""
Generate our k-shots for evaluation. 
**Note**: We provide the k-shots we used with our data release. 
Please do not run this script if you want to use the k-shots we used in our paper. 
"""
shot_strats = ["all"]
num_replicates: int = 5
for task in TASKS:
    for shot_strat in shot_strats:
        command = (
            'python3 ehrshot/6_generate_shots.py'
            ' --path_to_database EHRSHOT_ASSETS/femr/extract'
            ' --path_to_labels_dir EHRSHOT_ASSETS/custom_benchmark'
            f' --labeling_function {task}'
            f' --shot_strat {shot_strat}'
            f' --num_replicates {num_replicates}'
        )
        print(command)
        os.system(command)

# ---------------------------------------Script 7---------------------------------------
"""
Next, we train our baseline models and generate the metrics.
"""
shot_strats = ["all"]
num_replicates: int = 5

for task in TASKS:
    for shot_strat in shot_strats:
        command = (
            'python3 ehrshot/7_eval.py '
            ' --path_to_database EHRSHOT_ASSETS/femr/extract'
            f' --path_to_labels_dir EHRSHOT_ASSETS/custom_benchmark'
            f' --path_to_features_dir EHRSHOT_ASSETS/custom_features'
            f' --path_to_output_dir EHRSHOT_ASSETS/results'
            f' --labeling_function {task}'
            f' --shot_strat {shot_strat}'
            f' --num_threads {num_threads}'
        )
        print(command)
        os.system(command)

# ---------------------------------------Script 8---------------------------------------
"""
Generate plots included in the EHRSHOT paper.
"""
os.makedirs("EHRSHOT_ASSETS/figures", exist_ok=True)
shot_strat: str = "all"
command = (
    'python3 ehrshot/8_make_figures.py'
    ' --path_to_labels_and_feats_dir EHRSHOT_ASSETS/custom_benchmark'
    ' --path_to_results_dir EHRSHOT_ASSETS/results'
    ' --path_to_output_dir EHRSHOT_ASSETS/figures'
    ' --model_heads "[(\'clmbr\', \'lr_lbfgs\'), (\'count\', \'gbm\')]"'
    f' --shot_strat {shot_strat}'   
)
print(command)
os.system(command)

# ---------------------------------------Script 9---------------------------------------
"""
Generate cohort statistic tables included in the EHRSHOT paper.
"""
os.makedirs("EHRSHOT_ASSETS/cohort_stats", exist_ok=True)
shot_strat: str = "all"
command = (
    'python3 ehrshot/9_make_cohort_plots.py'
    ' --path_to_database EHRSHOT_ASSETS/femr/extract'
    ' --path_to_labels_and_feats_dir EHRSHOT_ASSETS/custom_benchmark'
    ' --path_to_input_dir EHRSHOT_ASSETS/data'
    ' --path_to_splits_dir EHRSHOT_ASSETS/splits'
    ' --path_to_output_dir EHRSHOT_ASSETS/cohort_stats'
    f' --num_threads {num_threads}'   
)
print(command)
os.system(command)