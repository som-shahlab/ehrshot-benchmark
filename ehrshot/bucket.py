"""
Buckets EHRSHOT patients by a specific stratification metric and reports metrics for each bucket.

Usage:
    python3 bucket.py --task guo_los
"""
import datetime
import argparse
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import collections
import pandas as pd
from tqdm import tqdm
from utils import (
    LABELING_FUNCTION_2_PAPER_NAME,
    MODEL_2_INFO,
    get_labels_and_features, 
    process_chexpert_labels, 
    convert_multiclass_to_binary_labels,
    CHEXPERT_LABELS, 
    get_patient_splits_by_idx
)
from femr.datasets import PatientDatabase
from jaxtyping import Float
from femr.labelers import load_labeled_patients, LabeledPatients
from hf_ehr.utils import load_tokenizer_from_path, load_model_from_path
from starr_eda import calc_n_gram_count, calc_inter_event_times

if __name__ == '__main__':
    
    # 1. You need to load the stratify.py outputs for this TASK
    # 2. You need to load the EHRSHOT results for this TASK
    # 3. You need to load the labeled patients for this TASK
    # 4. You need to split the patients into 4 buckets based on their df_stratify metrics
        # i.e. 0-25, 25-50, 50-75, 75-100th percentiles
    # 5. Recalculate AUROC within each bucket
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Type of task to perform', default='guo_los')
    parser.add_argument('--model', type=str, help='Model whose results we stratify', default='clmbr')
    args = parser.parse_args()

    # Constants
    PATH_TO_DATABASE: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract'
    PATH_TO_FEATURES_DIR: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/features_ehrshot'
    PATH_TO_RESULTS_DIR: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/results_ehrshot'
    PATH_TO_TOKENIZED_TIMELINES_DIR: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/tokenized_timelines_ehrshot'
    PATH_TO_LABELS_DIR: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark_ehrshot'
    PATH_TO_SPLIT_CSV: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/splits_ehrshot/person_id_map.csv'
    
    # Output directory
    path_to_output_dir: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/stratify/'
    os.makedirs(path_to_output_dir, exist_ok=True)

    # Load labeled patients for this task
    LABELING_FUNCTION: str = args.task
    PATH_TO_LABELED_PATIENTS: str =  os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION, 'labeled_patients.csv')
    femr_db = PatientDatabase(PATH_TO_DATABASE)
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)

    # Get features for patients
    model: str = 'gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last'
    # model: str = 'mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last'
    patient_ids, label_values, label_times, feature_matrixes = get_labels_and_features(labeled_patients, 
                                                                                        PATH_TO_FEATURES_DIR, 
                                                                                        PATH_TO_TOKENIZED_TIMELINES_DIR,
                                                                                        models_to_keep=[model,])
    train_pids_idx, val_pids_idx, test_pids_idx = get_patient_splits_by_idx(PATH_TO_SPLIT_CSV, patient_ids)
    label_times = [ x.astype(datetime.datetime) for x in label_times ] # cast to Python datetime
    assert len(train_pids_idx) + len(val_pids_idx) + len(test_pids_idx) == len(patient_ids)
    assert len(np.intersect1d(train_pids_idx, val_pids_idx)) == 0
    assert len(np.intersect1d(train_pids_idx, test_pids_idx)) == 0
    assert len(np.intersect1d(val_pids_idx, test_pids_idx)) == 0

    # Load EHRSHOT results
    ## Each model has its own results; Let's only examine results for k = -1
    head: str = 'lr_lbfgs'
    path_to_results_dir: str = os.path.join(PATH_TO_RESULTS_DIR, LABELING_FUNCTION, 'models')
    for model in os.listdir(path_to_results_dir):
        if model != args.model:
            continue
        path_to_results_file: str = os.path.join(path_to_results_dir, model, head, f'subtask={LABELING_FUNCTION}', 'k=-1', 'preds.csv')
        df_preds = pd.read_csv(path_to_results_file)
        df_preds['pid'] = patient_ids[train_pids_idx + val_pids_idx + test_pids_idx]
        df_preds['pid_idx'] = train_pids_idx + val_pids_idx + test_pids_idx
        df_preds['label_time'] = label_times

        # TODO -- finish