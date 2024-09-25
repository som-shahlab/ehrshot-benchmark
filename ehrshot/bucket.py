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
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Type of task to perform', default='guo_los')
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
        path_to_results_file: str = os.path.join(path_to_results_dir, model, head, f'subtask={LABELING_FUNCTION}', 'k=-1', 'preds.csv')
        df_preds = pd.read_csv(path_to_results_file)
        df_preds['pid'] = patient_ids[train_pids_idx + val_pids_idx + test_pids_idx]
        df_preds['pid_idx'] = train_pids_idx + val_pids_idx + test_pids_idx
        df_preds['label_time'] = label_times

    ########################################################
    ########################################################
    #
    # Irregularity
    #
    ########################################################
    ########################################################
    path_to_input_file: str = os.path.join(path_to_output_dir, f'df_stratify__{LABELING_FUNCTION}__inter_event_times__metrics.parquet')
    if not os.path.exists(path_to_input_file):
        raise FileNotFoundError(f'File not found: {path_to_input_file}. Please run `python3 stratify.py --task {LABELING_FUNCTION}` to generate this file.')
    df = pd.read_parquet(path_to_input_file)
    print(f'Loaded {df.shape[0]} rows')
    
    # Save all metrics
    df_std['metric'] = 'std'
    df_mean['metric'] = 'mean'
    df_iqr['metric'] = 'iqr'
    df_save = pd.concat([df_std, df_mean, df_iqr], axis=0)
    df_save['sub_task'] = LABELING_FUNCTION
    df_save.to_parquet(path_to_output_file.replace('.parquet', '__metrics.parquet'))
    
    ########################################################
    ########################################################
    #
    # Map each label to metrics for "repetitiveness of timeline"
    #
    ########################################################
    ########################################################
    path_to_output_file: str = os.path.join(path_to_output_dir, f'df_stratify__{LABELING_FUNCTION}__n_gram_count.parquet')
    if not os.path.exists(path_to_output_file):
        print(f'Calculating n-grams for {LABELING_FUNCTION}...')
        df = calc_n_gram_count(femr_db, pids, label_times)
        df.to_parquet(path_to_output_file)
        print(f'Loaded {df.shape[0]} rows')
    else:
        print(f'Loading n-grams for {LABELING_FUNCTION}...')
        df = pd.read_parquet(path_to_output_file)
        print(f'Loaded {df.shape[0]} rows')
    
    # Metric 1: Repetition Rate
    # RR = (# of repeated n-grams) / (# of n-grams)
    for n in df['n'].unique():
        df_n = df[df['n'] == n]
        df_n['repetition_rate'] = df_n['count'] / df_n['count'].sum()
    
    # Metric 2: Type-Token Ratio
    # TTR = (# of unique tokens) / (# of tokens)
    

    ########################################################
    ########################################################
    #
    # Map each label to metrics for "length of timeline"
    #
    ########################################################
    ########################################################
    path_to_output_file: str = os.path.join(path_to_output_dir, f'df_stratify__{LABELING_FUNCTION}__timeline_lengths.parquet')
    
    # Metric 1: number of tokens
    PAD_TOKEN_ID: int = 4
    n_tokens = (timelines != PAD_TOKEN_ID).sum(axis=1)
    
    # Metric 2: number of raw clinical events
    n_events = []
    for pid_idx, pid in enumerate(pids):
        n_event: int = 0
        for e in femr_db[pid].events:
            if label_times is not None and e.start > label_times[pid_idx]:
                # If label_times is provided, then calculate inter-event times only for events that occur before the label_times
                break
            n_event += 1
        n_events.append(n_event)

    df_save = pd.DataFrame({
        'pid' : pids,
        'pid_idx' : pids_idx,
        'label_time' : label_times,
        'n_events' : n_events,
        'n_tokens' : n_tokens,
    })
    df_save.to_parquet(path_to_output_file.replace('.parquet', '__metrics.parquet'))
    
    breakpoint()