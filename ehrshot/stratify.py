"""
Usage:
    python3 stratify.py --task guo_los
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
from functools import reduce

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
    # model: str = 'gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last'
    model: str = 'mamba-tiny-16384--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last'
    patient_ids, label_values, label_times, feature_matrixes = get_labels_and_features(labeled_patients, 
                                                                                        PATH_TO_FEATURES_DIR, 
                                                                                        PATH_TO_TOKENIZED_TIMELINES_DIR,
                                                                                        models_to_keep=[model,])
    train_pids_idx, val_pids_idx, test_pids_idx = get_patient_splits_by_idx(PATH_TO_SPLIT_CSV, patient_ids)
    assert len(train_pids_idx) + len(val_pids_idx) + len(test_pids_idx) == len(patient_ids)
    assert len(np.intersect1d(train_pids_idx, val_pids_idx)) == 0
    assert len(np.intersect1d(train_pids_idx, test_pids_idx)) == 0
    assert len(np.intersect1d(val_pids_idx, test_pids_idx)) == 0

    # Limit to test set
    pids_idx = test_pids_idx
    pids = patient_ids[pids_idx]
    label_times = label_times[pids_idx]
    label_times = [ x.astype(datetime.datetime) for x in label_times ] # cast to Python datetime
    timelines = feature_matrixes[model]['timelines'][pids_idx,:]
    assert timelines.shape[0] == len(pids)
    assert len(label_times) == len(pids)
    assert len(pids) == len(pids_idx)
    assert timelines.shape[1] == 16384

    ########################################################
    ########################################################
    #
    # Map each label to metrics for "irregularity of event times"
    #
    ########################################################
    ########################################################
    path_to_output_file: str = os.path.join(path_to_output_dir, f'df__{LABELING_FUNCTION}__inter_event_times.parquet')
    if not os.path.exists(path_to_output_file):
        print(f'Calculating inter-event times for {LABELING_FUNCTION}...')
        df = calc_inter_event_times(femr_db, pids, label_times)
        df.to_parquet(path_to_output_file)
        print(f'Loaded df.shape={df.shape}')
    else:
        print(f'Loading inter-event times for {LABELING_FUNCTION}...')
        df = pd.read_parquet(path_to_output_file)
        print(f'Loaded df.shape={df.shape}')
    
    # Metric 1: std of inter-event times
    df_std = df.groupby(['pid', 'pid_idx', 'label_time']).agg({'time': 'std'}).reset_index()
    
    # Metric 2: mean of inter-event times
    df_mean = df.groupby(['pid', 'pid_idx', 'label_time']).agg({'time': 'mean'}).reset_index()
    
    # Metric 3: IQR of inter-event times
    df_iqr = df.groupby(['pid', 'pid_idx', 'label_time']).agg({'time': lambda x: np.percentile(x, 75) - np.percentile(x, 25)}).reset_index()

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
    # Citation: https://aclanthology.org/2014.amta-researchers.13.pdf
    #
    ########################################################
    ########################################################
    path_to_output_file: str = os.path.join(path_to_output_dir, f'df__{LABELING_FUNCTION}__n_gram_count.parquet')
    if not os.path.exists(path_to_output_file):
        print(f'Calculating n-grams for {LABELING_FUNCTION}...')
        df = calc_n_gram_count(femr_db, pids, label_times)
        df.to_parquet(path_to_output_file)
        print(f'Loaded df.shape={df.shape}')
    else:
        print(f'Loading n-grams for {LABELING_FUNCTION}...')
        df = pd.read_parquet(path_to_output_file)
        print(f'Loaded df.shape={df.shape}')
    
    # Metric 1: Repetition Rate
    # RR = (# of repeated n-grams) / (# of n-grams)
    n2rr = {} # [key] = n, [value] = repetition rate
    for n in df['n'].unique():
        df_n = df[df['n'] == n]
        n2rr[n] = df_n.groupby(['pid', 'pid_idx', 'label_time']).agg({ 'count' : lambda x: (x > 1).sum() / len(x) }).reset_index().rename(columns={ 'count' : f'rr_{n}' })
    df_save = reduce(lambda left, right: pd.merge(left, right, on=['pid', 'pid_idx', 'label_time'], how='inner'), n2rr.values())
    df_save['rr'] = np.prod(np.concatenate([  df_save[f'rr_{n}'].values[None,:] for n in [ 1, 2, 3, 4 ] ], axis=0), axis=0) ** (1/4.)
    assert df_save.shape[0] == len(pids), f'Failed to merge n-gram repetition rates'

    # Metric 2: Type-Token Ratio
    # TTR = (# of unique tokens) / (# of tokens)
    df_save['ttr'] = df_save[f'rr_1']
    df_save.to_parquet(path_to_output_file.replace('.parquet', '__metrics.parquet'))

    ########################################################
    ########################################################
    #
    # Map each label to metrics for "length of timeline"
    #
    ########################################################
    ########################################################
    path_to_output_file: str = os.path.join(path_to_output_dir, f'df__{LABELING_FUNCTION}__timeline_lengths.parquet')
    
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
    assert all([ x >= y for x, y in zip(n_events, n_tokens) ]), f'Found more tokens than events in a timeline'

    df_save = pd.DataFrame({
        'pid' : pids,
        'pid_idx' : pids_idx,
        'label_time' : label_times,
        'n_events' : n_events,
        'n_tokens' : n_tokens,
    })
    df_save.to_parquet(path_to_output_file.replace('.parquet', '__metrics.parquet'))