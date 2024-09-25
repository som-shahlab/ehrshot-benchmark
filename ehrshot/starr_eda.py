"""
Usage:
    # Calculate inter-event times
        python3 starr_eda.py --task inter_event_times
    
    # Calculate inter-visit times
        python3 starr_eda.py --task inter_visit_times
    
    # Calculate n-gram repetitions
        python3 starr_eda.py --task n_gram_count
    
    # Calculate longest repeated substring
        python3 starr_eda.py --task longest_repeated_substring

Add --is_ehrshot flag to use EHRSHOT_ASSETS instead of SOM-RIT-PHI-STARR-PROD

Outputs:
    df_inter_event_times_{label}.parquet : DataFrame with columns ['pid', 'code', 'time' (seconds), 'idx']
"""
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from collections import Counter
from itertools import islice
import datetime
from femr.datasets import PatientDatabase
from hf_ehr.config import Event, SPLIT_TRAIN_CUTOFF, SPLIT_VAL_CUTOFF, SPLIT_SEED
import argparse
import difflib

def calc_longest_repeated_substring(femr_db, pids: List[int], label_times: Optional[List[datetime.datetime]] = None) -> pd.DataFrame:
    """
    Calculate longest repeated substring.
    If label_times is provided, then calculate inter-event times only for events that occur before the label_times.
    """
    def get_longest_repeated_substring(lst: List[str]) -> List[str]:
        subsequence_count = Counter(tuple(lst[i:j]) for i in range(len(lst)) for j in range(i + 1, len(lst) + 1))
        repeated_subsequences = [k for k, v in subsequence_count.items() if v > 1]
        if not repeated_subsequences:
            return []
        return max(repeated_subsequences, key=len)

    results = []
    for pid_idx, pid in enumerate(tqdm(pids, total=len(pids), desc=f'Calculating longest repeated substring')):
        sequence: List[str] = []
        for e in femr_db[pid].events:
            if label_times is not None and e.start > label_times[pid_idx]:
                # If label_times is provided, then calculate inter-event times only for events that occur before the label_times
                break
            sequence.append(e.code)
        subsequence: List[str] = get_longest_repeated_substring(sequence)
        results.append({
            'pid' : pid,
            'pid_idx' : pid_idx,
            'label_time' : label_times[pid_idx] if label_times is not None else None,
            'length' : len(subsequence),
            'subsequence' : subsequence,
        })
    df = pd.DataFrame(results)
    return df

def calc_n_gram_count(femr_db, pids, label_times: Optional[List[datetime.datetime]] = None) -> pd.DataFrame:
    """
    Calculate n-gram repetitions. Returns a dictionary with n as key and Counter as value.
    If label_times is provided, then calculate inter-event times only for events that occur before the label_times.
    """
    ns = [1, 2, 3, 4, 10, ]

    # Function to generate n-grams
    def generate_ngrams(sequence: List, n: int):
        return zip(*(islice(sequence, i, None) for i in range(n)))

    results = []
    for n in ns:
        if label_times is None and os.path.exists(f'df_n_gram_counts_{n}.parquet'):
            continue
        for pid_idx, pid in enumerate(tqdm(pids, total=len(pids), desc=f'Calculating n-gram repetitions for n={n}')):
            sequence: List[str] = []
            for e in femr_db[pid].events:
                if label_times is not None and e.start > label_times[pid_idx]:
                    # If label_times is provided, then calculate inter-event times only for events that occur before the label_times
                    break
                sequence.append(e.code)
            ngrams = generate_ngrams(sequence, n)
            counter = Counter()
            counter.update(ngrams)
            for ngram, count in counter.items():
                results.append({
                    'pid' : pid,
                    'pid_idx' : pid_idx,
                    'label_time' : label_times[pid_idx] if label_times is not None else None,
                    'ngram' : ngram,
                    'count' : count,
                    'n' : n,
                })

        # Checkpointing
        if label_times is None:
            df = pd.DataFrame(results)
            df.to_parquet(f'df_n_gram_counts_{n}.parquet')

    df = pd.DataFrame(results, dtype={'pid' : int, 'pid_idx' : int, 'label_time' : 'datetime64[ns]', 'ngram' : str, 'count' : int, 'n' : int})
    return df

def calc_inter_event_times(femr_db, pids: List[int], label_times: Optional[List[datetime.datetime]] = None) -> pd.DataFrame:
    """
    Calculate inter-event times.
    If label_times is provided, then calculate inter-event times only for events that occur before the label_times.
    """
    results: List[Dict] = []
    for pid_idx, pid in enumerate(tqdm(pids, total=len(pids))):
        if label_times is None and os.path.exists(f'df_inter_event_times_{pid_idx // 100_000}.parquet'):
            continue
        prev_event = None
        for e_idx, e in enumerate(femr_db[pid].events):
            if label_times is not None and e.start > label_times[pid_idx]:
                # If label_times is provided, then calculate inter-event times only for events that occur before the label_times
                break
            if prev_event is not None:
                results.append({
                    'pid' : pid,
                    'pid_idx' : pid_idx,
                    'label_time' : label_times[pid_idx] if label_times is not None else None,
                    'event_code' : e.code,
                    'time' : (e.start - prev_event.start).total_seconds(), # time in secs
                    'event_idx' : e_idx,
                })
            prev_event = e
            
        # Checkpointing
        if label_times is None and (pid_idx + 1) % 100_000 == 0:
            df = pd.DataFrame(results)
            df.to_parquet(f'df_inter_event_times_{pid_idx // 100_000}.parquet')
            print(f'Saved {df.shape[0]} rows')

    df = pd.DataFrame(results, dtype={'pid' : int, 'pid_idx' : int, 'label_time' : 'datetime64[ns]', 'event_code' : str, 'time' : float, 'event_idx' : int})
    return df

if __name__ == '__main__':
    path_to_output_dir: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/eda/'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_ehrshot', action='store_true', help='If TRUE, then use EHRSHOT_ASSETS, else use SOM-RIT-PHI-STARR-PROD')
    parser.add_argument('--task', type=str, help='Type of task to perform', default='inter_event_times')
    args = parser.parse_args()

    # Load FEMR DB
    if args.is_ehrshot:
        # EHRSHOT
        label: str = 'ehrshot'
        PATH_TO_DATABASE: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract'
        femr_db = PatientDatabase(PATH_TO_DATABASE)
        val_pids = pd.read_csv('/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/splits_ehrshot/person_id_map.csv')
        val_pids = val_pids[val_pids['split'] == 'val']['omop_person_id'].values
    else:
        # STARR
        label: str = 'starr'
        PATH_TO_DATABASE: str = '/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8_no_notes'
        femr_db = PatientDatabase(PATH_TO_DATABASE)
        all_pids: np.ndarray = np.array([ pid for pid in femr_db ])
        hashed_pids: np.ndarray = np.array([ femr_db.compute_split(SPLIT_SEED, pid) for pid in all_pids ])
        train_pids: np.ndarray = all_pids[np.where(hashed_pids < SPLIT_TRAIN_CUTOFF)[0]]
        val_pids: np.ndarray = all_pids[np.where((SPLIT_TRAIN_CUTOFF <= hashed_pids) & (hashed_pids < SPLIT_VAL_CUTOFF))[0]]
        test_pids: np.ndarray = all_pids[np.where(hashed_pids >= SPLIT_VAL_CUTOFF)[0]]
        assert np.intersect1d(train_pids, val_pids).shape[0] == 0
        assert np.intersect1d(train_pids, test_pids).shape[0] == 0
        assert np.intersect1d(val_pids, test_pids).shape[0] == 0

    # Calculate inter-event times
    if args.task == 'inter_event_times':
        df = calc_inter_event_times(femr_db, val_pids)
        path_to_output_file: str = os.path.join(path_to_output_dir, f'df__{label}__inter_event_times.parquet')
    # Calculate inter-visit times
    elif args.task == 'inter_visit_times':
        pass
    # Calculate n-gram repetitions
    elif args.task == 'n_gram_count':
        df = calc_n_gram_count(femr_db, val_pids)
        path_to_output_file: str = os.path.join(path_to_output_dir, f'df__{label}__n_gram_counts.parquet')
    # Calculate longest repeated substring
    elif args.task == 'longest_repeated_substring':
        df = calc_longest_repeated_substring(femr_db, val_pids)
        path_to_output_file: str = os.path.join(path_to_output_dir, f'df__{label}__longest_repeated_substrings.parquet')
    else:
        raise ValueError(f"Invalid task: {args.task}")
    print(f"Shape of df: {df.shape}")
    print(f'Saving to: {path_to_output_file}')
    df.to_parquet(f'{path_to_output_file}')