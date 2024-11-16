'''
Buckets EHRSHOT patients by a specific stratification metric and reports metrics for each bucket.

Usage:
    python3 bucket.py --task guo_los --model clmbr
'''
import datetime
import argparse
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import collections
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import brier_score_loss
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


def calculate_brier_score_per_tercile(df, strat_column, strat_method, model, task, part=None):
    results = []
    df.drop_duplicates(inplace=True)

    # Rank the strat_column and divide into terciles
    df['tercile'] = pd.qcut(df[strat_column].rank(method='first'), 3, labels=False)

    for tercile in df['tercile'].unique():
        tercile_df = df[df['tercile'] == tercile]

        # Calculate Brier score for each tercile, even if there's only one label
        try:
            brier_score = brier_score_loss(tercile_df['y'], tercile_df['pred_proba'])
        except ValueError as e:
            print(f"Error calculating Brier score for tercile {tercile + 1}: {e}")
            continue
        
        results.append({
            'model': model,
            'task': task,
            'strat_method': strat_method,
            'strat_metric': strat_column if not part else f'{strat_column}_{part}',
            'tercile': f'{tercile + 1}',
            'brier_score': brier_score
        })
    
    return results


def calculate_brier_score_per_quartile(df, strat_column, strat_method, model, task, part=None):
    results = []
    df.drop_duplicates(inplace=True)

    # Rank the strat_column and divide into quartiles
    df['quartile'] = pd.qcut(df[strat_column].rank(method='first'), 4, labels=False)

    for quartile in df['quartile'].unique():
        quartile_df = df[df['quartile'] == quartile]

        # Calculate Brier score for each quartile, even if there's only one label
        try:
            brier_score = brier_score_loss(quartile_df['y'], quartile_df['pred_proba'])
        except ValueError as e:
            print(f"Error calculating Brier score for quartile {quartile + 1}: {e}")
            continue
        
        results.append({
            'model': model,
            'task': task,
            'strat_method': strat_method,
            'strat_metric': strat_column if not part else f'{strat_column}_{part}',
            'quartile': f'{quartile + 1}',
            'brier_score': brier_score
        })
    
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Type of task to perform', default='guo_los')
    parser.add_argument('--model', type=str, help='Model whose results we stratify', default='clmbr')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("-" * 50)
    print(f"Computing stratified brier score for task: {args.task}, model: {args.model}.")
    print()
    
    # Constants
    PATH_TO_DATABASE: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract'
    PATH_TO_FEATURES_DIR: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/features_ehrshot'
    PATH_TO_RESULTS_DIR: str = '/share/pi/nigam/migufuen/ehrshot-benchmark/EHRSHOT_ASSETS/results_ehrshot'
    PATH_TO_TOKENIZED_TIMELINES_DIR: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/tokenized_timelines_ehrshot'
    PATH_TO_LABELS_DIR: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark_ehrshot'
    PATH_TO_SPLIT_CSV: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/splits_ehrshot/person_id_map.csv'
    
    # Output directory
    path_to_output_dir: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/stratify/'
    path_to_output_file: str = os.path.join(path_to_output_dir, f'metrics__{args.model}__{args.task}.csv')
    os.makedirs(path_to_output_dir, exist_ok=True)
    
    # Load labeled patients for this task
    LABELING_FUNCTION: str = args.task
    PATH_TO_LABELED_PATIENTS: str =  os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION, 'labeled_patients.csv')
    femr_db = PatientDatabase(PATH_TO_DATABASE)
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)

    # Get features for patients
    model: str = 'gpt2-base-512--clmbr_train-tokens-total_nonPAD-ckpt_val=2000000000-persist_chunk:last_embed:last'
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
    path_to_results_file: str = os.path.join(path_to_results_dir, 
                                                 args.model, 
                                                 head, 
                                                 f'subtask={LABELING_FUNCTION}', 
                                                 'k=-1', # always use k=-1
                                                 'preds.csv')
    assert os.path.exists(path_to_results_file), f'Path to results file does not exist: {path_to_results_file}'
    df_preds = pd.read_csv(path_to_results_file)
    df_preds['pid'] = patient_ids[train_pids_idx + val_pids_idx + test_pids_idx]
    df_preds['pid_idx'] = train_pids_idx + val_pids_idx + test_pids_idx
    df_preds['label_time'] = label_times
    df_preds = df_preds[df_preds['pid_idx'].isin(test_pids_idx)]
    strats = {
        'inter_event_times': {
            'strat_cols': ['time'],
            'sep': 'metric'    
        },
        'n_gram_count': {
            'strat_cols': ['rr_1', 'rr_2', 'rr_3', 'rr_4', 'rr_10', 'rr', 'ttr']
        }, 
        'timeline_lengths': {
            'strat_cols': ['n_events', 'n_unique_events', 'n_tokens', 'n_unique_tokens']
        }
    }
    results_df = pd.DataFrame()
    for strat, metrics in tqdm(strats.items(), desc='Computing stratified brier score'):
        strat_cols = metrics['strat_cols']
        sep = metrics.get('sep', None)
        df = pd.read_parquet(os.path.join(path_to_output_dir, f'df__{args.task}__{strat}__metrics.parquet'))
        for strat_col in strat_cols:
            # Merge df_preds with df based on pid
            if strat_col not in df.columns:
                print(f'{strat_col} not in df columns. Skipping...')
                continue
            if sep:
                for part in df['metric'].unique().tolist():
                    df2 = df[df['metric'] == part]
                    merged_df = pd.merge(df_preds, df2[['pid', strat_col]], on=['pid'])
                    strat_results = calculate_brier_score_per_quartile(merged_df, strat_col, strat, args.model, args.task, part)
                    results_df = pd.concat([results_df, pd.DataFrame(strat_results)], ignore_index=True)
            else:        
                merged_df = pd.merge(df_preds, df[['pid', strat_col]], on=['pid'])
                strat_results = calculate_brier_score_per_quartile(merged_df, strat_col, strat, args.model, args.task)
                results_df = pd.concat([results_df, pd.DataFrame(strat_results)], ignore_index=True)

    results_df.to_csv(path_to_output_file, index=False)
    print('Results saved to:', path_to_output_file)
    print('Done')
