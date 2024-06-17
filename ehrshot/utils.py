import json
import ast
import pickle
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import datetime
import torch.nn as nn
from sklearn.metrics import pairwise_distances
import femr
from femr.labelers import LabeledPatients
from femr.datasets import PatientDatabase
import femr.extension.dataloader
from loguru import logger

# SPLITS
SPLIT_SEED: int = 97
SPLIT_TRAIN_CUTOFF: int = 70
SPLIT_VAL_CUTOFF: int = 85

# Types of base models to test
MODEL_2_NAME: Dict[str, str] = {
    'count' : 'Count-based',
    'clmbr' : 'CLMBR',
    'motor': 'MOTOR',
    # 'motor_fixed': 'MOTOR',
    # 'gpt2-base' : 'GPT2-base (v9)',
    # 'gpt2-medium' : 'GP2-medium (v9)',
    # 'gpt2-large' : 'GP2-large (v9)',
    # 'bert-base' : 'BERT-base (v9)',
    # 'gpt2-base-v8_chunk:last_embed:last' : 'GPT2-base (v8)',
    # 'bert-base-v8_chunk:last_embed:last' : 'BERT-base (v8)',
}
BASE_MODELS: List[str] = list(MODEL_2_NAME.keys())

# Map each base model to a set of heads to test
BASE_MODEL_2_HEADS: Dict[str, List[str]] = {
    'count' : ['gbm', 'lr_lbfgs', 'rf', ], 
    'clmbr' : ['lr_lbfgs', ],
    'motor' : ['lr_lbfgs', ],
    'gpt2-base-v8_chunk:last_embed:last' : ['gbm', 'lr_lbfgs', 'rf', ], 
    'bert-base-v8_chunk:last_embed:last' : ['gbm', 'lr_lbfgs', 'rf', ], 
}
HEAD_2_NAME: Dict[str, str] = {
    'gbm' : 'GBM',
    'lr_lbfgs' : 'LR',
    'lr_femr' : 'LR',
    'lr_newton-cg' : 'LR',
    'protonet' : 'ProtoNet',
    'rf' : 'Random Forest',
}

# Labeling functions
LABELING_FUNCTIONS: List[str] = [
    # Guo et al. 2023
    "guo_los",
    "guo_readmission",
    "guo_icu",
    # New diagnosis
    'new_pancan',
    'new_celiac',
    'new_lupus',
    'new_acutemi',
    'new_hypertension',
    'new_hyperlipidemia',
    # Instant lab values
    "lab_thrombocytopenia",
    "lab_hyperkalemia",
    "lab_hypoglycemia",
    "lab_hyponatremia",
    "lab_anemia",
    # Custom tasks
    "chexpert"
]

LABELING_FUNCTION_2_PAPER_NAME = {
    "guo_los": "Long LOS",
    "guo_readmission": "30-day Readmission",
    "guo_icu": "ICU Admission",
    "lab_thrombocytopenia": "Thrombocytopenia",
    "lab_hyperkalemia": "Hyperkalemia",
    "lab_hypoglycemia": "Hypoglycemia",
    "lab_hyponatremia": "Hyponatremia",
    "lab_anemia": "Anemia",
    "new_hypertension": "Hypertension",
    "new_hyperlipidemia": "Hyperlipidemia",
    "new_pancan": "Pancreatic Cancer",
    "new_celiac": "Celiac",
    "new_lupus": "Lupus",
    "new_acutemi": "Acute MI",
    "chexpert": "Chest X-ray Findings"
}

TASK_GROUP_2_PAPER_NAME = {
    "operational_outcomes": "Operational Outcomes",
    "lab_values": "Anticipating Lab Test Results",
    "new_diagnoses": "Assignment of New Diagnoses",
    "chexpert": "Anticipating Chest X-ray Findings",
}

# CheXpert labels
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

TASK_GROUP_2_LABELING_FUNCTION = {
    "operational_outcomes": [
        "guo_los",
        "guo_readmission",
        "guo_icu"
    ],
    "lab_values": [
        "lab_thrombocytopenia",
        "lab_hyperkalemia",
        "lab_hypoglycemia",
        "lab_hyponatremia",
        "lab_anemia"
    ],
    "new_diagnoses": [
        "new_hypertension",
        "new_hyperlipidemia",
        "new_pancan",
        "new_celiac",
        "new_lupus",
        "new_acutemi"
    ],
    "chexpert": [
        "chexpert"
    ]
}

# Hyperparameter search
XGB_PARAMS = {
    'max_depth': [3, 6, -1],
    'learning_rate': [0.02, 0.1, 0.5],
    'num_leaves' : [10, 25, 100],
}
LR_PARAMS = {
    "C": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, 1e6], 
    "penalty": ['l2']
}
RF_PARAMS = {
    'n_estimators': [10, 20, 50, 100, 300],
    'max_depth' : [3, 5, 10, 20, 50],
}

# Few shot settings
SHOT_STRATS = {
    'few' : [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128],
    'long' : [-1],
    'all' : [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, -1],
    'debug' : [10],
}

# Plotting
SCORE_MODEL_HEAD_2_COLOR = {
    'auroc' : {
        'count' : {
            'gbm' : 'tab:red',
            'lr_lbfgs' : 'aqua',
            'rf' : 'tab:orange',
        },
        'clmbr' : {
            'lr_lbfgs' : 'tab:blue',
        },
    },
    'auprc' : {
        'count' : {
            'gbm' : 'tab:red',
            'lr_lbfgs' : 'aqua',
            'rf' : 'tab:orange',
        },
        'clmbr' : {
            'lr_lbfgs' : 'tab:blue',
        },
    },
}

def get_splits(path_to_split_csv: str,
                patient_ids: np.ndarray, 
                label_times: np.ndarray, 
                label_values: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return train/val/test splits for a given set of patients."""
    train_pids_idx, val_pids_idx, test_pids_idx = get_patient_splits_by_idx(path_to_split_csv, patient_ids)
    patient_ids: Dict[str, np.ndarray] = {
        'train' : patient_ids[train_pids_idx],
        'val' : patient_ids[val_pids_idx],
        'test' : patient_ids[test_pids_idx],
    }
    label_times: Dict[str, np.ndarray] = {
        'train' : label_times[train_pids_idx],
        'val' : label_times[val_pids_idx],
        'test' : label_times[test_pids_idx],
    }
    label_values: Dict[str, np.ndarray] = {
        'train' : label_values[train_pids_idx],
        'val' : label_values[val_pids_idx],
        'test' : label_values[val_pids_idx],
    }
    return patient_ids, label_values, label_times

def get_patient_splits_by_idx(path_to_split_csv: str, patient_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a list of patient IDs, split into train, val, and test sets.
        Returns the idxs for each split within `patient_ids`."""
    df_split = pd.read_csv(path_to_split_csv)
    split_2_idxs = { 'train' : [], 'val' : [], 'test' : [], }
    for split in ['train', 'val', 'test']:
        for idx, id in enumerate(patient_ids.tolist()):
            if id in df_split[df_split['split'] == split]['omop_person_id'].values:
                split_2_idxs[split].append(idx)
    return (
        split_2_idxs['train'],
        split_2_idxs['val'],
        split_2_idxs['test'],
    )

def get_labels_and_features(labeled_patients: LabeledPatients, path_to_features_dir: Optional[str]) -> Tuple[List[int], List[datetime.datetime], List[int], Dict[str, np.ndarray]]:
    """Given a path to a directory containing labels and features as well as a LabeledPatients object, returns
        the labels and features for each patient. Note that this function is more complex b/c we need to align
        the labels with their corresponding features based on their prediction times."""
    label_patient_ids, label_values, label_times = labeled_patients.as_numpy_arrays()
    label_times = label_times.astype("datetime64[us]")

    # Sort arrays by (1) patient ID and (2) label time
    sort_order: np.ndarray = np.lexsort((label_times, label_patient_ids))
    label_patient_ids, label_values, label_times = label_patient_ids[sort_order], label_values[sort_order], label_times[sort_order]

    # Just return labels, ignore features
    if path_to_features_dir is None:
        return label_patient_ids, label_values, label_times

    # Go through every featurization we've created (e.g. count, clmbr, motor)
    # and align the label times with the featurization times
    featurizations: Dict[str, np.ndarray] = {}
    for model in ('motor', ):
        path_to_feats_file: str = os.path.join(path_to_features_dir, f'{model}_features.pkl')
        logger.info(f"Loading features from: {path_to_feats_file}")
        assert os.path.exists(path_to_feats_file), f'Path to file containing `{model}` features does not exist at this path: {path_to_feats_file}. Maybe you forgot to run `generate_features.py` first?'
        
        with open(path_to_feats_file, 'rb') as f:
            # Load data and do type checking
            feats: Tuple[Any, np.ndarray, np.ndarray, np.ndarray] = pickle.load(f)
            if model == 'count':
                feature_matrix, feature_patient_ids, feature_times = (
                    feats[0],
                    feats[1],
                    feats[3], # NOTE: skip label_values in [2]
                )
            elif model == 'clmbr':
                feature_matrix, feature_patient_ids, feature_times = (
                    feats['data_matrix'],
                    feats['patient_ids'],
                    feats['labeling_time'],
                )
            elif model == 'motor':
                feature_matrix, feature_patient_ids, feature_times = (
                    feats['representations'],
                    feats['patient_ids'],
                    feats['prediction_times'],
                )
            else:
                assert False, f"Unsupported model: {model}"
            feature_patient_ids = feature_patient_ids.astype(label_patient_ids.dtype)
            feature_times = feature_times.astype(label_times.dtype)
            assert feature_patient_ids.dtype == label_patient_ids.dtype, f'Error -- mismatched types between feature_patient_ids={feature_patient_ids.dtype} and label_patient_ids={label_patient_ids.dtype}'
            assert feature_times.dtype == label_times.dtype, f'Error -- mismatched types between feature_times={feature_times.dtype} and label_times={label_times.dtype}'

            # Sort arrays by (1) patient ID and (2) label time
            sort_order: np.ndarray = np.lexsort((feature_times, feature_patient_ids))
            feature_patient_ids, feature_times = feature_patient_ids[sort_order], feature_times[sort_order]

            # Align label times with feature times
            join_indices = femr.extension.dataloader.compute_feature_label_alignment(label_patient_ids, 
                                                                                     label_times.astype(np.int64), 
                                                                                     feature_patient_ids, 
                                                                                     feature_times.astype(np.int64))
            feature_matrix = feature_matrix[sort_order[join_indices], :]

            # Validate that our alignment was successful
            assert np.all(feature_patient_ids[join_indices] == label_patient_ids)
            assert np.all(feature_times[join_indices] == label_times)

            featurizations[model] = feature_matrix
    
    return label_patient_ids, label_values, label_times, featurizations

def process_chexpert_labels(label_values):
    new_labels = []
    for label_value in label_values:
        label_str = bin(label_value)[2:]
        rem_bin = 14 - len(label_str)
        label_str = "0"*rem_bin + label_str
        label_list = [*label_str]
        label_list = [int(label) for label in label_list]
        new_labels.append(label_list)
    return np.array(new_labels)

def convert_multiclass_to_binary_labels(values, threshold: int = 1):
    values[values >= threshold] = 1
    return values

def check_file_existence_and_handle_force_refresh(path_to_file_or_dir: str, is_force_refresh: bool):
    """Checks if file/folder exists. If it does, deletes it if `is_force_refresh` is True."""
    if is_force_refresh:
        if os.path.exists(path_to_file_or_dir):
            if os.path.isdir(path_to_file_or_dir):
                logger.critical(f"Deleting existing directory at `{path_to_file_or_dir}`")
                os.system(f"rm -r {path_to_file_or_dir}")
            else:
                logger.critical(f"Deleting existing file at `{path_to_file_or_dir}`")
                os.system(f"rm {path_to_file_or_dir}")
    else:
        if os.path.exists(path_to_file_or_dir):
            if os.path.isdir(path_to_file_or_dir):
                raise ValueError(f"Error -- Directory already exists at `{path_to_file_or_dir}`. Please delete it and try again.")
            else:
                raise ValueError(f"Error -- File already exists at `{path_to_file_or_dir}`. Please delete it and try again.")
    if os.path.isdir(path_to_file_or_dir):
        os.makedirs(path_to_file_or_dir, exist_ok=True)

def type_tuple_list(s):
    """For parsing List[Tuple] from command line using `argparse`"""
    try:
        # Convert the string representation of list of tuples into actual list of tuples
        val = ast.literal_eval(s)
        if not isinstance(val, list):
            raise ValueError("Argument should be a list of tuples")
        for item in val:
            if not isinstance(item, tuple) or not all(isinstance(i, str) for i in item):
                raise ValueError("Argument items should be tuples of strings")
        return val
    except ValueError:
        raise ValueError("Argument should be a list of tuples of strings")

def filter_df(df: pd.DataFrame, 
            score: Optional[str] = None, 
            labeling_function: Optional[str] = None, 
            task_group: Optional[str] = None,
            sub_tasks: Optional[List[str]] = None,
            model_heads: Optional[List[Tuple[str, str]]] = None,
            k: Optional[int] = None) -> pd.DataFrame:
    """Filters results df based on various criteria."""
    df = df.copy()
    if score:
        df = df[df['score'] == score]
    if labeling_function:
        df = df[df['labeling_function'] == labeling_function]
    if task_group:
        labeling_functions: List[str] = TASK_GROUP_2_LABELING_FUNCTION[task_group]
        df = df[df['labeling_function'].isin(labeling_functions)]
    if sub_tasks:
        df = df[df['sub_task'].isin(sub_tasks)]
    if model_heads:
        mask = [ False ] * df.shape[0]
        for model_head in model_heads:
            mask = mask | ((df['model'] == model_head[0]) & (df['head'] == model_head[1]))
        df = df[mask]
    if k is not None:
        df = df[df['k'] == k]
    return df

class ProtoNetCLMBRClassifier(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, X_train, y_train):
        # (n_patients, clmbr_embedding_size)
        n_classes = len(set(y_train))
        
        # (n_classes, clmbr_embedding_size)
        self.prototypes = np.zeros((n_classes, X_train.shape[1]))
        for cls in range(n_classes):
            indices = np.nonzero(y_train == cls)[0]
            examples = X_train[indices]
            self.prototypes[cls, :] = np.mean(examples, axis=0)

    def predict_proba(self, X_test):
        # (n_patients, clmbr_embedding_size)    
        dists = pairwise_distances(X_test, self.prototypes, metric='euclidean')
        # Negate distance values
        neg_dists = -dists

        # Apply softmax function to convert distances to probabilities
        probabilities = np.exp(neg_dists) / np.sum(np.exp(neg_dists), axis=1, keepdims=True)

        return probabilities

    def predict(self, X_train):
        dists = self.predict_proba(X_train)
        predictions = np.argmax(dists, axis=1)
        return predictions
    
    def save_model(self, model_save_dir, model_name):
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir, exist_ok = True)
        np.save(self.prototypes, os.path.join(model_save_dir, f'{model_name}.npy'))


def write_table_to_latex(df: pd.DataFrame, path_to_file: str, is_ignore_index: bool = False):
    with open(path_to_file, 'a') as f:
        latex = df.to_latex(index=not is_ignore_index, escape=True)
        f.write("=======================================\n")
        f.write("=======================================\n")
        f.write("\n\nFigure:\n\n")
        f.write("\n")
        f.write(re.sub(r'\& +', '& ', latex))
        f.write("\n")
        f.write("=======================================\n")
        f.write("=======================================\n")


def get_rel_path(file: str, rel_path: str) -> str:
    """Transforms a relative path from a specific file in the package `eclair/src/eclair/ into an absolute path"""
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(file)), rel_path)
    )

def dump_patient_to_json(patient, file):
	events = []
	for event in patient.events:
		events.append({
			'start' : str(event.start),
			'end' : str(event.end),
			'code' : event.code,
			'table' : event.omop_table,
		})
	json.dump(events, open(file, 'w'), indent=2)