import pickle
import json
from typing import List
import os
import struct
import hashlib
import warnings
import struct
import hashlib
from loguru import logger
import numpy as np
import femr
import femr.labelers
from datetime import datetime

import femr.extension.dataloader

# Labeling functions
LABELING_FUNCTIONS: List[str] = [
    # Guo et al.
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
    # Instant lab value
    "lab_thrombocytopenia",
    "lab_hyperkalemia",
    "lab_hypoglycemia",
    "lab_hyponatremia",
    "lab_anemia",
    # Custom tasks
    "chexpert"
]

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

def get_pid_label_times_and_values(path_to_features, labeled_patients):
    label_pids, label_values, label_times = labeled_patients.as_numpy_arrays()


    order = np.lexsort((label_times, label_pids))
    label_pids = label_pids[order]
    label_values = label_values[order]
    label_times = label_times[order].astype("datetime64[us]")

    features = []
    for feature_name in ['count', 'clmbr', 'motor']:
        with open(os.path.join(path_to_features, f'{feature_name}_features.pkl'), 'rb') as f:
            feature_values, feature_pids, _, feature_times = pickle.load(f)
            feature_times = feature_times.astype("datetime64[us]")

            assert feature_pids.dtype == label_pids.dtype, f'{feature_pids.dtype}, {label_pids.dtype}'
            assert feature_times.dtype == label_times.dtype, f'{feature_times.dtype}, {label_times.dtype}'

            order = np.lexsort((feature_times, feature_pids))
            feature_pids = feature_pids[order]
            feature_times = feature_times[order]

            join_indices = femr.extension.dataloader.compute_feature_label_alignment(label_pids, label_times.astype(np.int64), feature_pids, feature_times.astype(np.int64))
            feature_values = feature_values[order[join_indices], :]

            assert np.all(feature_pids[join_indices] == label_pids)
            assert np.all(feature_times[join_indices] == label_times)

            features.append(feature_values)
    
    return [label_pids, label_times, label_values] + features

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

