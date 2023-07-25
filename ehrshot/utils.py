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
    # Guo et al. (CLMBR tasks)
    "guo_los",
    "guo_readmission",
    "guo_icu",
    # van Uden et al. (Few shot tasks)
    'uden_pancan',
    'uden_celiac',
    'uden_lupus',
    'uden_acutemi',
    'uden_hypertension',
    'uden_hyperlipidemia',
    # Instant lab value
    "thrombocytopenia_lab",
    "hyperkalemia_lab",
    "hypoglycemia_lab",
    "hyponatremia_lab",
    "anemia_lab",
    # Custom tasks
    "chexpert"
]

def save_data(data, filename):
    """
    Saves Python object to either pickle or JSON file, depending on file extension.

    Parameters:
    data (object): The Python object to be saved.
    filename (str): The name of the file to save to, including the extension.
    """

    # Determine file extension
    file_extension = filename.split('.')[-1]

    # Save to pickle file if extension is .pkl
    if file_extension == 'pkl':
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    # Save to JSON file if extension is .json
    elif file_extension == 'json':
        with open(filename, 'w') as f:
            json.dump(data, f)
    # Dump it to pickle
    else:
        warnings.warn("There is no file extension, so saving it as pickle")
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        # raise ValueError("Unsupported file extension. Only .pkl and .json are supported.")


def load_data(filename):
    """
    Loads Python object from either pickle or JSON file, depending on file extension.

    Parameters:
    filename (str): The name of the file to load from, including the extension.

    Returns:
    The loaded Python object.
    """

    # Determine file extension
    file_extension = filename.split('.')[-1]

    # Load from pickle file if extension is .pkl
    if file_extension == 'pkl':
        with open(filename, 'rb') as f:
            return pickle.load(f)
    # Load from JSON file if extension is .json
    elif file_extension == 'json':
        with open(filename, 'r') as f:
            return json.load(f)
    # Raise error if file extension is not supported
    else:
        warnings.warn("There is no file extension, so loading it as pickle")
        with open(filename, 'rb') as f:
            return pickle.load(f)
        # raise ValueError("Unsupported file extension. Only .pkl and .json are supported.")


def compute_split(person_id: int, seed: int = 97) -> int:
    """
    Rahul + ChatGPT's implementation 
    seed: 
    """
    network_patient_id = struct.pack("!I", person_id)
    network_seed = struct.pack("!I", seed)
    to_hash = network_seed + network_patient_id
    hashv = hashlib.sha256(to_hash).digest()
    
    result = 0
    for i in range(len(hashv)):
        result = (result * 256 + hashv[i]) % 100

    return result


def hash_person_id(person_id: int, seed: int = 97) -> int:
    """Convert person_id (patient unique ID) into a pseudo-random hash value.
    Assumes 32-bit (4 byte) int
    
    Args:
        person_id (int): unique patient identifier
        seed (int, optional): pack. Defaults to 97.

    Returns:
        int: person_id hash value
    """
    # convert ints to byte strings
    pid_as_bytes = person_id.to_bytes(4, "big")
    seed_as_bytes = seed.to_bytes(4, "big")
    # + operator will concatenate byte strings
    value = seed_as_bytes + pid_as_bytes
    # SHA256 hash
    return int.from_bytes(hashlib.sha256(value).digest(), "big")


def get_person_split(person_id: int):
    """Given a patient_id, return that patient's correponding 
     train/valid/test split assignment. This split is consistent 
     across all expertiments using FEMR pretrained models. 
     
     Note: Patients assigned to valid or test should only be used for
     evaluation purposes. 
     
    Args:
        person_id (int): patient unique identifier

    Returns:
        str: split \in {"train", "valid", "test"}
    """
    rand_num_mod_100 = hash_person_id(person_id, seed=97) % 100
    
    if 0 <= rand_num_mod_100 < 80:
        return "train"
    elif 80 <= rand_num_mod_100 < 85:
        return "valid"
    elif 85 <= rand_num_mod_100 < 100:
        return "test"
    else:
        raise ValueError("Random split ID out of range")


def rand_num_mod_100(person_id: int, seed: int = 97) -> int:
    """Convert person_id (patient unique ID) into a pseudo-random hash value.
    Assumes 32-bit (4 byte) int
    
    Args:
        person_id (int): unique patient identifier
        seed (int, optional): pack. Defaults to 97.

    Returns:
        int: person_id hash value
    """
    # convert ints to byte strings
    pid_as_bytes = person_id.to_bytes(4, "big")
    seed_as_bytes = seed.to_bytes(4, "big")
    # + operator will concatenate byte strings
    value = seed_as_bytes + pid_as_bytes
    # SHA256 hash
    return int.from_bytes(hashlib.sha256(value).digest(), "big") % 100


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

