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

def sort_tuples(lst, lst2):
    zipped_lists = list(zip(lst, lst2))
    zipped_lists.sort(key=lambda x: (x[0][0], x[0][1]))
    
    lst, lst2 = zip(*zipped_lists)
    return list(lst), list(lst2)


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


def get_pid_label_times_and_values(PATH_TO_DATA, labeling_function):

    # PATH_TO_LABELED_PATIENTS: str = os.path.join(PATH_TO_DATA, f"benchmark/{labeling_function}/labeled_patients.csv")
    PATH_TO_COUNT_FEATS: str = os.path.join(PATH_TO_DATA, f"benchmark/{labeling_function}/featurized_patients.pkl")
    PATH_TO_CLMBR_REPS = os.path.join(PATH_TO_DATA, f"clmbr_reps/{labeling_function}/clmbr_reprs")

    # labeled_patients = femr.labelers.core.load_labeled_patients(PATH_TO_LABELED_PATIENTS)

    # Count featurizations
    logger.success(f"Found count featurizations. Loading @ {PATH_TO_COUNT_FEATS}")
    
    logger.info(f"Path to CLMBR REPS: {PATH_TO_CLMBR_REPS}")
    clmbr_reps = load_data(PATH_TO_CLMBR_REPS)
    clmbr_feature_matrix, clmbr_patient_ids, clmbr_label_times = [
        clmbr_reps[k] for k in ("data_matrix", "patient_ids", "labeling_time")
    ]
    clmbr_patient_ids_label_times = [(pid, time) for pid, time in zip(clmbr_patient_ids, clmbr_label_times)]
    idxs = [i for i in range(len(clmbr_patient_ids_label_times))]
    _, sort_idxs = sort_tuples(clmbr_patient_ids_label_times, idxs)
    clmbr_feature_matrix = clmbr_feature_matrix[sort_idxs]
    clmbr_patient_ids = clmbr_patient_ids[sort_idxs]
    clmbr_label_times = clmbr_label_times[sort_idxs]

    count_feats = pickle.load(open(PATH_TO_COUNT_FEATS, 'rb'))
    count_feature_matrix, count_patient_ids, count_label_values, count_label_times = (
        count_feats[0],
        count_feats[1],
        count_feats[2],
        count_feats[3],
    )

    count_patient_ids_label_times = [(pid, time) for pid, time in zip(count_patient_ids, count_label_times)]
    idxs = [i for i in range(len(count_patient_ids_label_times))]
    _, sort_idxs = sort_tuples(count_patient_ids_label_times, idxs)
    count_feature_matrix = count_feature_matrix[sort_idxs]
    count_patient_ids = count_patient_ids[sort_idxs]
    count_label_values = count_label_values[sort_idxs]
    count_label_times = count_label_times[sort_idxs]

    count_patient_label_time_pairs = list(zip(count_patient_ids, [ x.astype('datetime64[m]') for x in count_label_times ]))
    clmbr_patient_label_time_pairs = list(zip(clmbr_patient_ids, [ np.datetime64(x).astype('datetime64[m]') for x in clmbr_label_times ]))
    intersection = set(clmbr_patient_label_time_pairs).intersection(set(count_patient_label_time_pairs))

    intersection_idxs = []
    already_seen = set()
    for idx, (patient_id, label_time) in enumerate(count_patient_label_time_pairs):
        # Get all unique pairs of (patient_id, label_time) present in both the CLMBR and count featurizations
        if (patient_id, label_time) in intersection and (patient_id, label_time) not in already_seen:
            intersection_idxs.append(idx)
            already_seen.add((patient_id, label_time))

    count_feature_matrix = count_feature_matrix[intersection_idxs]
    count_patient_ids = count_patient_ids[intersection_idxs]
    count_label_values = count_label_values[intersection_idxs]
    count_label_times = count_label_times[intersection_idxs]

    intersection_idxs = []
    already_seen = set()
    for idx, (patient_id, label_time) in enumerate(clmbr_patient_label_time_pairs):
        # Get all unique pairs of (patient_id, label_time) present in both the CLMBR and count featurizations
        if (patient_id, label_time) in intersection and (patient_id, label_time) not in already_seen:
            intersection_idxs.append(idx)
            already_seen.add((patient_id, label_time))
    clmbr_feature_matrix = clmbr_feature_matrix[intersection_idxs]
    clmbr_patient_ids = clmbr_patient_ids[intersection_idxs]
    clmbr_label_times = clmbr_label_times[intersection_idxs]

    count_label_times_new = []
    desired_format = "%Y-%m-%dT%H:%M:%S.%f"

    for label_time in count_label_times:
        dt_obj = datetime.strptime(str(label_time), desired_format)
        desired_datetime = datetime(dt_obj.year, dt_obj.month, dt_obj.day, dt_obj.hour, dt_obj.minute)
        count_label_times_new.append(desired_datetime)
    
    count_label_times = np.array(count_label_times_new)
    assert count_patient_ids.shape == clmbr_patient_ids.shape
    assert count_label_times.shape == clmbr_label_times.shape
    assert np.sum(count_patient_ids == clmbr_patient_ids) == len(clmbr_patient_ids)
    assert np.sum(count_label_times == clmbr_label_times) == len(count_label_times)

    # assert np.sum(count_label_times == clmbr_label_times) == len(clmbr_label_times)

    patient_ids = clmbr_patient_ids
    label_times = clmbr_label_times
    label_values = count_label_values

    logger.info(f"CLMBR Feature matrix shape: {clmbr_feature_matrix.shape}")
    logger.info(f"Count Feature matrix shape: {count_feature_matrix.shape}")
    logger.info(f"Patient IDs shape: {len(patient_ids)}")
    logger.info(f"Label values shape: {label_values.shape}")
    logger.info(f"Label times shape: {label_times.shape}")
    logger.info(f"Unique label values: {np.unique(label_values)} (total = {len(np.unique(label_values))})")
    logger.info(f"# of unique patients: {len(np.unique(patient_ids))}")

    return patient_ids, label_times, label_values, clmbr_feature_matrix, count_feature_matrix


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

