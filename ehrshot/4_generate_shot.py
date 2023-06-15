import argparse
import os
import pickle
import numpy as np
from loguru import logger
from utils import (LABELING_FUNCTIONS, load_data, save_data, 
                   sort_tuples, get_pid_label_times_and_values, 
                   process_chexpert_labels, convert_multiclass_to_binary_labels)
import femr.datasets
from datetime import datetime

XGB_PARAMS = {
    'max_depth': [3, 6, -1],
    'learning_rate': [0.02, 0.1, 0.5],
    'num_leaves' : [10, 25, 100],
}

LR_PARAMS = {
    "C": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e2, 1e3, 1e4, 1e5, 1e6], 
    "penalty": ['l2']
}

DEBUG_SHOTS = [10]
FEW_SHOTS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128]
LONG_SHOTS = [-1]

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

def get_k_samples(y, k: int, max_k: int, is_preserve_prevalence: bool = False, seed=0):
    """
        k = number of samples per class
        max_k = largest size of k that we'll feed into model. This is needed to ensure that 
            we always sample a subset of the larger k when sampling data points, to control for 
            randomness and isolate out the effect of increasing k.
        """
    # take all of training and testing data
    if k == -1:
        valid_idxs = [i for i in range(len(y))]
        return y, valid_idxs
    else:
        classes = np.unique(y)
        valid_idxs = []
        for c in classes:
            # Get idxs corresponding to this class
            class_idxs = np.where(y == c)[0]
            # Get k random labels
            np.random.seed(seed)
            # Always sample the same subset of examples for a given class
            idxs = np.random.choice(class_idxs, size=min(len(class_idxs), max_k), replace=False)
            if max_k > len(class_idxs):
                # Fill rest of `k` with oversampling if not enough examples to fill `k` without replacement
                idxs = np.hstack([idxs, np.random.choice(class_idxs, size=max_k - len(class_idxs), replace=True)])
                logger.warning(f"Oversampling class {c} with replacement from {k} -> {max_k} examples.")
            # If we want to preserve the prevalence of each class, then we need to adjust our sample size
            if is_preserve_prevalence:
                prev_k = max(1, int(len(classes) * k * len(class_idxs) / len(y)))
                idxs = idxs[:prev_k]
            else:
                idxs = idxs[:k]
            valid_idxs.extend(idxs)
        return y[valid_idxs], valid_idxs

def main(args):
    # Force settings
    labeling_function: str = args.labeling_function
    shot_strat: str = args.shot_strat
    num_replicates: int = args.num_replicates
    PATH_TO_DATA: str = args.path_to_data
    PATH_TO_SAVE: str = args.path_to_save
    PATH_TO_DATABASE: str = os.path.join(PATH_TO_DATA, "femr/extract")
    PATH_TO_SAVE_SHOTS: str = os.path.join(PATH_TO_SAVE, labeling_function)

    # Load PatientDatabase
    database = femr.datasets.PatientDatabase(PATH_TO_DATABASE)
    
    # Few v. long shot
    if shot_strat == 'few':
        SHOTS = FEW_SHOTS
    elif shot_strat == 'long':
        SHOTS = LONG_SHOTS
    elif shot_strat == 'debug':
        SHOTS = DEBUG_SHOTS
    elif shot_strat == 'both':
        SHOTS = FEW_SHOTS + LONG_SHOTS
    else:
        raise ValueError(f"Invalid shot_strat: {shot_strat}")

    patient_ids, label_times, label_values, _, _ = get_pid_label_times_and_values(PATH_TO_DATA, labeling_function)

    if labeling_function == "chexpert":
        label_values = process_chexpert_labels(label_values)
    elif labeling_function.endswith('_lab'):
        # Lab value is multi-class, convert to binary
        label_values = convert_multiclass_to_binary_labels(label_values, threshold=1)

    # Train/test splits
    split_seed: int = 97
    hashed_pids = np.array([database.compute_split(split_seed, pid) for pid in patient_ids])
    train_pids_idx = np.where(hashed_pids < 70)[0]
    val_pids_idx = np.where((70 <= hashed_pids) & (hashed_pids < 85))[0]
    test_pids_idx = np.where(hashed_pids >= 85)[0]

    patient_ids_train, label_times_train = patient_ids[train_pids_idx], label_times[train_pids_idx]
    patient_ids_val, label_times_val = patient_ids[val_pids_idx], label_times[val_pids_idx]
    patient_ids_test, label_times_test = patient_ids[test_pids_idx], label_times[test_pids_idx]

    y_train, y_val, y_test = label_values[train_pids_idx], label_values[val_pids_idx], label_values[test_pids_idx]

    prevalence = np.sum(y_test != 0) / len(y_test)
    logger.info(f"Test prevalence: {prevalence}")

    few_shots_dict = {}
    if labeling_function == 'chexpert':
        # Multilabel
        for idx, label_str in enumerate(CHEXPERT_LABELS):
            if label_str not in few_shots_dict:
                few_shots_dict[label_str] = {}
            y_train_one_label = y_train[:, idx]
            y_val_one_label = y_val[:, idx]
            y_test_one_label = y_test[:, idx]
            for k in SHOTS:
                if k not in few_shots_dict[label_str]:
                    few_shots_dict[label_str][k] = {}
                for replicate in range(num_replicates):
                    logger.critical(f"Label: {label_str} | k: {k} | Replicate: {replicate}")
                    y_train_k, train_idxs = get_k_samples(y_train_one_label, k=k, max_k=max(SHOTS), seed=replicate)
                    y_val_k, val_idxs = get_k_samples(y_val_one_label, k=k, max_k=max(SHOTS), seed=replicate)

                    patient_ids_train_k, label_times_train_k = patient_ids_train[train_idxs], label_times_train[train_idxs]
                    patient_ids_val_k, label_times_val_k = patient_ids_val[val_idxs], label_times_val[val_idxs]

                    patient_ids_train_k = [int(pid) for pid in patient_ids_train_k]
                    patient_ids_val_k = [int(pid) for pid in patient_ids_val_k]
                    train_idxs = [int(idx) for idx in train_idxs]
                    val_idxs = [int(idx) for idx in val_idxs]
                    label_times_train_k = [label_time.isoformat() for label_time in label_times_train_k]
                    label_times_val_k = [label_time.isoformat() for label_time in label_times_val_k]

                    shot_dict = {
                        "patient_ids_train_k": list(patient_ids_train_k), 
                        "label_times_train_k": list(label_times_train_k), 
                        "patient_ids_val_k": list(patient_ids_val_k), 
                        "label_times_val_k": list(label_times_val_k), 
                        "train_idxs": list(train_idxs), 
                        "val_idxs": list(val_idxs)
                    }
                    few_shots_dict[label_str][k][replicate] = shot_dict
    else:
        # Binary classification
        few_shots_dict[labeling_function] = {}
        for k in SHOTS:
            if k not in few_shots_dict[labeling_function]:
                few_shots_dict[labeling_function][k] = {}
            for replicate in range(num_replicates):
                logger.critical(f"k: {k} | Replicate: {replicate}")
                y_train_k, train_idxs = get_k_samples(y_train, k=k, max_k=max(SHOTS), seed=replicate)
                y_val_k, val_idxs = get_k_samples(y_val, k=k, max_k=max(SHOTS), seed=replicate)

                patient_ids_train_k, label_times_train_k = patient_ids_train[train_idxs], label_times_train[train_idxs]
                patient_ids_val_k, label_times_val_k = patient_ids_val[val_idxs], label_times_val[val_idxs]

                patient_ids_train_k = [int(pid) for pid in patient_ids_train_k]
                patient_ids_val_k = [int(pid) for pid in patient_ids_val_k]
                train_idxs = [int(idx) for idx in train_idxs]
                val_idxs = [int(idx) for idx in val_idxs]
                label_times_train_k = [label_time.isoformat() for label_time in label_times_train_k]
                label_times_val_k = [label_time.isoformat() for label_time in label_times_val_k]

                shot_dict = {
                    "patient_ids_train_k": list(patient_ids_train_k), 
                    "label_times_train_k": list(label_times_train_k), 
                    "patient_ids_val_k": list(patient_ids_val_k), 
                    "label_times_val_k": list(label_times_val_k), 
                    "train_idxs": list(train_idxs), 
                    "val_idxs": list(val_idxs)
                }
                few_shots_dict[labeling_function][k][replicate] = shot_dict
    path_to_save = os.path.join(PATH_TO_SAVE_SHOTS, f"{shot_strat}_shots_data.json")
    save_data(few_shots_dict, path_to_save)
    logger.info(f"Saved few shot data at: {path_to_save}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLMBR patient representations")
    # Paths
    parser.add_argument("--path_to_data", type=str, help=( "Path where you have all the data, including your clmbr representations, labeled_featururized patients" ), )
    parser.add_argument("--labeling_function", required=True, type=str, help="Name of labeling function to create.", choices=LABELING_FUNCTIONS, )
    parser.add_argument("--shot_strat", type=str, choices=['both', 'few', 'long', 'debug'], help="Type of sampling to do", required=True )
    parser.add_argument("--num_replicates", type=int, help="For std bars in plots", default=3, )
    parser.add_argument("--path_to_save", type=str, help=( "Path to save few shots data" ), )
    args = parser.parse_args()
    main(args)