"""Create a file at `PATH_TO_LABELS_AND_FEATS_DIR/LABELING_FUNCTION/{SHOT_STRAT}_shots_data.json` containing:

Output:
    few_shots_dict = {
        sub_task_1: : {
            k_1: {
                replicate_1: {
                    "patient_ids_train_k": List[int] = patient_ids['train'][train_idxs_k].tolist(),
                    "patient_ids_val_k": List[int] = patient_ids['val'][val_idxs_k].tolist(),
                    "label_times_train_k": List[str] = [ label_time.isoformat() for label_time in label_times['train'][train_idxs_k] ], 
                    "label_times_val_k": List[str] = [ label_time.isoformat() for label_time in label_times['val'][val_idxs_k] ], 
                    'label_values_train_k': List[int] = y_train[train_idxs_k].tolist(),
                    'label_values_val_k': List[int] = y_val[val_idxs_k].tolist(),
                    "train_idxs": List[int] = train_idxs_k.tolist(),
                    "val_idxs": List[int] = val_idxs_k.tolist(),
                },
                ... 
            },
            ...
        },
        ...
    }
"""
import argparse
import collections
import datetime
import json
import os
from typing import Dict, List, Union
import numpy as np
from loguru import logger
from utils import (
    LABELING_FUNCTION_2_PAPER_NAME, 
    CHEXPERT_LABELS, 
    SHOT_STRATS,
    get_labels_and_features, 
    process_chexpert_labels, 
    convert_multiclass_to_binary_labels,
    get_splits
)
import femr.datasets
from femr.labelers import LabeledPatients, load_labeled_patients

def get_k_samples(y: List[int], k: int, max_k: int, is_preserve_prevalence: bool = False, seed=0) -> List[int]:
    """_summary_

    Args:
        y (List[int]): Ground truth labels
        k (int): number of samples per class
        max_k (int): largest size of k that we'll feed into model. This is needed to ensure that 
            we always sample a subset of the larger k when sampling data points for smaller k's, 
            to control for randomness and isolate out the effect of increasing k.
        is_preserve_prevalence (bool, optional): If TRUE, then preserve prevalence of each class in k-shot sample. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        List[int]: List of idxs in `y` that are included in this k-shot sample
    """    
    valid_idxs: List[int] = []
    if k == -1:
        # Return all samples
        valid_idxs = list(range(len(y)))
    else:
        # Do sampling
        classes = np.unique(y)
        for c in classes:
            # Get idxs corresponding to this class
            class_idxs = np.where(y == c)[0]
            # Get k random labels
            # Note that instead of directly sampling `k` random samples, we instead
            # sample `max_k` examples all at once, and then take the first `k` subset of them.
            # This ensures that `k = N` always contains a superset of the examples sampled
            # for `k = \hat{N}`, where `\hat{N} < N`.
            np.random.seed(seed)
            idxs = np.random.choice(class_idxs, size=min(len(class_idxs), max_k), replace=False)
            if max_k > len(class_idxs):
                # Fill rest of `k` with oversampling if not enough examples to fill `k` without replacement
                idxs = np.hstack([idxs, np.random.choice(class_idxs, size=max_k - len(class_idxs), replace=True)])
                logger.warning(f"Oversampling class {c} with replacement from {len(class_idxs)} -> {max_k} examples.")
            # If we want to preserve the prevalence of each class, then we need to adjust our sample size
            if is_preserve_prevalence:
                prev_k = max(1, int(len(classes) * k * len(class_idxs) / len(y)))
                idxs = idxs[:prev_k]
            else:
                idxs = idxs[:k]
            valid_idxs.extend([ int(x) for x in idxs ]) # need to cast to normal `int` for JSON serializability
    return valid_idxs

def generate_shots(k: int, 
                   max_k: int, 
                   y_train: List[int], 
                   y_val: List[int], 
                   patient_ids: Dict[str, np.ndarray], 
                   label_times: Dict[str, np.ndarray], 
                   seed: int = 0) -> Dict[str, List[Union[int, str]]]:
    train_idxs_k: List[int] = get_k_samples(y_train, k=k, max_k=max_k, seed=seed)
    val_idxs_k: List[int] = get_k_samples(y_val, k=k, max_k=max_k, seed=seed)

    shot_dict: Dict[str, List[Union[int, str]]] = {
        "patient_ids_train_k": patient_ids['train'][train_idxs_k].tolist(),
        "patient_ids_val_k": patient_ids['val'][val_idxs_k].tolist(),
        "label_times_train_k": [ label_time.astype(datetime.datetime).isoformat() for label_time in label_times['train'][train_idxs_k] ], 
        "label_times_val_k": [ label_time.astype(datetime.datetime).isoformat() for label_time in label_times['val'][val_idxs_k] ], 
        'label_values_train_k': y_train[train_idxs_k].tolist(),
        'label_values_val_k': y_val[val_idxs_k].tolist(),
        "train_idxs": train_idxs_k,
        "val_idxs": val_idxs_k,
    }
    
    return shot_dict

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate few-shot data for eval")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_split_csv", required=True, type=str, help="Path to CSV of splits")
    parser.add_argument("--labeling_function", required=True, type=str, help="Labeling function for which we will create k-shot samples.", choices=LABELING_FUNCTION_2_PAPER_NAME.keys(), )
    parser.add_argument("--shot_strat", type=str, choices=SHOT_STRATS.keys(), help="What type of X-shot evaluation we are interested in.", required=True )
    parser.add_argument("--n_replicates", type=int, help="Number of replicates to run for each `k`. Useful for creating std bars in plots", default=3, )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    LABELING_FUNCTION: str = args.labeling_function
    SHOT_STRAT: str = args.shot_strat
    N_REPLICATES: int = args.n_replicates
    PATH_TO_DATABASE: str = args.path_to_database
    PATH_TO_LABELS_DIR: str = args.path_to_labels_dir
    PATH_TO_SPLIT_CSV: str = args.path_to_split_csv
    PATH_TO_LABELED_PATIENTS: str = os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION, 'labeled_patients.csv')
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION, f"{SHOT_STRAT}_shots_data.json")

    # Load PatientDatabase
    database = femr.datasets.PatientDatabase(PATH_TO_DATABASE)

    # Few v. long shot
    if SHOT_STRAT in SHOT_STRATS:
        SHOTS: List[int] = SHOT_STRATS[SHOT_STRAT]
    else:
        raise ValueError(f"Invalid `shot_strat`: {SHOT_STRAT}")

    # Load labels for this task
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)
    patient_ids, label_values, label_times = get_labels_and_features(labeled_patients, None, None)

    if LABELING_FUNCTION == "chexpert":
        # CheXpert is multilabel, convert to binary for EHRSHOT
        label_values = process_chexpert_labels(label_values)
    elif LABELING_FUNCTION.startswith('lab_'):
        # Lab values is multiclass, convert to binary for EHRSHOT
        label_values = convert_multiclass_to_binary_labels(label_values, threshold=1)

    # Train/val/test splits
    patient_ids, label_values, label_times = get_splits(PATH_TO_SPLIT_CSV, patient_ids, label_times, label_values)
    logger.info(f"Train prevalence: {np.sum(label_values['train'] != 0) / label_values['train'].size}")
    logger.info(f"Val prevalence: {np.sum(label_values['val'] != 0) / label_values['val'].size}")
    logger.info(f"Test prevalence: {np.sum(label_values['test'] != 0) / label_values['test'].size}")
    logger.info(f"Counts: train={len(label_values['train'])} | val={len(label_values['val'])} | test={len(label_values['test'])}")

    if LABELING_FUNCTION == 'chexpert':
        # Multilabel -- create one task per class
        sub_tasks: List[str] = CHEXPERT_LABELS
    else:
        # Binary classification
        sub_tasks: List[str] = [LABELING_FUNCTION]
    
    # Create shots
    few_shots_dict: Dict[str, Dict] = collections.defaultdict(dict)
    for idx, sub_task in enumerate(sub_tasks):
        few_shots_dict[sub_task]: Dict[int, Dict[int, Dict]] = collections.defaultdict(dict)
        # Get ground truth labels
        if LABELING_FUNCTION == 'chexpert':
            y_train, y_val = label_values['train'][:, idx], label_values['val'][:, idx]
        else:
            y_train, y_val = label_values['train'], label_values['val']
        # Create a sample for each k, for each replicate
        for k in SHOTS:
            for replicate in range(N_REPLICATES):
                if k == -1 and replicate > 0:
                    # Only need one copy of `all` dataset (for speed)
                    continue
                logger.critical(f"Label: {sub_task} | k: {k} | Replicate: {replicate}")
                shot_dict: Dict[str, List[Union[int, str]]] = generate_shots(k, 
                                                                                max_k=max(SHOTS), 
                                                                                y_train=y_train, 
                                                                                y_val=y_val, 
                                                                                patient_ids=patient_ids, 
                                                                                label_times=label_times, 
                                                                                seed=replicate)
                few_shots_dict[sub_task][k][replicate] = shot_dict
    
    # Save patients selected for each shot
    logger.info(f"Saving few shot data to: {PATH_TO_OUTPUT_FILE}")
    with open(PATH_TO_OUTPUT_FILE, 'w') as f:
        json.dump(few_shots_dict, f)
    logger.success(f"Done with {LABELING_FUNCTION}!")
    