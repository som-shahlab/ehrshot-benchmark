"""Create a file at `PATH_TO_LABELS_AND_FEATS_DIR/LABELING_FUNCTION/results.json` containing:

Output:
    results = {
        sub_task_1: : {
            model_1: {
                head_1: {
                    replicate_1: {
                        auroc: [ list of AUROC scores for each k-shot sample in replicate_1],
                        auprc: [ list of AUPRC scores for each k-shot sample in replicate_1],
                        brief: [ list of Brief scores for each k-shot sample in replicate_1],
                        ks: [ list of values of `k` tested in replicate_1],
                    },
                    ...
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
import json
import os
from typing import Dict, List, Union
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from loguru import logger
from sklearn.preprocessing import MaxAbsScaler
from ehrshot.utils import CHEXPERT_LABELS, LR_PARAMS, MODELS, XGB_PARAMS, ProtoNetCLMBRClassifier, get_patient_splits_by_idx
from utils import (
    LABELING_FUNCTIONS,
    SHOT_STRATS,
    BASE_MODELS,
    BASE_MODEL_2_HEADS,
    get_labels_and_features, 
    process_chexpert_labels, 
    convert_multiclass_to_binary_labels)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from scipy.sparse import issparse
import scipy
import lightgbm as lgb
import femr
import femr.datasets
import femr.models.conjugate_gradient
from femr.labelers import load_labeled_patients, LabeledPatients

def tune_hyperparams(X_train, X_val, y_train, y_val, model, param_grid: Dict[str, List], n_jobs: int = 1):
    """Use GridSearchCV to do hyperparam tuning, but we want to explicitly specify the train/val split.
        Thus, we ned to use `PredefinedSplit` to force the proper splits."""
    # First, concatenate train/val sets (NOTE: need to do concatenation slightly diff for sparse arrays)
    X: np.ndarray = scipy.sparse.vstack([X_train, X_val]) if issparse(X_train) else np.concatenate((X_train, X_val), axis=0)
    y: np.ndarray = np.concatenate((y_train, y_val), axis=0)
    # In PredefinedSplit, -1 = training example, and 0 = validation example
    test_fold: np.ndarray = -np.ones(X.shape[0])
    test_fold[X_train.shape[0]:] = 0
    # Fit model
    clf = GridSearchCV(model, param_grid, n_jobs=n_jobs, verbose=0, cv=PredefinedSplit(test_fold), refit=False)
    clf.fit(X, y)
    best_model = model.__class__(**clf.best_params_)
    best_model.fit(X_train, y_train) # refit on only training data so that we are truly do `k`-shot learning
    return best_model

def run_evaluation(X_train, X_val, X_test, y_train, y_val, y_test, model_head: str, n_jobs: int = 1):
    logger.critical(f"Start | Training {model_head}")
    logger.info(f"Train shape: X = {X_train.shape}, Y = {y_train.shape}")
    logger.info(f"Val shape: X = {X_val.shape}, Y = {y_val.shape}")
    logger.info(f"Test shape: X = {X_test.shape}, Y = {y_test.shape}")
    logger.info(f"Train prevalence:  {np.mean(y_train)}")
    logger.info(f"Val prevalence:  {np.mean(y_val)}")
    logger.info(f"Test prevalence:  {np.mean(y_test)}")
    
    # Shuffle training set
    np.random.seed(X_train.shape[0])
    train_shuffle_idx = np.arange(X_train.shape[0])
    np.random.shuffle(train_shuffle_idx)
    X_train = X_train[train_shuffle_idx]
    y_train = y_train[train_shuffle_idx]

    logger.critical(f"Start | Fitting {model_head}...")
    if model_head == "gbm":
        model = lgb.LGBMClassifier()
        # NOTE: Need to set `min_child_samples = 1`, which specifies the minimum number of samples required in a leaf (terminal node).
        # This is necessary for few-shot learning, since we may have very few samples in a leaf node.
        # Otherwise the GBM model will refuse to learn anything
        XGB_PARAMS['min_child_samples'] = [ 1 ]
        model = tune_hyperparams(X_train, X_val, y_train, y_val, model, XGB_PARAMS, n_jobs=n_jobs)
        logger.info(f"Best hparams: {model.get_params()}")
    elif model_head == "lr":
        # Logistic Regresion
        scaler = MaxAbsScaler().fit(X_train)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        model = LogisticRegression(n_jobs=1, penalty="l2", solver="lbfgs")
        model = tune_hyperparams(X_train, X_val, y_train, y_val, model, LR_PARAMS, n_jobs=n_jobs)
        logger.info(f"Best hparams: {model.get_params()}")
    elif model_head == "protonet":
        model = ProtoNetCLMBRClassifier()
        model = model.fit(X_train, y_train)
    else:
        raise ValueError(f"Model head `{model_head}` not supported.")
    logger.critical(f"Finish | Fitting {model_head}...")
    
    # AUROC
    y_train_proba = model.predict_proba(X_train)[::, 1]
    y_val_proba = model.predict_proba(X_val)[::, 1]
    y_test_proba = model.predict_proba(X_test)[::, 1]
    train_auroc = metrics.roc_auc_score(y_train, y_train_proba)
    val_auroc = metrics.roc_auc_score(y_val, y_val_proba)
    test_auroc = metrics.roc_auc_score(y_test, y_test_proba)
    logger.info(f"Train AUROC: {train_auroc}")
    logger.info(f"Val AUROC:   {val_auroc}")
    logger.info(f"Test AUROC:  {test_auroc}")
    
    # Brier Score
    train_brier = metrics.brier_score_loss(y_train, y_train_proba)
    val_brier = metrics.brier_score_loss(y_val, y_val_proba)
    test_brier = metrics.brier_score_loss(y_test, y_test_proba)
    logger.info(f"Train brier score: {train_brier}")
    logger.info(f"Val brier score:   {val_brier}")
    logger.info(f"Test brier score:  {test_brier}")
    
    # Precision
    train_auprc = metrics.average_precision_score(y_train, y_train_proba)
    val_auprc = metrics.average_precision_score(y_val, y_val_proba)
    test_auprc = metrics.average_precision_score(y_test, y_test_proba)
    logger.info(f"Train AUPRC: {train_auprc}")
    logger.info(f"Val AUPRC:   {val_auprc}")
    logger.info(f"Test AUPRC:  {test_auprc}")

    return model, {
        'auroc' : test_auroc,
        'auprc' : test_auprc,
        'brier' : test_brier,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EHRSHOT evaluation benchmark on a specific task.")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to directory where features will be saved")
    parser.add_argument("--shot_strat", type=str, choices=SHOT_STRATS.keys(), help="What type of X-shot evaluation we are interested in.", required=True )
    parser.add_argument("--labeling_function", required=True, type=str, help="Labeling function for which we will create k-shot samples.", choices=LABELING_FUNCTIONS, )
    parser.add_argument("--num_threads", type=int, help="Number of threads to use")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    LABELING_FUNCTION: str = args.labeling_function
    SHOT_STRAT: str = args.shot_strat
    NUM_THREADS: int = args.num_threads
    PATH_TO_DATABASE: str = args.path_to_database
    PATH_TO_FEATURES_DIR: str = args.path_to_features_dir
    PATH_TO_LABELS_DIR: str = args.path_to_labels_dir
    PATH_TO_LABELED_PATIENTS: str = os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION, 'labeled_patients.csv')
    PATH_TO_SHOTS: str = os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION, f"{SHOT_STRAT}_shots_data.json")
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_OUTPUT_DIR, LABELING_FUNCTION, 'results.json')
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # Load FEMR Patient Database
    database = femr.datasets.PatientDatabase(PATH_TO_DATABASE)

    # Load labels for this task
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)
    patient_ids, label_times, label_values, feature_matrixes = get_labels_and_features(labeled_patients, PATH_TO_FEATURES_DIR)
    train_pids_idx, val_pids_idx, test_pids_idx = get_patient_splits_by_idx(database, patient_ids)

    # Load shot assignments for this task
    with open(PATH_TO_SHOTS) as f:
        few_shots_dict: Dict[str, Dict] = json.load(f)

    # Preprocess certain non-binary labels
    if LABELING_FUNCTION == "chexpert":
        label_values = process_chexpert_labels(label_values)
        sub_tasks: List[str] = CHEXPERT_LABELS
        raise NotImplementedError
    elif LABELING_FUNCTION.startswith('lab_'):
       # Lab value is multi-class, convert to binary
        label_values = convert_multiclass_to_binary_labels(label_values, threshold=1)
        sub_tasks: List[str] = [LABELING_FUNCTION]
    else:
        # Binary classification
        sub_tasks: List[str] = [LABELING_FUNCTION]
        
    # Store results
    # Will have the form: 
    #       results[sub_task][model][head][replicate][score_name] = [ list of scores of type score_name for each k-shot sample in replicate]
    #       results[sub_task][model][head][replicate][ks] = [ list of values of `k` for each k-shot sample in replicate]
    results: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, List]]]]] = {}
    
    # For each base model we are evaluating...
    for model in BASE_MODELS:
        model_heads: List[str] = BASE_MODEL_2_HEADS[model]
        # For each head we can add to the top of this model...
        for head in model_heads:
            # Unpack each individual featurization we want to test
            assert model in feature_matrixes, f"Feature matrix not found for `{model}`. Are you sure you have generated features for this model? If not, you'll need to rerun `generate_features.py` or `generate_clmbr_representations.py`."
            X_train: np.ndarray = feature_matrixes[model][train_pids_idx]
            X_val: np.ndarray = feature_matrixes[model][val_pids_idx]
            X_test: np.ndarray = feature_matrixes[model][test_pids_idx]
            y_test: np.ndarray = label_values[test_pids_idx]
            
            # For each subtask in this task... 
            # (NOTE: The "subtask" is just the same thing as LABELING_FUNCTION for all binary tasks.
            # But for Chexpert, there are multiple subtasks, which of each represents a binary subtask
            for sub_task_idx, sub_task in enumerate(sub_tasks):
                ks: List[int] = sorted(list(few_shots_dict[sub_task].keys()))
                # `results_for_k`: [key] = replicate, [value] = { 'k' : list of k's, 'scores' : dict of scores, where [key] = score name, [value] = list of values }
                results_for_k: Dict[str, Dict[str, Union[List[int], Dict[str, List[float]]]]] = {} 
                
                # For each k-shot sample we are evaluating...
                for k in ks:
                    replicates: List[int] = sorted(list(few_shots_dict[sub_task][k].keys()))

                    # For each replicate of this k-shot sample...
                    for replicate in replicates:
                        logger.success(f"Model: {model} | Head: {head} | Task: {sub_task} | k: {k} | replicate: {replicate}")
                        
                        # Get X/Y train/val for this k-shot sample     
                        shot_dict: Dict[str, List[int]] = few_shots_dict[sub_task][k][replicate]               
                        X_train_k: np.ndarray = X_train[shot_dict["train_idxs"]]
                        X_val_k: np.ndarray = X_val[shot_dict["val_idxs"]]
                        y_train_k: List[int] = shot_dict['label_values_train_k']
                        y_val_k: List[int] = shot_dict['label_values_val_k']
                        y_test_k: List[int] = y_test

                        # CheXpert adjustment
                        if LABELING_FUNCTION == 'chexpert':
                            y_train_k = y_train_k[:, sub_task_idx]
                            y_val_k = y_val_k[:, sub_task_idx]
                            y_test_k = y_test[:, sub_task_idx]

                        # Fit model with hyperparameter tuning
                        best_model, scores = run_evaluation(X_train_k, X_val_k, X_test, y_train_k, y_val_k, y_test_k, model_head=head, n_jobs=NUM_THREADS)
                        
                        # Save results
                        for score_name, score_value in scores.items():
                            if replicate not in results_for_k: results_for_k[replicate] = {}
                            if 'k' not in results_for_k[replicate]: results_for_k[replicate]['k'] = []
                            if 'scores' not in results_for_k[replicate]: results_for_k[replicate]['scores'] = collections.defaultdict(list)
                            results_for_k[replicate]['scores'][score_name].append(score_value)
                            results_for_k[replicate]['k'].append(k)
                # Save results
                if sub_task not in results: results[sub_task] = {}
                if model not in results[sub_task]: results[sub_task][model] = {}
                results[sub_task][model][head] = results_for_k

    logger.info(f"Saving results to: {PATH_TO_OUTPUT_FILE}")
    with open(PATH_TO_OUTPUT_FILE, 'w') as f:
        json.dump(results, f)
    logger.success("Done!")
