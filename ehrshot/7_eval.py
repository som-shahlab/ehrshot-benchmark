"""Create a file at `PATH_TO_LABELS_AND_FEATS_DIR/LABELING_FUNCTION/{SHOT_STRAT}_results.csv` containing:
    Output is a CSV with headers:
        sub_task, model, head, replicate, score_name, score_value, k
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import collections
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from loguru import logger
from sklearn.preprocessing import MaxAbsScaler
from utils import (
    LABELING_FUNCTION_2_PAPER_NAME,
    SHOT_STRATS,
    MODEL_2_INFO,
    get_labels_and_features, 
    process_chexpert_labels, 
    convert_multiclass_to_binary_labels,
    CHEXPERT_LABELS, 
    LR_PARAMS, 
    XGB_PARAMS, 
    RF_PARAMS,
    ProtoNetCLMBRClassifier, 
    get_patient_splits_by_idx
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from scipy.sparse import issparse
import scipy
import lightgbm as lgb
import femr
import femr.datasets
from femr.labelers import load_labeled_patients, LabeledPatients

def tune_hyperparams(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, model, param_grid: Dict[str, List], n_jobs: int = 1):
    """Use GridSearchCV to do hyperparam tuning, but we want to explicitly specify the train/val split.
        Thus, we ned to use `PredefinedSplit` to force the proper splits."""
    # First, concatenate train/val sets (NOTE: need to do concatenation slightly diff for sparse arrays)
    X: np.ndarray = scipy.sparse.vstack([X_train, X_val]) if issparse(X_train) else np.concatenate((X_train, X_val), axis=0)
    y: np.ndarray = np.concatenate((y_train, y_val), axis=0)
    # In PredefinedSplit, -1 = training example, and 0 = validation example
    test_fold: np.ndarray = -np.ones(X.shape[0])
    test_fold[X_train.shape[0]:] = 0
    # Fit model
    clf = GridSearchCV(model, param_grid, scoring='roc_auc', n_jobs=n_jobs, verbose=0, cv=PredefinedSplit(test_fold), refit=False)
    clf.fit(X, y)
    best_model = model.__class__(**clf.best_params_)
    best_model.fit(X_train, y_train) # refit on only training data so that we are truly do `k`-shot learning
    return best_model

def run_evaluation(X_train: np.ndarray, 
                    X_val: np.ndarray, 
                    X_test: np.ndarray, 
                    y_train: np.ndarray, 
                    y_val: np.ndarray, 
                    y_test: np.ndarray, 
                    model_head: str, 
                    n_jobs: int = 1,
                    test_patient_ids: np.ndarray = None) -> Tuple[Any, Dict[str, float]]:
    logger.critical(f"Start | Training {model_head}")
    logger.info(f"Train shape: X = {X_train.shape}, Y = {y_train.shape}")
    logger.info(f"Val shape: X = {X_val.shape}, Y = {y_val.shape}")
    logger.info(f"Test shape: X = {X_test.shape}, Y = {y_test.shape}")
    logger.info(f"Train prevalence:  {np.mean(y_train)}")
    logger.info(f"Val prevalence:  {np.mean(y_val)}")
    logger.info(f"Test prevalence:  {np.mean(y_test)}")
    logger.info(f"Test pids:  {len(test_patient_ids)} | {len(y_test)} | {len(set(test_patient_ids))}")

    # Shuffle training set
    np.random.seed(X_train.shape[0])
    train_shuffle_idx = np.arange(X_train.shape[0])
    np.random.shuffle(train_shuffle_idx)
    X_train = X_train[train_shuffle_idx]
    y_train = y_train[train_shuffle_idx]

    logger.critical(f"Start | Fitting {model_head}...")
    model_head_parts: List[str] = model_head.split("_")
    model_head_base: str = model_head_parts[0]
    if model_head_base == "gbm":
        # XGBoost
        model = lgb.LGBMClassifier(random_state=0)
        # NOTE: Need to set `min_child_samples = 1`, which specifies the minimum number of samples required in a leaf (terminal node).
        # This is necessary for few-shot learning, since we may have very few samples in a leaf node.
        # Otherwise the GBM model will refuse to learn anything
        XGB_PARAMS['min_child_samples'] = [ 1 ]
        model = tune_hyperparams(X_train, X_val, y_train, y_val, model, XGB_PARAMS, n_jobs=n_jobs)
        logger.info(f"Best hparams: {model.get_params()}")
    elif model_head_base == "rf":
        RF_PARAMS['min_samples_leaf'] = [ 1 ]
        RF_PARAMS['min_samples_split'] = [ 2 ]
        model = RandomForestClassifier(random_state=0)
        model = tune_hyperparams(X_train, X_val, y_train, y_val, model, RF_PARAMS, n_jobs=n_jobs)
        logger.info(f"Best hparams: {model.get_params()}")
    elif model_head_base == "lr":
        # Logistic Regresion
        solver: str = model_head_parts[1] # "newton-cg" or "lbfgs" etc.
        # Use built-in SKLearn solver
        scaler = MaxAbsScaler().fit(X_train)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        model = LogisticRegression(n_jobs=1, penalty="l2", tol=0.0001, solver=solver, max_iter=1000, random_state=0)
        model = tune_hyperparams(X_train, X_val, y_train, y_val, model, LR_PARAMS, n_jobs=n_jobs)
        logger.info(f"Best hparams: {model.get_params()}")
    elif model_head_base == "protonet":
        # ProtoNet
        model = ProtoNetCLMBRClassifier()
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Model head `{model_head}` not supported.")
    logger.critical(f"Finish | Fitting {model_head}...")
    
    # Calculate probabilistic preds
    y_train_proba = model.predict_proba(X_train)[::, 1]
    y_val_proba = model.predict_proba(X_val)[::, 1]
    y_test_proba = model.predict_proba(X_test)[::, 1]
    
    metric_dict = {
        'auroc': metrics.roc_auc_score,
        'brier': metrics.brier_score_loss,
        'auprc': metrics.average_precision_score,
    }
    
    # Calculate metrics
    scores = {}
    for metric, func in metric_dict.items():
        scores[metric] = {}
        train_score = func(y_train, y_train_proba)
        val_score = func(y_val, y_val_proba)
        test_score = func(y_test, y_test_proba)

        logger.info(f"Train {metric} score: {train_score}")
        logger.info(f"Val {metric} score:   {val_score}")
        logger.info(f"Test {metric} score:  {test_score}")

        test_set = sorted(list(set(test_patient_ids)))

        score_list = []
        for i in range(1000): # 1k bootstrap replicates
            sample = sklearn.utils.resample(test_set, random_state=i)
            counts = collections.Counter(sample)
            weights = np.zeros_like(test_patient_ids)

            for i, p in enumerate(test_patient_ids):
                weights[i] = counts[p]

            score_val = func(y_test, y_test_proba, sample_weight=weights)
            score_list.append(score_val)

        # 95% CI
        lower, upper = np.percentile(score_list, [2.5, 97.5])

        # Std
        std = np.std(score_list, ddof=1)

        scores[metric]['score'] = test_score
        scores[metric]['std'] = std
        scores[metric]['lower'] = lower
        scores[metric]['mean'] = np.mean(score_list)
        scores[metric]['upper'] = upper

    return model, scores

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EHRSHOT evaluation benchmark on a specific task.")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to directory containing saved features")
    parser.add_argument("--path_to_output_dir", required=True, type=str, help="Path to directory where results will be saved")
    parser.add_argument("--path_to_split_csv", required=True, type=str, help="Path to CSV of splits")
    parser.add_argument("--shot_strat", type=str, choices=SHOT_STRATS.keys(), help="What type of X-shot evaluation we are interested in.", required=True )
    parser.add_argument("--labeling_function", required=True, type=str, help="Labeling function for which we will create k-shot samples.", choices=LABELING_FUNCTION_2_PAPER_NAME.keys(), )
    parser.add_argument("--num_threads", type=int, help="Number of threads to use")
    # TODO: Force refresh for debugging
    parser.add_argument("--is_force_refresh", action='store_true', default=True, help="If set, then overwrite all outputs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    LABELING_FUNCTION: str = args.labeling_function
    SHOT_STRAT: str = args.shot_strat
    NUM_THREADS: int = args.num_threads
    IS_FORCE_REFRESH: bool = args.is_force_refresh
    PATH_TO_DATABASE: str = args.path_to_database
    PATH_TO_FEATURES_DIR: str = args.path_to_features_dir
    PATH_TO_LABELS_DIR: str = args.path_to_labels_dir
    PATH_TO_SPLIT_CSV: str = args.path_to_split_csv
    PATH_TO_LABELED_PATIENTS: str = os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION, 'labeled_patients.csv')
    PATH_TO_SHOTS: str = os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION, f"{SHOT_STRAT}_shots_data.json")
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_OUTPUT_DIR, LABELING_FUNCTION, f'{SHOT_STRAT}_results.csv')
    os.makedirs(os.path.dirname(PATH_TO_OUTPUT_FILE), exist_ok=True)
    
    # If results already exist, then append new results to existing file
    df_existing: Optional[pd.DataFrame] = None
    if os.path.exists(PATH_TO_OUTPUT_FILE):
        logger.warning(f"Results already exist @ `{PATH_TO_OUTPUT_FILE}`.")
        df_existing = pd.read_csv(PATH_TO_OUTPUT_FILE)

    # Load FEMR Patient Database
    database = femr.datasets.PatientDatabase(PATH_TO_DATABASE)

    # Load labels for this task
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)
    patient_ids, label_values, label_times, feature_matrixes = get_labels_and_features(labeled_patients, PATH_TO_FEATURES_DIR, labeling_function=LABELING_FUNCTION)
    train_pids_idx, val_pids_idx, test_pids_idx = get_patient_splits_by_idx(PATH_TO_SPLIT_CSV, patient_ids)
    
    # Load shot assignments for this task
    with open(PATH_TO_SHOTS) as f:
        few_shots_dict: Dict[str, Dict] = json.load(f)

    # Preprocess certain non-binary labels
    if LABELING_FUNCTION == "chexpert":
        label_values = process_chexpert_labels(label_values)
        sub_tasks: List[str] = CHEXPERT_LABELS
    elif LABELING_FUNCTION.startswith('lab_'):
       # Lab value is multi-class, convert to binary
        label_values = convert_multiclass_to_binary_labels(label_values, threshold=1)
        sub_tasks: List[str] = [LABELING_FUNCTION]
    else:
        # Binary classification
        sub_tasks: List[str] = [LABELING_FUNCTION]
        
    # Results will be stored as a CSV with columns:
    #   sub_task, model, head, replicate, score_name, score_value, k
    results: List[Dict[str, Any]] = []
    
    # For each base model we are evaluating...
    for model in MODEL_2_INFO.keys():
        model_heads: List[str] = MODEL_2_INFO[model]['heads']
        # For each head we can add to the top of this model...
        for head in model_heads:
            # TODO: Better way to do this than hardcoding path
            # Check if in experimental LLM setting, then only consider LLM models
            if '/experiments/' in PATH_TO_FEATURES_DIR and model != 'llm':
                continue
            # Unpack each individual featurization we want to test
            assert model in feature_matrixes, f"Feature matrix not found for `{model}`. Are you sure you have generated features for this model? If not, you'll need to rerun `generate_features.py` or `generate_clmbr_representations.py`."
            X_train: np.ndarray = feature_matrixes[model][train_pids_idx]
            X_val: np.ndarray = feature_matrixes[model][val_pids_idx]
            X_test: np.ndarray = feature_matrixes[model][test_pids_idx]
            y_test: np.ndarray = label_values[test_pids_idx]
            
            test_patient_ids = patient_ids[test_pids_idx]
            
            # For each subtask in this task... 
            # NOTE: The "subtask" is just the same thing as LABELING_FUNCTION for all binary tasks.
            # But for Chexpert, there are multiple subtasks, which of each represents a binary subtask
            for sub_task_idx, sub_task in enumerate(sub_tasks):
                # Check if results already exist for this model/head/shot_strat in `results.csv`
                if df_existing is not None:
                    existing_rows: pd.DataFrame = df_existing[
                        (df_existing['labeling_function'] == LABELING_FUNCTION) 
                        & (df_existing['sub_task'] == sub_task) 
                        & (df_existing['model'] == model) 
                        & (df_existing['head'] == head)
                    ]
                    if existing_rows.shape[0] > 0:
                        # Overwrite
                        if IS_FORCE_REFRESH:
                            logger.warning(f"Results ALREADY exist for {model}/{head}:{LABELING_FUNCTION}/{sub_task} in `results.csv`. Overwriting these rows because `is_force_refresh` is TRUE.")
                        else:
                            logger.warning(f"Results ALREADY exist for {model}/{head}:{LABELING_FUNCTION}/{sub_task} in `results.csv`. Skipping this combination because `is_force_refresh` is FALSE.")
                            results += existing_rows.to_dict(orient='records')
                            continue
                    else:
                        # Append
                        logger.warning(f"Results DO NOT exist for {model}/{head}:{LABELING_FUNCTION}/{sub_task} in `results.csv`. Appending to this CSV.")
        
                ks: List[int] = sorted([ int(x) for x in few_shots_dict[sub_task].keys() ])
                
                # For each k-shot sample we are evaluating...
                for k in ks:
                    replicates: List[int] = sorted([ int(x) for x in few_shots_dict[sub_task][str(k)].keys() ])

                    # For each replicate of this k-shot sample...
                    for replicate in replicates:
                        logger.success(f"Model: {model} | Head: {head} | Task: {sub_task} | k: {k} | replicate: {replicate}")
                        
                        # Get X/Y train/val for this k-shot sample     
                        shot_dict: Dict[str, List[int]] = few_shots_dict[sub_task][str(k)][str(replicate)]               
                        X_train_k: np.ndarray = X_train[shot_dict["train_idxs"]]
                        X_val_k: np.ndarray = X_val[shot_dict["val_idxs"]]
                        y_train_k: np.ndarray = np.array(shot_dict['label_values_train_k'])
                        y_val_k: np.ndarray = np.array(shot_dict['label_values_val_k'])
                        y_test_k: np.ndarray = np.array(y_test)

                        # CheXpert adjustment
                        if LABELING_FUNCTION == 'chexpert':
                            y_test_k = y_test[:, sub_task_idx]

                        # Fit model with hyperparameter tuning
                        best_model, scores = run_evaluation(X_train_k, X_val_k, X_test, y_train_k, y_val_k, y_test_k, model_head=head, n_jobs=NUM_THREADS, test_patient_ids=test_patient_ids)

                        # Save results
                        for score_name, score_value in scores.items():
                            results.append({
                                'labeling_function' : LABELING_FUNCTION,
                                'sub_task' : sub_task,
                                'model' : model,
                                'head' : head,
                                'replicate' : replicate,
                                'k' : k,
                                'score' : score_name,
                                'value' : score_value['score'],
                                'std' : score_value['std'],
                                'lower' : score_value['lower'],
                                'mean' : score_value['mean'],
                                'upper' : score_value['upper'],
                            })

    logger.info(f"Saving results to: {PATH_TO_OUTPUT_FILE}")
    df: pd.DataFrame = pd.DataFrame(results)
    logger.info(f"Added {df.shape[0] - (df_existing.shape[0] if df_existing is not None else 0)} rows")
    df.to_csv(PATH_TO_OUTPUT_FILE)
    logger.success("Done!")
