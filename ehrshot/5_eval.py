import argparse
import os
from typing import List, Optional

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from loguru import logger
from utils import (LABELING_FUNCTIONS, load_data, save_data, 
                   get_pid_label_times_and_values, process_chexpert_labels, 
                   convert_multiclass_to_binary_labels)
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from scipy.sparse import issparse
import scipy
import lightgbm as lgb
import femr
import femr.datasets

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


def tune_hyperparams(X_train, y_train, X_val, y_val, model, params, num_threads: int = 1):
    # In `test_fold`, -1 indicates that the corresponding sample is used for training, and a value >=0 indicates the test set.
    # We use `PredefinedSplit` to specify our custom validation split
    if issparse(X_train):
        # Need to concatenate sparse matrices differently
        X = scipy.sparse.vstack([X_train, X_val])
    else:
        X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    test_fold = -np.ones(X.shape[0])
    test_fold[X_train.shape[0]:] = 1
    clf = GridSearchCV(model, params, n_jobs=6, verbose=1, cv=PredefinedSplit(test_fold=test_fold), refit=False)
    clf.fit(X, y)
    best_params = clf.best_params_
    best_model = model.__class__(**clf.best_params_)
    best_model.fit(X_train, y_train)
    return best_model

def generate_binary_classification_metrics(X_train, y_train, X_val, y_val, X_test, y_test, model="gbm", num_threads=1, is_tune_hyperparams=True):
    logger.critical(f"{model} | Training")
    logger.info(f"Train shape: X = {X_train.shape}, Y = {y_train.shape}")
    logger.info(f"Val shape: X = {X_val.shape}, Y = {y_val.shape}")
    logger.info(f"Test shape: X = {X_test.shape}, Y = {y_test.shape}")

    logger.info(f"Train prevalence:  {sum(y_train)/len(y_train)}")
    logger.info(f"Val prevalence: {sum(y_val)/len(y_val)}")
    logger.info(f"Test prevalence: {sum(y_test)/len(y_test)}")
    
    # Shuffle training set
    np.random.seed(X_train.shape[0])
    train_shuffle_idx = np.arange(X_train.shape[0])
    np.random.shuffle(train_shuffle_idx)
    X_train = X_train[train_shuffle_idx, :]
    y_train = y_train[train_shuffle_idx]

    if model == "gbm":
        model = lgb.LGBMClassifier()
        if is_tune_hyperparams:
            logger.info(f"Tuning hyperparameters for LightGBM...")
            # `min_child_samples`: Specifies the minimum number of samples required in a leaf (terminal node).
            XGB_PARAMS['min_child_samples'] = [ 1 ]
            model = tune_hyperparams(X_train, y_train, X_val, y_val, model, XGB_PARAMS, num_threads=num_threads)
            logger.info(f"Best params: {model.get_params()}")
        else:
            model.fit(X_train, y_train)
    elif model == "logistic":
        # Logistic Regresion
        scaler = MaxAbsScaler().fit(
            X_train
        )  # best for sparse data: see https://scikit-learn.org/stable/modules/preprocessing.html#scaling-sparse-data
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        model = LogisticRegression(n_jobs=1, penalty="l2", solver="lbfgs")
        if is_tune_hyperparams:
            logger.info(f"Tuning hyperparameters for LogReg...")
            model = tune_hyperparams(X_train, y_train, X_val, y_val, model, LR_PARAMS, num_threads=num_threads)
            logger.info(f"Best params: {model.get_params()}")
        else:
            model.fit(X_train, y_train)
    elif model == "protonet":
        model = ProtoNetCLMBRClassifier()
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Model {model} not supported.")
    y_train_proba = model.predict_proba(X_train)[::, 1]
    y_val_proba = model.predict_proba(X_val)[::, 1]
    y_test_proba = model.predict_proba(X_test)[::, 1]
    train_auroc = metrics.roc_auc_score(y_train, y_train_proba)
    val_auroc = metrics.roc_auc_score(y_val, y_val_proba)
    test_auroc = metrics.roc_auc_score(y_test, y_test_proba)
    
    logger.info(f"Train AUROC: {train_auroc}")
    logger.info(f"Val AUROC: {val_auroc}")
    logger.info(f"Test AUROC: {test_auroc}")
    train_calibration = metrics.brier_score_loss(y_train, y_train_proba)
    val_calibration = metrics.brier_score_loss(y_val, y_val_proba)
    test_calibration = metrics.brier_score_loss(y_test, y_test_proba)
    logger.info(f"Train calibration: {train_calibration}")
    logger.info(f"Val calibration: {val_calibration}")
    logger.info(f"Test calibration: {test_calibration}")
    train_auprc = metrics.average_precision_score(y_train, y_train_proba)
    val_auprc = metrics.average_precision_score(y_val, y_val_proba)
    test_auprc = metrics.average_precision_score(y_test, y_test_proba)
    logger.info(f"Train AUPRC: {train_auprc}")
    logger.info(f"Val AUPRC: {val_auprc}")
    logger.info(f"Test AUPRC: {test_auprc}")

    return model, {
        'auroc' : test_auroc,
        'auprc' : test_auprc,
        'calibration' : test_calibration,
    }

def plot_results(label_dict: dict, 
                 labeling_function: str, 
                 size: int = 14,
                 path_to_save: str = "./"):
    """label_dict[labeling_function][model_name][replicate][scores][auroc]"""
    task = label_dict[labeling_function]
    models = list(task.keys())
    scores = list(task[models[0]][0]["scores"].keys()) # e.g. auroc, ap, mse
    n_replicates: int = len(task[models[0]])
    for s in sorted(scores, reverse=True):
        if s == "auroc":
            colors = ["green", "red", 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        elif s == "auprc":
            colors = ["blue", "orange", 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        else:
            continue
        fig, ax = plt.subplots()
        for idx, m in enumerate(models):
            if m == "Codes_Only":
                legend_name = "CLMBR"
            elif m == "Count_based_GBM":
                legend_name = "GBM"
            all_values = [] # Collect values across all replicates
            for replicate in task[m].keys():
                all_values.append(task[m][replicate]["scores"][s])
            all_values = np.array(all_values)
            means = np.mean(all_values, axis=0)
            stds = np.std(all_values, axis=0)
            x = task[m][0]['k']
            x = [x_i*4 for x_i in x]
            color = colors[idx]
            plt.plot(x, means, color=color, label=legend_name, linestyle='--', marker='o')
            plt.plot(x, means - 0.5 * stds, color=color, alpha=0.1)
            plt.plot(x, means + 0.5 * stds, color=color, alpha=0.1)
            plt.fill_between(x, means - 0.5 * stds, means + 0.5 * stds, color=color, alpha=0.2)
        
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        # plt.legend(fontsize=size)
        # plt.xlabel("# of Train Examples per Class", fontsize=size)
        # plt.ylabel(s.upper(), fontsize=size)
        plt.title(labeling_function_to_paper_name[labeling_function], size=8)
        plt.xticks(fontsize=size)
        plt.yticks(fontsize=size)
        plt.tight_layout()
        new_path_to_save = os.path.join(path_to_save, f"{labeling_function}_{s}.png")
        plt.savefig(new_path_to_save, dpi=300)
        plt.close('all')

def create_file_name(shot_strat: str, model_head: str):
    return f"{shot_strat}_{model_head}"

def main(args):
    # Force settings
    labeling_function: str = args.labeling_function
    shot_strat: str = args.shot_strat
    model_head: str = args.model_head
    num_replicates: int = args.num_replicates
    PATH_TO_DATA: str = args.path_to_data
    CLMBR_MODELS: List[str] = args.clmbr_models

    PATH_TO_DATABASE: str = os.path.join(PATH_TO_DATA, "femr/extract")

    # Set up logging
    PATH_TO_SAVE: str = os.path.join(args.path_to_save, labeling_function)
    os.makedirs(PATH_TO_SAVE, exist_ok=True)

    PATH_TO_SHOTS: str = os.path.join(PATH_TO_DATA, f"benchmark/{labeling_function}/{shot_strat}_shots_data.json")
    logger.info(f"Args: {PATH_TO_SHOTS}")

    path_to_log_file: str = os.path.join(PATH_TO_SAVE, f"{labeling_function}_{create_file_name(shot_strat, model_head)}.log")
    if os.path.exists(path_to_log_file):
        os.remove(path_to_log_file)
    logger.add(path_to_log_file)
    logger.info(f"Args: {args}")

    logger.info(f"Args: {args}")
    
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

    patient_ids, label_times, label_values, clmbr_feature_matrix, count_feature_matrix = get_pid_label_times_and_values(PATH_TO_DATA, labeling_function)

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
    X_train_count, X_val_count, X_test_count = count_feature_matrix[train_pids_idx], count_feature_matrix[val_pids_idx], count_feature_matrix[test_pids_idx]
    X_train_clmbr, X_val_clmbr, X_test_clmbr = clmbr_feature_matrix[train_pids_idx], clmbr_feature_matrix[val_pids_idx], clmbr_feature_matrix[test_pids_idx]

    prevalence = np.sum(y_test != 0) / len(y_test)
    logger.info(f"CLMBR Train shape: X = {X_train_clmbr.shape}, Y = {y_train.shape}")
    logger.info(f"Count Train shape: X = {X_train_count.shape}, Y = {y_train.shape}")
    logger.info(f"CLMBR Val shape: X = {X_val_clmbr.shape}, Y = {y_val.shape}")
    logger.info(f"Count Val shape: X = {X_val_count.shape}, Y = {y_val.shape}")
    logger.info(f"CLMBR Test shape: X = {X_test_clmbr.shape}, Y = {y_test.shape}")
    logger.info(f"COUNT Test shape: X = {X_test_count.shape}, Y = {y_test.shape}")
    logger.info(f"Test prevalence: {prevalence}")
    
    # Load PatientDatabase

    # Store results
    label_dict = {}
    if labeling_function == "chexpert":
        for label_str in CHEXPERT_LABELS:
            label_dict[label_str] = {}
    else:
        label_dict[labeling_function] = {}

    for model_name in CLMBR_MODELS:
        logger.critical(f"Running model: {model_name}")
        if model_name.startswith("Count_based_"):
            # Load count-based featurizations
            # Select only the patients that are in the CLMBR reprs
            # Choose model head
            if model_name == 'Count_based_GBM':
                model_head = 'gbm'
                X_train = X_train_count
                X_val = X_val_count
                X_test = X_test_count
            else:
                raise ValueError(f"Invalid count-based feats value for: {model_name}")
        else:
            X_train = X_train_clmbr
            X_val = X_val_clmbr
            X_test = X_test_clmbr
            # Choose model head = Logistic Regression
            model_head = args.model_head
        logger.critical(f"Using head: {model_head}")

        few_shots_dict = load_data(PATH_TO_SHOTS)
        if labeling_function == 'chexpert':
            # Multilabel
            for replicate in range(num_replicates):
                for idx, label_str in enumerate(CHEXPERT_LABELS):
                    few_shots_results = []
                    for k in SHOTS:
                        k = str(k)
                        replicate = str(replicate)
                        logger.critical(f"Label: {label_str} | k: {k} | Replicate: {replicate}")
                        train_idxs = few_shots_dict[label_str][k][replicate]["train_idxs"]
                        val_idxs = few_shots_dict[label_str][k][replicate]["val_idxs"]
                        X_train_k, y_train_k = X_train[train_idxs], y_train[train_idxs]
                        X_val_k, y_val_k = X_val[val_idxs], y_val[val_idxs]

                        y_train_k_one_label = y_train_k[:, idx]
                        y_val_k_one_label = y_val_k[:, idx]
                        y_test_one_label = y_test[:, idx]
                        model, score = generate_binary_classification_metrics(X_train_k, y_train_k_one_label, X_val_k, y_val_k_one_label, X_test, y_test_one_label, model=model_head, is_tune_hyperparams=args.is_tune_hyperparams)
                        few_shots_results.append(score)
                    if not model_name in label_dict[label_str]:
                        label_dict[label_str][model_name] = {}
                    label_dict[label_str][model_name][replicate] = {
                        # For `scores`, convert list of dicts to dict of lists
                        'scores' : collections.defaultdict(list, {k: [d[k] for d in few_shots_results] for k in set().union(*few_shots_results)}),
                        'k' : SHOTS,
                        'best_params': model.get_params()
                    }
        else:
            # Binary classification
            for replicate in range(num_replicates):
                few_shots_results = []
                for k in SHOTS:
                    k = str(k)
                    replicate = str(replicate)
                    logger.critical(f"Label: {labeling_function} | k: {k} | Replicate: {replicate}")
                    train_idxs = few_shots_dict[labeling_function][str(k)][replicate]["train_idxs"]
                    val_idxs = few_shots_dict[labeling_function][str(k)][replicate]["val_idxs"]
                    X_train_k, y_train_k = X_train[train_idxs], y_train[train_idxs]
                    X_val_k, y_val_k = X_val[val_idxs], y_val[val_idxs]
                    model, score = generate_binary_classification_metrics(X_train_k, y_train_k, X_val_k, y_val_k, X_test, y_test, model=model_head, is_tune_hyperparams=args.is_tune_hyperparams)
                    few_shots_results.append(score)
                if not model_name in label_dict[labeling_function]:
                    label_dict[labeling_function][model_name] = {}
                label_dict[labeling_function][model_name][replicate] = {
                    # For `scores`, convert list of dicts to dict of lists
                    'scores' : collections.defaultdict(list, {k: [d[k] for d in few_shots_results] for k in set().union(*few_shots_results)}),
                    'k' : SHOTS,
                    'best_params': model.get_params()
                }
    # Save results
    path_to_save: str = os.path.join(PATH_TO_SAVE, f"{shot_strat}_tune_params_{args.is_tune_hyperparams}.json")
    save_data(label_dict, path_to_save)
    logger.success("DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLMBR patient representations")
    # Paths
    parser.add_argument("--path_to_data", type=str, help=( "Path where you have all the data, including your clmbr representations, labeled_featururized patients" ), )
    parser.add_argument("--path_to_save", type=str, help=( "Path to save evaluation data" ), )
    parser.add_argument("--labeling_function", required=True, type=str, help="Name of labeling function to create.", choices=LABELING_FUNCTIONS, )
    parser.add_argument("--shot_strat", type=str, choices=['both', 'few', 'long', 'debug'], help="Type of sampling to do", required=True )
    
    # Logistics
    parser.add_argument("--num_replicates", type=int, help="For std bars in plots", default=3, )
    parser.add_argument("--model_head", type=str, choices=['gbm', 'logistic', 'protonet'], default='logistic', help="CLMBR Head")
    
    # Subsampling labeled patients
    # Eval pipeline tuning
    parser.add_argument("--is_tune_hyperparams", action='store_true', default=False, help="Tune parameters or not")
    parser.add_argument("--is_preserve_prevalence", action='store_true', default=False, help="Preserve Prevalence or not")

    parser.add_argument('--clmbr_models', nargs='+', default=["Count_based_GBM", "Codes_Only"], help="Pass models to run. ")
    args = parser.parse_args()
    main(args)
