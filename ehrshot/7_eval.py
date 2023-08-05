import argparse
import json
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
import femr.models.conjugate_gradient

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


def train_and_compute(patient_ids, label_times, label_values, count_feature_mtarix, clmbr_feature_matrix, motor_feature_matrix, train_mask, val_mask):
    # Train/test splits
    split_seed: int = 97
    hashed_pids = np.array([database.compute_split(split_seed, pid) for pid in patient_ids])
    train_pids_idx = np.where(np.logical_and(hashed_pids < 70, train_mask))[0]
    val_pids_idx = np.where(np.logical_and((70 <= hashed_pids) & (hashed_pids < 85), val_mask))[0]
    test_pids_idx = np.where(hashed_pids >= 85)[0]

    patient_ids_train, label_times_train = patient_ids[train_pids_idx], label_times[train_pids_idx]
    patient_ids_val, label_times_val = patient_ids[val_pids_idx], label_times[val_pids_idx]
    patient_ids_test, label_times_test = patient_ids[test_pids_idx], label_times[test_pids_idx]

    y_train, y_val, y_test = label_values[train_pids_idx], label_values[val_pids_idx], label_values[test_pids_idx]
    X_train_count, X_val_count, X_test_count = count_feature_matrix[train_pids_idx], count_feature_matrix[val_pids_idx], count_feature_matrix[test_pids_idx]
    X_train_clmbr, X_val_clmbr, X_test_clmbr = clmbr_feature_matrix[train_pids_idx], clmbr_feature_matrix[val_pids_idx], clmbr_feature_matrix[test_pids_idx]
    X_train_motor, X_val_motor, X_test_motor = motor_feature_matrix[train_pids_idx], motor_feature_matrix[val_pids_idx], motor_feature_matrix[test_pids_idx]

    prevalence = np.sum(y_test != 0) / len(y_test)

    assert X_train_clmbr.shape[0] == y_train.shape[0]
    assert X_train_motor.shape[0] == y_train.shape[0]
    assert X_train_count.shape[0] == y_train.shape[0]

    logger.info(f"Num Train: {y_train.shape[0]}, Num Valid: {y_val.shape[0]}, Num Test: {y_test.shape[0]}")
    logger.info(f"Test prevalence: {prevalence}")

    clmbr_model = femr.models.conjugate_gradient.train_logistic_regression(X_train_clmbr, y_train.astype(float), X_val_clmbr, y_val.astype(float))
    
    y_train_proba = np.dot(X_train_clmbr, clmbr_model)
    y_val_proba = np.dot(X_val_clmbr, clmbr_model)
    y_test_proba = np.dot(X_test_clmbr, clmbr_model)

    train_auroc = metrics.roc_auc_score(y_train, y_train_proba)
    val_auroc = metrics.roc_auc_score(y_val, y_val_proba)
    test_auroc = metrics.roc_auc_score(y_test, y_test_proba)

    results = {}

    results['clmbr'] = {
        'train': train_auroc,
        'val': val_auroc,
        'test': test_auroc,
    }
    
    logger.info(f"CLMBR Train AUROC: {train_auroc}")
    logger.info(f"CLMBR Val AUROC: {val_auroc}")
    logger.info(f"CLMBR Test AUROC: {test_auroc}")
    
    motor_model = femr.models.conjugate_gradient.train_logistic_regression(X_train_motor, y_train.astype(float), X_val_motor, y_val.astype(float))
    
    y_train_proba = np.dot(X_train_motor, motor_model)
    y_val_proba = np.dot(X_val_motor, motor_model)
    y_test_proba = np.dot(X_test_motor, motor_model)

    train_auroc = metrics.roc_auc_score(y_train, y_train_proba)
    val_auroc = metrics.roc_auc_score(y_val, y_val_proba)
    test_auroc = metrics.roc_auc_score(y_test, y_test_proba)

    results['motor'] = {
        'train': train_auroc,
        'val': val_auroc,
        'test': test_auroc,
    }

    logger.info(f"MOTOR Train AUROC: {train_auroc}")
    logger.info(f"MOTOR Val AUROC: {val_auroc}")
    logger.info(f"MOTOR Test AUROC: {test_auroc}")

    if False:
        model = lgb.LGBMClassifier()
        logger.info(f"Tuning hyperparameters for LightGBM...")
        # `min_child_samples`: Specifies the minimum number of samples required in a leaf (terminal node).
        XGB_PARAMS['min_child_samples'] = [ 1 ]
        model = tune_hyperparams(X_train_count, y_train, X_val_count, y_val, model, XGB_PARAMS, num_threads=args.num_threads)
        logger.info(f"Best params: {model.get_params()}")


        y_train_proba = model.predict_proba(X_train_count)[::, 1]
        y_val_proba = model.predict_proba(X_val_count)[::, 1]
        y_test_proba = model.predict_proba(X_test_count)[::, 1]
        train_auroc = metrics.roc_auc_score(y_train, y_train_proba)
        val_auroc = metrics.roc_auc_score(y_val, y_val_proba)
        test_auroc = metrics.roc_auc_score(y_test, y_test_proba)
        
        logger.info(f"GBM Train AUROC: {train_auroc}")
        logger.info(f"GBM Val AUROC: {val_auroc}")
        logger.info(f"GBM Test AUROC: {test_auroc}")
        
        results['gbm'] = {
            'train': train_auroc,
            'val': val_auroc,
            'test': test_auroc,
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CLMBR patient representations")
    # Paths
    parser.add_argument("--path_to_database", type=str)
    parser.add_argument("--path_to_labels", type=str)
    parser.add_argument("--path_to_features", type=str)
    parser.add_argument("--path_to_save", type=str)
    parser.add_argument("--num_threads", type=int)
    
    args = parser.parse_args()

    # Set up logging
    os.makedirs(args.path_to_save, exist_ok=True)
    

    database = femr.datasets.PatientDatabase(args.path_to_database)

    labeled_patients = femr.labelers.load_labeled_patients(os.path.join(args.path_to_labels, 'labeled_patients.csv'))

    patient_ids, label_times, label_values, count_feature_matrix, clmbr_feature_matrix, motor_feature_matrix = get_pid_label_times_and_values(args.path_to_features, labeled_patients)

    label_name = os.path.basename(args.path_to_labels)

    logger.info(f"Task name: {label_name}")

    if label_name == "chexpert":
        label_values = process_chexpert_labels(label_values)
        os.exit()
    elif label_name.endswith('_lab'):
       # Lab value is multi-class, convert to binary
        label_values = convert_multiclass_to_binary_labels(label_values, threshold=1)
    
    results = {
       'long': train_and_compute(patient_ids, label_times, label_values, 
            count_feature_matrix, clmbr_feature_matrix, motor_feature_matrix,
            np.ones_like(patient_ids), np.ones_like(patient_ids))
    }

    with open(os.path.join(args.path_to_labels, 'few_shots_data.json')) as f:
        few_shots = json.load(f)

    if False:
        results['few'] = {}

        for k, k_vals in few_shots.items():
            results['few'][k] = {}
            for r, r_instance in k_vals.items():
                print(k, r, r_instance)
                train_mask = np.zeros_like(patient_ids)
                val_mask = np.zeros_like(patient_ids)

                for name, mask in (('train', train_mask), ('val', val_mask)):
                    pids = np.array(r_instance[f'patient_ids_{name}_k'], dtype=np.int64)
                    times = np.array(r_instance[f'label_times_{name}_k'], dtype="datetime64[us]")
                
                    order = np.lexsort((times, pids))
                    pids = pids[order]
                    times = times[order]
                
                    join_indices = femr.extension.dataloader.compute_feature_label_alignment(pids, times.astype(np.int64), patient_ids, label_times.astype("datetime64[us]").astype(np.int64))
                    mask[join_indices] = 1

                results['few'][k][r] = train_and_compute(patient_ids, label_times, label_values, count_feature_matrix, clmbr_feature_matrix, train_mask, val_mask)

    with open(os.path.join(args.path_to_save, 'results.json'), 'w') as f:
        json.dump(results, f)
