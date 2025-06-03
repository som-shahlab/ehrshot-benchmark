"""Create a file at `PATH_TO_LABELS_AND_FEATS_DIR/LABELING_FUNCTION/{SHOT_STRAT}_results.csv` containing:
    Output is a CSV with headers:
        sub_task, model, head, replicate, score_name, score_value, k
"""
import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import collections
import pandas as pd
from tqdm import tqdm
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch.optim.lr_scheduler import LambdaLR
from loguru import logger
from sklearn.preprocessing import MaxAbsScaler
from ehrshot.utils import (
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
import torch
from jaxtyping import Float
from femr.labelers import load_labeled_patients, LabeledPatients

def fit_logreg_lbfgs(model: torch.nn.Module, 
                     X: Float[torch.Tensor, 'B H'], 
                     y: Float[torch.Tensor, 'B'],
                     lr: float = 1.0,
                     C: float = 1.0,
                     penalty: Optional[str] = 'l2',
                     max_iter: float = 1000) -> None:
    """Train a logistic regression model with (optional) regularization using the LBFGS optimizer."""
    model.weight.data.zero_()
    model.bias.data.zero_()
    X = X.to(model.weight.device)
    y = y.long().to(model.weight.device)
    
    # LBFGS
    opt = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=max_iter)
    
    # Forward/backward pass
    def closure():
        if torch.is_grad_enabled():
            opt.zero_grad()
        preds: Float[torch.Tensor, 'B C'] = model(X)
        
        # Compute CE loss v. true binary labels
        ce_loss = torch.nn.functional.cross_entropy(preds, y)
        
        # Regularization (optional)
        if penalty == 'l1':
            reg_loss = 1/C * 1/(2*X.shape[0]) * (model.weight.abs()).sum()
        elif penalty == 'l2':
            reg_loss = 1/C * 1/(2*X.shape[0]) * (model.weight**2).sum()
        else:
            reg_loss = 0

        loss = reg_loss + ce_loss
        if loss.requires_grad:
            loss.backward()
        return loss

    # Run LBFGS
    opt.step(closure)

    
def lr_lambda(current_step: int, total_steps: int) -> float:
    # Linearly scale the learning rate from 0 to 1 over the total steps
    if current_step >= total_steps:
        return 1.0
    return current_step / float(max(1, total_steps))

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

def compare_models(model1, model2):
    """For sanity checking unfrozen/frozen layers of model checkpoints"""
    differences = []
    for idx, (param1, param2) in enumerate(zip(model1.parameters(), model2.parameters())):
        if not torch.equal(param1, param2):
            differences.append(idx)
            print(f'Difference found in layer: #{idx}')
            print(f'Parameters: {param1.shape} v. {param2.shape}')
            if param1.shape == param2.shape:
                print(f'Diff: {param1 - param2}')
    
    if not differences:
        print('No differences found between the models.')
    else:
        print(f'Found {len(differences)} differences between the models.')
    return differences

def calc_metrics(y_train: Float[np.ndarray, 'N'], 
                 y_train_proba: Float[np.ndarray, 'N'], 
                 y_val: Float[np.ndarray, 'N'], 
                 y_val_proba: Float[np.ndarray, 'N'], 
                 y_test: Float[np.ndarray, 'N'], 
                 y_test_proba: Float[np.ndarray, 'N'], 
                 test_patient_ids: List[int]) -> Dict[str, Dict[str, float]]:
    """Calculates AUROC, AUPRC, and Brier scores for train, val, and test sets.
        NOTE: Expects `y_train_proba`, `y_val_proba`, and `y_test_proba` to be the probability of the positive class.
    """

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
        scores[metric]['score_val'] = val_score
        scores[metric]['score_train'] = train_score
        scores[metric]['std'] = std
        scores[metric]['lower'] = lower
        scores[metric]['mean'] = np.mean(score_list)
        scores[metric]['upper'] = upper
    return scores

def setup_finetuning(model: 'CookbookModelWithClassificationHead', finetune_strat: str):
    # Start by freezing all `base_model` params
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # Find layers we can unfreeze
    if model.base_model_name == 'mamba':
        layers = model.base_model.layers
    elif model.base_model_name == 'hyena':
        layers = model.base_model.layers
    elif model.base_model_name == 'gpt2':
        layers = model.base_model.h
    elif model.base_model_name == 'llama':
        layers = model.base_model.layers
    elif model.base_model_name == 'bert':
        layers = model.base_model.encoder.layer
    else:
        raise ValueError(f"Base model `{model.base_model_name}` not supported.")

    # Selectively unfreeze, depending on `finetune_strat`
    if finetune_strat.startswith('layers'):
        n_layers: int = int(finetune_strat.split('=')[-1])
        for layer in layers[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    elif finetune_strat.startswith("full"):
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True
    elif finetune_strat.startswith('frozen'):
        pass
    else:
        raise ValueError(f"Fine-tuning strategy `{finetune_strat}` not supported.")
    return model, layers

def sanity_check_finetuning_model(model: 'CookbookModelWithClassificationHead', layers, finetune_strat):
    """For sanity checking that the output of `setup_finetuning` has correctly unfrozen the proper layers.
        NOTE: This takes ~10 seconds, so don't use if trying to be fast.
    """
    if finetune_strat.startswith('full'):
        for l in layers:
            for param in l.parameters():
                assert param.requires_grad, "All layers should be unfrozen"
    elif finetune_strat.startswith('frozen'):
        for l in layers:
            for param in l.parameters():
                assert not param.requires_grad, "All layers should be frozen"
    elif finetune_strat.startswith('layers'):
        n_layers: int = int(finetune_strat.split('=')[-1])
        for l in layers[-n_layers:]:
            for param in l.parameters():
                assert param.requires_grad, f"Last {n_layers} layers should be frozen"
        for l in layers[:-n_layers]:
            for param in l.parameters():
                assert not param.requires_grad, f"First {len(layers) - n_layers} layers should be frozen"
    else:
        raise ValueError(f"Fine-tuning strategy `{finetune_strat}` not supported.")

def finetune_pytorch_model(X_train_timelines: torch.Tensor, 
                           X_train: np.ndarray,
                           X_val: np.ndarray,
                           y_train: np.ndarray, 
                           y_val: np.ndarray, 
                           model: torch.nn.Module, 
                           optimizer,
                           scheduler,
                           criterion,
                           logreg_C: float,
                           logreg_penalty: Optional[str],
                           is_finetune_logreg_first: bool,
                           pad_token_id: int,
                           model_name: str, 
                           model_head: str,
                           batch_size: int,
                           n_epochs: int,
                           replicate: int,
                           device: str) -> torch.nn.Module:
    torch.manual_seed(replicate)
    model.train()
    
    # First, finetune logreg head (if applicable)
    if is_finetune_logreg_first:
        logger.info(f"Start | Finetuning 'logreg_first'")
        fit_logreg_lbfgs(model.classifier, torch.from_numpy(X_train), torch.from_numpy(y_train), lr=1.0, C=logreg_C, penalty=logreg_penalty, max_iter=1000)
        y_train_proba: Float[np.ndarray, 'N 2'] = torch.nn.functional.softmax(model.classifier(torch.from_numpy(X_train).to(device)), dim=-1).detach().cpu().numpy()
        y_val_proba: Float[np.ndarray, 'N 2'] = torch.nn.functional.softmax(model.classifier(torch.from_numpy(X_val).to(device)), dim=-1).detach().cpu().numpy()
        logger.warning(f"Train AUROC from logreg_first: {metrics.roc_auc_score(y_train, y_train_proba[:,1])}")
        logger.warning(f"Val AUROC from logreg_first: {metrics.roc_auc_score(y_val, y_val_proba[:,1])}")
        logger.info(f"Finish | Finetuning 'logreg_first'")

    for epoch in range(n_epochs):
        for batch_start in tqdm(range(0, X_train_timelines.shape[0], batch_size), desc=f'Finetuning: epoch={epoch} | model={model_name[:15]} | head={model_head[:15]} | k={X_train_timelines.shape[0]}', total=X_train_timelines.shape[0] // batch_size):
            # Subset training batch
            X_train_batch = X_train_timelines[batch_start:batch_start+batch_size]
            y_train_batch = y_train[batch_start:batch_start+batch_size]

            # Drop last batch if it's too small (otherwise torch.compile complains about non-static shaped inputs)
            # Ignore case where batch_size > X_train_timelines.shape[0] (i.e. we have fewer samples than batch_size so there's just one batch)
            if X_train_batch.shape[0] != batch_size and batch_size < X_train_timelines.shape[0]:
                continue
            
            # Tokenize batch
            input_ids = torch.tensor(X_train_batch, device=device)
            attention_mask = (input_ids != pad_token_id).int()
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
            if "hyena" in model_name:
                batch.pop('attention_mask')

            # Run model to get logits for each class
            logits: Float[torch.Tensor, 'B C'] = model(**batch)
            assert logits.shape == (X_train_batch.shape[0], 2)

            # Compute CE loss v. true binary labels
            binary_labels: Float[torch.Tensor, 'B'] = torch.tensor(y_train_batch, device=device).long()
            loss = criterion(logits, binary_labels)
            
            # Regularization (optional)
            if logreg_penalty == 'l1':
                reg_loss = 1/logreg_C * 1/(2*X_train_batch.shape[0]) * (model.classifier.weight.abs()).sum()
            elif logreg_penalty == 'l2':
                reg_loss = 1/logreg_C * 1/(2*X_train_batch.shape[0]) * (model.classifier.weight**2).sum()
            else:
                reg_loss = 0
            loss += reg_loss

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # LR scheduler
            if scheduler:
                scheduler.step()
    return model

def eval_pytorch_model(X_train_timelines: np.ndarray,
                        X_val_timelines: np.ndarray,
                        X_test_timelines: np.ndarray,
                        y_train: np.ndarray,
                        y_val: np.ndarray,
                        y_test: np.ndarray,
                        model: torch.nn.Module, 
                        pad_token_id: int,
                        model_name: str, 
                        model_head: str, 
                        batch_size: int,
                        device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_train_proba, y_val_proba, y_test_proba = [], [], []
    with torch.no_grad():
        for X, y, split in zip([X_train_timelines, X_val_timelines, X_test_timelines], [y_train_proba, y_val_proba, y_test_proba], ['train', 'val', 'test']):
            for batch_start in tqdm(range(0, X.shape[0], batch_size), desc=f'Inference: split={split} | model={model_name[:15]} | head={model_head[:15]}', total=X.shape[0] // batch_size):
                X_batch = X[batch_start:batch_start+batch_size]
                input_ids = torch.tensor(X_batch, device=device)
                attention_mask = (input_ids != pad_token_id).int()
                batch = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                }
                if "hyena" in model_name:
                    batch.pop('attention_mask')
                y.append(model.predict_proba(**batch)[:, 1].detach().cpu().numpy())
    y_train_proba = np.concatenate(y_train_proba) if len(y_train_proba) > 0 else np.array([])
    y_val_proba = np.concatenate(y_val_proba) if len(y_val_proba) > 0 else np.array([])
    y_test_proba = np.concatenate(y_test_proba) if len(y_test_proba) > 0 else np.array([])
    return y_train_proba, y_val_proba, y_test_proba

def run_finetune_evaluation(X_train: np.ndarray,
                            X_val: np.ndarray, 
                            X_test: np.ndarray, 
                            y_train: np.ndarray, 
                            y_val: np.ndarray, 
                            y_test: np.ndarray,
                            X_train_timelines: np.ndarray,
                            X_val_timelines: np.ndarray,
                            X_test_timelines: np.ndarray,
                            model_name: str, 
                            model_head: str, 
                            path_to_ckpt: str,
                            batch_size: int = 4,
                            n_epochs: int = 2,
                            lr: float = 1e-5,
                            logreg_C: Union[float, List[float]] = [1.0],
                            logreg_penalty: Optional[str] = 'l2',
                            warmup_epochs: int = 0,
                            replicate: int = 0,
                            n_jobs: int = 1,
                            test_patient_ids: List[int] = None) -> Tuple[Any, Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """Run fine-tuning evaluation for a given model and hyperparameter configuration.
    
    ! NOTE: Requires the `hf_ehr` package (installable via `pip install hf-ehr`)

    Args:
        X_train (np.ndarray): Training features
        X_val (np.ndarray): Validation features 
        X_test (np.ndarray): Test features
        y_train (np.ndarray): Training labels
        y_val (np.ndarray): Validation labels
        y_test (np.ndarray): Test labels
        X_train_timelines (np.ndarray): Training timeline features
        X_val_timelines (np.ndarray): Validation timeline features 
        X_test_timelines (np.ndarray): Test timeline features
        model_name (str): Name of the model architecture
        model_head (str): Type of model head/fine-tuning strategy
        path_to_ckpt (str): Path to model checkpoint
        batch_size (int, optional): Batch size for training. Defaults to 4.
        n_epochs (int, optional): Number of training epochs. Defaults to 2.
        lr (float, optional): Learning rate. Defaults to 1e-5.
        logreg_C (Union[float, List[float]], optional): Logistic regression regularization parameter(s). Defaults to [1.0].
        logreg_penalty (Optional[str], optional): Type of regularization penalty. Defaults to 'l2'.
        warmup_epochs (int, optional): Number of warmup epochs. Defaults to 0.
        replicate (int, optional): Random seed for reproducibility. Defaults to 0.
        n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
        test_patient_ids (List[int], optional): List of patient IDs in test set. Defaults to None.

    Returns:
        Tuple[Any, Dict[str, Dict[str, float]], Dict[str, np.ndarray]]: Tuple containing:
            - Best model state dictionary
            - Dictionary of evaluation metrics
            - Dictionary of model predictions
    """
    
    from hf_ehr.utils import load_tokenizer_from_path, load_model_from_path
    from hf_ehr.eval.ehrshot import CookbookModelWithClassificationHead

    logger.debug(f"Start | Training {model_head} | replicate={replicate}")
    logger.info(f"Train shape: X = {X_train.shape}, Y = {y_train.shape}")
    logger.info(f"Val shape: X = {X_val.shape}, Y = {y_val.shape}")
    logger.info(f"Test shape: X = {X_test.shape}, Y = {y_test.shape}")
    logger.info(f"Train prevalence:  {np.mean(y_train)}")
    logger.info(f"Val prevalence:  {np.mean(y_val)}")
    logger.info(f"Test prevalence:  {np.mean(y_test)}")
    logger.info(f"Test pids:  {len(test_patient_ids)} | {len(y_test)} | {len(set(test_patient_ids))}")

    # Shuffle training set
    np.random.seed(replicate)
    train_shuffle_idx = np.arange(X_train.shape[0])
    np.random.shuffle(train_shuffle_idx)
    X_train_timelines = X_train_timelines[train_shuffle_idx]
    X_train = X_train[train_shuffle_idx]
    y_train = y_train[train_shuffle_idx]

    # Load tokenizer
    device: str = 'cuda'
    tokenizer = load_tokenizer_from_path(path_to_ckpt)
    pad_token_id: int = tokenizer.pad_token_id
    finetune_strat = model_head.split("_")[1] # "layers=n" or "full"
    embed_strat: str = [ x for x in model_name.split("_") if x.split(":")[0] == 'embed' ][0].split(":")[1] # "mean" or "last"

    # Load original model -- weights won't change
    orig_model = load_model_from_path(path_to_ckpt)
    orig_model = CookbookModelWithClassificationHead(orig_model, embed_strat, 2)
    # Load model that will get finetuned for each logreg_C -- weights will change
    model = load_model_from_path(path_to_ckpt)
    model = CookbookModelWithClassificationHead(model, embed_strat, 2)
    model, layers = setup_finetuning(model, finetune_strat)
    model: torch.nn.Module = torch.compile(model)  # type: ignore
    # Sanity checks
    # sanity_check_finetuning_model(model, layers, finetune_strat)

    if isinstance(logreg_C, float):
        logreg_C = [ logreg_C ]

    best_model_state_dict, best_val_auroc, best_hparams = None, None, None
    for C in logreg_C:
        logger.info(f"Start | Finetuning logreg_C=`{C}` | model=`{model_name}` | head=`{model_head}`")
        # Reload original weights into model
        # ! NOTE: This needs to be done inside the loop, since we need to reset the model each time 
        #         (otherwise we'll be finetuning the same model multiple times)
        model._orig_mod.load_state_dict(orig_model.state_dict())
        ## Sanity check
        for key in orig_model.state_dict().keys():
            assert torch.equal(model._orig_mod.state_dict()[key].to('cpu'), orig_model.state_dict()[key].to('cpu')), f"Model weights not equal for key={key}"
        model.to(device)

        # Optimizer + Loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.functional.cross_entropy

        # Initialize the warmup scheduler (if applicable)
        steps_per_epoch = X_train_timelines.shape[0] // batch_size
        scheduler = LambdaLR(optimizer, lr_lambda=lambda x: lr_lambda(x, steps_per_epoch)) if warmup_epochs > 0 else None
        
        # Finetune `model`
        is_finetune_logreg_first: bool = "logregfirst" in finetune_strat
        model = finetune_pytorch_model(X_train_timelines, X_train, X_val, y_train, y_val, model, optimizer, scheduler, criterion, 
                                        C, logreg_penalty, is_finetune_logreg_first,
                                        pad_token_id, model_name, model_head, batch_size, n_epochs, replicate, device)
        logger.info(f"Finish | Finetuning logreg_C=`{C}` | model=`{model_name}` | head=`{model_head}`")
        
        # Eval model on val set
        logger.info(f"Start | Evaling only VAL for model=`{model_name}` | head=`{model_head}` | logreg_C=`{C}`")
        __, y_val_proba, __ = eval_pytorch_model(np.array([]), X_val_timelines, np.array([]), np.array([]), y_val, np.array([]), model, pad_token_id, model_name, model_head, batch_size, device)
        logger.info(f"Finish | Evaling only VAL for model=`{model_name}` | head=`{model_head}` | logreg_C=`{C}`")
        
        # Calculate AUROC, AUPRC, and Brier scores
        val_auroc = metrics.roc_auc_score(y_val, y_val_proba)

        # Save best model (based on val AUROC)
        if best_val_auroc is None or val_auroc > best_val_auroc:
            logger.debug(f"New best model found with val AUROC {val_auroc} > {best_val_auroc if best_val_auroc is not None else -1} | logreg_C={C}")
            best_model_state_dict = model._orig_mod.state_dict()
            best_val_auroc = val_auroc
            best_hparams = { 
                'logreg_C' : C, 
                'logreg_penalty' : logreg_penalty, 
                'finetune_strat' : finetune_strat, 
                'warmup_epochs' : warmup_epochs,
                'n_epochs' : n_epochs,
                'batch_size' : batch_size,
                'lr' : lr,
            }
    
    # Load best model
    model._orig_mod.load_state_dict(best_model_state_dict)
    model._orig_mod.get_params = lambda : best_hparams
 
    # Eval best model on train/val/test sets
    logger.info(f"Start | Evaling best model=`{model_name}` | head=`{model_head}` | logreg_C=`{C}`")
    y_train_proba, y_val_proba, y_test_proba = eval_pytorch_model(X_train_timelines, X_val_timelines, X_test_timelines, 
                                                                  y_train, y_val, y_test, 
                                                                  model, pad_token_id, 
                                                                  model_name, model_head, batch_size, device)
    logger.info(f"Finish | Evaling best model=`{model_name}` | head=`{model_head}` | logreg_C=`{C}`")
        
    # Calculate AUROC, AUPRC, and Brier scores
    best_scores = calc_metrics(y_train, y_train_proba, y_val, y_val_proba, y_test, y_test_proba, test_patient_ids)
    best_metrics: Dict[str, np.ndarray] = { 'train' : y_train_proba, 'val' : y_val_proba, 'test' : y_test_proba }

    # Return best model -- if torch.compile() is used, then we need to unwrap it to return the original model
    logger.debug(f"Finish | Training {model_head} | replicate={replicate}")
    model.to('cpu')
    return model._orig_mod, best_scores, best_metrics

def run_frozen_feature_evaluation(X_train: np.ndarray, 
                                    X_val: np.ndarray, 
                                    X_test: np.ndarray, 
                                    y_train: np.ndarray, 
                                    y_val: np.ndarray, 
                                    y_test: np.ndarray, 
                                    model_head: str, 
                                    replicate: int = 0,
                                    n_jobs: int = 1,
                                    test_patient_ids: List[int] = None) -> Tuple[Any, Dict[str, float]]:
    logger.debug(f"Start | Training {model_head} | replicate={replicate}")
    logger.info(f"Train shape: X = {X_train.shape}, Y = {y_train.shape}")
    logger.info(f"Val shape: X = {X_val.shape}, Y = {y_val.shape}")
    logger.info(f"Test shape: X = {X_test.shape}, Y = {y_test.shape}")
    logger.info(f"Train prevalence:  {np.mean(y_train)}")
    logger.info(f"Val prevalence:  {np.mean(y_val)}")
    logger.info(f"Test prevalence:  {np.mean(y_test)}")
    logger.info(f"Test pids:  {len(test_patient_ids)} | {len(y_test)} | {len(set(test_patient_ids))}")

    # Shuffle training set
    np.random.seed(replicate)
    train_shuffle_idx = np.arange(X_train.shape[0])
    np.random.shuffle(train_shuffle_idx)
    X_train = X_train[train_shuffle_idx]
    y_train = y_train[train_shuffle_idx]
    
    if model_head == "gbm":
        # XGBoost
        logger.info(f"XGBoost")
        model = lgb.LGBMClassifier(random_state=replicate)
        # NOTE: Need to set `min_child_samples = 1`, which specifies the minimum number of samples required in a leaf (terminal node).
        # This is necessary for few-shot learning, since we may have very few samples in a leaf node.
        # Otherwise the GBM model will refuse to learn anything
        XGB_PARAMS['min_child_samples'] = [ 1 ]
        model = tune_hyperparams(X_train, X_val, y_train, y_val, model, XGB_PARAMS, n_jobs=n_jobs)
        logger.info(f"Best hparams: {model.get_params()}")
    elif model_head == "rf":
        RF_PARAMS['min_samples_leaf'] = [ 1 ]
        RF_PARAMS['min_samples_split'] = [ 2 ]
        model = RandomForestClassifier(random_state=replicate)
        model = tune_hyperparams(X_train, X_val, y_train, y_val, model, RF_PARAMS, n_jobs=n_jobs)
        logger.info(f"Best hparams: {model.get_params()}")
    elif model_head.startswith("lr"):
        # Logistic Regresion
        solver: str = model_head.split("_")[-1] # "newton-cg" or "lbfgs" etc.
        # Use built-in SKLearn solver
        scaler = MaxAbsScaler().fit(X_train)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        model = LogisticRegression(n_jobs=1, penalty="l2", tol=0.0001, solver=solver, max_iter=1000, random_state=replicate)
        model = tune_hyperparams(X_train, X_val, y_train, y_val, model, LR_PARAMS, n_jobs=n_jobs)
        logger.info(f"Best hparams: {model.get_params()}")
    elif model_head == "protonet":
        # ProtoNet
        model = ProtoNetCLMBRClassifier()
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Model head `{model_head}` not supported.")
    
    # Calculate probabilistic preds
    y_train_proba: Float[np.ndarray, 'N'] = model.predict_proba(X_train)[:, 1]
    y_val_proba: Float[np.ndarray, 'N'] = model.predict_proba(X_val)[:, 1]
    y_test_proba: Float[np.ndarray, 'N'] = model.predict_proba(X_test)[:, 1]
    
    # Calculate AUROC, AUPRC, and Brier scores
    scores = calc_metrics(y_train, y_train_proba, y_val, y_val_proba, y_test, y_test_proba, test_patient_ids)

    logger.debug(f"Finish | Training {model_head} | replicate={replicate}")
    return model, scores, { 'train' : y_train_proba, 'val' : y_val_proba, 'test' : y_test_proba }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EHRSHOT evaluation benchmark on a specific task.")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to directory containing saved features")
    parser.add_argument("--path_to_output_dir", required=True, type=str, help="Path to directory where results will be saved")
    parser.add_argument("--path_to_split_csv", required=True, type=str, help="Path to CSV of splits")
    parser.add_argument("--path_to_tokenized_timelines_dir", required=False, type=str, default=None, help="Path to directory containing tokenized timelines (if applicable)")
    parser.add_argument("--shot_strat", type=str, choices=SHOT_STRATS.keys(), help="What type of X-shot evaluation we are interested in.", required=True )
    parser.add_argument("--labeling_function", required=True, type=str, help="Labeling function for which we will create k-shot samples.", choices=LABELING_FUNCTION_2_PAPER_NAME.keys(), )
    parser.add_argument("--is_force_refresh", action='store_true', default=False, help="If set, then overwrite all outputs")
    parser.add_argument("--models", required=True, help="Comma separated list. If specified, then only consider models in this list, e.g. `clmbr,count`")
    parser.add_argument("--heads", default=None, help="Comma separated list. If specified, then only consider heads in this list, e.g. `finetune_layers=1,finetune_layers=2`")
    parser.add_argument("--ks", type=str, default=None, help="Comma separated list. If specified, then only use these k-shot values to evaluate")
    # Frozen model, finetuning Sklearn head
    parser.add_argument("--num_threads", type=int, help="Number of threads to use")
    # Finetuning model with PyTorch head
    parser.add_argument("--logreg_C", type=str, default="1e-8,1e-2,1e-1", help="Command separated list of regularization strengths for logistic regression head")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for finetuning. NOTE: This must be a small value (e.g. 4) for torch.compile() to correctly work")
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of epochs for finetuning")
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Number of epochs for lr warmup")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    LABELING_FUNCTION: str = args.labeling_function
    VALID_HEADS: Optional[List[str]] = None if args.heads is None else args.heads.split(',')
    VALID_MODELS: List[str] = args.models.split(',')
    SHOT_STRAT: str = args.shot_strat
    KS: Optional[str] = args.ks
    NUM_THREADS: int = args.num_threads
    BATCH_SIZE: int = args.batch_size
    N_EPOCHS: int = args.n_epochs
    warmup_epochs: int = args.warmup_epochs
    logreg_C: List[float] = [ float(x) for x in args.logreg_C.split(',') ]
    logreg_penalty: Optional[str] = 'l2'
    PATH_TO_TOKENIZED_TIMELINES_DIR: Optional[str] = args.path_to_tokenized_timelines_dir
    IS_FORCE_REFRESH: bool = args.is_force_refresh
    PATH_TO_FEATURES_DIR: str = args.path_to_features_dir
    PATH_TO_LABELS_DIR: str = args.path_to_labels_dir
    PATH_TO_SPLIT_CSV: str = args.path_to_split_csv
    PATH_TO_LABELED_PATIENTS: str = os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION, 'labeled_patients.csv')
    PATH_TO_SHOTS: str = os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION, f"{SHOT_STRAT}_shots_data.json")
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_OUTPUT_DIR, LABELING_FUNCTION, f'{SHOT_STRAT}_results.csv')
    os.makedirs(os.path.dirname(PATH_TO_OUTPUT_FILE), exist_ok=True)
    assert BATCH_SIZE <= 4, f"You should only use a --batch_size <= 4 for optimal use of torch.compile()"

    # Determine which models to load
    logger.critical(f"Only running models: {VALID_MODELS}")
    logger.critical(f"Only running heads: {VALID_HEADS}")

    # Determine which ks to load
    VALID_KS = None
    if KS is not None and KS != "":
        VALID_KS = [ int(x) for x in KS.split(',') ]
        logger.critical(f"Only running ks: {VALID_KS}")

    # Load all labeled patients
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)
    logger.info(f"Loading task {LABELING_FUNCTION} with {len(labeled_patients)} labeled patients.")
    
    # Load shot assignments for this task
    with open(PATH_TO_SHOTS) as f:
        few_shots_dict: Dict[str, Dict] = json.load(f)
    
    # For each base model we are evaluating...
    for model in VALID_MODELS:
        model_base_name: str = model.split("-")[0] # count -> count; gpt-base-1024 -> gpt
        model_heads: List[str] = MODEL_2_INFO[model_base_name]['heads']

        # For each head we can add to the top of this model...
        for head in model_heads:
            if VALID_HEADS is not None and head not in VALID_HEADS:
                # Skip heads (if specified)
                continue
            
            if 'finetune' in head:
                assert PATH_TO_TOKENIZED_TIMELINES_DIR is not None, "Must specify `--path_to_tokenized_timelines_dir` for finetuning experiments"

            # Load labels/features for this task + model_head
            patient_ids, label_values, label_times, feature_matrixes = get_labels_and_features(labeled_patients, 
                                                                                               PATH_TO_FEATURES_DIR, 
                                                                                               PATH_TO_TOKENIZED_TIMELINES_DIR,
                                                                                               models_to_keep=[ model ])
            train_pids_idx, val_pids_idx, test_pids_idx = get_patient_splits_by_idx(PATH_TO_SPLIT_CSV, patient_ids)

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
                
            assert model in feature_matrixes, f"Feature matrix not found for `{model}`. Are you sure you have generated features for this model? If not, you'll need to rerun `generate_features.py` or `generate_clmbr_representations.py`."
            
            # Unpack each individual featurization we want to test
            if head.startswith('finetune'):
                # In this case, X_train, X_val, X_test are raw sequences of tokenized timelines
                assert torch.cuda.is_available(), "CUDA must be available to run finetuning experiments."
                X_train_timelines: np.ndarray = feature_matrixes[model]['timelines'][train_pids_idx]
                X_val_timelines: np.ndarray = feature_matrixes[model]['timelines'][val_pids_idx]
                X_test_timelines: np.ndarray = feature_matrixes[model]['timelines'][test_pids_idx]
                X_train: np.ndarray = feature_matrixes[model]['frozen'][train_pids_idx]
                X_val: np.ndarray = feature_matrixes[model]['frozen'][val_pids_idx]
                X_test: np.ndarray = feature_matrixes[model]['frozen'][test_pids_idx]
            else:
                # In this case, X_train, X_val, X_test are frozen feature representations from a previous model
                X_train_timelines = None
                X_val_timelines = None
                X_test_timelines = None
                X_train: np.ndarray = feature_matrixes[model]['frozen'][train_pids_idx]
                X_val: np.ndarray = feature_matrixes[model]['frozen'][val_pids_idx]
                X_test: np.ndarray = feature_matrixes[model]['frozen'][test_pids_idx]
            
            # Test labels
            y_test: np.ndarray = label_values[test_pids_idx]
            test_patient_ids = patient_ids[test_pids_idx]
            
            # For each subtask in this task... 
            # NOTE: The "subtask" is just the same thing as LABELING_FUNCTION for all binary tasks.
            # But for Chexpert, there are multiple subtasks, which of each represents a binary subtask
            for sub_task_idx, sub_task in enumerate(sub_tasks):
                ks: List[int] = sorted([ int(x) for x in few_shots_dict[sub_task].keys() ])

                # Filter out k-shot values (if specified)
                if VALID_KS is not None:
                    ks = [ k for k in ks if k in VALID_KS ]
                
                # For each k-shot sample we are evaluating...
                for k in ks:
                    replicates: List[int] = sorted([ int(x) for x in few_shots_dict[sub_task][str(k)].keys() ])

                    # Save best model according to val AUROC
                    best_model_val_auroc: float = -1

                    # If k = -1, then we use all data points. So each replicate is the exact same thing, just repeat 5 times so we can get a std
                    if k == -1:
                        replicates = [0, 1, 2, 3, 4, ]
                        for replicate in replicates:
                            few_shots_dict[sub_task][str(k)][str(replicate)] = few_shots_dict[sub_task][str(k)][str(0)]
                        assert len(replicates) == 5

                    # For each replicate of this k-shot sample...
                    for replicate in replicates:
                
                        ############################
                        # ! Do loading of previous results here so that we reload any results that
                        # ! were generated in a concurrently running SLURM job
                        # If results already exist, then append new results to existing file
                        df_existing: Optional[pd.DataFrame] = None
                        if os.path.exists(PATH_TO_OUTPUT_FILE):
                            logger.warning(f"Results already exist @ `{PATH_TO_OUTPUT_FILE}`.")
                            df_existing = pd.read_csv(PATH_TO_OUTPUT_FILE)
                        ############################
                        
                        # Check if results already exist for this model/head/shot_strat in `results.csv`
                        if df_existing is not None:
                            existing_rows: pd.DataFrame = df_existing[
                                (df_existing['labeling_function'] == LABELING_FUNCTION) 
                                & (df_existing['sub_task'] == sub_task) 
                                & (df_existing['model'] == model) 
                                & (df_existing['head'] == head)
                                & (df_existing['k'] == k)
                                & (df_existing['replicate'] == replicate)
                            ]
                            if existing_rows.shape[0] > 0:
                                # Overwrite
                                if IS_FORCE_REFRESH:
                                    logger.warning(f"Results ALREADY exist for {model}/{head}/{LABELING_FUNCTION}/{sub_task}/k={k}/r={replicate} in `results.csv`.\nOVERWRITING these rows because `is_force_refresh` is TRUE.")
                                else:
                                    logger.warning(f"Results ALREADY exist for {model}/{head}/{LABELING_FUNCTION}/{sub_task}/k={k}/r={replicate} in `results.csv`.\nSKIPPING this combination because `is_force_refresh` is FALSE.")
                                    continue
                            else:
                                # Append
                                logger.warning(f"Results DO NOT exist for {model}/{head}/{LABELING_FUNCTION}/{sub_task}/k={k}/r={replicate} in `results.csv`. Appending to this CSV.")
                
                
                        logger.success(f"Model: {model} | Head: {head} | Task: {sub_task} | k: {k} | replicate: {replicate}")
                        shot_dict: Dict[str, List[int]] = few_shots_dict[sub_task][str(k)][str(replicate)]               

                        # Get X/Y train/val frozen features for this k-shot sample
                        X_train_k: np.ndarray = X_train[shot_dict["train_idxs"]]
                        X_val_k: np.ndarray = X_val[shot_dict["val_idxs"]]
                        y_train_k: np.ndarray = np.array(shot_dict['label_values_train_k'])
                        y_val_k: np.ndarray = np.array(shot_dict['label_values_val_k'])
                        if X_train_timelines is not None:
                            X_train_timelines_k: np.ndarray = X_train_timelines[shot_dict["train_idxs"]]
                            X_val_timelines_k: np.ndarray = X_val_timelines[shot_dict["val_idxs"]]
                        
                        # Test labels
                        y_test_k: np.ndarray = np.array(y_test)

                        # CheXpert adjustment
                        if LABELING_FUNCTION == 'chexpert':
                            y_test_k = y_test[:, sub_task_idx]

                        # Get path to saved model
                        path_to_model_dir: str = os.path.abspath(os.path.join(PATH_TO_FEATURES_DIR, '../models', model))
                        if model == 'count': 
                            os.makedirs(path_to_model_dir, exist_ok=True)
                        
                        # Fit model with hyperparameter tuning
                        if head.startswith('finetune'):
                            assert os.path.exists(path_to_model_dir), f"Path to .ckpt directory for model={model},head={head} does not exist: `{path_to_model_dir}`"
                            path_to_ckpt: str = os.path.join(path_to_model_dir, [ x for x in os.listdir(path_to_model_dir) if x.endswith('.ckpt') ][0])
                            logger.info(f"Loaded model `{model}` from .ckpt at: `{path_to_ckpt}`")
                            best_model, scores, preds_proba = run_finetune_evaluation(X_train_k, X_val_k, X_test, y_train_k, y_val_k, y_test_k, 
                                                                                       X_train_timelines_k, X_val_timelines_k, X_test_timelines,
                                                                                       model_name=model, model_head=head, path_to_ckpt=path_to_ckpt, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS, 
                                                                                       logreg_C=logreg_C, logreg_penalty=logreg_penalty, 
                                                                                       warmup_epochs=warmup_epochs,
                                                                                       replicate=replicate,
                                                                                       test_patient_ids=test_patient_ids)
                        else:
                            best_model, scores, preds_proba = run_frozen_feature_evaluation(X_train_k, X_val_k, X_test, 
                                                                                            y_train_k, y_val_k, y_test_k, 
                                                                                            model_head=head, 
                                                                                            replicate=replicate,
                                                                                            n_jobs=NUM_THREADS, 
                                                                                            test_patient_ids=test_patient_ids)

                        # Save best model (according to val AUROC)
                        if scores['auroc']['score_val'] > best_model_val_auroc:
                            # Create folder
                            path_to_best_model_dir: str = os.path.join(PATH_TO_OUTPUT_DIR, LABELING_FUNCTION, 'models', model, head, f"subtask={sub_task}", f"k={k}")
                            os.makedirs(path_to_best_model_dir, exist_ok=True)
                            logger.warning(f"Achieved best val AUROC: {scores['auroc']['score_val']} > {best_model_val_auroc} with most recent model. Saving ckpt + preds to `{path_to_best_model_dir}`")
                            best_model_val_auroc = scores['auroc']['score_val']
                            model_hparams = best_model.get_params() if hasattr(best_model, 'get_params') else None
                            if head.startswith('finetune'):
                                # Save Pytorch model
                                if hasattr(best_model, 'get_params'):
                                    del best_model.get_params
                                torch.save(best_model, os.path.join(path_to_best_model_dir,  'ckpt.pt'))
                            else:
                                # Save SKLearn model
                                pickle.dump(best_model, open(os.path.join(path_to_best_model_dir,  'ckpt.pkl'), 'wb'))
                            # Save preds
                            df_preds = pd.DataFrame({ 
                                'labeling_function' : LABELING_FUNCTION,
                                'sub_task' : sub_task,
                                'model' : model,
                                'head' : head,
                                'replicate' : replicate,
                                'k' : k,
                                'y' : np.hstack([ y_train_k, y_val_k, y_test_k ]), 
                                'pred_proba' : np.hstack([ preds_proba['train'], preds_proba['val'], preds_proba['test'] ]), 
                                'split' : [ 'train' ] * preds_proba['train'].shape[0] + [ 'val' ] * preds_proba['val'].shape[0] + [ 'test' ] * preds_proba['test'].shape[0]
                            })
                            df_preds.to_csv(os.path.join(path_to_best_model_dir, "preds.csv"), index=False)
                            if model_hparams is not None:
                                json.dump({
                                    'labeling_function' : LABELING_FUNCTION,
                                    'sub_task' : sub_task,
                                    'model' : model,
                                    'head' : head,
                                    'replicate' : replicate,
                                    'k' : k,
                                    'scores' : scores,
                                    'model_hparams' : model_hparams
                                }, open(os.path.join(path_to_best_model_dir, 'model_hparams.json'), 'w'), indent=2)

                        # Save results
                        results: List[Dict[str, Any]] = []
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
                                'model_hparams' : best_model.get_params() if hasattr(best_model, 'get_params') else None,
                            })

                        ############################
                        # Reload to ensure we have most updated version of file
                        if os.path.exists(PATH_TO_OUTPUT_FILE):
                            df_existing = pd.read_csv(PATH_TO_OUTPUT_FILE)
                            results += df_existing.to_dict(orient='records')
                        ############################

                        # Save results to CSV after each (model, head, sub_task, k, replicate) is calculated
                        logger.critical(f"Saving results for {model} + {head} + k={k} + replicate={replicate} to: {PATH_TO_OUTPUT_FILE}")
                        df: pd.DataFrame = pd.DataFrame(results)
                        logger.critical(f"Added {df.shape[0] - (df_existing.shape[0] if df_existing is not None else 0)} rows")
                        df.to_csv(PATH_TO_OUTPUT_FILE, index=False)
    logger.success("Done!")
