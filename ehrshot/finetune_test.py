"""
    Usage:

    python3 finetune_test.py --task guo_los --model_head finetune_full --k -1 --device cuda:0
    python3 finetune_test.py --task guo_icu --model_head finetune_full --k -1 --device cuda:0
    python3 finetune_test.py --task guo_readmission --model_head finetune_full --k -1 --device cuda:0
    python3 finetune_test.py --task new_pancan --model_head finetune_full --k -1 --device cuda:0
    python3 finetune_test.py --task new_hypertension --model_head finetune_full --k -1 --device cuda:0
    python3 finetune_test.py --task lab_hypoglycemia --model_head finetune_full --k -1 --device cuda:0
    python3 finetune_test.py --task lab_anemia --model_head finetune_full --k -1 --device cuda:0
    python3 finetune_test.py --task new_acutemi --model_head finetune_full --k -1 --device cuda:0
    python3 finetune_test.py --task new_hyperlipidemia --model_head finetune_full --k -1 --device cuda:0
    python3 finetune_test.py --task lab_thrombocytopenia --model_head finetune_full --k -1 --device cuda:0
    python3 finetune_test.py --task lab_hyperkalemia --model_head finetune_full --k -1 --device cuda:0
    python3 finetune_test.py --task lab_hyponatremia --model_head finetune_full --k -1 --device cuda:0

    python3 finetune_test.py --task guo_los --model_head finetune_frozen --k -1 --device cuda:1
    python3 finetune_test.py --task guo_icu --model_head finetune_frozen --k -1 --device cuda:1
    python3 finetune_test.py --task guo_readmission --model_head finetune_frozen --k -1 --device cuda:1
    python3 finetune_test.py --task new_pancan --model_head finetune_frozen --k -1 --device cuda:1
    python3 finetune_test.py --task new_hypertension --model_head finetune_frozen --k -1 --device cuda:1
    python3 finetune_test.py --task lab_hypoglycemia --model_head finetune_frozen --k -1 --device cuda:1
    python3 finetune_test.py --task lab_anemia --model_head finetune_frozen --k -1 --device cuda:1
    python3 finetune_test.py --task new_acutemi --model_head finetune_frozen --k -1 --device cuda:1
    python3 finetune_test.py --task new_hyperlipidemia --model_head finetune_frozen --k -1 --device cuda:1
    python3 finetune_test.py --task lab_thrombocytopenia --model_head finetune_frozen --k -1 --device cuda:1
    python3 finetune_test.py --task lab_hyperkalemia --model_head finetune_frozen --k -1 --device cuda:1
    python3 finetune_test.py --task lab_hyponatremia --model_head finetune_frozen --k -1 --device cuda:1
    
    python3 finetune_test.py --task guo_los --model_head finetune_layers=1 --k -1 --device cuda:2
    python3 finetune_test.py --task guo_icu --model_head finetune_layers=1 --k -1 --device cuda:2
    python3 finetune_test.py --task guo_readmission --model_head finetune_layers=1 --k -1 --device cuda:2
    python3 finetune_test.py --task new_pancan --model_head finetune_layers=1 --k -1 --device cuda:2
    python3 finetune_test.py --task new_hypertension --model_head finetune_layers=1 --k -1 --device cuda:2
    python3 finetune_test.py --task lab_hypoglycemia --model_head finetune_layers=1 --k -1 --device cuda:2
    python3 finetune_test.py --task lab_anemia --model_head finetune_layers=1 --k -1 --device cuda:2
    python3 finetune_test.py --task new_acutemi --model_head finetune_layers=1 --k -1 --device cuda:2
    python3 finetune_test.py --task new_hyperlipidemia --model_head finetune_layers=1 --k -1 --device cuda:2
    python3 finetune_test.py --task lab_thrombocytopenia --model_head finetune_layers=1 --k -1 --device cuda:2
    python3 finetune_test.py --task lab_hyperkalemia --model_head finetune_layers=1 --k -1 --device cuda:2
    python3 finetune_test.py --task lab_hyponatremia --model_head finetune_layers=1 --k -1 --device cuda:2
    
    python3 finetune_test.py --task guo_los --model_head finetune_layers=2 --k -1 --device cuda:3
    python3 finetune_test.py --task guo_icu --model_head finetune_layers=2 --k -1 --device cuda:3
    python3 finetune_test.py --task guo_readmission --model_head finetune_layers=2 --k -1 --device cuda:3
    python3 finetune_test.py --task new_pancan --model_head finetune_layers=2 --k -1 --device cuda:3
    python3 finetune_test.py --task new_hypertension --model_head finetune_layers=2 --k -1 --device cuda:3
    python3 finetune_test.py --task lab_hypoglycemia --model_head finetune_layers=2 --k -1 --device cuda:3
    python3 finetune_test.py --task lab_anemia --model_head finetune_layers=2 --k -1 --device cuda:3
    python3 finetune_test.py --task new_acutemi --model_head finetune_layers=2 --k -1 --device cuda:3
    python3 finetune_test.py --task new_hyperlipidemia --model_head finetune_layers=2 --k -1 --device cuda:3
    python3 finetune_test.py --task lab_thrombocytopenia --model_head finetune_layers=2 --k -1 --device cuda:3
    python3 finetune_test.py --task lab_hyperkalemia --model_head finetune_layers=2 --k -1 --device cuda:3
    python3 finetune_test.py --task lab_hyponatremia --model_head finetune_layers=2 --k -1 --device cuda:3
"""
import argparse
import torch
import numpy as np
import os
import json
import sklearn
import sklearn.metrics
from jaxtyping import Float
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from femr.labelers import load_labeled_patients, LabeledPatients
from hf_ehr.utils import load_tokenizer_from_path, load_model_from_path
from hf_ehr.eval.ehrshot import CookbookModelWithClassificationHead
from utils import convert_multiclass_to_binary_labels, get_labels_and_features, get_patient_splits_by_idx, process_chexpert_labels
import sys

def eprint(*args, **kwargs):
    """Print to stderr"""
    print(*args, file=sys.stderr, **kwargs)

def setup_finetuning(model: CookbookModelWithClassificationHead, finetune_strat: str):
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
    elif finetune_strat == "full":
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True
    elif finetune_strat == 'frozen':
        pass
    else:
        raise ValueError(f"Fine-tuning strategy `{finetune_strat}` not supported.")
    return model, layers

def load_model(model_name: str, model_head: str, path_to_ckpt: str, device: str):
    # Load model
    finetune_strat = model_head.split("_")[1] # "layers=n" or "full"
    embed_strat: str = [ x for x in model_name.split("_") if x.split(":")[0] == 'embed' ][0].split(":")[1] # "mean" or "last"
    tokenizer = load_tokenizer_from_path(path_to_ckpt)
    model = load_model_from_path(path_to_ckpt)
    model = CookbookModelWithClassificationHead(model, embed_strat, 2)
    model, layers = setup_finetuning(model, finetune_strat)
    model.to(device)

    return model, tokenizer.pad_token_id

def load_data(model_name: str, sub_task, PATH_TO_SHOTS: str, PATH_TO_LABELED_PATIENTS: str, PATH_TO_FEATURES_DIR: str, PATH_TO_SPLIT_CSV: str):
    # Load all labeled patients
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)

    # Load shot assignments for this task
    with open(PATH_TO_SHOTS) as f:
        few_shots_dict: Dict[str, Dict] = json.load(f)

    # Load labels/features for this task + model_head
    patient_ids, label_values, label_times, feature_matrixes = get_labels_and_features(labeled_patients, PATH_TO_FEATURES_DIR, models_to_keep=[ model_name ])
    train_pids_idx, val_pids_idx, test_pids_idx = get_patient_splits_by_idx(PATH_TO_SPLIT_CSV, patient_ids)

    # Preprocess certain non-binary labels
    if sub_task == "chexpert":
        label_values = process_chexpert_labels(label_values)
    elif sub_task.startswith('lab_'):
        # Lab value is multi-class, convert to binary
        label_values = convert_multiclass_to_binary_labels(label_values, threshold=1)
    else:
        # Binary classification
        pass
    
    assert model_name in feature_matrixes, f"Feature matrix not found for `{model_name}`. Are you sure you have generated features for this model? If not, you'll need to rerun `generate_features.py` or `generate_clmbr_representations.py`."

    # In this case, X_train, X_val, X_test are raw sequences of tokenized timelines
    X_train: np.ndarray = feature_matrixes[model_name]['timelines'][train_pids_idx]
    X_val: np.ndarray = feature_matrixes[model_name]['timelines'][val_pids_idx]
    X_test: np.ndarray = feature_matrixes[model_name]['timelines'][test_pids_idx]

    # Test labels
    y_test: np.ndarray = label_values[test_pids_idx]
    test_patient_ids = patient_ids[test_pids_idx]

    return X_train, X_val, X_test, y_test, few_shots_dict

def train_model(model, optimizer, criterion, pad_token_id, X_train, X_val, X_test, y_train, y_val, y_test, batch_size: int, n_epochs: int, device: str):
    torch.manual_seed(X_train.shape[0])

    # Run initial eval
    metrics = []
    metric = eval_model(model, pad_token_id, X_train, X_val, X_test, y_train, y_val, y_test, batch_size, device)
    metrics.append(metric)

    model.train()
    train_losses = []
    for epoch in range(n_epochs):

        # Run train for 1 epoch
        for batch_start in tqdm(range(0, X_train.shape[0], batch_size), desc=f'Finetuning', total=X_train.shape[0] // batch_size):
            # Subset training batch
            X_train_batch = X_train[batch_start:batch_start+batch_size]
            y_train_batch = y_train[batch_start:batch_start+batch_size]
            
            # Tokenize batch
            input_ids = torch.tensor(X_train_batch, device=device)
            attention_mask = (input_ids != pad_token_id).int()
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }

            # Run model to get logits for each class
            logits: Float[torch.Tensor, 'B C'] = model(**batch)
            assert logits.shape == (X_train_batch.shape[0], 2)

            # Compute CE loss v. true binary labels
            binary_labels: Float[torch.Tensor, 'B'] = torch.tensor(y_train_batch, device=device).long()
            loss = criterion(logits, binary_labels)
            train_losses.append(loss.detach().cpu().item())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Run eval post-epoch
        metric = eval_model(model, pad_token_id, X_train, X_val, X_test, y_train, y_val, y_test, batch_size, device)
        metrics.append(metric)
    return model, train_losses, metrics

def eval_model(model, pad_token_id, X_train, X_val, X_test, y_train, y_val, y_test, batch_size: int, device: str):
    model.eval()
    with torch.no_grad():
        y_train_proba, y_val_proba, y_test_proba = [], [], []
        for X, y, split in zip([X_train, X_val, X_test], [y_train_proba, y_val_proba, y_test_proba], ['train', 'val', 'test']):
            for batch_start in tqdm(range(0, X.shape[0], batch_size), desc=f'Inference: split={split}', total=X.shape[0] // batch_size):
                X_batch = X[batch_start:batch_start+batch_size]
                input_ids = torch.tensor(X_batch, device=device)
                attention_mask = (input_ids != pad_token_id).int()
                batch = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                }
                y.append(model.predict_proba(**batch)[::, 1].detach().cpu().numpy())
        y_train_proba = np.concatenate(y_train_proba)
        y_val_proba = np.concatenate(y_val_proba)
        y_test_proba = np.concatenate(y_test_proba)
    model.train()
    eprint("______ TEST AUROC:", sklearn.metrics.roc_auc_score(y_test, y_test_proba))
    return {
        'auroc' : {
            'train' : sklearn.metrics.roc_auc_score(y_train, y_train_proba),
            'val' : sklearn.metrics.roc_auc_score(y_val, y_val_proba),
            'test' : sklearn.metrics.roc_auc_score(y_test, y_test_proba),
        },
        'precision' : {
            'train' : sklearn.metrics.precision_score(y_train, y_train_proba >= 0.5),
            'val' : sklearn.metrics.precision_score(y_val, y_val_proba >= 0.5),
            'test' : sklearn.metrics.precision_score(y_test, y_test_proba >= 0.5),
        },
        'recall' : {
            'train' : sklearn.metrics.recall_score(y_train, y_train_proba >= 0.5),
            'val' : sklearn.metrics.recall_score(y_val, y_val_proba >= 0.5),
            'test' : sklearn.metrics.recall_score(y_test, y_test_proba >= 0.5),
        },
        'accuracy' : {
            'train' : (y_train  == (y_train_proba >= 0.5)).mean(),
            'val' : (y_val  == (y_val_proba >= 0.5)).mean(),
            'test' : (y_test  == (y_test_proba >= 0.5)).mean(),
        },
    }

def run_hparams(model, pad_token_id, few_shots_dict, X_train, X_val, X_test, y_test, sub_task: str, lr: float, k: int, n_epochs: int, batch_size: int, device: str):
    # Optimizer + Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.functional.cross_entropy

    # For each k-shot sample we are evaluating...
    replicates: List[int] = sorted([ int(x) for x in few_shots_dict[sub_task][str(k)].keys() ])
    if k == -1: replicates = [ replicates[0] ] * 5

    # Default to first replicate
    replicate = replicates[0]
    shot_dict: Dict[str, List[int]] = few_shots_dict[sub_task][str(k)][str(replicate)]               

    # Get X/Y train/val frozen features for this k-shot sample
    X_train_k: np.ndarray = X_train[shot_dict["train_idxs"]]
    X_val_k: np.ndarray = X_val[shot_dict["val_idxs"]]
    y_train_k: np.ndarray = np.array(shot_dict['label_values_train_k'])
    y_val_k: np.ndarray = np.array(shot_dict['label_values_val_k'])
    y_test_k: np.ndarray = np.array(y_test)

    model, train_losses, metrics = train_model(model, optimizer, criterion, pad_token_id, X_train_k, X_val_k, X_test, y_train_k, y_val_k, y_test_k, batch_size, n_epochs, device)
    return model, train_losses, metrics
    

def main(model: str, model_head: str, sub_task: str, k: int, device: str):
    if model == 'gpt-base-1024':
        model_name = 'gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last'
        path_to_ckpt = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/models/gpt2-base-1024--clmbr_train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist_chunk:last_embed:last/train-tokens-total_nonPAD-true_val=600000064-ckpt_val=600000000-persist.ckpt'
        batch_size: int = 4
    elif model == 'gpt-base-512':
        model_name = 'gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last'
        path_to_ckpt = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/models/gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last/train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist.ckpt'
        batch_size: int = 16
    else:
        model_name = 'gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last'
        path_to_ckpt = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/models/gpt2-base-512--clmbr_train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist_chunk:last_embed:last/train-tokens-total_nonPAD-true_val=600000896-ckpt_val=600000000-persist.ckpt'
        batch_size: int = 16

    n_epochs: int = 10 if k == -1 else 20

    # Paths
    PATH_TO_LABELS_DIR: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark'
    PATH_TO_SPLIT_CSV: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/splits/person_id_map.csv'
    PATH_TO_LABELED_PATIENTS: str = os.path.join(PATH_TO_LABELS_DIR, sub_task, 'labeled_patients.csv')
    PATH_TO_SHOTS: str = os.path.join(PATH_TO_LABELS_DIR, sub_task, f"all_shots_data.json")
    PATH_TO_FEATURES_DIR: str = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/features'
    path_to_output_dir: str = f'/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/finetunes/{model_name}/{model_head}/{sub_task}/k={k}'
    os.makedirs(path_to_output_dir, exist_ok=True)
    eprint("====> Path to output dir:", path_to_output_dir)

    # Load data
    X_train, X_val, X_test, y_test, few_shots_dict = load_data(model_name, sub_task, PATH_TO_SHOTS, PATH_TO_LABELED_PATIENTS, PATH_TO_FEATURES_DIR, PATH_TO_SPLIT_CSV)

    for lr in [1e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 1e-3, 1e-2, 1]:
        eprint("====> Running lr=", lr, "<====")
        path_to_output_json: str = os.path.join(path_to_output_dir, f'lr={str(lr)}.json')
        if os.path.exists(path_to_output_json):
            eprint(f"Skipping 'lr={str(lr)}.json' as it already exists.")
            continue
        # Load fresh model for finetuning
        model, pad_token_id = load_model(model_name, model_head, path_to_ckpt, device)
        model, losses, metrics = run_hparams(model, pad_token_id, few_shots_dict, X_train, X_val, X_test, y_test, sub_task, lr, k, n_epochs, batch_size, device)
        eprint("====> Saving to:", path_to_output_json)
        with open(path_to_output_json, 'w') as fd:
            json.dump({
                'losses': losses,
                'metrics': metrics,
                'n_epochs': n_epochs,
                'sub_task' : sub_task,
                'model' : model_name,
                'head' : model_head,
                'k' : k,
                'lr' : lr,
            }, fd, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True) # 'guo_los'
    parser.add_argument('--model', type=str, default='gpt-base-512')
    parser.add_argument('--model_head', type=str, required=True) # 'finetune_full'
    parser.add_argument('--device', type=str, required=True) # 'cuda:0'
    parser.add_argument('--k', type=int, default=-1)
    args = parser.parse_args()

    if args.task == 'all':
        eprint("Doing `all` tasks")
        for task in [
            'guo_los',
            'guo_icu',
            'guo_readmission',
            'new_pancan',
            'new_hypertension',
            'lab_hypoglycemia',
            'lab_anemia',
            'new_acutemi',
            'new_hyperlipidemia',
            'lab_thrombocytopenia',
            'lab_hyperkalemia',
            'lab_hyponatremia',
        ]:
            main(args.model, args.model_head, task, args.k, args.device)
    else:
        main(args.model, args.model_head, args.task, args.k, args.device)