import numpy as np
import os
import pandas as pd
import femr.datasets
from hf_ehr.config import PATH_TO_FEMR_EXTRACT_v8, PATH_TO_FEMR_EXTRACT_MIMIC4, SPLIT_SEED, SPLIT_TRAIN_CUTOFF, SPLIT_VAL_CUTOFF

def make_split(path_to_femr_extract: str, name: str):
    femr_db = femr.datasets.PatientDatabase(path_to_femr_extract)
    all_pids: np.ndarray = np.array([ pid for pid in femr_db ])
    hashed_pids: np.ndarray = np.array([ femr_db.compute_split(SPLIT_SEED, pid) for pid in all_pids ])
    train_pids: np.ndarray = all_pids[np.where(hashed_pids < SPLIT_TRAIN_CUTOFF)[0]]
    val_pids: np.ndarray = all_pids[np.where((SPLIT_TRAIN_CUTOFF <= hashed_pids) & (hashed_pids < SPLIT_VAL_CUTOFF))[0]]
    test_pids: np.ndarray = all_pids[np.where(hashed_pids >= SPLIT_VAL_CUTOFF)[0]]

    # Confirm disjoint train/val/test
    assert np.intersect1d(train_pids, val_pids).shape[0] == 0
    assert np.intersect1d(train_pids, test_pids).shape[0] == 0
    assert np.intersect1d(val_pids, test_pids).shape[0] == 0
    assert len(train_pids) + len(val_pids) + len(test_pids) == len(all_pids)

    df = pd.DataFrame({
        'split': ['test'] * len(test_pids) + ['val'] * len(val_pids) + ['train'] * len(train_pids),
        'omop_person_id': np.concatenate([test_pids, val_pids, train_pids])
    })
    path_to_output_dir = f'../../EHRSHOT_ASSETS/splits_{name}'
    os.makedirs(path_to_output_dir, exist_ok=True)
    df.to_csv(os.path.join(path_to_output_dir, 'person_id_map.csv'), index=False)


    print(f"==== {name} Split Counts ====")
    print(f"Train: {len(train_pids)} ({len(train_pids) / len(all_pids) * 100:.2f}%)")
    print(f"Val: {len(val_pids)} ({len(val_pids) / len(all_pids) * 100:.2f}%)")
    print(f"Test: {len(test_pids)} ({len(test_pids) / len(all_pids) * 100:.2f}%)")
    print("=============================")

# STARR
make_split(PATH_TO_FEMR_EXTRACT_v8, 'starr')
# MIMIC-IV
make_split(PATH_TO_FEMR_EXTRACT_MIMIC4, 'mimic4')
