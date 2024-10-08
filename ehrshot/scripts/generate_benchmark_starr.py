import os
import pandas as pd
from tqdm import tqdm

# Convert `benchmark_starr_all_labels_per_patient` to `benchmark_starr` with one label per patient

path_to_dir: str = '../../EHRSHOT_ASSETS/benchmark_starr_all_labels_per_patient'

for folder in os.listdir(path_to_dir):
    if not os.path.isdir(os.path.join(path_to_dir, folder)):
        continue
    df = pd.read_csv(os.path.join(path_to_dir, folder, 'labeled_patients.csv'))
    print("Loaded", folder)
    # Sample one row per patient_id
    sampled_df = df.sample(frac=1, random_state=42).reset_index(drop=True).drop_duplicates(subset='patient_id').reset_index(drop=True)
    sampled_df.to_csv(os.path.join(path_to_dir, '../', 'benchmark_starr', folder, 'labeled_patients.csv'))
    print("Shape: ", df.shape, "=>", sampled_df.shape)
    print("# unique patients: ", df['patient_id'].nunique(), "=>", sampled_df['patient_id'].nunique())
    assert df['patient_id'].nunique() == sampled_df['patient_id'].nunique()
    assert sampled_df['patient_id'].nunique() == sampled_df.shape[0]