"""
Convert EHRSHOT to MEDS format

Note: Takes ~10 mins
"""
import datetime
from tqdm import tqdm
import json
import jsonschema
import femr.datasets
from typing import List, Dict, Optional, Any, Set
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from meds import (
    code_metadata_schema,
    data_schema,
    dataset_metadata_schema,
    subject_split_schema,
    label_schema,
    train_split,
    tuning_split,
    held_out_split,
)

path_to_output_dir: str = '../../EHRSHOT_ASSETS/meds'
os.makedirs(path_to_output_dir, exist_ok=True)
os.makedirs(os.path.join(path_to_output_dir, 'metadata'), exist_ok=True)
os.makedirs(os.path.join(path_to_output_dir, 'data'), exist_ok=True)
print(f"Saving output to {path_to_output_dir}")

# Dataset
metadata: Dict[str, str] = {
    "dataset_name": "EHRSHOT",
    "dataset_version": "1.0",
    "etl_name": "EHRSHOT => MEDS ETL",
    "etl_version": "1.0",
    "meds_version" : "0.3.3",
    "created_at" : str(datetime.datetime.now()),
}
jsonschema.validate(instance=metadata, schema=dataset_metadata_schema)
print("Writing dataset.json")
json.dump(metadata, open(os.path.join(path_to_output_dir, 'metadata/dataset.json'), 'w'))

# Splits
print("Converting splits...")
df_splits = pd.read_csv('../../EHRSHOT_ASSETS/splits_ehrshot/person_id_map.csv')
df_splits = df_splits.rename(columns={ 'omop_person_id' : 'subject_id', })
df_splits['split'] = df_splits['split'].apply(lambda x: train_split if x == 'train' else (tuning_split if x == 'val' else held_out_split))
pa_splits = pa.Table.from_pandas(df_splits, schema=subject_split_schema)
print("Writing metadata/subject_splits.parquet")
pq.write_table(pa_splits, os.path.join(path_to_output_dir, 'metadata/subject_splits.parquet'))
## Sanity check
df_ = pd.read_parquet(os.path.join(path_to_output_dir, 'metadata/subject_splits.parquet'))
assert pa_splits.schema.equals(subject_split_schema), "subject split schema does not match"
assert df_splits.shape[0] == df_.shape[0], "splits shape does not match"
print(df_.head())
print("Shape:", df_.shape)

# Events + Codes
print("Converting events...")
femr_db = femr.datasets.PatientDatabase('../../EHRSHOT_ASSETS/femr/extract')
events: List[Dict] = []
codes: List[Dict] = []
codes_set: Set[str] = set()
for pid in tqdm(femr_db, total=len(femr_db), desc='events'):
    for event in femr_db[pid].events:
        ## Event
        e: Dict[str, Any] = {
            'subject_id' : pid,
            'time' : event.start,
            'code' : event.code,
            'unit' : event.unit,
            'omop_table' : event.omop_table,
            'numeric_value' : None,
        }
        if event.value is not None:
            if isinstance(event.value, str):
                e['text_value'] = event.value
            elif isinstance(event.value, float) or isinstance(event.value, int):
                e['numeric_value'] = event.value
        events.append(e)
        ## Code
        if event.code not in codes_set:
            codes.append({
                'code' : event.code,
                'description' : '',
            })
            codes_set.add(event.code)

## Events
print("Writing data/data.parquet")
events_schema = data_schema([
    ("text_value", pa.string()), 
    ("unit", pa.string()), 
    ("omop_table", pa.string())
])
pa_events = pa.Table.from_pylist(events, schema=events_schema)
pq.write_table(pa_events, os.path.join(path_to_output_dir, 'data/data.parquet'))
## Sanity check
df_ = pd.read_parquet(os.path.join(path_to_output_dir, 'data/data.parquet'))
assert pa_events.schema.equals(events_schema), "events schema does not match"
assert pa_events.shape[0] == df_.shape[0], "events shape does not match"
print(df_.head())
print("Shape:", df_.shape)

## Codes
print("Writing metadata/codes.parquet")
codes_schema = code_metadata_schema()
pa_codes = pa.Table.from_pylist(codes, schema=codes_schema)
pq.write_table(pa_codes, os.path.join(path_to_output_dir, 'metadata/codes.parquet'))
## Sanity check
df_ = pd.read_parquet(os.path.join(path_to_output_dir, 'metadata/codes.parquet'))
assert pa_codes.schema.equals(codes_schema), "codes schema does not match"
assert pa_codes.shape[0] == df_.shape[0], "events shape does not match"
print(df_.head())
print("Shape:", df_.shape)

# Labels
print("Converting labels...")
path_to_labels_dir: str = '../../EHRSHOT_ASSETS/benchmark_ehrshot/'
for task in tqdm(os.listdir(path_to_labels_dir), desc='labels'):
    if not os.path.isdir(os.path.join(path_to_labels_dir, task)):
        continue
    path_to_labels: str = os.path.join(path_to_labels_dir, task, 'labeled_patients.csv')
    df_labels = pd.read_csv(path_to_labels)
    df_labels = df_labels.rename(columns={ 'patient_id' : 'subject_id', })
    label_type: str = df_labels['label_type'].iloc[0]
    if label_type == 'boolean':
        df_labels = df_labels.rename(columns={ 'value' : 'boolean_value', })
        df_labels['integer_value'] = None
        df_labels['float_value'] = None
        df_labels['categorical_value'] = None
    elif label_type == 'categorical':
        df_labels = df_labels.rename(columns={ 'value' : 'integer_value', })
        df_labels['boolean_value'] = None
        df_labels['float_value'] = None
        df_labels['categorical_value'] = None
    else:
        raise ValueError(f"Unexpected label type: {label_type}")
    df_labels = df_labels.drop(columns=['label_type'])
    df_labels['prediction_time'] = pd.to_datetime(df_labels['prediction_time'])
    pa_labels = pa.Table.from_pandas(df_labels, schema=label_schema)
    print(f"Writing labels/{task}/labels.parquet")
    os.makedirs(os.path.join(path_to_output_dir, 'labels', task), exist_ok=True)
    pq.write_table(pa_labels, os.path.join(path_to_output_dir, 'labels', task, 'labels.parquet'))
    ## Sanity check
    df_ = pd.read_parquet(os.path.join(path_to_output_dir, 'labels', task, 'labels.parquet'))
    assert pa_labels.schema.equals(label_schema), "labels schema does not match"
    assert df_labels.shape[0] == df_.shape[0], "labels shape does not match"
    print(df_.head())
    print("Shape:", df_.shape)

# Confirm it worked
import meds_reader
print("Running meds_reader to confirm MEDS extract is valid...")
status: int = os.system(f"meds_reader_convert {path_to_output_dir} {path_to_output_dir}_db --num_threads 5")
if status != 1:
    raise ValueError("Failed to convert to MEDS DB -- Couldn't run meds_reader on it")

print("Done!")