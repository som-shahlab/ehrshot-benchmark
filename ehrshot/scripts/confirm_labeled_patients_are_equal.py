import os
import pandas as pd

path1 = '../../EHRSHOT_ASSETS/benchmark_ehrshot'
path2 = '../../EHRSHOT_ASSETS/benchmark'

for task in os.listdir(path1):
    if task.endswith('.csv'): continue
    df1 = pd.read_csv(f'{path1}/{task}/labeled_patients.csv').sort_values(['patient_id', 'prediction_time', 'value']).reset_index()
    df2 = pd.read_csv(f'{path2}/{task}/labeled_patients.csv').sort_values(['patient_id', 'prediction_time', 'value']).reset_index()
    print(task, "|", (df1[['patient_id', 'prediction_time', 'value']] != df2[['patient_id', 'prediction_time', 'value']]).sum().sum())