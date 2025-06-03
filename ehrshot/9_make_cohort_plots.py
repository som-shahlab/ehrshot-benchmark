import argparse
import datetime
import multiprocessing
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import os
import femr.datasets
from loguru import logger
from tqdm import tqdm

from utils import (
    write_table_to_latex,
)
from plot import (
    plot_column_per_patient,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Make plots/tables of cohort stats")
    parser.add_argument("--path_to_labels_and_feats_dir", required=True, type=str, help="Path to directory containing saved labels and featurizers")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_input_dir", required=True, type=str, help="Path to folder containing all EHRSHOT cohort CSVs")
    parser.add_argument("--path_to_splits_dir", required=True, type=str, help="Path to directory containing splits")
    parser.add_argument("--path_to_output_dir", required=True, type=str, help="Path to directory to save figures")
    parser.add_argument("--num_threads", default=1, type=int, help="Number of processes to launch")
    return parser.parse_args()

def create_df_demo(path_to_database: str, path_to_output_dir: str, path_to_splits_dir: str, num_threads: int):
    # Get list of all patient IDs in labeled patients
    patient_database = femr.datasets.PatientDatabase(path_to_database)
    all_patient_ids = np.array(sorted(list([ pid for pid in patient_database ])))
    logger.info("# of patients:", len(all_patient_ids))

    # Split patients across threads
    patient_ids_chunks: List[List[int]] = np.array_split(all_patient_ids, num_threads)
    paths_to_csvs: List[str] = [
        os.path.join(path_to_output_dir, f"pretrain_thread_{i}.csv") 
        for i in range(len(patient_ids_chunks))
    ]

    logger.success("Start | Writing CSVs in parallel")
    task_args: List[Tuple[str, List[int]]] = [
        (paths_to_csvs[i], path_to_database, path_to_splits_dir, patient_ids_chunks[i])
        for i in range(len(patient_ids_chunks))
    ]
    with multiprocessing.get_context("forkserver").Pool(num_threads) as pool:
        parallel_result = list(tqdm(pool.imap(compute_demographics, task_args), total=num_threads))
    logger.success("Finish | Writing CSVs in parallel")

    logger.info("Start | Merging CSVs together")

    # Re-read all CSVs
    dfs: List[pd.DataFrame] = []
    for filename in paths_to_csvs:
        dfs.append(pd.read_csv(filename))
    df_merged: pd.DataFrame = pd.concat(dfs)

    # Sort by patient ID
    df_merged = df_merged.sort_values(by=['patient_id',])

    # Save merged CSV to disk
    df_merged.to_csv(os.path.join(path_to_output_dir, 'df_demo.csv'), index=False)
    logger.info(f"Merged CSV file saved @ `{os.path.join(path_to_output_dir, 'df_demo.csv')}`")
    logger.success("Finish | Merging CSVs together")
    
    # Remove old `pretrain_thread_*.csv` files
    for filename in paths_to_csvs:
        os.remove(filename)
    logger.success(f"Done creating df_demo!")

def get_splits(splits_dir):
    splits_df = pd.read_csv(os.path.join(splits_dir, 'person_id_map.csv'))
    splits_json = {'train':[],'test':[],'val':[]}

    for index,row in splits_df.iterrows():
        if row['split'] == 'val':
            splits_json['val'].append(row['omop_person_id'])
        elif row['split'] == 'test':
            splits_json['test'].append(row['omop_person_id'])
        else:
            splits_json['train'].append(row['omop_person_id'])

    return splits_json

def compute_demographics(args):
    path_to_csv, path_to_database, path_to_splits_dir, patient_ids = args
    patient_database = femr.datasets.PatientDatabase(path_to_database)
    
    # Load splits
    df_split = pd.read_csv(os.path.join(path_to_splits_dir, 'person_id_map.csv'))
    splits: Dict[str, List[int]] = { # [key] = split, [value] = list of patient IDs
        split: df_split[df_split['split'] == split]['omop_person_id'].tolist() for split in df_split['split'].unique()
    }

    rows = []
    for idx, pid in enumerate(patient_ids):
        is_male: bool = False
        age: int = -1
        is_hispanic: bool = False
        race: str = 'unknown'
        # gender
        for e in patient_database[pid].events:
            if e.code == 'Gender/M':
                is_male = True
                break
            elif e.code == 'Gender/F':
                is_male = False
                break
        # age
        end_age = patient_database[pid].events[-1].start
        start_age = patient_database[pid].events[0].start
        age = end_age - start_age
        # ethnicity
        for e in patient_database[pid].events:
            if e.code == 'Ethnicity/Hispanic':
                is_hispanic = True
                break
            elif e.code == 'Ethnicity/Not Hispanic':
                is_hispanic = False
                break
        # race
        for e in patient_database[pid].events:
            if e.code == 'Race/5':
                race = 'white'
                break
            elif e.code == 'Race/4':
                race = 'pacific_islander'
                break
            elif e.code == 'Race/3':
                race = 'black'
                break
            elif e.code == 'Race/2':
                race = 'asian'
                break
            elif e.code == 'Race/1':
                race = 'american_indian'
                break
        # number of events
        num_events: int = len(patient_database[pid].events)
        # number of visits
        unique_visits = set()
        for e in patient_database[pid].events:
            if e.visit_id is not None:
                unique_visits.add(e.visit_id)
        num_visits: int = len(unique_visits)
        # split
        split: str = 'train' if pid in splits['train'] else ('val' if pid in splits['val'] else 'test')
        rows.append({
            'split' : split,
            'patient_id' : pid,
            'age' : age.days / 365.25,
            'age_20' : age <= datetime.timedelta(days=20*365),
            'age_40' : datetime.timedelta(days=20*365) < age <= datetime.timedelta(days=40*365),
            'age_60' : datetime.timedelta(days=40*365) < age < datetime.timedelta(days=60*365),
            'age_80' : datetime.timedelta(days=60*365) < age < datetime.timedelta(days=80*365),
            'age_plus' : datetime.timedelta(days=80*365) < age,
            'is_male' : is_male,
            'is_hispanic' : is_hispanic,
            'race' : race,
            'num_events' : num_events,
            'timeline_length' : age.days / 365.25,
            'num_visits' : num_visits,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path_to_csv, index=False)

if __name__ == "__main__":
    args = parse_args()
    PATH_TO_DATABASE: str = args.path_to_database
    PATH_TO_INPUT_DIR: str = args.path_to_input_dir
    PATH_TO_LABELS_AND_FEATS_DIR: str = args.path_to_labels_and_feats_dir
    PATH_TO_SPLITS_DIR: str = args.path_to_splits_dir
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    PATH_TO_LATEX_TABLES_FILE: str = os.path.join(PATH_TO_OUTPUT_DIR, 'latex_tables.txt')
    NUM_THREADS: int = args.num_threads
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)
    if os.path.exists(PATH_TO_LATEX_TABLES_FILE):
        os.remove(PATH_TO_LATEX_TABLES_FILE)

    # Load splits
    df_split = pd.read_csv(os.path.join(PATH_TO_SPLITS_DIR, 'person_id_map.csv'))
    splits: Dict[str, List[int]] = { # [key] = split, [value] = list of patient IDs
        split: df_split[df_split['split'] == split]['omop_person_id'].tolist() for split in df_split['split'].unique()
    }

    # Load all labels from CSVs
    logger.info("Start | Creating df_labels.csv")
    if not os.path.exists(os.path.join(PATH_TO_OUTPUT_DIR, 'df_labels.csv')):
        dfs: List[pd.DataFrame] = []
        for labeling_function in os.listdir(PATH_TO_LABELS_AND_FEATS_DIR):
            if os.path.isdir(os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, labeling_function)):
                df_ = pd.read_csv(os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, f"{labeling_function}/labeled_patients.csv"))
                df_['task'] = labeling_function
                df_['split'] = df_['patient_id'].apply(lambda x: 'train' if x in splits['train'] else ('val' if x in splits['val'] else 'test'))
                dfs.append(df_)
        df_labels: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        df_labels.to_csv(os.path.join(PATH_TO_OUTPUT_DIR, 'df_labels.csv'), index=False)
    df_labels = pd.read_csv(os.path.join(PATH_TO_OUTPUT_DIR, 'df_labels.csv'))
    logger.info("Finish | Creating df_labels.csv")

    # Load all patient events from CSVs
    logger.info("Start | Creating df_merged.csv")
    if not os.path.exists(os.path.join(PATH_TO_OUTPUT_DIR, 'df_merged.csv')):
        dfs: List[pd.DataFrame] = []
        for cohort_csv in os.listdir(PATH_TO_INPUT_DIR):
            dfs.append(pd.read_csv(os.path.join(PATH_TO_INPUT_DIR, cohort_csv)))
        df_merged: pd.DataFrame = pd.concat(dfs, ignore_index=True)
        df_merged.to_csv(os.path.join(PATH_TO_OUTPUT_DIR, 'df_merged.csv'), index=False)
    df_merged = pd.read_csv(os.path.join(PATH_TO_OUTPUT_DIR, 'df_merged.csv'))
    logger.info("Finish | Creating df_merged.csv")

    # Load pre-computed patient demographics
    logger.info("Start | Creating df_demo.csv")
    if not os.path.exists(os.path.join(PATH_TO_OUTPUT_DIR, 'df_demo.csv')):
        create_df_demo(PATH_TO_DATABASE, PATH_TO_OUTPUT_DIR, PATH_TO_SPLITS_DIR, NUM_THREADS)
    df_demo = pd.read_csv(os.path.join(PATH_TO_OUTPUT_DIR, 'df_demo.csv'))
    logger.info("Finish | Creating df_demo.csv")

    # Unique # of visits + obs
    logger.info(f"# of obs: {df_merged.shape[0]}")
    logger.info(f"# of unique codes: {df_merged['code'].nunique()}")
    logger.info(f"# of visits: {df_merged['visit_id'][~df_merged['visit_id'].isna()].nunique()}")
    logger.info(f"# of patients: {df_merged['patient_id'].nunique()}")
    logger.info(f"# of minors (<19 yrs old): {(df_demo['timeline_length'] < 19).sum()}")
    logger.info(f"# of seniors (>89 yrs old): {(df_demo['timeline_length'] > 89).sum()}")

    ########################################################
    ########################################################
    #
    # Precompute values so that code below runs faster
    #
    ########################################################
    ########################################################
    logger.info("Start | Precomputing dictionaries")
    pid_2_is_male: Dict[int, bool] = {
        row['patient_id']: row['is_male'] for idx, row in df_demo.iterrows()
    }
    pid_2_age: Dict[int, int] = {
        row['patient_id']: row['age'] for idx, row in df_demo.iterrows()
    }
    pid_2_is_hispanic: Dict[int, bool] = {
        row['patient_id']: row['is_hispanic'] for idx, row in df_demo.iterrows()
    }
    pid_2_race: Dict[int, str] = {
        row['patient_id']: row['race'] for idx, row in df_demo.iterrows()
    }
    logger.success("Finish | Precomputing dictionaries")

    ########################################################
    ########################################################
    #
    # Create task-level stats per split
    #
    ########################################################
    ########################################################
    logger.info("Start | Creating df_task.csv")
    if not os.path.exists(os.path.join(PATH_TO_OUTPUT_DIR, 'df_task.csv')):
        rows = []
        for split in df_labels['split'].unique():
            for lf in df_labels['task'].unique():
                df_ = df_labels[(df_labels['split'] == split) & (df_labels['task'] == lf)]
                if lf == 'chexpert':
                    neg_values: List = ['8192']
                    # Patients
                    n_pos_patients = df_[~df_['value'].isin(neg_values)]['patient_id'].nunique()
                    n_neg_patients = df_[df_['value'].isin(neg_values)]['patient_id'].nunique()
                    n_total_patients = df_['patient_id'].nunique()
                    # Labels
                    n_pos_labels = df_[~df_['value'].isin(neg_values)].shape[0]
                    n_neg_labels = df_[df_['value'].isin(neg_values)].shape[0]
                    n_total_labels = df_.shape[0]
                else:
                    true_values: List = ['1', 1, 'True', True]
                    # Patients
                    n_pos_patients = df_[df_['value'].isin(true_values)]['patient_id'].nunique()
                    n_neg_patients = df_[~df_['value'].isin(true_values)]['patient_id'].nunique()
                    n_total_patients = df_['patient_id'].nunique()
                    # Labels
                    n_pos_labels = df_[df_['value'].isin(true_values)].shape[0]
                    n_neg_labels = df_[~df_['value'].isin(true_values)].shape[0]
                    n_total_labels = df_.shape[0]
                # % Male
                n_male = sum([ 1 if pid_2_is_male[pid] else 0 for pid in df_['patient_id'].unique() ])
                # Median Age
                ages = [ pid_2_age[pid] for pid in df_['patient_id'].unique() ]
                median_age = np.median(ages)
                # Save data
                rows.append({
                    'lf' : lf,
                    'split' : split,
                    'n_pos_patients' : n_pos_patients,
                    'n_neg_patients' : n_neg_patients,
                    'n_total_patients' : n_total_patients,
                    'n_pos_labels' : n_pos_labels,
                    'n_neg_labels' : n_neg_labels,
                    'n_total_labels' : n_total_labels,
                    'n_male' : n_male,
                    'median_age_yrs' : median_age,
                })
        df_task = pd.DataFrame(rows)
        df_task.to_csv(os.path.join(PATH_TO_OUTPUT_DIR, 'df_task.csv'), index=True)
    else:
        df_task = pd.read_csv(os.path.join(PATH_TO_OUTPUT_DIR, 'df_task.csv'))
    logger.success("Finish | Creating df_task.csv")


            
    ########################################################
    ########################################################
    #
    # SPLIT SUMMARY STATISTICS TABLE
    #
    # Format:
    """
        split	train	val	test
        Events Min	5	6	6
        Events Mean	6208	5430	5863
        Events Max	126051	129281	179161
        Visits Min	0	0	0
        Visits Mean	142	125	131
        Visits Max	3288	2441	3701
        Age Min	0	0	0
        Age Mean	51	50	51
        Age Max	88	88	88
    """
    #
    #########################################################
    #########################################################

    df_figure = df_demo[['split', 'patient_id', 'num_events', 'timeline_length', 'num_visits', ]].copy()
    df_figure = df_figure.groupby(['split']).agg(num_events_min=('num_events', 'min'),
                                                num_events_mean=('num_events', 'mean'),
                                                num_events_max=('num_events', 'max'),
                                                num_visits_min=('num_visits', 'min'),
                                                num_visits_mean=('num_visits', 'mean'),
                                                num_visits_max=('num_visits', 'max'),
                                                timeline_length_min=('timeline_length', 'min'),
                                                timeline_length_mean=('timeline_length', 'mean'),
                                                timeline_length_max=('timeline_length', 'max'))
    df_figure = df_figure.rename(columns={'num_events_min' : 'Events Min', 
                                        'num_events_mean' : 'Events Mean',
                                        'num_events_max' : 'Events Max',
                                        'num_visits_min' : 'Visits Min',
                                        'num_visits_mean' : 'Visits Mean',
                                        'num_visits_max' : 'Visits Max',
                                        'timeline_length_min' : 'Age Min',
                                        'timeline_length_mean' : 'Age Mean',
                                        'timeline_length_max' : 'Age Max',
                    })
    df_figure = df_figure.T.astype(int)
    write_table_to_latex(df_figure, PATH_TO_LATEX_TABLES_FILE)

    ########################################################
    ########################################################
    #
    # DEMOGRAPHICS GROUPED BY SPLIT TABLE
    #
    # Format:
    """
        split	train	val	test	total
        age_20	387	384	340	1111
        age_40	366	367	373	1106
        age_60	509	462	476	1447
        age_80	804	750	720	2274
        age_plus	263	255	256	774
        race__american_indian	7	9	12	28
        race__asian	344	370	318	1032
        race__black	94	91	91	276
        race__pacific_islander	31	37	36	104
        race__unknown	563	567	537	1667
        race__white	1290	1144	1171	3605
        is_hispanic__False	1948	1798	1786	5532
        is_hispanic__True	381	420	379	1180
        is_male__False	1196	1063	1104	3363
        is_male__True	1133	1155	1061	3349
        age_total	2329	2218	2165	6712
        ethnicity_total	2329	2218	2165	6712
        gender_total	2329	2218	2165	6712
        race_total	2329	2218	2165	6712
    """
    #
    #########################################################
    #########################################################

    df_figure = df_demo[['split', 'patient_id', 'race', 'is_hispanic', 'is_male', 'age_20', 'age_40', 'age_60', 'age_80', 'age_plus']].copy()

    # One hot encode categorical variables
    oh_race = pd.get_dummies(df_figure['race'], prefix='race_')
    oh_hisp = pd.get_dummies(df_figure['is_hispanic'], prefix='is_hispanic_')
    oh_male = pd.get_dummies(df_figure['is_male'], prefix='is_male_')
    df_figure = pd.concat([df_figure, oh_race, oh_hisp, oh_male], axis=1)

    df_figure = df_figure.groupby(['split']).agg('sum')
    df_figure['age_total'] = df_figure['age_20'] + df_figure['age_40'] + df_figure['age_60'] + df_figure['age_80'] + df_figure['age_plus']
    df_figure['ethnicity_total'] = df_figure['is_hispanic__True'] + df_figure['is_hispanic__False']
    df_figure['gender_total'] = df_figure['is_male__True'] + df_figure['is_male__False']
    df_figure['race_total'] = df_figure['race__american_indian'] + df_figure['race__white'] + df_figure['race__black'] + df_figure['race__asian'] + df_figure['race__unknown'] + + df_figure['race__pacific_islander'] 

    df_figure = df_figure.drop(columns=['patient_id', 'is_hispanic', 'is_male', 'race',  ])
    df_figure = df_figure.T.astype(int)
    df_figure['total'] = df_figure.sum(axis=1)
    df_figure = df_figure.iloc[:, [1, 2, 0, 3, ]]
    write_table_to_latex(df_figure, PATH_TO_LATEX_TABLES_FILE, is_ignore_index=False)

    ########################################################
    ########################################################
    #
    # TASK LABEL PREVALENCE TABLE
    #
    # Format:
    """
    lf	# Patients (# Positive) train	# Labels (# Positive) train	Spacer	# Patients (# Positive) test	# Patients (# Positive) val
        2	Long Length of Stay	1424 (458)	2695 (743)		1248 (394)	1254 (400)
        3	30-day Readmission	1342 (186)	2658 (373)		1175 (151)	1137 (147)
        1	ICU Transfer	1311 (100)	2460 (119)		1139 (78)	1167 (75)
        7	Thrombocytopenia	1951 (798)	59462 (8529)		1793 (707)	1796 (680)
        4	Hyperkalemia	1923 (391)	70653 (1292)		1758 (346)	1777 (350)
        5	Hypoglycemia	1983 (430)	104433 (1128)		1804 (355)	1829 (354)
        6	Hyponatremia	1922 (1198)	74009 (17962)		1760 (1094)	1777 (1064)
        0	Anemia	1968 (1211)	64130 (8997)		1806 (1045)	1808 (1017)
        11	Hypertension	953 (129)	1551 (194)		848 (128)	846 (128)
        10	Hyperlipidemia	1066 (132)	1912 (179)		952 (135)	928 (134)
        13	Pancreatic Cancer	1429 (128)	2688 (146)		1252 (42)	1248 (47)
        9	Celiac	1437 (82)	2688 (105)		1264 (17)	1263 (12)
        12	Lupus	1427 (88)	2649 (124)		1250 (25)	1245 (25)
        8	Acute MI	1415 (130)	2613 (200)		1245 (117)	1244 (116)
    """
    #
    #########################################################
    #########################################################

    df_figure = df_task.copy()
    lf_to_col_name = {
        'new_celiac' : 'Celiac',
        'new_lupus' : 'Lupus',
        'new_pancan' : 'Pancreatic Cancer',
        'new_acutemi' : 'Acute MI',
        'new_hypertension' : 'Hypertension',
        'new_hyperlipidemia' : 'Hyperlipidemia',
        'lab_anemia' : 'Anemia',
        'lab_hyperkalemia' : 'Hyperkalemia',
        'lab_hypoglycemia' : 'Hypoglycemia',
        'lab_hyponatremia' : 'Hyponatremia',
        'lab_thrombocytopenia' : 'Thrombocytopenia',
        'guo_readmission' : '30-day Readmission',
        'guo_icu' : 'ICU Transfer',
        'guo_los' : 'Long Length of Stay',
        'chexpert' : 'Chest X-Ray Findings',
    }
    df_figure = df_figure[['lf', 'split', 'n_total_patients', 'n_pos_patients', 'n_total_labels', 'n_pos_labels', 'median_age_yrs']]
    df_figure = df_figure[~df_figure['lf'].str.startswith('chexpert_')]
    df_figure['# Patients (# Positive)'] = df_figure['n_total_patients'].astype(int).astype(str) + ' (' + df_figure['n_pos_patients'].astype(str) + ')'
    df_figure['# Labels (# Positive)'] = df_figure['n_total_labels'].astype(int).astype(str) + ' (' + df_figure['n_pos_labels'].astype(str) + ')'
    df_figure = df_figure.drop(columns=['n_total_patients', 'n_pos_patients', 'n_total_labels', 'n_pos_labels', 'median_age_yrs'])
    # Pivot columns
    df_figure = df_figure.pivot(index='lf', columns='split')
    df_figure.columns = [' '.join(col).strip() for col in df_figure.columns.values]
    df_figure.reset_index(inplace=True)
    # Order columns
    new_col_order = sorted(df_figure.columns[1:], key=lambda x: -1 if 'train' in x else (0 if 'val' in x else 1))
    df_figure = df_figure[['lf'] + new_col_order]
    # Sort rows
    new_row_order = [
        'guo_los',
        'guo_readmission',
        'guo_icu',
        'lab_thrombocytopenia',
        'lab_hyperkalemia',
        'lab_hypoglycemia',
        'lab_hyponatremia',
        'lab_anemia',
        'new_hypertension',
        'new_hyperlipidemia',
        'new_pancan',
        'new_celiac',
        'new_lupus',
        'new_acutemi',
        'chexpert',
    ]
    df_figure = df_figure.sort_values(by='lf', key=lambda x: x.map({name: i for i, name in enumerate(new_row_order)}))
    df_figure['lf'] = df_figure['lf'].apply(lambda x: lf_to_col_name[x])
    df_figure.insert(3, 'Spacer', '')
    df_figure.insert(6, 'Spacer 2', '')
    write_table_to_latex(df_figure, PATH_TO_LATEX_TABLES_FILE, is_ignore_index=True)

    ########################################################
    ########################################################
    #
    # Make plots
    #
    ########################################################
    ########################################################
    logger.info("Start | Making plots")
    plot_column_per_patient(df_demo, PATH_TO_OUTPUT_DIR, 'num_events', title='events', x_label='# of Events', max_clamp=40000)
    plot_column_per_patient(df_demo, PATH_TO_OUTPUT_DIR, 'num_visits', title='visits', x_label='# of Visits', max_clamp=1000)
    plot_column_per_patient(df_demo, PATH_TO_OUTPUT_DIR, 'timeline_length', title='timeline length', x_label='Timeline Length (yrs)')
    logger.success("Finish | Making plots")
