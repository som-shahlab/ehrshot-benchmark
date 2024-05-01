import datetime
import femr
from utils import get_rel_path, dump_patient_to_json
import os
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import collections
from femr.labelers import load_labeled_patients
import json
from tqdm import tqdm

LABELING_FUNCTION = 'guo_icu'
PATH_TO_DATABASE = get_rel_path(__file__, '../EHRSHOT_ASSETS/database/')
PATH_TO_LABELS_DIR = get_rel_path(__file__, '../EHRSHOT_ASSETS/labels/')
PATH_TO_LABELED_PATIENTS = os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION, 'labeled_patients.csv')

database_new = femr.datasets.PatientDatabase(PATH_TO_DATABASE)
database_orig_raw = femr.datasets.PatientDatabase('/share/pi/nigam/data/som-rit-phi-starr-prod.starr_omop_cdm5_deid_2023_02_08_extract_v8')
database_orig = femr.datasets.PatientDatabase('/share/pi/nigam/mwornow/ehrshot-benchmark-hf_ehr/EHRSHOT_ASSETS/femr/extract')
df_orig_splits = pd.read_csv('/share/pi/nigam/mwornow/ehrshot-private/cohort/ignore/starr-compliant/patient_ids.csv')

df_mapping = pd.read_csv('/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/mapping.csv')

codes = []
for new_patient_id in tqdm(df_mapping['new'].values.tolist()):
    events = []
    old_patient_id = df_mapping[df_mapping['new'] == new_patient_id]['old'].values[0]
    new_bdate = database_new[new_patient_id].events[0].start
    old_bdate = database_orig_raw[old_patient_id].events[0].start
    jitter = new_bdate - old_bdate
    for e in database_new[new_patient_id].events:
        if e.start > (datetime.datetime(2023, 2, 8) + jitter):
            break
        events.append(e.code)
    codes.append({
        'new_patient_id' : new_patient_id,
        'events' : events,
    })
df_codes = pd.DataFrame(codes)
df_codes.to_csv('/share/pi/nigam/mwornow/ehrshot-benchmark/ehrshot/new_patient_codes.csv')
exit()


# Fix CLMBR CARE_SITE dict

import msgpack
path_to_dict = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/models/clmbr/dictionary'
path_to_care_site_map = '/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/care_site_map.csv'

with open(path_to_dict, 'rb') as file: 
    file_data = file.read()
    data = msgpack.unpackb(file_data)

care_site_mapping = pd.read_csv(path_to_care_site_map)[['care_site_id_old', 'care_site_id_new']]

for idx, __ in enumerate(data['regular']):
    old_code = data['regular'][idx]['code_string']
    if old_code.startswith('CARE_SITE'):
        old_id = old_code.split("/")[-1]
        if old_id in care_site_mapping['care_site_id_old'].astype(str).values.tolist():
            new_id = care_site_mapping[care_site_mapping['care_site_id_old'].astype(str).values == old_id]['care_site_id_new'].values[0]
            data['regular'][idx]['code_string'] = f"CARE_SITE/{new_id}"

packed_data = msgpack.packb(data)
with open(path_to_dict, 'wb') as file:
    file.write(packed_data)


# lengths = []
# for new_patient_id in database_new:
#     count = 0
#     for e in database_new[new_patient_id].events:
#         if e.start > datetime.datetime(2023, 2, 8):
#             break
#         count += 1
#     lengths.append({
#         'new_patient_id' : new_patient_id,
#         'n_events' : count,
#     })
# df_lengths = pd.DataFrame(lengths)
# df_lengths.to_csv('new_patient_lengths.csv')

exit()

new_total_events: int = sum([ len(database[x].events) for x in database_new ])
print("# of events in new ")
# Load labels for this task
labeled_patients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)

df_labels = pd.read_csv('/share/pi/nigam/mwornow/ehrshot-benchmark/EHRSHOT_ASSETS/labels/guo_icu/labeled_patients.csv')
df_old_labels = pd.read_csv('/share/pi/nigam/mwornow/ehrshot-benchmark-hf_ehr/EHRSHOT_ASSETS/custom_benchmark/guo_icu/labeled_patients.csv')
df_old_patient_id_2_random_id = json.load(open('/share/pi/nigam/mwornow/ehrshot-private/cohort/ignore/starr-compliant/patient_id_2_random_id.json', 'r'))

pids = set(df_labels['patient_id'].unique()).intersection(set(df_mapping['new']))

pid_2_labels = {}
for pid in tqdm(sorted(list(pids))):
    try:
        pid_2_labels[pid] = {
            'new' : df_labels[df_labels['patient_id'] == pid].values.tolist(),
            'old' : df_old_labels[df_old_labels['patient_id'] == df_old_patient_id_2_random_id[str(df_mapping[df_mapping['new'] == pid]['old'].values[0])]].values.tolist(),
            'new_pid' : pid,
            'old_pid' : int(df_mapping[df_mapping['new'] == pid]['old'].values[0]),
        }
    except:
        pass
json.dump(pid_2_labels, open('guo_icu.json', 'w'), indent=2)
exit()
# lengths = []
# for idx, row in tqdm(df_mapping.iterrows()):
#     if abs((database_orig[row['old_patient_id']].events[0].start 
#            - database_new[row['new_patient_id']].events[0].start).days) > 60:
#         continue
#     lengths.append({
#         'old' : len(database_orig[row['old_patient_id']].events),
#         'new' : len(database_new[row['new_patient_id']].events),
#         'old_patient_id' : row['old_patient_id'],
#         'new_patient_id' : row['new_patient_id'],
#     })
# df_lengths = pd.DataFrame(lengths)
# df_lengths.to_csv('lengths.csv')
# breakpoint()


# birth_date_2_new_patient_ids = collections.defaultdict(list)
# for patient_id in tqdm(database_new):
#     birth_date_2_new_patient_ids[database_new[patient_id].events[0].start].append(patient_id)

# breakpoint()
# old_patient_id_2_possible_new_patient_ids = collections.defaultdict(list)
# for idx, row in tqdm(df_orig_splits.iterrows(), total=df_orig_splits.shape[0]):
#     old_birth_date = database_orig[row['patient_id']].events[0].start

#     for delta in range(-30, 31):
#         offset = old_birth_date + timedelta(days=delta)
#         if offset in birth_date_2_new_patient_ids:
#             old_patient_id_2_possible_new_patient_ids[row['patient_id']].extend(birth_date_2_new_patient_ids[offset])
# breakpoint()

[ (key, val) for key, val in old_patient_id_2_possible_new_patient_ids.items() if len(val) == 1 ]
# df_orig_splits['birth_date'] = birth_dates
# df_orig_splits['new_ehrshot_id'] = new_ids

# df_orig_splits.to_csv('mapping_old_to_new_ehrshot.csv')