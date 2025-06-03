import datetime
import numpy as np
from femr.datasets import PatientDatabase
from utils import (
    get_labels_and_features, 
    get_patient_splits_by_idx
)
from femr.labelers import load_labeled_patients, LabeledPatients

# Load ontology
database = PatientDatabase(PATH_TO_PATIENT_DATABASE)
ontology = database.get_ontology()

# Load all labeled patients
labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELED_PATIENTS)

model = 'clmbr'

# Load labels/features for this task + model_head
patient_ids, label_values, label_times, feature_matrixes = get_labels_and_features(labeled_patients, 
                                                                                    PATH_TO_FEATURES_DIR, 
                                                                                    PATH_TO_TOKENIZED_TIMELINES_DIR,
                                                                                    models_to_keep=[ model ])
__, __, test_pids_idx = get_patient_splits_by_idx(PATH_TO_SPLIT_CSV, patient_ids)

# Test labels
y_test: np.ndarray = label_values[test_pids_idx]
test_patient_ids = patient_ids[test_pids_idx]
label_times = [ x.astype(datetime.datetime) for x in label_times ] # cast to Python datetime