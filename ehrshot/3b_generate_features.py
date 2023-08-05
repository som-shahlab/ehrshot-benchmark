import argparse
import pickle
import datetime
import os
import json
from loguru import logger
from utils import LABELING_FUNCTIONS, save_data
import pandas as pd

import femr
from femr.datasets import PatientDatabase
import femr.labelers
from femr.labelers.core import LabeledPatients
from femr.featurizers.core import FeaturizerList
from femr.featurizers.featurizers import AgeFeaturizer, CountFeaturizer
from femr.labelers.core import NLabelsPerPatientLabeler, TimeHorizon
from femr.labelers.omop import (
    MortalityCodeLabeler,
    ChexpertLabeler,
)

def parse_args(args) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate count-based featurizations for GBM models (for all tasks at once)")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_and_feats_dir", required=True, type=str, help="Path to save labels and featurizers")
    parser.add_argument("--num_threads", type=int, help="Number of threads to use")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    NUM_THREADS: int = args.num_threads
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_LABELS_AND_FEATS_DIR = args.path_to_labels_and_feats_dir

    # Load consolidated labels across all patients for all tasks
    labeled_patients = femr.labelers.load_labeled_patients(os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, 'all_labels.csv'))

    # Combine two featurizations of each patient: one for the patient's age, and one for the count of every code
    # they've had in their record up to the prediction timepoint for each label
    age = femr.featurizers.AgeFeaturizer()
    count = femr.featurizers.CountFeaturizer(is_ontology_expansion=True)
    featurizer_age_count = femr.featurizers.FeaturizerList([age, count])

    # Preprocessing the featurizers -- this includes processes such as normalizing age
    logger.info("Start | Preprocess featurizers")
    featurizer_age_count.preprocess_featurizers(PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS)
    logger.info("Finish | Preprocess featurizers")

    # Run actual featurization for each patient
    logger.info("Start | Featurize patients")
    results = featurizer_age_count.featurize(PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS)
    logger.info("Finish | Featurize patients")

    # Save results
    with open(os.path.join(args.path_to_labels_and_feats_dir, 'count_features.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # Logging
    feature_matrix, patient_ids, label_values, label_times = (
        results[0],
        results[1],
        results[2],
        results[3],
    )
    logger.info("FeaturizedPatient stats:\n"
                f"feature_matrix={repr(feature_matrix)}\n"
                f"patient_ids={repr(patient_ids)}\n"
                f"label_values={repr(label_values)}\n"
                f"label_times={repr(label_times)}")