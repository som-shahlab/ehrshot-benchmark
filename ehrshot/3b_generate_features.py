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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate all labels to speed up featurization")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to femr database")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to save labeles and featurizers")
    parser.add_argument("--num_threads", type=int, help="The number of threads to use")

    args = parser.parse_args()

    labeled_patients = femr.labelers.load_labeled_patients(os.path.join(args.path_to_features_dir, 'all_labels.csv'))

    age = femr.featurizers.AgeFeaturizer()
    count = femr.featurizers.CountFeaturizer(is_ontology_expansion=True)
    featurizer_age_count = femr.featurizers.FeaturizerList([age, count])

    # Preprocessing the featurizers, which includes processes such as normalizing age.
    logger.info("Start | Preprocess featurizers")
    featurizer_age_count.preprocess_featurizers(args.path_to_database, labeled_patients, args.num_threads)
    logger.info("Finish | Preprocess featurizers")

    logger.info("Start | Featurize patients")
    results = featurizer_age_count.featurize(args.path_to_database, labeled_patients, args.num_threads)

    with open(os.path.join(args.path_to_features_dir, 'count_features.pkl'), 'wb') as f:
        pickle.dump(results, f)

    logger.info("Finish | Featurize patients")
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