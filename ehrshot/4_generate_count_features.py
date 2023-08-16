import argparse
import pickle
import os
from typing import Any, Dict
from loguru import logger
from femr.featurizers import AgeFeaturizer, CountFeaturizer, FeaturizerList
from femr.labelers import LabeledPatients, load_labeled_patients
from utils import check_file_existence_and_handle_force_refresh

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate count-based featurizations for GBM models (for all tasks at once)")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to directory where features will be saved")
    parser.add_argument("--num_threads", type=int, help="Number of threads to use")
    parser.add_argument("--is_force_refresh", action='store_true', default=False, help="If set, then overwrite all outputs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    NUM_THREADS: int = args.num_threads
    IS_FORCE_REFRESH = args.is_force_refresh
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_LABELS_DIR = args.path_to_labels_dir
    PATH_TO_FEATURES_DIR = args.path_to_features_dir
    PATH_TO_LABELS_FILE: str = os.path.join(PATH_TO_LABELS_DIR, 'all_labels.csv')
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_FEATURES_DIR, 'count_features.pkl')

    # Force refresh
    check_file_existence_and_handle_force_refresh(PATH_TO_OUTPUT_FILE, IS_FORCE_REFRESH)

    # Load consolidated labels across all patients for all tasks
    logger.info(f"Loading LabeledPatients from `{PATH_TO_LABELS_FILE}`")
    labeled_patients: LabeledPatients = load_labeled_patients(PATH_TO_LABELS_FILE)

    # Combine two featurizations of each patient: one for the patient's age, and one for the count of every code
    # they've had in their record up to the prediction timepoint for each label
    age = AgeFeaturizer()
    count = CountFeaturizer(is_ontology_expansion=True)
    featurizer_age_count = FeaturizerList([age, count])

    # Preprocessing the featurizers -- this includes processes such as normalizing age
    logger.info("Start | Preprocess featurizers")
    featurizer_age_count.preprocess_featurizers(PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS)
    logger.info("Finish | Preprocess featurizers")

    # Run actual featurization for each patient
    logger.info("Start | Featurize patients")
    results = featurizer_age_count.featurize(PATH_TO_PATIENT_DATABASE, labeled_patients, NUM_THREADS)
    feature_matrix, patient_ids, label_values, label_times = (
        results[0],
        results[1],
        results[2],
        results[3],
    )
    logger.info("Finish | Featurize patients")

    # Save results
    logger.info(f"Saving results to `{PATH_TO_OUTPUT_FILE}`")
    with open(PATH_TO_OUTPUT_FILE, 'wb') as f:
        pickle.dump(results, f)

    # Logging
    logger.info("FeaturizedPatient stats:\n"
                f"feature_matrix={repr(feature_matrix)}\n"
                f"patient_ids={repr(patient_ids)}\n"
                f"label_values={repr(label_values)}\n"
                f"label_times={repr(label_times)}")
    logger.success("Done!")
    