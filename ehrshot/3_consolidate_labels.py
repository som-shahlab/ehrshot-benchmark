import argparse
import datetime
import os
from typing import Dict, List, Set
from loguru import logger
import collections

from femr.labelers import load_labeled_patients, LabeledPatients, Label

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate all labels to speed up featurization")
    parser.add_argument("--path_to_labels_and_feats_dir", required=True, type=str, help="Path to save labeles and featurizers")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to save labeles and featurizers")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    PATH_TO_LABELS_AND_FEATS_DIR: str = args.path_to_labels_and_feats_dir

    # Get all labels currently in `PATH_TO_LABELS_AND_FEATS_DIR`
    labeling_functions: List[str] = os.listdir(PATH_TO_LABELS_AND_FEATS_DIR)

    # Merge all labels into a single file, saving only their prediction times
    # so that we can later generate features for all of them at once.
    logger.info("Start | Consolidate patients")
    patient_2_label_times: Dict[int, Set[datetime.datetime]] = collections.defaultdict(set)
    for lf in labeling_functions:
        labeled_patients = load_labeled_patients(os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, 
                                                                  lf, 
                                                                  'labeled_patients.csv'))
        for patient_id, labels in labeled_patients.items():
            for label in labels:
                patient_2_label_times[patient_id].add(label.time)
    logger.info("Finish | Consolidate patients")

    # Resort all labels to be in chronological order
    logger.info("Start | Resort labels chronologically")
    patient_2_labels: Dict[int, List[Label]] = {}
    for patient_id, timestamps in patient_2_label_times.items():
        patient_2_labels[patient_id] = sorted([
            Label(time=timestamp, value=None) 
            for timestamp in timestamps
        ], key=lambda x: x.time)
    logger.info("Finish | Resort labels chronologically")

    # Save as LabeledPatient object to `PATH_TO_LABELS_AND_FEATS_DIR`
    labeled_patients: LabeledPatients = LabeledPatients(patient_2_labels, "empty")
    labeled_patients.save(os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, 'all_labels.csv'))
    logger.info(f"Saved new LabeledPatients object to: {PATH_TO_LABELS_AND_FEATS_DIR}")
    