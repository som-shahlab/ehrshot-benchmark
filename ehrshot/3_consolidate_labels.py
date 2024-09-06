import argparse
import datetime
import os
from typing import Dict, List, Set
from loguru import logger
import collections
from utils import check_file_existence_and_handle_force_refresh

from femr.labelers import load_labeled_patients, LabeledPatients, Label

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate all labels to speed up featurization")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--is_force_refresh", action='store_true', default=False, help="If set, then overwrite all outputs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    PATH_TO_LABELS_DIR: str = args.path_to_labels_dir
    IS_FORCE_REFRESH: bool = args.is_force_refresh
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_LABELS_DIR, 'all_labels.csv')
    
    # Force refresh
    check_file_existence_and_handle_force_refresh(PATH_TO_OUTPUT_FILE, IS_FORCE_REFRESH)

    # Get all labels currently in `PATH_TO_LABELS_DIR`, filtering out non-labeling directories
    labeling_functions: List[str] = [
        lf for lf in os.listdir(PATH_TO_LABELS_DIR) 
        if os.path.isdir(os.path.join(PATH_TO_LABELS_DIR, lf)) and 
        os.path.exists(os.path.join(PATH_TO_LABELS_DIR, lf, 'labeled_patients.csv'))
    ]
    logger.info(f"Found {len(labeling_functions)} labeling functions to merge: {labeling_functions}")
    
    # Merge all predictions times for all labels across all tasks into a single file,
    # so that we can later generate features for all of them at once.
    logger.info("Start | Consolidate patients")
    patient_2_label_times: Dict[int, Set[datetime.datetime]] = collections.defaultdict(set)
    for lf in labeling_functions:
        labeled_patients: LabeledPatients = load_labeled_patients(os.path.join(PATH_TO_LABELS_DIR, lf, 'labeled_patients.csv'))
        for patient_id, labels in labeled_patients.items():
            for label in labels:
                patient_2_label_times[patient_id].add(label.time)
    logger.info("Finish | Consolidate patients")

    # Resort all labels to be in chronological order AND force minute-level time resolution
    logger.info("Start | Resort labels chronologically")
    patient_2_merged_labels: Dict[int, List[Label]] = {}
    for patient_id, timestamps in patient_2_label_times.items():
        patient_2_merged_labels[patient_id] = [
            Label(time=timestamp, value=False) 
            for timestamp in sorted(timestamps)
        ]
    logger.info("Finish | Resort labels chronologically")

    # Save as LabeledPatient object to `PATH_TO_LABELS_DIR`
    labeled_patients: LabeledPatients = LabeledPatients(patient_2_merged_labels, "boolean")
    labeled_patients.save(PATH_TO_OUTPUT_FILE)
    logger.info(f"Saved new LabeledPatients object to: `{PATH_TO_OUTPUT_FILE}`")
    logger.success("Done!")
    