import argparse
import datetime
import os
import json
from loguru import logger
import collections

import femr.labelers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate all labels to speed up featurization")
    parser.add_argument("--path_to_output_dir", required=True, type=str, help="Path to save labeles and featurizers")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to save labeles and featurizers")

    args = parser.parse_args()

    label_names = os.listdir(args.path_to_output_dir)

    all_labels = collections.defaultdict(set)

    num_initial_patients = 0

    for label_name in label_names:
        labels = femr.labelers.load_labeled_patients(os.path.join(args.path_to_output_dir, label_name, 'labeled_patients.csv'))
        for patient_id, ls in labels.items():
            num_initial_patients += 1
            for label in ls:
                all_labels[patient_id].add(label.time)

    label_dict = {}

    for patient_id, timestamps in all_labels.items():
        label_dict[patient_id] = sorted([femr.labelers.Label(time=timestamp, value=None) for timestamp in timestamps], key=lambda a:a.time)

    num_final_patients = len(label_dict)

    logger.info(f"Went from {num_initial_patients} to {num_final_patients}")
    
    labeled_patients = femr.labelers.LabeledPatients(label_dict, "empty")

    labeled_patients.save(os.path.join(args.path_to_features_dir, 'all_labels.csv'))