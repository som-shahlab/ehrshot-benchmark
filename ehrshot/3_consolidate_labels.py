import argparse
import datetime
import os
from typing import Dict, List, Set, Tuple
from loguru import logger
import collections
import csv
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
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_LABELS_DIR, 'all_labels_out.csv')
    # Also write out labels by task for models that might distinguish labels by task (e.g. instructed LLM)
    PATH_TO_OUTPUT_TASKS_FILE: str = os.path.join(PATH_TO_LABELS_DIR, 'all_labels_tasks_out.csv')
    
    # Force refresh
    check_file_existence_and_handle_force_refresh(PATH_TO_OUTPUT_FILE, IS_FORCE_REFRESH)
    check_file_existence_and_handle_force_refresh(PATH_TO_OUTPUT_TASKS_FILE, IS_FORCE_REFRESH)

    # Get all labels currently in `PATH_TO_LABELS_DIR`
    labeling_functions: List[str] = os.listdir(PATH_TO_LABELS_DIR)
    # TODO: For faster testing only include tasks 'new_*' and 'guo_*'
    labeling_functions = [x for x in labeling_functions if x.startswith('new_') or x.startswith('guo_')]
    logger.info(f"Found {len(labeling_functions)} labeling functions to merge: {labeling_functions}")

    # Merge all predictions times for all labels across all tasks into a single file,
    # and track the task (labeling function) used for each label.
    logger.info("Start | Consolidate patients")
    patient_2_label_times: Dict[int, Set[datetime.datetime]] = collections.defaultdict(set)
    # Labels are reduced to prediction times, which leads to reduction of 1,178,654 to 406,379
    # Keep them separate for each task for task specific processing
    patient_2_label_times_tasks: Dict[int, Set[Tuple[datetime.datetime, str]]] = collections.defaultdict(set)

    for lf in labeling_functions:
        labeled_patients: LabeledPatients = load_labeled_patients(os.path.join(PATH_TO_LABELS_DIR, lf, 'labeled_patients.csv'))
        for patient_id, labels in labeled_patients.items():
            for label in labels:
                patient_2_label_times[patient_id].add(label.time)
                patient_2_label_times_tasks[patient_id].add((label.time, lf))
    logger.info("Finish | Consolidate patients")

    # Resort all labels to be in chronological order AND force minute-level time resolution
    logger.info("Start | Resort labels chronologically")
    patient_2_merged_labels: Dict[int, List[Label]] = {}
    for patient_id, timestamps in patient_2_label_times.items():
        patient_2_merged_labels[patient_id] = [
            Label(time=timestamp, value=False) 
            for timestamp in sorted(timestamps)
        ]
    all_labels_tasks = []
    for patient_id, time_task_pairs in sorted(patient_2_label_times_tasks.items()):
        sorted_time_task_pairs = sorted(time_task_pairs, key=lambda x: x[0])
        # Collect task information in the same order as the labels
        all_labels_tasks.extend([(patient_id, timestamp, task) for timestamp, task in sorted_time_task_pairs])

    logger.info("Finish | Resort labels chronologically")

    # Save as LabeledPatient object to `PATH_TO_LABELS_DIR`
    labeled_patients: LabeledPatients = LabeledPatients(patient_2_merged_labels, "boolean")
    labeled_patients.save(PATH_TO_OUTPUT_FILE)
    logger.info(f"Saved new LabeledPatients object to: `{PATH_TO_OUTPUT_FILE}`")

    # Save the task information to `all_labels_tasks.csv`
    logger.info("Start | Save task information")
    with open(PATH_TO_OUTPUT_TASKS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["patient_id", "prediction_time", "label_type", "value", "task"])  # Header
        for patient_id, timestamp, task in all_labels_tasks:
            writer.writerow([patient_id, timestamp, 'boolean', 'False', task])
    logger.info(f"Saved task information to: `{PATH_TO_OUTPUT_TASKS_FILE}`")

    logger.success("Done!")
