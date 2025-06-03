import argparse
import os
import json
from tqdm import tqdm
from typing import List
from loguru import logger
from utils import LABELING_FUNCTION_2_PAPER_NAME
import pandas as pd
import random
from multiprocessing import Pool, Manager

from femr.datasets import PatientDatabase
from ehrshot.labelers.core import LabeledPatients, Label
from ehrshot.labelers.ehrshot import (
    Guo_LongLOSLabeler,
    Guo_30DayReadmissionLabeler,
    Guo_ICUAdmissionLabeler,
    PancreaticCancerCodeLabeler,
    CeliacDiseaseCodeLabeler,
    LupusCodeLabeler,
    AcuteMyocardialInfarctionCodeLabeler,
    EssentialHypertensionCodeLabeler,
    HyperlipidemiaCodeLabeler,
    HyponatremiaInstantLabValueLabeler,
    ThrombocytopeniaInstantLabValueLabeler,
    HyperkalemiaInstantLabValueLabeler,
    HypoglycemiaInstantLabValueLabeler,
    AnemiaInstantLabValueLabeler,
    ChexpertLabeler,
)
# from ehrshot.labelers.mimic import (
#     Mimic_MortalityLabeler,
#     Mimic_ReadmissionLabeler
# )

# Function to handle label generation for a subset of patient IDs
def process_patient_ids(pid_subset, labeled_patients, path_to_database, results_dict):
    local_results = {}
    database = PatientDatabase(path_to_database)
    
    for pid in pid_subset:
        random.seed(int(pid))
        labels = labeled_patients.get_labels_from_patient_idx(pid)

        # Filter out labels that occur <= 18 yrs of age
        birthdate = database[pid].events[0].start.year
        keep_label_start_idx = 0
        for l in labels:
            if l.time.year - birthdate < 18:
                keep_label_start_idx += 1
            else:
                break
        labels = labels[keep_label_start_idx:]

        # Randomly sample one label
        if len(labels) == 0:
            local_results[pid] = []
        elif len(labels) == 1:
            local_results[pid] = labels
        else:
            local_results[pid] = [random.choice(labels)]

    # Save local results to the shared dictionary
    results_dict.update(local_results)

# Main function to parallelize the task
def parallel_label_generation(labeled_patients, path_to_database, num_processes: int = 4):
    manager = Manager()
    results_dict = manager.dict()

    # Get all patient IDs and split into subsets for each process
    pids: List[int] = labeled_patients.get_all_patient_ids()
    pid_subsets = [pids[i::num_processes] for i in range(num_processes)]

    with Pool(num_processes) as pool:
        pool.starmap(
            process_patient_ids, 
            [(pid_subset, labeled_patients, path_to_database, results_dict) for pid_subset in pid_subsets]
        )

    results = dict(results_dict)
    return results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate labels for a specific task")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_chexpert_csv", type=str, help="Path to CheXpert CSV file. Specific to CheXpert labeler", default=None,)
    parser.add_argument("--labeling_function", required=True, type=str, help="Name of task for which we are creating labels", choices=LABELING_FUNCTION_2_PAPER_NAME.keys(), )
    parser.add_argument("--is_sample_one_label_per_patient", action="store_true", default=False, help="If specified, only keep one random label per patient")
    parser.add_argument("--num_threads", type=int, help="Number of threads to use", default=1, )
    return parser.parse_args()

def save_labeled_patients_to_csv(labeled_patients: LabeledPatients, path_to_csv: str) -> pd.DataFrame:
    """Converts a LabeledPatient object -> pd.DataFrame and saves as CSV to `path_to_csv`"""
    rows = []
    for patient, labels in labeled_patients.items():
        omop_patient_id = patient # for some reason the pipeline uses the OMOP ID for labelers as the patient ID
        for l in labels:
            rows.append((omop_patient_id, l.time, l.value, labeled_patients.labeler_type))
    df = pd.DataFrame(rows, columns = ['patient_id', 'prediction_time', 'value', 'label_type', ]).sort_values(['patient_id', 'prediction_time', 'value'])
    df.to_csv(path_to_csv, index=False)
    return df

if __name__ == "__main__":
    args = parse_args()
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_LABELS_DIR: str = args.path_to_labels_dir
    NUM_THREADS: int = args.num_threads
    LABELING_FUNCTION: str = args.labeling_function
    is_sample_one_label_per_patient: bool = args.is_sample_one_label_per_patient
    PATH_TO_OUTPUT_DIR: str = os.path.join(PATH_TO_LABELS_DIR, LABELING_FUNCTION)
    PATH_TO_OUTPUT_FILE: str = os.path.join(PATH_TO_OUTPUT_DIR, "labeled_patients.csv")
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # Logging
    path_to_log_file: str = os.path.join(PATH_TO_OUTPUT_DIR, 'info.log')
    if os.path.exists(path_to_log_file):
        os.remove(path_to_log_file)
    logger.add(path_to_log_file, level="INFO")  # connect logger to file
    logger.info(f"Task: {LABELING_FUNCTION}")
    logger.info(f"Loading patient database from: {PATH_TO_PATIENT_DATABASE}")
    logger.info(f"Saving one label per patient? {'Yes' if is_sample_one_label_per_patient else 'No'}")
    logger.info(f"Saving output to: {PATH_TO_OUTPUT_DIR}")
    logger.info(f"# of threads: {NUM_THREADS}")
    with open(os.path.join(PATH_TO_OUTPUT_DIR, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Load PatientDatabase + Ontology
    logger.info(f"Start | Load PatientDatabase")
    database = PatientDatabase(PATH_TO_PATIENT_DATABASE)
    ontology = database.get_ontology()
    logger.info(f"Finish | Load PatientDatabase")

    # Select the appropriate labeling function
    #   EHRSHOT: Guo et al. 2023 tasks
    if LABELING_FUNCTION == "guo_los":
        labeler = Guo_LongLOSLabeler(ontology)
    elif LABELING_FUNCTION == "guo_readmission":
        labeler = Guo_30DayReadmissionLabeler(ontology)
    elif LABELING_FUNCTION == "guo_icu":
        labeler = Guo_ICUAdmissionLabeler(ontology)
    #   EHRSHOT: New diagnosis tasks
    elif LABELING_FUNCTION == "new_pancan":
        labeler = PancreaticCancerCodeLabeler(ontology)
    elif LABELING_FUNCTION == 'new_celiac':
        labeler = CeliacDiseaseCodeLabeler(ontology)
    elif LABELING_FUNCTION == 'new_lupus':
        labeler = LupusCodeLabeler(ontology)
    elif LABELING_FUNCTION == 'new_acutemi':
        labeler = AcuteMyocardialInfarctionCodeLabeler(ontology)
    elif LABELING_FUNCTION == 'new_hypertension':
        labeler = EssentialHypertensionCodeLabeler(ontology)
    elif LABELING_FUNCTION == 'new_hyperlipidemia':
        labeler = HyperlipidemiaCodeLabeler(ontology)
    #   EHRSHOT: Lab values
    elif LABELING_FUNCTION == "lab_thrombocytopenia":
        labeler = ThrombocytopeniaInstantLabValueLabeler(ontology)
    elif LABELING_FUNCTION == "lab_hyperkalemia":
        labeler = HyperkalemiaInstantLabValueLabeler(ontology)
    elif LABELING_FUNCTION == "lab_hypoglycemia":
        labeler = HypoglycemiaInstantLabValueLabeler(ontology)
    elif LABELING_FUNCTION == "lab_hyponatremia":
        labeler = HyponatremiaInstantLabValueLabeler(ontology)
    elif LABELING_FUNCTION == "lab_anemia":
        labeler = AnemiaInstantLabValueLabeler(ontology)
    #   EHRSHOT: Radiology
    elif LABELING_FUNCTION == "chexpert":
        assert args.path_to_chexpert_csv is not None, f"The argument --path_to_chexpert_csv must be specified"
        labeler = ChexpertLabeler(args.path_to_chexpert_csv)
    #   MIMIC-IV specific labelers
    # elif LABELING_FUNCTION == "mimic4_los":
    #     labeler = Guo_LongLOSLabeler(ontology)
    # elif LABELING_FUNCTION == "mimic4_readmission":
    #     labeler = Mimic_ReadmissionLabeler(ontology)
    # elif LABELING_FUNCTION == "mimic4_mortality":
    #     labeler = Mimic_MortalityLabeler(ontology)
    else:
        raise ValueError(
            f"Labeling function `{LABELING_FUNCTION}` not supported. Must be one of: {LABELING_FUNCTION_2_PAPER_NAME.keys()}."
        )

    logger.info("Start | Label patients")
    if LABELING_FUNCTION.startswith('mimic4_'):
        
        # NOTE: For MIMIC-IV labelers, we need to overwrite the EHRSHOT definition of inpatient Visits to include "Visit/ERIP"
        import unittest
        def mimic_4_get_inpatient_admission_concepts() -> List[str]:
            return ["Visit/IP", "Visit/ERIP"]
        
        # Overwrite `get_inpatient_admission_concepts()`
        with unittest.mock.patch('labelers.omop.get_inpatient_admission_concepts', mimic_4_get_inpatient_admission_concepts):
            if NUM_THREADS != 1:
                logger.info(f"unittest.mock.patch only works with NUM_THREADS=1 (not {NUM_THREADS}), so downgrading to 1 thread for this labeler")
                NUM_THREADS = 1
            
            # Run labeler
            labeled_patients = labeler.apply(
                path_to_patient_database=PATH_TO_PATIENT_DATABASE,
                num_threads=NUM_THREADS,
            )
            assert labeled_patients.labeler_type == 'boolean'
    else:
        # Run labeler
        labeled_patients = labeler.apply(
            path_to_patient_database=PATH_TO_PATIENT_DATABASE,
            num_threads=NUM_THREADS,
        )
    logger.info("Finish | Label patients")
    
    # Randomly sample (if applicable)
    if is_sample_one_label_per_patient:
        pids: List[int] = labeled_patients.get_all_patient_ids()
        results = parallel_label_generation(labeled_patients, PATH_TO_PATIENT_DATABASE, num_processes=20)
        for pid, labels in results.items():
            labeled_patients.patients_to_labels[pid] = labels
        assert all([ len(labeled_patients.get_labels_from_patient_idx(x)) <= 1 for x in pids ]), f"Found a patient with != 1 label"

    # Force labels to be minute-level resolution for FEMR compatibility
    for patient, labels in labeled_patients.items():
        new_labels: List[Label] = [ Label(time=l.time.replace(second=0, microsecond=0), value=l.value) for l in labels ]
        labeled_patients[patient] = new_labels

    # Save labeled patients to simple CSV pipeline format
    logger.info(f"Saving labeled patients to `{PATH_TO_OUTPUT_FILE}`")
    save_labeled_patients_to_csv(labeled_patients, PATH_TO_OUTPUT_FILE)
    
    # Logging
    label_values = labeled_patients.as_numpy_arrays()[1].sum() if labeled_patients.labeler_type == 'boolean' else (labeled_patients.as_numpy_arrays()[1] != 0).sum()
    logger.info("LabeledPatient stats:\n"
                f"Total # of patients = {labeled_patients.get_num_patients(is_include_empty_labels=True)}\n"
                f"Total # of patients with at least one label = {labeled_patients.get_num_patients(is_include_empty_labels=False)}\n"
                f"Total # of labels = {labeled_patients.get_num_labels()}\n"
                f"Total # of positive labels = {label_values}")
    logger.success("Done!")


