import argparse
import os
import json
from loguru import logger
from utils import LABELING_FUNCTIONS
import pandas as pd

from femr.datasets import PatientDatabase
from femr.labelers.core import LabeledPatients
from femr.labelers.omop import (
    ChexpertLabeler,
)
from femr.labelers.benchmarks import (
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
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate labels for a specific task")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_chexpert_csv", type=str, help="Path to CheXpert CSV file. Specific to CheXpert labeler", default=None,)
    parser.add_argument("--labeling_function", required=True, type=str, help="Name of task for which we are creating labels", choices=LABELING_FUNCTIONS, )
    parser.add_argument("--num_threads", type=int, help="Number of threads to use", default=1, )
    return parser.parse_args()

def save_labeled_patients_to_csv(labeled_patients: LabeledPatients, path_to_csv: str) -> pd.DataFrame:
    """Converts a LabeledPatient object -> pd.DataFrame and saves as CSV to `path_to_csv`"""
    rows = []
    for patient, labels in labeled_patients.items():
        omop_patient_id = patient # for some reason the pipeline uses the OMOP ID for labelers as the patient ID
        for l in labels:
            rows.append((omop_patient_id, l.time, l.value, labeled_patients.labeler_type))
    df = pd.DataFrame(rows, columns = ['patient_id', 'prediction_time', 'value', 'label_type', ])
    df.to_csv(path_to_csv, index=False)
    return df

if __name__ == "__main__":
    args = parse_args()
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_LABELS_DIR: str = args.path_to_labels_dir
    NUM_THREADS: int = args.num_threads
    LABELING_FUNCTION: str = args.labeling_function
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
    #   Guo et al. 2023 tasks (CLMBR tasks)
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
    else:
        raise ValueError(
            f"Labeling function `{LABELING_FUNCTION}` not supported. Must be one of: {LABELING_FUNCTIONS}."
        )

    logger.info("Start | Label patients")
    labeled_patients = labeler.apply(
        path_to_patient_database=PATH_TO_PATIENT_DATABASE,
        num_threads=NUM_THREADS
    )
    logger.info("Finish | Label patients")

    # Save labeled patients to simple CSV pipeline format
    logger.info(f"Saving labeled patients to `{PATH_TO_OUTPUT_FILE}`")
    save_labeled_patients_to_csv(labeled_patients, PATH_TO_OUTPUT_FILE)
    
    # Logging
    logger.info("LabeledPatient stats:\n"
                f"Total # of patients = {labeled_patients.get_num_patients(is_include_empty_labels=True)}\n"
                f"Total # of patients with at least one label = {labeled_patients.get_num_patients(is_include_empty_labels=False)}\n"
                f"Total # of labels = {labeled_patients.get_num_labels()}")
    logger.success("Done!")


