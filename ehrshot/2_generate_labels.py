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
    Harutyunyan_LengthOfStayLabeler,
    Harutyunyan_MortalityLabeler, 
    Harutyunyan_DecompensationLabeler,
    Guo_LongLOSLabeler,
    Guo_30DayReadmissionLabeler,
    Guo_ICUAdmissionLabeler,
    PancreaticCancerCodeLabeler,
    CeliacDiseaseCodeLabeler,
    LupusCodeLabeler,
    AcuteMyocardialInfarctionCodeLabeler,
    CTEPHCodeLabeler,
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
    parser.add_argument("--path_to_labels_and_feats_dir", required=True, type=str, help="Path to save labels and featurizers")
    parser.add_argument("--labeling_function", required=True, type=str, help="Name of task for which we are creating labels", choices=LABELING_FUNCTIONS, )
    parser.add_argument("--num_threads", type=int, help="Number of threads to use", default=1, )
    parser.add_argument("--max_labels_per_patient", type=int, help="Max number of labels to keep per patient (excess labels are randomly discarded)", default=None, )
    parser.add_argument("--path_to_chexpert_csv", type=str, help="Path to CheXpert CSV file. Specific to CheXpert labeler", default=None,)
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
    NUM_THREADS: int = args.num_threads
    MAX_LABELS_PER_PATIENT: int = args.max_labels_per_patient
    LABELING_FUNCTION: str = args.labeling_function
    PATH_TO_LABELS_AND_FEATS_DIR = os.path.join(args.path_to_labels_and_feats_dir, LABELING_FUNCTION)
    PATH_TO_SAVE_LABELED_PATIENTS: str = os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, "labeled_patients.csv")
    os.makedirs(PATH_TO_LABELS_AND_FEATS_DIR, exist_ok=True)

    # Logging
    path_to_log_file: str = os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, 'info.log')
    if os.path.exists(path_to_log_file):
        os.remove(path_to_log_file)
    logger.add(path_to_log_file, level="INFO")  # connect logger to file
    logger.info(f"Task: {LABELING_FUNCTION}")
    logger.info(f"Loading patient database from: {PATH_TO_PATIENT_DATABASE}")
    logger.info(f"Saving output to: {PATH_TO_LABELS_AND_FEATS_DIR}")
    logger.info(f"Max # of labels per patient: {MAX_LABELS_PER_PATIENT}")
    logger.info(f"# of threads: {NUM_THREADS}")
    with open(os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Load PatientDatabase + Ontology
    logger.info(f"Start | Load PatientDatabase")
    database = PatientDatabase(PATH_TO_PATIENT_DATABASE)
    ontology = database.get_ontology()
    logger.info(f"Finish | Load PatientDatabase")

    # Select the appropriate labeling function
    #    Hartutyunyan et al. 2019 (MIMIC-3 benchmark tasks)
    if LABELING_FUNCTION == "hartutyunyan_mortality":
        labeler = Harutyunyan_MortalityLabeler(ontology)
    elif LABELING_FUNCTION == "hartutyunyan_decompensation":
        labeler = Harutyunyan_DecompensationLabeler(ontology)
    elif LABELING_FUNCTION == "hartutyunyan_los":
        labeler = Harutyunyan_LengthOfStayLabeler(ontology)
    #   Guo et al. 2023 tasks (CLMBR tasks)
    elif LABELING_FUNCTION == "guo_los":
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
    elif LABELING_FUNCTION == 'new_cteph':
        labeler = CTEPHCodeLabeler(ontology)
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
        assert args.path_to_chexpert_csv is not None, f"path_to_chexpert_csv cannot be {args.path_to_chexpert_csv}"
        labeler = ChexpertLabeler(args.path_to_chexpert_csv)
    else:
        raise ValueError(
            f"Labeling function `{LABELING_FUNCTION}` not supported. Must be one of: {LABELING_FUNCTIONS}."
        )

    # Determine how many labels to keep per patient
    if MAX_LABELS_PER_PATIENT is not None and LABELING_FUNCTION != "chexpert":
        labeler = NLabelsPerPatientLabeler(labeler, seed=0, num_labels=MAX_LABELS_PER_PATIENT)

    logger.info("Start | Label patients")
    if LABELING_FUNCTION != "chexpert":
        labeled_patients = labeler.apply(
            path_to_patient_database=PATH_TO_PATIENT_DATABASE,
            num_threads=NUM_THREADS
        )
    else:
        labeled_patients = labeler.apply(
            path_to_patient_database=PATH_TO_PATIENT_DATABASE,
            num_threads=NUM_THREADS,
            num_labels=MAX_LABELS_PER_PATIENT,
        )
    logger.info("Finish | Label patients")

    # Save labeled patients to simple CSV pipeline format
    save_labeled_patients_to_csv(labeled_patients, PATH_TO_SAVE_LABELED_PATIENTS)
    
    # Logging
    logger.info("LabeledPatient stats:\n"
                f"Total # of patients = {labeled_patients.get_num_patients(is_include_empty_labels=True)}\n"
                f"Total # of patients with at least one label = {labeled_patients.get_num_patients(is_include_empty_labels=False)}\n"
                f"Total # of labels = {labeled_patients.get_num_labels()}")
    with open(os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, "done.txt"), "w") as f:
        f.write("done")
    logger.info("Done!")


