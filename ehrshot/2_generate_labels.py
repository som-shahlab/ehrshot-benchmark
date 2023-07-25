import argparse
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
    parser = argparse.ArgumentParser(description="Run femr featurization")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to femr database")
    parser.add_argument("--path_to_output_dir", required=True, type=str, help="Path to save labeles and featurizers")
    parser.add_argument("--labeling_function", required=True, type=str, help="Name of labeling function to create.", choices=LABELING_FUNCTIONS, )
    parser.add_argument("--num_threads", type=int, help="The number of threads to use", default=1, )
    parser.add_argument("--max_labels_per_patient", type=int, help="Max number of labels to keep per patient (excess labels are randomly discarded)", default=None, )
    parser.add_argument("--path_to_chexpert_csv", type=str, help="Path to chexpert labeled csv file. Specific to chexpert labeler", default=None,)
    parser.add_argument("--is_skip_label", action="store_true", help="If specified, skip labeling step", default=False)

    args = parser.parse_args()
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_OUTPUT_DIR = args.path_to_output_dir
    NUM_THREADS: int = args.num_threads
    MAX_LABELS_PER_PATIENT: int = args.max_labels_per_patient

    PATH_TO_OUTPUT_DIR = os.path.join(PATH_TO_OUTPUT_DIR, args.labeling_function)

    # Logging
    path_to_log_file: str = os.path.join(PATH_TO_OUTPUT_DIR, 'info.log')
    if os.path.exists(path_to_log_file):
        os.remove(path_to_log_file)
    logger.add(path_to_log_file, level="INFO")  # connect logger to file
    logger.info(f"Labeling function: {args.labeling_function}")
    logger.info(f"Loading patient database from: {PATH_TO_PATIENT_DATABASE}")
    logger.info(f"Saving output to: {PATH_TO_OUTPUT_DIR}")
    logger.info(f"Max # of labels per patient: {MAX_LABELS_PER_PATIENT}")
    logger.info(f"# of threads: {NUM_THREADS}")
    with open(os.path.join(PATH_TO_OUTPUT_DIR, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # create directories to save files
    PATH_TO_SAVE_LABELED_PATIENTS: str = os.path.join(PATH_TO_OUTPUT_DIR, "labeled_patients.csv")
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # Load PatientDatabase + Ontology
    logger.info(f"Start | Load PatientDatabase")
    database = PatientDatabase(PATH_TO_PATIENT_DATABASE)
    ontology = database.get_ontology()
    logger.info(f"Finish | Load PatientDatabase")

# Define the labeling function.
    # Hartutyunyan et al. 2019 (MIMIC-3 benchmark tasks)
    if args.labeling_function == "hartutyunyan_mortality":
         # GOOD
        labeler = Harutyunyan_MortalityLabeler(ontology)
    elif args.labeling_function == "hartutyunyan_decompensation":
         # GOOD
        labeler = Harutyunyan_DecompensationLabeler(ontology)
    elif args.labeling_function == "hartutyunyan_los":
         # GOOD
        labeler = Harutyunyan_LengthOfStayLabeler(ontology)
    # Guo et al. 2023 tasks (CLMBR tasks)
    elif args.labeling_function == "guo_los":
        labeler = Guo_LongLOSLabeler(ontology)
    elif args.labeling_function == "guo_readmission":
        labeler = Guo_30DayReadmissionLabeler(ontology)
    elif args.labeling_function == "guo_icu":
        labeler = Guo_ICUAdmissionLabeler(ontology)
    # van Uden et al. (Fewshot tasks)
    elif args.labeling_function == "uden_pancan":
        labeler = PancreaticCancerCodeLabeler(ontology)
    elif args.labeling_function == 'uden_celiac':
        labeler = CeliacDiseaseCodeLabeler(ontology)
    elif args.labeling_function == 'uden_lupus':
        labeler = LupusCodeLabeler(ontology)
    elif args.labeling_function == 'uden_acutemi':
        labeler = AcuteMyocardialInfarctionCodeLabeler(ontology)
    elif args.labeling_function == 'uden_cteph':
        labeler = CTEPHCodeLabeler(ontology)
    elif args.labeling_function == 'uden_hypertension':
        labeler = EssentialHypertensionCodeLabeler(ontology)
    elif args.labeling_function == 'uden_hyperlipidemia':
        labeler = HyperlipidemiaCodeLabeler(ontology)
    # Lab values
    elif args.labeling_function == "thrombocytopenia_lab":
         # GOOD
        labeler = ThrombocytopeniaInstantLabValueLabeler(ontology)
    elif args.labeling_function == "hyperkalemia_lab":
         # GOOD
        labeler = HyperkalemiaInstantLabValueLabeler(ontology)
    elif args.labeling_function == "hypoglycemia_lab":
         # GOOD
        labeler = HypoglycemiaInstantLabValueLabeler(ontology)
    elif args.labeling_function == "hyponatremia_lab":
         # GOOD
        labeler = HyponatremiaInstantLabValueLabeler(ontology)
    elif args.labeling_function == "anemia_lab":
         # GOOD
        labeler = AnemiaInstantLabValueLabeler(ontology)
    # Custom
    elif args.labeling_function == "12_month_mortality":
        # GOOD
        labeler = MortalityCodeLabeler(ontology, 
                                       TimeHorizon(datetime.timedelta(days=0), datetime.timedelta(days=365)), 
                                       prediction_codes=femr.labelers.omop.get_inpatient_admission_codes(ontology),
                                       prediction_time_adjustment_func=lambda x: x)
    elif args.labeling_function == "chexpert":
        assert args.path_to_chexpert_csv is not None, f"path_to_chexpert_csv cannot be {args.path_to_chexpert_csv}"
        labeler = ChexpertLabeler(args.path_to_chexpert_csv)
    else:
        raise ValueError(
            f"Labeling function `{args.labeling_function}` not supported. Must be one of: {LABELING_FUNCTIONS}."
        )

    # Determine how many labels to keep per patient
    if args.max_labels_per_patient is not None and args.labeling_function != "chexpert":
        labeler = NLabelsPerPatientLabeler(labeler, seed=0, num_labels=MAX_LABELS_PER_PATIENT)

    if args.is_skip_label:
        logger.critical(f"Skipping labeling step. Loading labeled patients from @ {PATH_TO_SAVE_LABELED_PATIENTS}")
        labeled_patients = femr.labelers.core.load_labeled_patients(PATH_TO_SAVE_LABELED_PATIENTS)
    else:
        logger.info("Start | Label patients")
        if args.labeling_function != "chexpert":
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
        logger.info("LabeledPatient stats:\n"
                    f"Total # of patients = {labeled_patients.get_num_patients(is_include_empty_labels=True)}\n"
                    f"Total # of patients with at least one label = {labeled_patients.get_num_patients(is_include_empty_labels=False)}\n"
                    f"Total # of labels = {labeled_patients.get_num_labels()}")

        # Save labeled patients to simple CSV pipeline format
        save_labeled_patients_to_csv(labeled_patients, PATH_TO_SAVE_LABELED_PATIENTS)
    
    with open(os.path.join(PATH_TO_OUTPUT_DIR, "done.txt"), "w") as f:
        f.write("done")
    
    logger.info("Done!")


