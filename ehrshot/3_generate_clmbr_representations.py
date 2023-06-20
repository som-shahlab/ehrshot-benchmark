import argparse
import os
from typing import Optional
from loguru import logger
import shutil
from utils import save_data, load_data, sort_tuples, LABELING_FUNCTIONS
import femr.datasets

def main(args):
    CLMBR_DATA_PATH: str = args.path_to_clmbr_data
    PATH_TO_PATIENT_DATABASE: str = args.path_to_database
    PATH_TO_DATA: str = args.path_to_labeled_featurized_data
    TASK_PATH: str = args.path_to_save
    PATH_TO_EMBEDDINGS_CONSTDB: str = args.path_to_embeddings_constdb

    MODEL_PATH: str = os.path.join(CLMBR_DATA_PATH, "clmbr_model")
    DICTIONARY_PATH: str = os.path.join(CLMBR_DATA_PATH, "dictionary")
    
    labeling_function: str = args.labeling_function

    # Logistics
    cuda_visible_devices: Optional[str] = args.cuda_visible_devices
    is_force_refresh: bool = args.is_force_refresh

    TASKS_PATH: str = os.path.join(TASK_PATH, f"{args.labeling_function}")
    
    # Setup logging
    path_to_log_file: str = os.path.join(TASKS_PATH, "info.log")
    if os.path.exists(path_to_log_file):
        os.remove(path_to_log_file)
    logger.add(path_to_log_file)
    logger.info(f"Logging to: {path_to_log_file}")
    logger.info(f"Args: {args}")
    
    database = femr.datasets.PatientDatabase(PATH_TO_PATIENT_DATABASE)
    logger.info(f"Total number of patients in database: {len(database)}")

    PATH_TO_LABELED_PATIENTS = os.path.join(PATH_TO_DATA, f"{labeling_function}/labeled_patients.csv")

    # Get LabeledPatients
    logger.info(f"Path to LabeledPatients: {PATH_TO_LABELED_PATIENTS}")
    logger.info(f"Path to PatientDatabase: {PATH_TO_PATIENT_DATABASE}")
    logger.info(f"Path to ConstDB Note Embeddings: {PATH_TO_EMBEDDINGS_CONSTDB}")

    # Force refresh
    if is_force_refresh:
        logger.critical(f"Force refresh is set to True. Deleting directory @ {TASKS_PATH}")
        shutil.rmtree(TASKS_PATH, ignore_errors=True)
    os.makedirs(TASKS_PATH, exist_ok=True)
    logger.info(f"Saving tasks to {TASKS_PATH}")

    # Generate batches for tasks
    PATH_TO_TASK_BATCHES = os.path.join(TASKS_PATH, f"task_batches")
    if not os.path.exists(PATH_TO_TASK_BATCHES):
        command: str = (
            f"clmbr_create_batches {PATH_TO_TASK_BATCHES}"
            f" --data_path {PATH_TO_PATIENT_DATABASE}"
            f" --dictionary {DICTIONARY_PATH}"
            f" --task labeled_patients"
            f" --labeled_patients_path {PATH_TO_LABELED_PATIENTS}" +
            (f" --note_embedding_data {PATH_TO_EMBEDDINGS_CONSTDB}" if args.is_include_note is True else "")
        )
        logger.info(f"Creating batches @ {PATH_TO_TASK_BATCHES}")
        logger.critical(f"Command run:\n{command}")
        assert 0 == os.system(command)
        logger.success(f"Done creating batches @ {PATH_TO_TASK_BATCHES}")
    else:
        logger.critical(f"Skipping clmbr_create_batches because batches already exist @ {PATH_TO_TASK_BATCHES}")
        
    # Generate patient representations for tasks
    REPRESENTATIONS = os.path.join(TASKS_PATH, f"clmbr_reprs")
    if not os.path.exists(REPRESENTATIONS):
        command: str = (
            (f"export CUDA_VISIBLE_DEVICES={cuda_visible_devices} && " if cuda_visible_devices else '') +
            f"clmbr_compute_representations {REPRESENTATIONS}"
            f" --data_path {PATH_TO_PATIENT_DATABASE}"
            f" --batches_path {PATH_TO_TASK_BATCHES}"
            f" --model_dir {MODEL_PATH}"
        )
        logger.info(f"Creating representations @ {REPRESENTATIONS}")
        logger.critical(f"Command run:\n{command}")
        assert 0 == os.system(command)
        logger.success(f"Done creating representations @ {REPRESENTATIONS}")
    else:
        logger.critical(f"Skipping clmbr_compute_representations because reprs already exist @ {REPRESENTATIONS}")
        
    # Logging
    reps = load_data(REPRESENTATIONS)

    # Sorting the reps for determinism
    patient_ids_labeling_time = [(pid, time) for pid, time in zip(reps["patient_ids"], reps["labeling_time"])]
    idx = [i for i in range(len(patient_ids_labeling_time))]
    _, sort_idx = sort_tuples(patient_ids_labeling_time, idx)
    reps["data_matrix"] = reps["data_matrix"][sort_idx]
    reps["patient_ids"] = reps["patient_ids"][sort_idx]
    reps["labeling_time"] = reps["labeling_time"][sort_idx]
    reps["label_values"] = reps["label_values"][sort_idx]

    save_data(reps, REPRESENTATIONS)

    logger.info(reps.keys())
    logger.info("Pulling the data for the first label")
    logger.info("Patient id", reps["patient_ids"][:10])
    logger.info("Label time", reps["labeling_time"][:10])
    logger.info("Representation", reps["data_matrix"][0, :16])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLMBR patient representations")
    parser.add_argument("--path_to_clmbr_data", type=str, help=( "Path to save files CLMBR training data" " Example: '/local-scratch/nigam/projects/clmbr_text_assets/data/clmbr_data'" ), )
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to femr database")
    parser.add_argument("--path_to_labeled_featurized_data", type=str, help=( "This is the path to the labeled featurized parent folder" ), )
    parser.add_argument("--path_to_save", type=str, help=( "Path to save the tasks representations" ), )
    parser.add_argument("--path_to_embeddings_constdb", type=str, default=None, help=( "Path to text embeddings constdb for clmbr_text model" ), )
    parser.add_argument("--labeling_function", required=True, type=str, help="Name of labeling function to create.", choices=LABELING_FUNCTIONS, )

    # Logistics
    parser.add_argument("--cuda_visible_devices", type=str, default=None, help="Set 'CUDA_VISIBLE_DEVICES'. Specify as comma separated list of GPUs, or don't specify to use all GPUs.")
    parser.add_argument("--is_force_refresh", action='store_true', default=False, help="If specified, delete all existing data and re-run everything from scratch.")
    parser.add_argument("--is_include_note", action='store_true', default=False, help="Include notes or not")
    
    args = parser.parse_args()
    main(args)
