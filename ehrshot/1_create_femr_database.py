import os
import argparse
import femr.datasets
from loguru import logger
from utils import check_file_existence_and_handle_force_refresh

def delete_files_not_starting_with_csv(folder_path):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is a regular file (not a folder)
        if os.path.isfile(file_path):
            # Check if the file does not start with ".csv"
            if not filename.endswith('.csv'):
                # Delete the file
                os.remove(file_path)
                logger.info(f"Deleted file: {filename}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create FEMR patient database from EHRSHOT raw CSVs")
    parser.add_argument("--path_to_input_dir", required=True, type=str, help="Path to folder containing all EHRSHOT cohort CSVs")
    parser.add_argument("--path_to_output_dir", required=True, type=str, help="Path to save FEMR patient database")
    parser.add_argument("--path_to_athena_download", type=str, help="Path to where your Athena download folder is located (which contains your ontologies)")
    parser.add_argument("--num_threads", type=int, help="Number of threads to use")
    parser.add_argument("--is_force_refresh", action='store_true', default=False, help="If set, then overwrite all outputs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    PATH_TO_INPUT_DIR: str = args.path_to_input_dir
    PATH_TO_OUTPUT_DIR: str = args.path_to_output_dir
    PATH_TO_ATHENA_DOWNLOAD: str = args.path_to_athena_download
    NUM_THREADS: int = args.num_threads
    IS_FORCE_REFRESH: bool = args.is_force_refresh
    PATH_TO_FEMR_LOGS: str = os.path.join(PATH_TO_OUTPUT_DIR, "logs")
    PATH_TO_FEMR_EXTRACT: str = os.path.join(PATH_TO_OUTPUT_DIR, "extract")
    
    # Force refresh
    check_file_existence_and_handle_force_refresh(PATH_TO_OUTPUT_DIR, IS_FORCE_REFRESH)
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # `etl_simple_femr` command will crash if it sees any non-csv files in the input directory
    # Thus, we need to make sure we delete any non-CSV files in our input directory
    delete_files_not_starting_with_csv(PATH_TO_INPUT_DIR) 

    # Run the ETL pipeline to transform: EHRSHOT CSVs -> FEMR patient database
    logger.info(f"Start | Create FEMR PatientDatabase")
    os.system(f"etl_simple_femr {PATH_TO_INPUT_DIR} {PATH_TO_FEMR_EXTRACT} {PATH_TO_FEMR_LOGS} --num_threads {NUM_THREADS} --athena_download {PATH_TO_ATHENA_DOWNLOAD}")
    logger.info(f"Finish | Create FEMR PatientDatabase")

    # Logging
    database = femr.datasets.PatientDatabase(PATH_TO_FEMR_EXTRACT)
    all_patient_ids = list(database)
    patient_id: int = all_patient_ids[0]
    patient = database[patient_id]
    events = patient.events
    logger.info(f"FEMR database saved to: {PATH_TO_OUTPUT_DIR}")
    logger.info(f"Num patients: {len(database)}")
    logger.info(f"Number of events in patient '{patient_id}': {len(events)}")
    logger.info(f"First event of patient '{patient_id}': {events[0]}")
    logger.success("Done!")