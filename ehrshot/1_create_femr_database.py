import os
import argparse
from loguru import logger
import femr.datasets

"""
python3 1_create_femr_database.py \
    --path_to_input ../data/cohort \
    --path_to_target ../data/femr\
    --athena_download ../data/athena_download \
    --num_threads 10
"""

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run femr text featurizer")
    parser.add_argument(
        "--path_to_input",
        type=str,
        help="Path to folder with all the csv files",
    )
    parser.add_argument(
        "--path_to_target",
        type=str,
        help="Path to your target directory to save femr",
    )
    parser.add_argument(
        "--athena_download",
        type=str,
        help="Path to athena download",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads",
    )

    args = parser.parse_args()
    INPUT_DIR = args.path_to_input
    TARGET_DIR = args.path_to_target
    athena_download = args.athena_download
    num_threads = args.num_threads

    delete_files_not_starting_with_csv(INPUT_DIR)  # We just want csv files inside this folder
    os.makedirs(TARGET_DIR)

    LOG_DIR = os.path.join(TARGET_DIR, "logs")
    EXTRACT_DIR = os.path.join(TARGET_DIR, "extract")

    os.system(f"etl_simple_femr {INPUT_DIR} {EXTRACT_DIR} {LOG_DIR} --num_threads {num_threads} --athena_download {athena_download}")

    logger.info(f"Femr database saved in path: {TARGET_DIR}")
    logger.info("Testing the database")

    database = femr.datasets.PatientDatabase(EXTRACT_DIR)
    logger.info("Num patients", len(database))
    all_patient_ids = list(database)
    omop_id = all_patient_ids[0]
    patient = database[omop_id]
    events = patient.events
    logger.info(f"Number of events in patients with omop_id {omop_id}: {events}")
    logger.info(f"First event of the patient: {events[0]}")
