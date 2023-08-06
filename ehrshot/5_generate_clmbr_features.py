import argparse
import os
from loguru import logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CLMBR / MOTOR patient representations (for all tasks at once)")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_and_feats_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_output_dir", type=str, help="Path to directory where model will be saved")
    parser.add_argument("--model", type=str, help="Type of foundation model to train. Options: 'motor' or 'clmbr'")
    return parser.parse_args()

# TODO - verify what `PATH_TO_OUTPUT_DIR` is doing

if __name__ == "__main__":
    
    args = parse_args()
    MODEL: str = args.model
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_LABELS_AND_FEATS_DIR: str = args.path_to_labels_and_feats_dir
    PATH_TO_OUTPUT_DIR = os.path.join(args.path_to_output_dir, MODEL)
    PATH_TO_MODEL: str = os.path.join(PATH_TO_OUTPUT_DIR, "model")
    PATH_TO_DICTIONARY: str = os.path.join(PATH_TO_OUTPUT_DIR, "dictionary")

    if MODEL == "motor":
        hier_flag = "--is_hierarchical "
    else:
        hier_flag = ""
    
    # Generate batches for all tasks
    PATH_TO_TASK_BATCHES: str = os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, f"{MODEL}_batches")
    if not os.path.exists(PATH_TO_TASK_BATCHES):
        command: str = (
            f"clmbr_create_batches {PATH_TO_TASK_BATCHES}"
            f" --data_path {PATH_TO_PATIENT_DATABASE}"
            f" --dictionary {PATH_TO_DICTIONARY}"
            f" --task labeled_patients"
            f" --batch_size 131072"
            f" {hier_flag}"
            f" --labeled_patients_path {os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, 'all_labels.csv')}"
        )
        logger.info(f"Start | Create CLMBR batches @ {PATH_TO_TASK_BATCHES}")
        logger.critical(f"Command run:\n{command}")
        assert 0 == os.system(command)
        logger.success(f"Finish | Create CLMBR batches @ {PATH_TO_TASK_BATCHES}")
    else:
        logger.critical(f"Skipping `clmbr_create_batches` because batches already exist @ {PATH_TO_TASK_BATCHES}")
        
    # Generate patient representations for tasks
    PATH_TO_REPRESENTATIONS = os.path.join(PATH_TO_LABELS_AND_FEATS_DIR, f"{MODEL}_features.pkl")
    if not os.path.exists(PATH_TO_REPRESENTATIONS):
        command: str = (
            f"clmbr_compute_representations {PATH_TO_REPRESENTATIONS}"
            f" --data_path {PATH_TO_PATIENT_DATABASE}"
            f" --batches_path {PATH_TO_TASK_BATCHES}"
            f" --model_dir {PATH_TO_MODEL}"
        )
        logger.info(f"Start | Create CLMBR representations @ {PATH_TO_REPRESENTATIONS}")
        logger.critical(f"Command run:\n{command}")
        assert 0 == os.system(command)
        logger.success(f"Finish | Create CLMBR representations @ {PATH_TO_REPRESENTATIONS}")
    else:
        logger.critical(f"Skipping `clmbr_compute_representations` because representations already exist @ {PATH_TO_REPRESENTATIONS}")