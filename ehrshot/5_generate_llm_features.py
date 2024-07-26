import argparse
import os
from loguru import logger

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CLMBR / MOTOR patient representations (for all tasks at once)")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to directory where features will be saved")
    parser.add_argument("--path_to_models_dir", type=str, help="Path to directory where models are saved")
    parser.add_argument("--model", type=str, help="Name of foundation model to load. Options: 'motor' or 'clmbr'")
    parser.add_argument("--is_force_refresh", action='store_true', default=False, help="If set, then overwrite all outputs")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    MODEL: str = args.model
    IS_FORCE_REFRESH: bool = args.is_force_refresh
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_LABELS_DIR: str = args.path_to_labels_dir
    PATH_TO_FEATURES_DIR: str = args.path_to_features_dir
    PATH_TO_LABELED_PATIENTS: str = os.path.join(PATH_TO_LABELS_DIR, 'all_labels.csv')
    PATH_TO_MODEL_DIR = os.path.join(args.path_to_models_dir, MODEL)
    PATH_TO_MODEL: str = os.path.join(PATH_TO_MODEL_DIR, "clmbr_model")
    PATH_TO_DICTIONARY: str = os.path.join(PATH_TO_MODEL_DIR, "dictionary")
    PATH_TO_TASK_BATCHES: str = os.path.join(PATH_TO_FEATURES_DIR, f"{MODEL}_batches")
    PATH_TO_REPRESENTATIONS = os.path.join(PATH_TO_FEATURES_DIR, f"{MODEL}_features.pkl")
    
    # Check that requested model exists
    assert os.path.exists(PATH_TO_MODEL), f"No model for `{MODEL}` exists @ `{PATH_TO_MODEL}`"
    assert os.path.exists(PATH_TO_DICTIONARY), f"No model dictionary for `{MODEL}` exists @ `{PATH_TO_DICTIONARY}`"

    if MODEL == "motor":
        hier_flag = "--is_hierarchical "
    else:
        hier_flag = ""
    
    if IS_FORCE_REFRESH:
        os.system(f"rm -rf {PATH_TO_TASK_BATCHES}")
        os.system(f"rm -rf {PATH_TO_REPRESENTATIONS}")
    
    # Generate batches for all tasks
    if IS_FORCE_REFRESH or not os.path.exists(PATH_TO_TASK_BATCHES):
        command: str = (
            f"clmbr_create_batches {PATH_TO_TASK_BATCHES}"
            f" --data_path {PATH_TO_PATIENT_DATABASE}"
            f" --dictionary {PATH_TO_DICTIONARY}"
            f" --task labeled_patients"
            f" --batch_size 131072"
            f" --val_start 70"
            f" {hier_flag}"
            f" --labeled_patients_path {PATH_TO_LABELED_PATIENTS}"
        )
        logger.info(f"Start | Create CLMBR batches @ {PATH_TO_TASK_BATCHES}")
        logger.critical(f"Command run:\n{command}")
        assert 0 == os.system(command)
        logger.success(f"Finish | Create CLMBR batches @ {PATH_TO_TASK_BATCHES}")
    else:
        logger.critical(f"Skipping `clmbr_create_batches` because batches already exist @ {PATH_TO_TASK_BATCHES}")
        
    # Generate patient representations for tasks
    if IS_FORCE_REFRESH or not os.path.exists(PATH_TO_REPRESENTATIONS):
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
    logger.success("Done!")