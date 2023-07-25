import argparse
import os
from typing import Optional
from loguru import logger
import shutil
import femr.datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLMBR patient representations")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to femr database")
    parser.add_argument("--path_to_features", type=str, help=( "This is the path to the labeled featurized parent folder" ), )
    parser.add_argument("--path_to_models", type=str, help=( "This is the path to the labeled featurized parent folder" ), )
    parser.add_argument("--model_name")
    
    args = parser.parse_args()

    MODEL_DATA = os.path.join(args.path_to_models, args.model_name)

    MODEL_PATH: str = os.path.join(MODEL_DATA, "model")
    DICTIONARY_PATH: str = os.path.join(MODEL_DATA, "dictionary")

    if args.model_name == "motor":
        hier_flag = "--is_hierarchical "
    else:
        hier_flag = ""
    
    # Generate batches for tasks
    PATH_TO_TASK_BATCHES = os.path.join(args.path_to_features, f"{args.model_name}_batches")
    if not os.path.exists(PATH_TO_TASK_BATCHES):
        command: str = (
            f"clmbr_create_batches {PATH_TO_TASK_BATCHES}"
            f" --data_path {args.path_to_database}"
            f" --dictionary {DICTIONARY_PATH}"
            f" --task labeled_patients"
            f" --batch_size 131072"
            f" {hier_flag}"
            f" --labeled_patients_path {os.path.join(args.path_to_features, 'all_labels.csv')}"
        )
        logger.info(f"Creating batches @ {PATH_TO_TASK_BATCHES}")
        logger.critical(f"Command run:\n{command}")
        assert 0 == os.system(command)
        logger.success(f"Done creating batches @ {PATH_TO_TASK_BATCHES}")
    else:
        logger.critical(f"Skipping clmbr_create_batches because batches already exist @ {PATH_TO_TASK_BATCHES}")
        
    # Generate patient representations for tasks
    REPRESENTATIONS = os.path.join(args.path_to_features, f"{args.model_name}_features.pkl")
    if not os.path.exists(REPRESENTATIONS):
        command: str = (
            f"clmbr_compute_representations {REPRESENTATIONS}"
            f" --data_path {args.path_to_database}"
            f" --batches_path {PATH_TO_TASK_BATCHES}"
            f" --model_dir {MODEL_PATH}"
        )
        logger.info(f"Creating representations @ {REPRESENTATIONS}")
        logger.critical(f"Command run:\n{command}")
        assert 0 == os.system(command)
        logger.success(f"Done creating representations @ {REPRESENTATIONS}")
    else:
        logger.critical(f"Skipping clmbr_compute_representations because reprs already exist @ {REPRESENTATIONS}")