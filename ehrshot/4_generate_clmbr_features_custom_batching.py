import argparse
import os
import pickle
from typing import List
import femr.models.transformer
from typing import Any, Dict, List, Optional
import meds
import torch
from loguru import logger
import femr.models.tokenizer
import femr.models.processor
import datasets
from utils import get_rel_path, convert_csv_labels_to_meds
import numpy as np
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CLMBR-T-Base patient representations (for all tasks at once)")
    parser.add_argument("--path_to_dataset", default=get_rel_path(__file__, "../assets/ehrshot-meds-stanford/"), type=str, help="Path to MEDS formatted version of EHRSHOT")
    parser.add_argument("--path_to_labels_csv", default=get_rel_path(__file__, "../assets/labels/merged_labels.csv"), type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", default=get_rel_path(__file__, "../assets/features/"), type=str, help="Path to directory where features will be saved")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str, help="GPU device to use (if available)")
    parser.add_argument("--num_threads", default=5, type=int, help="Number of threads to use")
    return parser.parse_args()

def compute_features(
    dataset: datasets.Dataset,
    model_path: str,
    device: Optional[torch.device] = None,
    ontology = None,
) -> Dict[str, np.ndarray]:
    """
    Taken from: https://github.com/som-shahlab/femr/blob/6b2f778afd3a346d0beef3098b1868912d870df4/src/femr/models/transformer.py#L324
    """
    model = femr.models.transformer.FEMRModel.from_pretrained(model_path)
    tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(model_path, ontology=ontology)
    processor = femr.models.processor.FEMRBatchProcessor(tokenizer)

    if device:
        model = model.to(device)
        
    all_patient_ids = []
    all_feature_times = []
    all_representations = []
    print("Running inference...")
    for patient in tqdm(dataset, total=len(dataset)):
        batch = processor.collate([processor.convert_patient(patient, tensor_type="pt")])['batch']
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = batch[key].to(device)
        for key, val in batch['transformer'].items():
            if isinstance(val, torch.Tensor):
                batch['transformer'][key] = batch['transformer'][key].to(device)
        with torch.no_grad():
            __, result = model(batch)
            all_patient_ids.append(result["patient_ids"].cpu().numpy())
            all_feature_times.append(result["timestamps"].cpu().numpy())
            all_representations.append(result["representations"].cpu().numpy())

    return {
        "patient_ids": np.concatenate(all_patient_ids),
        "feature_times": np.concatenate(all_feature_times).astype("datetime64[s]"),
        "features": np.concatenate(all_representations),
    }

if __name__ == "__main__":
    args = parse_args()
    path_to_dataset: str = os.path.join(args.path_to_dataset, 'data/*.parquet')
    path_to_labels_csv: str = args.path_to_labels_csv
    path_to_features_dir: str = args.path_to_features_dir
    num_threads: int = args.num_threads
    device: str = args.device
    os.makedirs(path_to_features_dir, exist_ok=True)

    assert os.path.exists(args.path_to_dataset), f"Path to dataset does not exist: {args.path_to_dataset}"
    assert os.path.exists(path_to_labels_csv), f"Path to labels CSV does not exist: {path_to_labels_csv}"
    assert os.path.exists(path_to_features_dir), f"Path to features directory does not exist: {path_to_features_dir}"

    model_name: str = "StanfordShahLab/clmbr-t-base"

    # Load EHRSHOT dataset
    dataset = datasets.Dataset.from_parquet(path_to_dataset)
    
    print("==> Len dataset:", len(dataset))

    # Load consolidated labels across all patients for all tasks
    labels: List[meds.Label] = convert_csv_labels_to_meds(path_to_labels_csv)
    
    # Load model
    print("Loading model", model_name, "to device", device)
    model = femr.models.transformer.FEMRModel.from_pretrained(model_name)
    
    # Generate features
    results: Dict[str, Any] = compute_features(dataset, model_name, device=device)

    # Save results
    path_to_output_file = os.path.join(path_to_features_dir, f"clmbr_custom_batch_features.pkl")
    logger.info(f"Saving results to `{path_to_output_file}`")
    with open(path_to_output_file, 'wb') as f:
        pickle.dump(results, f)

    # Logging
    patient_ids, feature_times, features = results['patient_ids'], results['feature_times'], results['features']
    logger.info("FeaturizedPatient stats:\n"
                f"patient_ids={repr(patient_ids)}\n"
                f"features={repr(features)}\n"
                f"feature_times={repr(feature_times)}\n")
    logger.success("Done!")