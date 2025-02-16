import os
import argparse
import pandas as pd
import numpy as np
import pickle

RESULTS_FILE_NAME = 'llm_features.pkl'

def main(args):

    # Load source pickle files
    llm_source_embeddings = pd.read_pickle(args.llm_source_embeddings)
    clmbr_source_embeddings = pd.read_pickle(args.clmbr_source_embeddings)
    
    # Iterate over clmbr_source_embeddings to get the patient_id and prediction_time
    clmbr_mapping = dict()
    for idx in range(0, len(clmbr_source_embeddings['data_matrix'])):
        clmbr_mapping[(clmbr_source_embeddings['patient_ids'][idx], pd.to_datetime(clmbr_source_embeddings['labeling_time'][idx]))] = idx
    
    # Use the mapping to assign idx to llm_source_labels based on 'patient_id' and 'prediction_time'
    llm_clmbr_idx = [clmbr_mapping.get((llm_source_embeddings[1][i], pd.to_datetime(llm_source_embeddings[3][i]))) for i in range(0, len(llm_source_embeddings[0]))]
    assert None not in llm_clmbr_idx, "There are None values in llm_clmbr_idx"
    assert len(llm_clmbr_idx) == len(llm_source_embeddings[0]), "Length of llm_clmbr_idx is not equal to len(llm_source_embeddings[0])"
    
    # Convert the tuple to a list so it is mutable
    llm_source_embeddings = list(llm_source_embeddings)
    
    # Concatenate embeddings along axis=1 (feature dimension)
    llm_source_embeddings[0] = np.concatenate(
        (llm_source_embeddings[0],
         clmbr_source_embeddings['data_matrix'][llm_clmbr_idx]),
        axis=1
    )

    # Save the results to a pickle file
    llm_source_embeddings = tuple(llm_source_embeddings)
    output_dir = os.path.join(args.output_dir, RESULTS_FILE_NAME)
    with open(output_dir, 'wb') as f:
        pickle.dump(llm_source_embeddings, f)
    
    print(f"Saved results to {output_dir}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine embeddings of LLM and CLMBR")
    parser.add_argument("--llm_source_embeddings", required=True, help="LLM source embeddings")
    parser.add_argument("--clmbr_source_embeddings", required=True, help="CLMBR source embeddings")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    main(args)