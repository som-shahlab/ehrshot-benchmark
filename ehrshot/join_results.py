import os
import argparse
import pandas as pd
from pathlib import Path

LABELING_FUNCTIONS = ['new_celiac', 'new_lupus', 'new_pancan', 'new_acutemi', 'new_hypertension', 'new_hyperlipidemia',
                      'lab_anemia', 'lab_hyperkalemia', 'lab_hypoglycemia', 'lab_hyponatremia', 'lab_thrombocytopenia',
                      'guo_readmission', 'guo_icu', 'guo_los',
                      'chexpert']
RESULTS_FILE_NAME = 'all_results.csv'

def read_and_filter_results(source_dir, source_filter):
    # Read all_results.csv from source
    source_file = os.path.join(source_dir, RESULTS_FILE_NAME)
    source_df = pd.read_csv(source_file)
    
    # Filter source_df by list of tuples, if ('model', 'head') is in the list, keep it
    # Get indicies of enties that occur in list
    indices = []
    for model, head in source_filter:
        indices += source_df[(source_df['model'] == model) & (source_df['head'] == head)].index.tolist()
        
    return source_df.iloc[indices]        

def parse_filter(filter_str):
    # Contains a list of tuples separate by ;, i.e. (model1, head1); (model2, head2), ...
    filter_list = filter_str.split(';')
    filter_list = [tuple(filter_item.split(',')) for filter_item in filter_list]
    return filter_list

def main(args):
    
    for labeling_function in LABELING_FUNCTIONS:
        # Create folder in output directory
        output_dir = os.path.join(args.output_dir, labeling_function)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load filter and read results with filters
        first_filter = parse_filter(args.first_source_filter)
        second_filter = parse_filter(args.second_source_filter)
        # Add labeling function to source dir paths
        first_source_dir = os.path.join(args.first_source_dir, labeling_function)
        second_source_dir = os.path.join(args.second_source_dir, labeling_function)
        first_filtered_entries = read_and_filter_results(first_source_dir, first_filter)
        second_filtered_entries = read_and_filter_results(second_source_dir, second_filter)
        
        # Ensure some properties of the dataframes
        assert first_filtered_entries.columns.tolist() == second_filtered_entries.columns.tolist()
        assert first_filtered_entries.shape[0] / len(first_filter) == second_filtered_entries.shape[0] / len(second_filter)
        
        # Join results and write to output directory
        joined_results = pd.concat([first_filtered_entries, second_filtered_entries])
        # Rename first column to empty string and fill it with continuous numbers
        joined_results.columns = [''] + joined_results.columns.tolist()[1:]
        joined_results[''] = range(0, joined_results.shape[0])
        # Write results to output directory
        joined_results.to_csv(os.path.join(output_dir, RESULTS_FILE_NAME), index=False) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EHRShot experiments")
    parser.add_argument("--first_source_dir", required=True, help="First source directory")
    parser.add_argument("--first_source_filter", required=True, help="First source filter")
    parser.add_argument("--second_source_dir", required=True, help="Second source directory")
    parser.add_argument("--second_source_filter", required=True, help="Second source filter")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    main(args)