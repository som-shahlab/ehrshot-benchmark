import argparse
import pickle
import os
from loguru import logger
from utils import check_file_existence_and_handle_force_refresh
from typing import Dict, List, Tuple
import numpy as np
from serialization.text_encoder import TextEncoder, LLM2VecLlama3_1_7B_InstructSupervisedEncoder, GTEQwen2_7B_InstructEncoder, GTEQwen2_1_5B_InstructEncoder, STGTELargeENv15Encoder, BertEncoder, LLM2VecLlama2_Sheared_1_3B_SupervisedEncoder, GTEQwen2_7B_InstructChunkedEncoder, LLM2VecLlama3_1_7B_InstructSupervisedChunkedEncoder
from serialization.ehr_serializer import ListEventsStrategy, ListVisitsWithEventsStrategy, ListVisitsWithEventsDetailedAggrStrategy, UniqueThenListVisitsStrategy, UniqueThenListVisitsWithValuesStrategy, UniqueThenListVisitsWOAllCondsStrategy, UniqueThenListVisitsWOAllCondsWithValuesStrategy
from datetime import datetime
from llm_featurizer import LLMFeaturizer, preprocess_llm_featurizer, featurize_llm_featurizer, load_labeled_patients_with_tasks
import json
from utils import LABELING_FUNCTION_2_PAPER_NAME

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text-based featurizations for LLM models (for all tasks at once)")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to FEMR patient database")
    parser.add_argument("--path_to_labels_dir", required=True, type=str, help="Path to directory containing saved labels")
    parser.add_argument("--path_to_features_dir", required=True, type=str, help="Path to directory where features will be saved")
    parser.add_argument("--task_to_instructions", type=str, default="", help="Path to task to instructions file")
    parser.add_argument("--num_threads", type=int, help="Number of threads to use")
    parser.add_argument("--is_force_refresh", action='store_true', default=False, help="If set, then overwrite all outputs")
    parser.add_argument("--text_encoder", type=str, help="Text encoder to use")
    parser.add_argument("--serialization_strategy", required=True, type=str, help="Serialization strategy to use")
    parser.add_argument("--excluded_ontologies", type=str, default="", help="Ontologies to exclude")
    parser.add_argument("--unique_events", type=str, default="true", help="Whether to use unique events")
    parser.add_argument("--numeric_values", type=str, default="false", help="Whether to use numeric values")
    parser.add_argument("--medication_entry", type=str, default="false", help="Whether to use a designated medication entry")
    parser.add_argument("--num_aggregated", type=int, default=0, help="Number of aggregated values to use")
    parser.add_argument("--add_parent_concepts", required=True, type=str, help="Category for parent concepts")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    NUM_THREADS: int = args.num_threads
    IS_FORCE_REFRESH = args.is_force_refresh
    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_LABELS_DIR = args.path_to_labels_dir
    PATH_TO_FEATURES_DIR = args.path_to_features_dir
    PATH_TO_LABELS_FILE: str = os.path.join(PATH_TO_LABELS_DIR, 'all_labels_tasks.csv')
    PATH_TO_TASK_TO_INSTRUCTIONS_FILE: str = args.task_to_instructions
    EXCLUDED_ONTOLOGIES: List[str] = ['LOINC', 'Domain', 'CARE_SITE', 'ICDO3'] if args.excluded_ontologies == 'no_labs' else \
        ['LOINC', 'Domain', 'CARE_SITE', 'ICDO3', 'RxNorm', 'RxNorm Extension'] if args.excluded_ontologies == 'no_labs_meds' else \
        ['LOINC', 'Domain', 'CARE_SITE', 'ICDO3', 'Medicare Specialty', 'CMS Place of Service', 'OMOP Extension', 'Condition Type'] if args.excluded_ontologies == 'no_labs_single' else \
        ['LOINC', 'Domain', 'CARE_SITE', 'ICDO3', 'RxNorm', 'RxNorm Extension', 'Medicare Specialty', 'CMS Place of Service', 'OMOP Extension', 'Condition Type']  if args.excluded_ontologies == 'no_labs_meds_single' else []
    UNIQUE_EVENTS: bool = args.unique_events == 'true'
    NUMERIC_VALUES: bool = args.numeric_values == 'true'
    MEDICATION_ENTRY: bool = args.medication_entry == 'true'
    NUM_AGGREGATED_EVENTS: int = args.num_aggregated  # Default: 0
    FILTER_AGGREGATED_EVENTS: bool = NUM_AGGREGATED_EVENTS > 0
    ADD_CONDITIONS_PARENT_CONCEPTS: bool = args.add_parent_concepts == 'conditions'
        
    # Serialization strategies
    if args.serialization_strategy == 'list_events':
        serialization_strategy = ListEventsStrategy(UNIQUE_EVENTS, NUMERIC_VALUES, MEDICATION_ENTRY, NUM_AGGREGATED_EVENTS)
        max_input_length = 8192
    elif args.serialization_strategy == 'list_visits_with_events':
        serialization_strategy = ListVisitsWithEventsStrategy(UNIQUE_EVENTS, NUMERIC_VALUES, MEDICATION_ENTRY, NUM_AGGREGATED_EVENTS)
        max_input_length = 8192
    elif args.serialization_strategy == 'list_visits_with_events_detailed_aggr':
        serialization_strategy = ListVisitsWithEventsDetailedAggrStrategy(UNIQUE_EVENTS, NUMERIC_VALUES, MEDICATION_ENTRY, NUM_AGGREGATED_EVENTS)
        max_input_length = 8192
    elif args.serialization_strategy == 'unique_then_list_visits_wo_allconds_w_values':
        serialization_strategy = UniqueThenListVisitsWOAllCondsWithValuesStrategy(NUM_AGGREGATED_EVENTS, MEDICATION_ENTRY)
        max_input_length = 8192
    elif args.serialization_strategy == 'unique_then_list_visits_wo_allconds_w_values_4k':
        serialization_strategy = UniqueThenListVisitsWOAllCondsWithValuesStrategy(NUM_AGGREGATED_EVENTS, MEDICATION_ENTRY)
        max_input_length = 4096
    elif args.serialization_strategy == 'unique_then_list_visits_wo_allconds_w_values_2k':
        serialization_strategy = UniqueThenListVisitsWOAllCondsWithValuesStrategy(NUM_AGGREGATED_EVENTS, MEDICATION_ENTRY)
        max_input_length = 2048
    elif args.serialization_strategy == 'unique_then_list_visits_wo_allconds_w_values_1k':
        serialization_strategy = UniqueThenListVisitsWOAllCondsWithValuesStrategy(NUM_AGGREGATED_EVENTS, MEDICATION_ENTRY)
        max_input_length = 1024
    elif args.serialization_strategy == 'unique_then_list_visits_wo_allconds_w_values_512':
        serialization_strategy = UniqueThenListVisitsWOAllCondsWithValuesStrategy(NUM_AGGREGATED_EVENTS, MEDICATION_ENTRY)
        max_input_length = 512
    elif args.serialization_strategy == 'unique_then_list_visits_wo_allconds':
        serialization_strategy = UniqueThenListVisitsWOAllCondsStrategy(NUM_AGGREGATED_EVENTS, MEDICATION_ENTRY)
        max_input_length = 8192
    elif args.serialization_strategy == 'unique_then_list_visits_wo_allconds_4k':
        serialization_strategy = UniqueThenListVisitsWOAllCondsStrategy(NUM_AGGREGATED_EVENTS, MEDICATION_ENTRY)
        max_input_length = 4096
    elif args.serialization_strategy == 'unique_then_list_visits_w_values':
        serialization_strategy = UniqueThenListVisitsWithValuesStrategy(NUM_AGGREGATED_EVENTS, MEDICATION_ENTRY)
        max_input_length = 8192
    elif args.serialization_strategy == 'unique_then_list_visits_w_values_4k':
        serialization_strategy = UniqueThenListVisitsWithValuesStrategy(NUM_AGGREGATED_EVENTS, MEDICATION_ENTRY)
        max_input_length = 4096
    elif args.serialization_strategy == 'unique_then_list_visits':
        serialization_strategy = UniqueThenListVisitsStrategy(NUM_AGGREGATED_EVENTS, MEDICATION_ENTRY)
        max_input_length = 8192
    elif args.serialization_strategy == 'unique_then_list_visits_4k':
        serialization_strategy = UniqueThenListVisitsStrategy(NUM_AGGREGATED_EVENTS, MEDICATION_ENTRY)
        max_input_length = 4096
    else:
        raise ValueError(f"Serialization strategy `{args.serialization_strategy}` not recognized")
    logger.info(f"Use serialization strategy: {serialization_strategy.__class__}")
    logger.info(f"    Unique events: {UNIQUE_EVENTS}")
    logger.info(f"    Numeric values: {NUMERIC_VALUES}")
    logger.info(f"    Medication entry: {MEDICATION_ENTRY}")
    logger.info(f"    Num aggregated events: {NUM_AGGREGATED_EVENTS}")
    logger.info(f"    Max input length: {max_input_length}")
    logger.info(f"    Exclude ontologies: {EXCLUDED_ONTOLOGIES}")
    
    # Mapping of text encoder names to their corresponding classes
    encoder_mapping = {
        'llm2vec_llama3_1_7b_instruct_supervised': LLM2VecLlama3_1_7B_InstructSupervisedEncoder,
        'llm2vec_llama3_1_7b_instruct_supervised_chunked_2k': lambda max_input_length: LLM2VecLlama3_1_7B_InstructSupervisedChunkedEncoder(max_input_length=2048),
        'llm2vec_llama3_1_7b_instruct_supervised_chunked_1k': lambda max_input_length: LLM2VecLlama3_1_7B_InstructSupervisedChunkedEncoder(max_input_length=1024),
        'llm2vec_llama3_1_7b_instruct_supervised_chunked_512': lambda max_input_length: LLM2VecLlama3_1_7B_InstructSupervisedChunkedEncoder(max_input_length=512),
        'llm2vec_llama2_sheared_1_3b_supervised': LLM2VecLlama2_Sheared_1_3B_SupervisedEncoder,
        'gteqwen2_7b_instruct': GTEQwen2_7B_InstructEncoder,
        'gteqwen2_7b_instruct_chunked_2k': lambda max_input_length: GTEQwen2_7B_InstructChunkedEncoder(max_input_length=2048),
        'gteqwen2_7b_instruct_chunked_1k': lambda max_input_length: GTEQwen2_7B_InstructChunkedEncoder(max_input_length=1024),
        'gteqwen2_7b_instruct_chunked_512': lambda max_input_length: GTEQwen2_7B_InstructChunkedEncoder(max_input_length=512),
        'gteqwen2_1_5b_instruct': GTEQwen2_1_5B_InstructEncoder,
        'st_gte_large_en_v15': STGTELargeENv15Encoder,
        'bioclinicalbert': lambda max_input_length: BertEncoder(max_input_length=max_input_length, bert_identifier='emilyalsentzer/Bio_ClinicalBERT', embedding_size=768, model_max_input_length=512), 
        'bert_base': lambda max_input_length: BertEncoder(max_input_length=max_input_length, bert_identifier='bert-base-uncased', embedding_size=768, model_max_input_length=512),
        'bert_large': lambda max_input_length: BertEncoder(max_input_length=max_input_length, bert_identifier='bert-large-uncased', embedding_size=1024, model_max_input_length=512),
        'deberta_v3_base': lambda max_input_length: BertEncoder(max_input_length=max_input_length, bert_identifier='microsoft/deberta-v3-base', embedding_size=768, model_max_input_length=512),
        'deberta_v3_large': lambda max_input_length: BertEncoder(max_input_length=max_input_length, bert_identifier='microsoft/deberta-v3-large', embedding_size=1024, model_max_input_length=512),
        # Modern Bert models could handle 8192 tokens, but only use them with 4096 tokens
        'modernbert_base': lambda max_input_length: BertEncoder(max_input_length=max_input_length, bert_identifier='answerdotai/ModernBERT-base', embedding_size=768, model_max_input_length=4096),
        'modernbert_large': lambda max_input_length: BertEncoder(max_input_length=max_input_length, bert_identifier='answerdotai/ModernBERT-large', embedding_size=1024, model_max_input_length=4096),
    }

    # First check custom llm2vec model, than look up in mapping
    if args.text_encoder.startswith('llm2vec_llama3_1_7b_instruct_mimic_'):
        custom_path = args.text_encoder.removeprefix('llm2vec_llama3_1_7b_instruct_mimic_')
        text_encoder = TextEncoder(LLM2VecLlama3_1_7B_InstructSupervisedEncoder(max_input_length=max_input_length, custom_path=custom_path))
    elif args.text_encoder in encoder_mapping:
        text_encoder = TextEncoder(encoder_mapping[args.text_encoder](max_input_length=max_input_length))
    else:
        raise ValueError(f"Text encoder `{args.text_encoder}` not recognized")
    logger.info(f"Use text encoder: {text_encoder.encoder.__class__} with max length: {text_encoder.encoder.max_input_length}")
        
    # Load task to instructions json
    if PATH_TO_TASK_TO_INSTRUCTIONS_FILE:
        with open(PATH_TO_TASK_TO_INSTRUCTIONS_FILE, 'r') as f:
            task_to_instructions = json.load(f)
            assert all([isinstance(v, str) for v in task_to_instructions.values()]), "All values of task_to_instructions must be strings"
            if set(LABELING_FUNCTION_2_PAPER_NAME.keys()) - set(task_to_instructions.keys()):
                # Print differences
                logger.error(f"Task to instructions file does not contain all tasks. Missing: {set(LABELING_FUNCTION_2_PAPER_NAME.keys()) - set(task_to_instructions.keys())}")
    else:
        task_to_instructions = {}
    use_instructions = task_to_instructions is not None
    logger.info("Use no instructions." if not use_instructions else f"Use instructions from: {PATH_TO_TASK_TO_INSTRUCTIONS_FILE}")

    # Add date and time (hh-mm-ss) to name
    # output_file_name = f'llm_features_{args.text_encoder}_{args.serialization_strategy}{"_instr" if use_instructions else ""}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pkl'
    output_file_name = 'llm_features.pkl'
    PATH_TO_OUTPUT_FILE = os.path.join(PATH_TO_FEATURES_DIR, output_file_name)
    # Force refresh
    check_file_existence_and_handle_force_refresh(PATH_TO_OUTPUT_FILE, IS_FORCE_REFRESH)

    # Load consolidated labels across all patients for all tasks
    logger.info(f"Loading LabeledPatients from `{PATH_TO_LABELS_FILE}`")
    patients_to_labels: Dict[int, List[Tuple[datetime, str]]] = load_labeled_patients_with_tasks(PATH_TO_LABELS_FILE)
    # TODO Debug: Consider subset of patients
    # patients_to_labels = {k: v for k, v in list(patients_to_labels.items())[:10]}
    logger.info(f"Loaded {len(patients_to_labels)} patients with {sum([len(v) for v in patients_to_labels.values()])} labels")

    # Combine two featurizations of each patient: one for the patient's age, and one for the text of every code
    # they've had in their record up to the prediction timepoint for each label
    logger.info("Start | Preprocess featurizers")
    llm_featurizer = LLMFeaturizer(
        embedding_size=text_encoder.encoder.embedding_size,
        serialization_strategy=serialization_strategy,
        task_to_instructions=task_to_instructions,
        excluded_ontologies=EXCLUDED_ONTOLOGIES,
        filter_aggregated_events=FILTER_AGGREGATED_EVENTS,
        add_condition_parent_concepts=ADD_CONDITIONS_PARENT_CONCEPTS
    ) 
    llm_featurizer = preprocess_llm_featurizer(PATH_TO_PATIENT_DATABASE, llm_featurizer, patients_to_labels, NUM_THREADS)

    logger.info("Finish | Preprocess featurizers")
    
    # Run text encoding on serializations of patients - must be done separately to prevent multiprocessing issue with CUDA
    llm_featurizer.encode_serializations(text_encoder, cache_dir=PATH_TO_FEATURES_DIR)
    # Encoder not necessary anymore
    del text_encoder

    # Featurization only performs serial copying of embeddings.
    # Hence, one thread faster than multiple threads.
    logger.info("Start | Featurize patients")
    results = featurize_llm_featurizer(PATH_TO_PATIENT_DATABASE, patients_to_labels, llm_featurizer, num_threads=1)
    feature_matrix, patient_ids, label_values, label_times, label_tasks = (
        results[0],
        results[1],
        results[2],
        results[3],
        results[4],
    )
    logger.info("Finish | Featurize patients")
    
    # Ensure that all final features sum up to the same value as the generated embeddings
    assert np.allclose(llm_featurizer.embeddings, feature_matrix)

    # Save results
    logger.info(f"Saving results to `{PATH_TO_OUTPUT_FILE}`")
    with open(PATH_TO_OUTPUT_FILE, 'wb') as f:
        pickle.dump(results, f)

    # Logging
    logger.info("FeaturizedPatient stats:\n"
                f"feature_matrix={repr(feature_matrix)}\n"
                f"patient_ids={repr(patient_ids)}\n"
                f"label_values={repr(label_values)}\n"
                f"label_times={repr(label_times)}")
    logger.success("Done!")
    