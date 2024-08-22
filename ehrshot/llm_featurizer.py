from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
import re
from serialization.ehr_serializer import EHRSerializer, ListUniqueEventsStrategy
from femr import Event
from serialization.text_encoder import TextEncoder
from femr import Patient
from femr.featurizers.core import ColumnValue
from femr.featurizers.featurizers import get_patient_birthdate
from nptyping import NDArray
from dataclasses import dataclass
import csv
import collections
import multiprocessing
from femr.labelers import Label
from femr.extension import datasets as extension_datasets

PatientDatabase = extension_datasets.PatientDatabase
Ontology = extension_datasets.Ontology


""" Reimplemented FeaturizerList methods specificlly for LLM Featurizer to cater for task-specific labels """

@dataclass
class LabelTask(Label):
    task: str

    def __init__(self, time: datetime, value: bool, task: str):
        super().__init__(time, value)
        self.task = task

def load_labeled_patients_with_tasks(filename: str) -> Dict[int, List[Tuple[datetime, str]]]:
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) != 0, "Must have at least one label to load it"

        labeled_patients_with_tasks: Dict[int, List[Tuple[datetime, str]]] = collections.defaultdict(list)
        for row in rows:
            time = datetime.fromisoformat(row["prediction_time"])
            if time.second != 0:
                time = time.replace(second=0)
            task = row["task"]

            labeled_patients_with_tasks[int(row["patient_id"])].append((time, task))
        return labeled_patients_with_tasks

def _run_llm_preprocess_featurizer(task):
    database_path, patient_ids, patients_to_labels, featurizer = task
    for patient_id in patient_ids:
        featurizer.preprocess(database_path, patient_id, patients_to_labels[patient_id])
    return featurizer

def preprocess_llm_featurizer(
    database_path: str,
    patients_to_labels: Dict[int, List[Tuple[datetime, str]]],
    embedding_size: int,
    num_threads: int = 1,
):
    # Create an instance of the featurizer
    featurizer = LLMFeaturizer(embedding_size)

    # Split patients across multiple threads
    patient_ids: List[int] = list(patients_to_labels.keys())
    patient_ids_per_thread: List[NDArray] = np.array_split(patient_ids, num_threads * 10)
    tasks = [
        (database_path, patient_ids, patients_to_labels, featurizer) for patient_ids in patient_ids_per_thread
    ]

    # Preprocess in parallel using multiprocessing
    with multiprocessing.Pool(num_threads) as pool:
        preprocessed_featurizers: List[LLMFeaturizer] = list(pool.imap(_run_llm_preprocess_featurizer, tasks))

    # Aggregate the results from all featurizers
    aggregated_featurizer = LLMFeaturizer.aggregate_preprocessed_featurizers(preprocessed_featurizers)

    return aggregated_featurizer

    
def _run_llm_featurizer(args: Tuple[str, NDArray, Dict[int, List[Tuple[datetime, str]]], LLMFeaturizer]) -> Tuple[Any, Any, Any, Any, Any]:
    database_path: str = args[0]
    patient_ids: NDArray = args[1]
    patients_to_labels: Dict[int, List[Tuple[datetime, str]]] = args[2]
    featurizer: LLMFeaturizer = args[3]

    # Load patients + ontology
    database: PatientDatabase = PatientDatabase(database_path)
    ontology: Ontology = database.get_ontology()

    label_data: List[Tuple[int, Any, np.datetime64, str]] = []
    # For each Patient...
    features_list: List[NDArray] = []
    for patient_id in patient_ids:
        patient: Patient = database[patient_id] # type: ignore
        labels: List[Tuple[datetime, str]] = patients_to_labels[patient_id]

        if len(labels) == 0:
            continue

        # Featurize the patient using LLMFeaturizer
        features: NDArray = featurizer.featurize(patient, labels, ontology)
        assert features.shape == (len(labels), featurizer.embedding_dim)
        features_list.append(features)

        for label_idx, feature_row in enumerate(features):
            # Store the label, patient ID, label time, and task (labeling function)
            # datetime to np.datetime64 conversion is necessary for compatibility with pandas
            label_data.append((patient_id, False, np.datetime64(labels[label_idx][0]), labels[label_idx][1]))
        
    features = np.concatenate(features_list, axis=0)
    assert features.shape[0] == len(label_data)

    label_pids = np.array([x[0] for x in label_data], dtype=np.int64)
    label_values = np.array([x[1] for x in label_data])
    label_times = np.array([x[2] for x in label_data], dtype='datetime64[ns]')
    label_tasks = np.array([x[3] for x in label_data])

    return features, label_pids, label_values, label_times, label_tasks

def featurize_llm_featurizer(
        database_path: str,
        patients_to_labels: Dict[int, List[Tuple[datetime, str]]],
        llm_featurizer: LLMFeaturizer,
        num_threads: int = 1,
    ) -> Tuple[
        Any,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        # Split patient IDs into chunks for parallel processing
        patient_ids: List[int] = list(patients_to_labels.keys())
        patient_ids_per_thread: List[np.ndarray] = np.array_split(patient_ids, num_threads * 10)
        tasks = [
            (database_path, patient_ids_chunk, patients_to_labels, llm_featurizer)
            for patient_ids_chunk in patient_ids_per_thread
            if len(patient_ids_chunk) > 0
        ]

        # Run featurization in parallel
        with multiprocessing.Pool(num_threads) as pool:
            results: List[Tuple[Any, Any, Any, Any, Any]] = list(pool.imap(_run_llm_featurizer, tasks))

        # Filter out empty results
        results = [res for res in results if res[2].shape[0] > 0]

        # Join results into final matrices and arrays
        data_matrix = np.concatenate([x[0] for x in results])
        label_pids = np.concatenate([x[1] for x in results])
        label_values = np.concatenate([x[2] for x in results])
        label_times = np.concatenate([x[3] for x in results])
        label_tasks = np.concatenate([x[4] for x in results])

        return data_matrix, label_pids, label_values, label_times, label_tasks

class LLMFeaturizer():
    """
    Produces LLM-encoded representation of patient.
    """

    def __init__(self, embedding_size: int):

        self.embedding_dim = embedding_size

        # Filled during preprocessing
        # A dictionary mapping patient ID and label index to the serialization of the patient's EHR
        self.pid_label_idx_serializations: Dict[Tuple[int, int], int] = {}
        
        # Filled during aggregation
        # Numpy array of embeddings according to order of .items() of pid_label_idx_serializations
        self.serializations: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        self.pid_to_embedding_idx: Dict[int, List[int]] = {}
        
        # Custom ontologies
        cpt4 = pd.read_csv('ehrshot/custom_ontologies/cpt4.csv')
        cpt4 = cpt4.set_index('com.medigy.persist.reference.type.clincial.CPT.code')['label'].to_dict()
        icd10pcs = pd.read_csv('ehrshot/custom_ontologies/PClassR_v2023-1.csv', skiprows=1)
        icd10pcs.columns = icd10pcs.columns.str.strip("'")
        icd10pcs['ICD-10-PCS CODE'] = icd10pcs['ICD-10-PCS CODE'].str.strip("'")
        icd10pcs = icd10pcs.set_index('ICD-10-PCS CODE')['ICD-10-PCS CODE DESCRIPTION'].to_dict()
        cvx = pd.read_csv('ehrshot/custom_ontologies/cvx.csv', sep="|", header=None)
        cvx = dict(zip(cvx.iloc[:, 0], cvx.iloc[:, 2]))
        custom_ontologies = {
            'CPT4': cpt4,
            'ICD10PCS': icd10pcs,
            'CVX': cvx,
        }
        # Make all keys to strings, remove all whitespaces around, and lowerspace them
        for name, ontology in custom_ontologies.items():
            custom_ontologies[name] = {str(k).strip().lower(): v for k, v in ontology.items()}
        self.custom_ontologies = custom_ontologies
        self.re_custom_ontologies = re.compile(r"^(" + "|".join(custom_ontologies.keys()) + ")\/")

        # Remove some non-informative semantic codes / descriptions
        exclude_description_prefixes = ['Birth']
        # exclude_description_prefixes = ['CARE\_SITE\/', 'Birth', 'Plan Stop Reason', 'Patient discharge']
        self.re_exclude_description_prefixes = None if exclude_description_prefixes == [] else re.compile(r"^(" + "|".join(exclude_description_prefixes) + ")")
    
    def get_num_columns(self) -> int:
        return self.embedding_dim
    
    def resolve_code_with_custom_ontologies(
        self,
        ontology: extension_datasets.Ontology,
        code: str
    ) -> str:
        # Resolve semantic code to its description with default and custom onotologies
        description = ontology.get_text_description(code)
        
        # Exclude some custom ontologies / descriptions
        if self.re_exclude_description_prefixes is not None and self.re_exclude_description_prefixes.match(description):
            return None

        # Check if custom ontology is applicable
        if self.re_custom_ontologies.match(code):
            ontology_name = code.split('/')[0]
            code = code.split('/')[1].lower()
            if code in self.custom_ontologies[ontology_name]:
                description = self.custom_ontologies[ontology_name][code]

        # TODO: Post-process the description
        # # Remove some irrelevant artifacts
        # # Remove all [*] - often correspond to units
        # description = re.sub(r"\[.*\]", "", description)
        # # Remove suffixes
        # description = re_exclude_description_suffixes.sub("", description)
        # # Remove repeated whitespaces
        # description = re.sub(r"\s+", " ", description)
        description = description.strip()

        return description

    def preprocess(
        self,
        database_path: str,
        patient_id: int,
        labels: Tuple[datetime, str]
    ):
        # Load patients + ontology
        database: PatientDatabase = PatientDatabase(database_path)
        ontology: Ontology = database.get_ontology()
        patient: Patient = database[patient_id] # type: ignore
        labels: List[LabelTask] = [LabelTask(time, False, task) for time, task in labels]
        
        # Initialize mapping from pids to embedding indices based on number of labels
        self.pid_to_embedding_idx[patient.patient_id] = [-1] * len(labels)
        
        def is_visit_event(event: Event) -> bool:
            return event.code.startswith('Visit/')
        
        def resolve_code(code: str) -> str:
            return self.resolve_code_with_custom_ontologies(ontology, code)
        
        for label_idx, label in enumerate(labels):
            # According to existing feature processing, all events before or at the label time are included
            events_until_label = [event for event in patient.events if event.start <= label.time]
            serializer = EHRSerializer()
            serializer.load_from_femr_events(events_until_label, resolve_code, is_visit_event)
            
            # text = serialize_unique_codes([event for event in patient.events if event.start <= label.time])
            text = serializer.serialize(ListUniqueEventsStrategy())
        
            # Add age manually
            patient_birth_date: datetime = get_patient_birthdate(patient)
            age = int((label.time - patient_birth_date).days / 365)
            text = f"- Age: {age}\n" + text
        
            # TODO: Get task specific labels for be able to add task specific instructions
            # Add instruction
            instruction = ""
            text = instruction + text

            self.pid_label_idx_serializations[(patient.patient_id, label_idx)] = text
        
        return
                    
    @classmethod
    def aggregate_preprocessed_featurizers(  # type: ignore[override]
        cls,
        featurizers: List[LLMFeaturizer]
    ) -> LLMFeaturizer:
        """After preprocessing a LLMFeaturizer using multiprocessing (resulting in the list of featurizers
        contained in `featurizers`), this method aggregates all those featurizers into one LLMFeaturizer.
        """
        if len(featurizers) == 0:
            raise ValueError("You must pass in at least one featurizer to `aggregate_preprocessed_featurizers`")

        # Helper function to get all patient IDs from a dictionary
        def get_pids(d: Dict[Tuple[int, int], int]) -> Set[int]:
            return set(map(lambda d: d[0], d.keys()))

        # Combine all self.serialization of all featurizers
        merged_pid_label_idx_serializations: dict[int, List[str]] = {}
        merged_pid_to_embedding_idx: dict[int, List[int]] = {}
        for featurizer in featurizers:
            # Check that each featurizer considered disjoint set of patients
            assert get_pids(merged_pid_label_idx_serializations).isdisjoint(get_pids(featurizer.pid_label_idx_serializations))
            merged_pid_label_idx_serializations.update(featurizer.pid_label_idx_serializations)
            assert merged_pid_to_embedding_idx.keys().isdisjoint(featurizer.pid_to_embedding_idx.keys())
            merged_pid_to_embedding_idx.update(featurizer.pid_to_embedding_idx)
            assert len(get_pids(merged_pid_label_idx_serializations)) == len(merged_pid_to_embedding_idx)

        # Fix ordering of serializations
        merged_pid_label_idx_serializations = dict(sorted(merged_pid_label_idx_serializations.items()))
        serializations = []
        for idx, ((pid, label_idx), serialization) in enumerate(merged_pid_label_idx_serializations.items()):
            serializations.append(serialization)
            merged_pid_to_embedding_idx[pid][label_idx] = idx
        # Check that merged_pid_to_embedding_idx contains all indices from 0 to len(serializations) - 1
        assert set([idx for indices in merged_pid_to_embedding_idx.values() for idx in indices]) == set(range(len(serializations)))
        
        # Set template featurizer
        template_featurizer: LLMFeaturizer = featurizers[0]
        # Add serializations to template featurizer
        template_featurizer.pid_label_idx_serializations = merged_pid_label_idx_serializations
        # Ensure same ordering of serializations as at the beginning to create the list of serializations using .items()
        template_featurizer.serializations = serializations
        template_featurizer.embeddings = None
        template_featurizer.pid_to_embedding_idx = merged_pid_to_embedding_idx

        return template_featurizer
    
    def encode_serializations(
        self,
        text_encoder: TextEncoder
    ) -> None:
        """ Encode all serializations into embeddings. Outside of featurizer functions to prevent CUDA issues. """
        assert self.embeddings is None, "Embeddings already exist"
        self.embeddings = text_encoder.encoder.encode(self.serializations)

    def featurize(
        self,
        patient: Patient,
        labels: List[Tuple[datetime, str]],
        ontology: Optional[extension_datasets.Ontology],
    ) -> NDArray:
        """ Return text representation of patient at each label. """
        
        assert ontology is not None, "Ontology cannot be `None` for LLMFeaturizer"
        
        # Use the dictionary for fast lookups
        # This is a numpy array with dimension (num_labels, embedding_dim)
        patient_labels_embeddings = self.embeddings[self.pid_to_embedding_idx[patient.patient_id]]
        assert patient_labels_embeddings.shape == (len(labels), self.embedding_dim)
            
        # To return List[List[ColumnValue]: Create list of list of column values using list comprehension
        # all_columns: List[List[ColumnValue]] = []
        # all_columns = [[ColumnValue(i, patient_labels_embeddings[label_idx, i]) for i in range(self.embedding_dim)] for label_idx, _ in enumerate(labels)]
        
        return patient_labels_embeddings

    def is_needs_preprocessing(self) -> bool:
        return True 

    def __repr__(self) -> str:
        # return f"LLMFeaturizer(number of included codes={self.num_columns})"
        return "LLMFeaturizer()"

    def get_column_name(self, column_idx: int) -> str:
        return "Embedding"
        