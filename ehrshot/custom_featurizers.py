from __future__ import annotations

import collections
import datetime
import functools
import random
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np

# Following original femr.featurizers
from femr import Event, Patient
from femr.extension import datasets as extension_datasets
from femr.labelers import Label
from femr.featurizers.core import ColumnValue, Featurizer
from femr.featurizers.utils import OnlineStatistics
from femr.featurizers.featurizers import get_patient_birthdate

# Additional imports
from femr.featurizers.featurizers import _reshuffle_count_time_bins, exclusion_helper, ReservoirSampler
# TODO: Remove count featurizer
from femr.featurizers.featurizers import CountFeaturizer

import torch
from llm2vec import LLM2Vec

class LLMFeaturizer(Featurizer):
    """
    Produces LLM-encoded representation of patient.
    """

    def __init__(self):
        self.column_names = ['age', 'race', 'gender']
        self.embedding_dim = 4096
        # Filled during preprocessing
        self.pid_serializations: Dict[int, str] = {}

        # Filled during aggregation
        self.pids: List[int] = []
        self.serializations: List[str] = []
        self.encodings: np.ndarray = np.array([])

    def get_num_columns(self) -> int:
        return self.embedding_dim

    def preprocess(self, patient: Patient, labels: List[Label], ontology: extension_datasets.Ontology):
        
        data = {'age': 0, 'race': 0, 'gender': 0}
        race_dict = {1: 'american indian', 2: 'asian', 3: 'black', 4: 'pacific islander', 5: 'white'}
        gender_dict = {'F': 'female', 'M': 'male'}
        patient_birth_date: datetime = get_patient_birthdate(patient)
        
        for event in patient.events:
            if event.code.startswith('Race/'):
                data['race'] = race_dict[int(event.code.split('/')[1])]
            elif event.code.startswith('Gender/'):
                data['gender'] = gender_dict[event.code.split('/')[1]]
        # Debug: assume same age across all labels for first test run
        label = labels[0]
        data['age'] = int((label.time - patient_birth_date).days / 365)

        # Text serialization of all data
        instruction = "Classify the following description of a patient as being at risk for death or not: "
        text = f"The patient is a {data['age']} year-old {data['gender']} of {data['race']} race."
        text = instruction + text
        
        self.pid_serializations[patient.patient_id] = text
        return
        # text_events = []
        
        # for event in patient.events:
        #     # Check for excluded events
        #     if self.excluded_event_filter is not None and self.excluded_event_filter(event):
        #         continue

        #     # # Sequenital processing of all codes
        #     # if event.value is None:
        #     #     # for code in self.get_codes(event.code, ontology):
        #     #         # If we haven't seen this code before, then add it to our list of included codes
        #     #     text_events.append(f"{event.start}, {ontology.get_text_description(event.code)} ({event.code})")
        #     # elif type(event.value) is str:
        #     #     if self.string_value_combination:
        #     #         # TODO: Might add self.characters_for_string_values to the string value
        #     #         text_events.append(f"{event.start}, {ontology.get_text_description(event.code)} ({event.code}): {event.value}")
        #     # else:
        #     #     if self.numeric_value_decile:
        #     #         text_events.append(f"{event.start}, {ontology.get_text_description(event.code)} ({event.code}): {event.value}")

        #     # Extract some relevant information

        #     basic_information = {}
        #     if event.code == 'SNOMED/3950001':
        #         # get age from substracting 2022 from event.start
        #         # TODO: This is an approximation - unclear prediction times (I guess different for tasks)
        #         basic_information['born'] = int(event.start.year)
        #     elif event.code.startswith('Race/'):
        #         race_dict = {1: 'american_indian', 2: 'asian', 3: 'black', 4: 'pacific_islander', 5: 'white'}
        #         basic_information['race'] = race_dict[event.code.split('/')[1]]
        #     elif event.code.startswith('Gener/'):
        #         gender_dict = {'F': 'female', 'M': 'male'}
        #         basic_information['gender'] = gender_dict[event.code.split('/')[1]]
        #     elif event.code == ('Inpatient Visit (Visit/IP)'):
        #         if 'hospital_visits' not in basic_information:
        #             basic_information['hospital_visits'] = 0
        #         basic_information['hospital_visits'] += 1
        #     elif event.code == ('Outpatient Visit (Visit/OP)'):
        #         if 'outpatient_visits' not in basic_information:
        #             basic_information['outpatient_visits'] = 0
        #         basic_information['outpatient_visits'] += 1
        #     
        #     # Some relevant diseases
        #     # Essential hypertension (SNOMED/59621000)
        #     # Diabetes -> not in examples (SNOMED/73211009)
        #     # Hyperlipidemia (SNOMED/55822004)
        #     # Low back pain (SNOMED/279039007)
        #     # Chronic obstructive lung disease (SNOMED/13645005)
        #     elif event.code.startswith('SNOMED/'):
        #         cond_dict = {
        #             '59621000': 'hypertension',
        #             '73211009': 'diabetes',
        #             '55822004': 'hyperlipidemia',
        #             '279039007': 'low_back_pain',
        #             '13645005': 'copd',
        #         }
        #         code = event.code.split('/')[1]
        #         if code in cond_dict:
        #             basic_information[cond_dict[code]] == 1

        #     self.patient_basic_information[patient.patient_id] = basic_information
                    

    @classmethod
    def aggregate_preprocessed_featurizers(  # type: ignore[override]
        cls, featurizers: List[CountFeaturizer]
    ) -> CountFeaturizer:
        """After preprocessing a LLMFeaturizer using multiprocessing (resulting in the list of featurizers
        contained in `featurizers`), this method aggregates all those featurizers into one LLMFeaturizer.
        """
        if len(featurizers) == 0:
            raise ValueError("You must pass in at least one featurizer to `aggregate_preprocessed_featurizers`")

        # Combine all self.serialization of all featurizers
        merged_serializations: dict[int, str] = {}
        for featurizer in featurizers:
            merged_serializations.update(featurizer.pid_serializations)

        # Fix ordering of patients
        pids = list(merged_serializations.keys())
        serializations = [merged_serializations[pid] for pid in pids]

        model = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )
        def encode_texts(texts: List[str]) -> np.ndarray:
            return np.array(model.encode(texts, batch_size=32))
        merged_encodings = encode_texts(serializations) 

        template_featurizer: LLMFeaturizer = featurizers[0]
        template_featurizer.pids = pids
        template_featurizer.serializations = serializations
        template_featurizer.encodings = merged_encodings

        return template_featurizer


    def featurize(
        self,
        patient: Patient,
        labels: List[Label],
        ontology: Optional[extension_datasets.Ontology],
    ) -> List[List[ColumnValue]]:
        """ Return text representation of patient at each label. """
        
        assert ontology is not None, "Ontology cannot be `None` for LLMFeaturizer"
        all_columns: List[List[ColumnValue]] = []
        # Outer list is per label
        # Inner list is the list of features for that label

        patient_encoding = self.encodings[self.pids.index(patient.patient_id)]
        
        for label in labels:
            # Copy encoding for each label
            # TODO: Add label specific encodings
            all_columns.append([ColumnValue(i, patient_encoding[i]) for i in range(self.embedding_dim)])

        return all_columns

    def is_needs_preprocessing(self) -> bool:
        return True 

    def __repr__(self) -> str:
        # return f"LLMFeaturizer(number of included codes={self.num_columns})"
        return f"LLMFeaturizer()"

    def get_column_name(self, column_idx: int) -> str:
        return "Embedding"
        