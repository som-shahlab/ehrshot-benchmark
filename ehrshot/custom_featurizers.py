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
import re
import pandas as pd

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
        self.pid_to_index: Dict[int, int] = {}
        self.serializations: List[str] = []
        self.encodings: np.ndarray = np.array([])
        
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

    def get_num_columns(self) -> int:
        return self.embedding_dim

    def preprocess(self, patient: Patient, labels: List[Label], ontology: extension_datasets.Ontology):
        
        data = {'age': 0, 'race': 0, 'gender': 0}
        race_dict = {1: 'american indian', 2: 'asian', 3: 'black', 4: 'pacific islander', 5: 'white'}
        gender_dict = {'F': 'female', 'M': 'male'}
        patient_birth_date: datetime = get_patient_birthdate(patient)
        
        # Single sentence for age, gender, race
        # for event in patient.events:
        #     if event.code.startswith('Race/'):
        #         data['race'] = race_dict[int(event.code.split('/')[1])]
        #     elif event.code.startswith('Gender/'):
        #         data['gender'] = gender_dict[event.code.split('/')[1]]
        # # Debug: assume same age across all labels for first test run
        # label = labels[0]
        # data['age'] = int((label.time - patient_birth_date).days / 365)
        # text = f"The patient is a {data['age']} year-old {data['gender']} of {data['race']} race."
        
        # Serialization of all codes
        text_events = []
        text_set = set()
        
        # Add age manually
        # Debug: assume same age across all labels for first test run
        label = labels[0]
        age = int((label.time - patient_birth_date).days / 365)
        text_events.append(f"- Age: {age}")
        
        # Exclude some non-informative codes
        re_exlude_description_prefixes = re.compile(r"^(CARE\_SITE\/|Birth|Plan Stop Reason|Patient discharge)")
        
        # Custom onlogies
        re_ontologies = re.compile(r"^(CPT4\/|ICD10PCS\/|CVX\/)")
        
        # Remove some suffixes:
        # 'in Serum or Plasma', 'Serum or Plasma', ' - Serum or Plasma', 'in Serum', 'in Plasma'
        # 'in Blood', ' - Blood', 'in Blood by Automated count', 'by Automated count', ', automated'
        # 'by Manual count'
        re_exclude_description_suffixes = re.compile(r"( in Serum or Plasma| Serum or Plasma| - Serum or Plasma| in Serum| in Plasma| in Blood| - Blood| in Blood by Automated count| by Automated count|, automated| by Manual count)")
        
        for event in patient.events:
            description = ontology.get_text_description(event.code)
            # if len(text_events) >= 40:
            #     break
            if re_exlude_description_prefixes.match(description):
                continue
            if re_ontologies.match(event.code):
                ontology_name = event.code.split('/')[0]
                code = event.code.split('/')[1].lower()
                if code in self.custom_ontologies[ontology_name]:
                    description = self.custom_ontologies[ontology_name][code]
                else:
                    continue
                
            # Remove some irrelevant artifacts
            # Remove all [*] - often correspond to units
            description = re.sub(r"\[.*\]", "", description)
            # Remove suffixes
            description = re_exclude_description_suffixes.sub("", description)
            # Remove repeated whitespaces
            description = re.sub(r"\s+", " ", description)
            description = description.strip()
            
            # Each code only once
            if description in text_set:
                continue
            text_set.add(description)
            
            if type(event.value) is str:
                text_events.append(f"- {description}: {event.value}")
            # TODO: Add handling of numeric values
            else:
                text_events.append(f"- {description}")
            
        text = '\n'.join(text_events)
        # print('-'*20)
        # print('-'*20)
        # print(text)

        # Text serialization of all data
        # instruction = "Classify the following description of a patient as being at risk for death or not: "
        instruction = ""
        text = instruction + text
        
        self.pid_serializations[patient.patient_id] = text
        return
                    
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
        pid_to_index: Dict[int, int] = {}
        for i, pid in enumerate(pids):
            pid_to_index[pid] = i
        serializations = [merged_serializations[pid] for pid in pids]

        model = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            max_length=8192,
            doc_max_length=8192,
        )
        def encode_texts(texts: List[str]) -> np.ndarray:
            return np.array(model.encode(texts, batch_size=8))
        merged_encodings = encode_texts(serializations)
        del model

        template_featurizer: LLMFeaturizer = featurizers[0]
        template_featurizer.pids = pids
        template_featurizer.pid_to_index = pid_to_index
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
        
        # Use the dictionary for fast lookups
        patient_index = self.pid_to_index.get(patient.patient_id)
        patient_encoding = self.encodings[patient_index]

        # Precompute the common feature list for the patient
        # common_features = [ColumnValue(i, patient_encoding[i]) for i in range(self.embedding_dim)]
        # Do the same with map function
        common_features = list(map(lambda i: ColumnValue(i, patient_encoding[i]), range(self.embedding_dim)))
        # TODO: Currently duplicate the features for each label
        all_columns = [common_features for _ in labels]

        # patient_encoding = self.encodings[self.pids.index(patient.patient_id)]
        
        # for _ in labels:
        #     # Copy encoding for each label
        #     # TODO: Add label specific encodings
        #     all_columns.append([ColumnValue(i, patient_encoding[i]) for i in range(self.embedding_dim)])

        return all_columns

    def is_needs_preprocessing(self) -> bool:
        return True 

    def __repr__(self) -> str:
        # return f"LLMFeaturizer(number of included codes={self.num_columns})"
        return f"LLMFeaturizer()"

    def get_column_name(self, column_idx: int) -> str:
        return "Embedding"
        