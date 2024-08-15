from __future__ import annotations

import datetime
from typing import Callable, Deque, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import numpy as np

# Following original femr.featurizers
from femr import Patient
from femr.extension import datasets as extension_datasets
from femr.labelers import Label
from femr.featurizers.core import ColumnValue, Featurizer
from femr.featurizers.featurizers import get_patient_birthdate

# Additional imports
# TODO: Remove count featurizer
from femr.featurizers.featurizers import CountFeaturizer

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import re
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from llm2vec import LLM2Vec
from tqdm import tqdm
import random
from serialization.ehr_serializer import EHRSerializer, ListUniqueEventsStrategy
from femr import Event


class LLMFeaturizer(Featurizer):
    """
    Produces LLM-encoded representation of patient.
    """

    def __init__(self):
        # Llama 3(.1)
        self.embedding_dim = 4096
        # Qwen 2
        # self.embedding_dim = 3584

        # Filled during preprocessing
        # A dictionary mapping patient ID and label index to the serialization of the patient's EHR
        self.pid_label_idx_serializations: Dict[Tuple[int, int], int] = {}
        
        # Filled during aggregation
        # Numpy array of embeddings according to order of .items() of pid_label_idx_serializations
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
    
    def resolve_code_with_custom_ontologies(self, ontology: extension_datasets.Ontology, code: str) -> str:
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

    def preprocess(self, patient: Patient, labels: List[Label], ontology: extension_datasets.Ontology):
        
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
        cls, featurizers: List[CountFeaturizer]
    ) -> CountFeaturizer:
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
        
        ###########################################################################################
        # Qwen models
        # def last_token_pool(last_hidden_states: Tensor,
        #                     attention_mask: Tensor) -> Tensor:
        #     left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        #     if left_padding:
        #         return last_hidden_states[:, -1]
        #     else:
        #         sequence_lengths = attention_mask.sum(dim=1) - 1
        #         batch_size = last_hidden_states.shape[0]
        #         return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

        # def get_detailed_instruct(task_description: str, query: str) -> str:
        #     return f'Instruct: {task_description}\nQuery: {query}'

        # # TODO: If want to use intructions, check format
        # # Each query must come with a one-sentence instruction that describes the task
        # # task = 'Given a web search query, retrieve relevant passages that answer the query'
        # # queries = [get_detailed_instruct(task, 'how much protein should a female eat')]

        # tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True, torch_dtype=torch.float16).to(device)
        
        # class TextsDataset(Dataset):
        #     def __init__(self, texts):
        #         self.texts = texts 

        #     def __len__(self):
        #         return len(self.texts)

        #     def __getitem__(self, idx):
        #         return self.texts[idx]

        # def encode(texts: List[str], batch_size: int) -> np.ndarray:
        #     with torch.no_grad():
        #         dataloader = DataLoader(TextsDataset(serializations), batch_size=batch_size, shuffle=False)
        #         all_embeddings = []
        #         for batch in tqdm(dataloader, desc="Processing Batches"):
        #             batch_dict = tokenizer(batch, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(device)
        #             outputs = model(**batch_dict)
        #             embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        #             normalized_embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach().numpy()
        #             all_embeddings.append(normalized_embeddings)
        #         return np.concatenate(all_embeddings, axis=0)
        
        # merged_embeddings = encode(serializations, batch_size=8)
        # del model

        ###########################################################################################
        # LLM2Vec models
        
        # LLM2Vec-Llama3-supervised
        # model = LLM2Vec.from_pretrained(
        #     "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        #     peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
        #     device_map="cuda" if torch.cuda.is_available() else "cpu",
        #     torch_dtype=torch.bfloat16,
        #     max_length=8192,
        #     doc_max_length=8192,
        # )

        # LLM2Vec-Llama3.1-supervised
        model_path = "/home/sthe14/llm2vec/output"
        model = LLM2Vec.from_pretrained(
            model_path + "/mntp/Meta-Llama-3.1-8B-Instruct",
            peft_model_name_or_path=model_path + "/mntp-supervised/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-64_l-512_bidirectional-True_e-3_s-42_w-300_lr-0.0002_lora_r-16/checkpoint-1000",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            max_length=8192,
            doc_max_length=8192,
        )
        def encode_texts(texts: List[str], model) -> np.ndarray:
            return np.array(model.encode(texts, batch_size=8))
        merged_embeddings = encode_texts(serializations, model)
        
        del model

        # Add serializations to template featurizer
        template_featurizer: LLMFeaturizer = featurizers[0]
        template_featurizer.pid_label_idx_serializations = merged_pid_label_idx_serializations
        # Ensure same ordering of serializations as at the beginning to create the list of serializations using .items()
        template_featurizer.embeddings = merged_embeddings
        template_featurizer.pid_to_embedding_idx = merged_pid_to_embedding_idx

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
        # This is a numpy array with dimension (num_labels, embedding_dim)
        patient_labels_embeddings = self.embeddings[self.pid_to_embedding_idx[patient.patient_id]]
        
        # Create list of list of column values
        # all_columns = []
        # for label_idx, _ in enumerate(labels):
        #     all_columns.append(list(map(lambda i: ColumnValue(i, patient_labels_embeddings[label_idx, i]), range(self.embedding_dim))))
            
        # Create list of list of column values using list comprehension
        all_columns = [[ColumnValue(i, patient_labels_embeddings[label_idx, i]) for i in range(self.embedding_dim)] for label_idx, _ in enumerate(labels)]
        
        return all_columns

    def is_needs_preprocessing(self) -> bool:
        return True 

    def __repr__(self) -> str:
        # return f"LLMFeaturizer(number of included codes={self.num_columns})"
        return f"LLMFeaturizer()"

    def get_column_name(self, column_idx: int) -> str:
        return "Embedding"
        