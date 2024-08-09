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
from tqdm import tqdm

class LLMFeaturizer(Featurizer):
    """
    Produces LLM-encoded representation of patient.
    """

    def __init__(self):
        self.column_names = ['age', 'race', 'gender']
        # Llama 3
        # self.embedding_dim = 4096
        # Qwen 2
        self.embedding_dim = 3584
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
            
        # Single sentence for age, gender, race
        # data = {'age': 0, 'race': 0, 'gender': 0}
        # race_dict = {1: 'american indian', 2: 'asian', 3: 'black', 4: 'pacific islander', 5: 'white'}
        # gender_dict = {'F': 'female', 'M': 'male'}
        # for event in patient.events:
        #     if event.code.startswith('Race/'):
        #         data['race'] = race_dict[int(event.code.split('/')[1])]
        #     elif event.code.startswith('Gender/'):
        #         data['gender'] = gender_dict[event.code.split('/')[1]]
        # # Debug: assume same age across all labels for first test run
        # label = labels[0]
        # data['age'] = int((label.time - patient_birth_date).days / 365)
        # text = f"The patient is a {data['age']} year-old {data['gender']} of {data['race']} race."
        
        def serialize_unique_codes(events):
            # Serialization of all codes
            text_events = []
            text_set = set()
            
            # Exclude some non-informative codes
            re_exlude_description_prefixes = re.compile(r"^(CARE\_SITE\/|Birth|Plan Stop Reason|Patient discharge)")
            
            # Custom onlogies
            re_ontologies = re.compile(r"^(CPT4\/|ICD10PCS\/|CVX\/)")
            
            # Remove some suffixes:
            # 'in Serum or Plasma', 'Serum or Plasma', ' - Serum or Plasma', 'in Serum', 'in Plasma'
            # 'in Blood', ' - Blood', 'in Blood by Automated count', 'by Automated count', ', automated'
            # 'by Manual count'
            re_exclude_description_suffixes = re.compile(r"( in Serum or Plasma| Serum or Plasma| - Serum or Plasma| in Serum| in Plasma| in Blood| - Blood| in Blood by Automated count| by Automated count|, automated| by Manual count)")
            
            for event in events:
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
            return text
        
        text = serialize_unique_codes(patient.events)
        
        # Add age manually
        label = labels[0]
        patient_birth_date: datetime = get_patient_birthdate(patient)
        age = int((label.time - patient_birth_date).days / 365)
        text = f"- Age: {age}\n" + text
        
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
        
        ###########################################################################################
        # Qwen models
        def last_token_pool(last_hidden_states: Tensor,
                            attention_mask: Tensor) -> Tensor:
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f'Instruct: {task_description}\nQuery: {query}'

        # TODO: If want to use intructions, check format
        # Each query must come with a one-sentence instruction that describes the task
        # task = 'Given a web search query, retrieve relevant passages that answer the query'
        # queries = [get_detailed_instruct(task, 'how much protein should a female eat')]

        tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True, torch_dtype=torch.float16).to(device)
        
        class TextsDataset(Dataset):
            def __init__(self, texts):
                self.texts = texts 

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                return self.texts[idx]

        def encode(texts: List[str], batch_size: int) -> np.ndarray:
            with torch.no_grad():
                dataloader = DataLoader(TextsDataset(serializations), batch_size=batch_size, shuffle=False)
                all_embeddings = []
                for batch in tqdm(dataloader, desc="Processing Batches"):
                    batch_dict = tokenizer(batch, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(device)
                    outputs = model(**batch_dict)
                    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                    normalized_embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach().numpy()
                    all_embeddings.append(normalized_embeddings)
                return np.concatenate(all_embeddings, axis=0)
        
        merged_encodings = encode(serializations, batch_size=8)
        del model

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

        # LLM2Vec-Llama3-supervised
        # model_path = "/home/sthe14/llm2vec/output"
        # llm2vec_llama3_1_8b_supervised = ModelMeta(
        #     loader=_loader(
        #         LLM2VecWrapper,
        #         base_model_name_or_path=model_path + "/mntp/Meta-Llama-3.1-8B-Instruct",
        #         peft_model_name_or_path=model_path + "/mntp-supervised/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-64_l-512_bidirectional-True_e-3_s-42_w-300_lr-0.0002_lora_r-16/checkpoint-1000",
        #         device_map="auto",
        #         torch_dtype=torch.bfloat16,
        #     ),
        #     name="local/LLM2Vec-Meta-Llama-3.1-8B-Instruct-mntp-supervised",
        #     languages=["eng_Latn"],
        #     open_source=True,
        #     revision=None,
        #     release_date="2024-04-09",
        # )
        
        # LLM2Vec-Llama3.1-supervised
        # model = LLM2Vec.from_pretrained(
        #     model_path + "/mntp/Meta-Llama-3.1-8B-Instruct",
        #     peft_model_name_or_path=model_path + "/mntp-supervised/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-64_l-512_bidirectional-True_e-3_s-42_w-300_lr-0.0002_lora_r-16/checkpoint-1000",
        #     device_map="cuda" if torch.cuda.is_available() else "cpu",
        #     torch_dtype=torch.bfloat16,
        #     max_length=8192,
        #     doc_max_length=8192,
        # )
        # def encode_texts(texts: List[str]) -> np.ndarray:
        #     return np.array(model.encode(texts, batch_size=8))
        # merged_encodings = encode_texts(serializations)
        # del model

        # Add serializations to template featurizer
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
        