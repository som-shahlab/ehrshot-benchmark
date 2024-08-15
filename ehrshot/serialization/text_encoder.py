from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np
import torch
from llm2vec import LLM2Vec
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class LLMEncoder(ABC):
    def __init__(self, embedding_size: int):
        self.embedding_size = embedding_size
        
    @abstractmethod
    def encode(self, serializations: List[str], batch_size: int, **kwargs) -> List[Any]:
        pass

class LLM2VecLlama3_7B_InstructSupervisedEncoder(LLMEncoder):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(embedding_size=4096)
        self.model = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            max_length=8192,
            doc_max_length=8192,
        )

    def encode(self, texts: List[str], batch_size: int = 8) -> List[Any]:
        return np.array(self.model.encode(texts, batch_size=batch_size))
        
class LLM2VecLlama3_1_7B_InstructSupervisedEncoder(LLMEncoder):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(embedding_size=4096)
        model_path = "/home/sthe14/llm2vec/output"
        self.model = LLM2Vec.from_pretrained(
            model_path + "/mntp/Meta-Llama-3.1-8B-Instruct",
            peft_model_name_or_path=model_path + "/mntp-supervised/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-64_l-512_bidirectional-True_e-3_s-42_w-300_lr-0.0002_lora_r-16/checkpoint-1000",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            max_length=8192,
            doc_max_length=8192,
        )
    
    def encode(self, texts: List[str], batch_size: int = 8) -> List[Any]:
        return np.array(self.model.encode(texts, batch_size=batch_size))
    
class GTEQwen2_7B_InstructEncoder(LLMEncoder):
    
    def __init__(self, **kwargs) -> None:
        super().__init__(embedding_size=3584)
        self.tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
                
    @staticmethod
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

    def encode(self, serializations: List[str], batch_size: int = 8) -> List[Any]:
        
        class TextsDataset(Dataset):
            def __init__(self, texts):
                self.texts = texts 

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                return self.texts[idx]
        
        with torch.no_grad():
            dataloader = DataLoader(TextsDataset(serializations), batch_size=batch_size, shuffle=False)
            all_embeddings = []
            for batch in tqdm(dataloader, desc="Processing Batches"):
                batch_dict = self.tokenizer(batch, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(self.device)
                outputs = self.model(**batch_dict)
                embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                normalized_embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach().numpy()
                all_embeddings.append(normalized_embeddings)
            return np.concatenate(all_embeddings, axis=0)
        

class TextEncoder:
    def __init__(self, encoder: LLMEncoder):
        self.encoder = encoder

    def encode_texts(self, texts: List[str], batch_size: int) -> List[Any]:
        return self.encoder.encode(texts, batch_size)
