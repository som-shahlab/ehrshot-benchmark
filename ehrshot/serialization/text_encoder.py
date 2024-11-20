from abc import ABC, abstractmethod
from typing import List, Any
from numpy.typing import NDArray
import numpy as np
import torch
from llm2vec import LLM2Vec
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
from tqdm import tqdm
from typing import Tuple
    
        
class TextsDataset(TorchDataset):
    def __init__(self, texts):
        self.texts = texts 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
    
    
class LLMEncoder(ABC):
    def __init__(self, embedding_size: int, model_max_input_length: int, max_input_length: int) -> None:
        self.embedding_size = embedding_size
        self.max_input_length = min(model_max_input_length, max_input_length)

        # Use simple heuristic to determine batch size
        # TODO: Adapt based on available GPU memory
        def determine_llm_batch_size():
            default_batch_size = 4
            if max_input_length > 32768:
                return 1
            elif max_input_length > 8192:
                return 2
            elif model_max_input_length == 512:
                return 32
            return default_batch_size
        self.batch_size: int = determine_llm_batch_size()
        
        # Ensure that tokenizer and model are set, but this is done in subclasses
        self.tokenizer = None
        self.model = None   

    def add_instruction(self, instruction: str, text: str) -> Any:
        # Per default: ignore instruction
        return text
        
    @abstractmethod
    def _encode(self, inputs: List, **kwargs) -> NDArray[Any]:
        pass
            

class BERTLLMEncoder(LLMEncoder):
    def __init__(self, embedding_size: int, model_max_input_length: int, max_input_length: int) -> None:
        super().__init__(embedding_size, model_max_input_length, max_input_length)     
    
    # Want to reproduce Jiang et al. Health system-scale language models are all-purpose prediction engines 2023.
    # However, unclear what MLM classification head exactly means, so will use the cls token for now.
    def get_cls_embedding(self, batch):
        inputs = self.tokenizer(batch['text'], padding=True, truncation=True, max_length=self.max_input_length, return_tensors='pt').to(self.device) # type: ignore
        outputs = self.model(**inputs, output_hidden_states=True) # type: ignore
        # Get last layer of hidden states
        last_hidden_states = outputs.hidden_states[-1]
        cls_embedding = last_hidden_states[:,0,:]
        return {'embedding': cls_embedding}
    
    # Get average of all hidden states in the first and laster hidden layer
    # Shown to be superior to cls token or only last hidden state (http://arxiv.org/pdf/2103.15316)
    # See: https://github.com/autoliuweijie/BERT-whitening-pytorch/blob/b5cfbd606bd19fc3b3adf9e074dc0bfd830ef597/all_utils.py#L33
    def get_first_last_avg_embedding(self, batch):
        inputs = self.tokenizer(batch['text'], padding=True, truncation=True, max_length=self.max_input_length, return_tensors='pt').to(self.device) # type: ignore
        hidden_states = self.model(**inputs, output_hidden_states=True).hidden_states # type: ignore
        # Average over all states in the first and the last layer (each layer return [batch_size, seq_len, hidden_size])
        first_last_avg_embedding = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        return {'embedding': first_last_avg_embedding}


class LLM2VecLlamaLLMEncoder(LLMEncoder):
    def __init__(self, embedding_size: int, model_max_input_length: int, max_input_length: int) -> None:
        super().__init__(embedding_size, model_max_input_length, max_input_length)
        
    def add_instruction(self, instruction: str, text: str) -> Any:
        def llm2vec_instruction(instruction):
            if len(instruction) > 0 and instruction[-1] != ":":
                instruction = instruction.strip(".") + ":"
            return instruction
        return [llm2vec_instruction(instruction), text]
    
    def _encode(self, inputs: List, **kwargs) -> NDArray[Any]:
        return np.array(self.model.encode(inputs, batch_size=self.batch_size)) # type: ignore
        
    
class Qwen2LLMEncoder(LLMEncoder):
    def __init__(self, embedding_size: int, model_max_input_length: int, max_input_length: int) -> None:
        super().__init__(embedding_size, model_max_input_length, max_input_length)
        
    def add_instruction(self, instruction: str, text: str) -> Any:
        # From https://huggingface.co/Alibaba-NLP/gte-Qwen1.5-7B-instruct
        if instruction is not None and len(instruction) > 0:
            return f'Instruct: {instruction}\nQuery:\n{text}'
        return text
        
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
        
    def _encode(self, inputs: List, **kwargs) -> NDArray[Any]:
        with torch.no_grad():
            dataloader = DataLoader(TextsDataset(inputs), batch_size=self.batch_size, shuffle=False)
            all_embeddings = []
            for batch in tqdm(dataloader, desc="Processing Batches"):
                batch_dict = self.tokenizer(batch, max_length=self.max_input_length, padding=True, truncation=True, return_tensors='pt').to(self.device) # type: ignore
                outputs = self.model(**batch_dict) # type: ignore
                embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']) # type: ignore
                normalized_embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach().numpy()
                all_embeddings.append(normalized_embeddings)
            return np.concatenate(all_embeddings, axis=0)


class LLM2VecLlama3_7B_InstructSupervisedEncoder(LLM2VecLlamaLLMEncoder):
    
    def __init__(self, max_input_length: int, **kwargs) -> None:
        super().__init__(embedding_size=4096, model_max_input_length=8192, max_input_length=max_input_length)
        self.model = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            max_length=self.max_input_length,
            doc_max_length=self.max_input_length,
        )


class LLM2VecLlama3_1_7B_InstructSupervisedEncoder(LLM2VecLlamaLLMEncoder):
    
    def __init__(self, max_input_length: int, **kwargs) -> None:
        super().__init__(embedding_size=4096, model_max_input_length=128000, max_input_length=max_input_length)
        model_path = "/home/sthe14/llm2vec/output"
        self.model = LLM2Vec.from_pretrained(
            model_path + "/mntp/Meta-Llama-3.1-8B-Instruct",
            peft_model_name_or_path=model_path + "/mntp-supervised/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-64_l-512_bidirectional-True_e-3_s-42_w-300_lr-0.0002_lora_r-16/checkpoint-1000",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            max_length=self.max_input_length,
            doc_max_length=self.max_input_length,
        )

    
class GTEQwen2_7B_InstructEncoder(Qwen2LLMEncoder):
    
    def __init__(self, max_input_length: int, **kwargs) -> None:
        super().__init__(embedding_size=3584, model_max_input_length=128000, max_input_length=max_input_length) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True, torch_dtype=torch.float16).to(self.device)  

    
class GTEQwen2_1_5B_InstructEncoder(Qwen2LLMEncoder):
    
    def __init__(self, max_input_length: int, **kwargs) -> None:
        super().__init__(embedding_size=1536, model_max_input_length=128000, max_input_length=max_input_length) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-1.5B-instruct', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-1.5B-instruct', trust_remote_code=True, torch_dtype=torch.float16).to(self.device)  

        
class STGTELargeENv15Encoder(LLMEncoder):
    
    def __init__(self, max_input_length: int, **kwargs) -> None:
        super().__init__(embedding_size=1024, model_max_input_length=8192, max_input_length=max_input_length)  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True).to(self.device)
        
    def _encode(self, inputs: List, **kwargs) -> NDArray[Any]:
        with torch.no_grad():
            dataloader = DataLoader(TextsDataset(inputs), batch_size=self.batch_size, shuffle=False)
            all_embeddings = []
            for batch in tqdm(dataloader, desc="Processing Batches"):
                batch_dict = self.tokenizer(batch, max_length=self.max_input_length, padding=True, truncation=True, return_tensors='pt').to(self.device)
                outputs = self.model(**batch_dict)
                embeddings = outputs.last_hidden_state[:, 0]
                normalized_embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach().numpy()
                all_embeddings.append(normalized_embeddings)
            return np.concatenate(all_embeddings, axis=0)
        

class BioClinicalBert(BERTLLMEncoder):
    
    def __init__(self, max_input_length: int, **kwargs) -> None:
        super().__init__(embedding_size=768, model_max_input_length=512, max_input_length=max_input_length)  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(self.device)
        self.handle_long_texts = kwargs.get('handle_long_texts', 'truncate')    
    
    def chunk_texts(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        all_chunks = []
        chunk_counts = []
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            chunks = []
            for i in range(0, len(tokens), self.max_input_length):
                chunk = tokens[i:i + self.max_input_length]
                chunks.append(self.tokenizer.convert_tokens_to_string(chunk))
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))
        return all_chunks, chunk_counts

    def _encode(self, inputs: List, **kwargs) -> NDArray[Any]:
        
        if self.handle_long_texts not in ['truncate', 'average_chunks']:
            raise ValueError(f"handle_long_texts must be 'truncate' or 'average_chunks', but got {self.handle_long_texts}")
        
        chunk_counts = []
        if self.handle_long_texts == 'average_chunks':
            inputs, chunk_counts = self.chunk_texts(inputs)
             
        dataset = Dataset.from_dict({"text": inputs})
        dataset = dataset.map(self.get_first_last_avg_embedding, batched=True, batch_size=self.batch_size)
        all_embeddings = np.array(dataset['embedding'])
        
        if self.handle_long_texts == 'average_chunks':
            # Average embeddings for each original text
            final_embeddings = []
            start_idx = 0
            for count in chunk_counts:
                text_embeddings = all_embeddings[start_idx:start_idx + count]
                avg_embedding = np.mean(text_embeddings, axis=0)
                final_embeddings.append(avg_embedding)
                start_idx += count
    
            all_embeddings = np.array(final_embeddings)
            
        return all_embeddings


class LongformerLargeEncoder(BERTLLMEncoder):
        
    def __init__(self, max_input_length: int, **kwargs) -> None:
        super().__init__(embedding_size=1024, model_max_input_length=4096, max_input_length=max_input_length)  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.biomedical = kwargs.get('biomedical', False)
        if self.biomedical:
            self.tokenizer = AutoTokenizer.from_pretrained("kiddothe2b/biomedical-longformer-large")
            self.model = AutoModel.from_pretrained("kiddothe2b/biomedical-longformer-large").to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096")
            self.model = AutoModel.from_pretrained("allenai/longformer-large-4096").to(self.device)

    def _encode(self, inputs: List, **kwargs) -> NDArray[Any]:
        dataset = Dataset.from_dict({"text": inputs})
        dataset = dataset.map(self.get_first_last_avg_embedding, batched=True, batch_size=self.batch_size)
        all_embeddings = np.array(dataset['embedding'])
        return all_embeddings


class TextEncoder:
    def __init__(self, encoder: LLMEncoder):
        self.encoder = encoder

    def encode_texts(self, instructions: List[str], texts: List[str]) -> NDArray[Any]:
        if all([instruction is None or len(instruction) == 0 for instruction in instructions]):
            inputs = texts
        else:
            inputs = [self.encoder.add_instruction(instruction, text) for instruction, text in zip(instructions, texts)]
        
        return self.encoder._encode(inputs)
