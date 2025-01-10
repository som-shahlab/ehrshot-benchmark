from abc import ABC, abstractmethod
from typing import List, Any
from numpy.typing import NDArray
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset
from tqdm import tqdm
from typing import Tuple
import hashlib
import os
from llm2vec import LLM2Vec
import torch
# Workaround for ModernBert
# 1. Use different environment with github transformers version
# 2. Remove LLM2Vec import as it cannot handle new transformers version and add stumbs for LLM2Vec classes
# 3. For multi-GPU support, add "TORCHDYNAMO_DISABLE=1" to command

        
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
            # For max_input_length = 8192 (2 for 40 GB, 16 (llama), 8 for Qwen for 80 GB)
            batch_size = 8
            if self.__class__.__name__.startswith('LLM2VecLlama3'):
                batch_size = 16
        
            if max_input_length > 32768:
                batch_size = 1
            elif max_input_length > 8192:
                batch_size = 2
            elif model_max_input_length == 512:
                batch_size = 64
        
            # BERT models can use larger batch size, since they are generally smaller
            if self.__class__.__name__.startswith('Bert'):
                if model_max_input_length <= 512:
                    batch_size = batch_size * 4
                else:
                    batch_size = batch_size * 2
                
            return batch_size
        
        
        self.batch_size: int = determine_llm_batch_size()
        
        # Ensure that tokenizer and model are set, but this is done in subclasses
        self.tokenizer = None
        self.model = None   

    def add_instruction(self, instruction: str, text: str) -> Any:
        # Per default: ignore instruction
        return text
    
    def get_chunked_dataset(self, texts: List[str], tokenizer, max_chunks=None) -> Tuple[Dataset, List[int]]:
        # Create chunks of size max_input_length tokens for each text
        batch_size=8192 
        max_input_length = self.max_input_length - 8  # Subtract 8 to account for potential special tokens
        
        all_chunks = []
        chunk_counts = []
        
        start_idx = 0
        while start_idx < len(texts):
            print(f"  Chunking batch {start_idx // batch_size + 1} of {len(texts) // batch_size + 1}")
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            batch_offsets = tokenizer(batch_texts, add_special_tokens=False, return_offsets_mapping=True, truncation=False, padding=False)["offset_mapping"]
        
            for text, offsets in zip(batch_texts, batch_offsets):
                num_offsets = len(offsets)
                
                # Pre-limit how many indices we'll iterate over, so we generate at most `max_chunks` slices.
                if max_chunks is not None:
                    limit = max_chunks * max_input_length
                    end = min(num_offsets, limit)
                else:
                    end = num_offsets
                
                text_chunks = [
                    text[offsets[i][0]:offsets[min(i + max_input_length, num_offsets) - 1][1]]
                    for i in range(0, end, max_input_length)
                ]
                    
                chunk_counts.append(len(text_chunks))
                all_chunks.extend(text_chunks)
                
            start_idx = end_idx
        print()
            
        return all_chunks, chunk_counts
    
    def get_averaged_chunks(self, all_embeddings: NDArray[Any], chunk_counts: List[int]) -> NDArray[Any]:
        current_index = 0
        averaged_embeddings = []
        for count in chunk_counts:
            chunk_embeddings = all_embeddings[current_index:current_index + count]
            averaged_embeddings.append(np.mean(chunk_embeddings, axis=0))
            current_index += count
        return np.array(averaged_embeddings)
        
    @abstractmethod
    def _encode(self, inputs: List, **kwargs) -> NDArray[Any]:
        pass
            

class BERTLLMEncoder(LLMEncoder):
    def __init__(self, embedding_size: int, model_max_input_length: int, max_input_length: int) -> None:
        super().__init__(embedding_size, model_max_input_length, max_input_length)     
        

class LLM2VecLLMEncoder(LLMEncoder):
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
            gpu_factor = 1 if torch.cuda.device_count() <= 1 else int(torch.cuda.device_count() / 2)  # Divide by 2 to ensure not too large batch per GPU
            dataloader = DataLoader(TextsDataset(inputs), batch_size=self.batch_size * gpu_factor, shuffle=False)
            all_embeddings = []
            for batch in tqdm(dataloader, desc="Processing Batches"):
                batch_dict = self.tokenizer(batch, max_length=self.max_input_length, padding=True, truncation=True, return_tensors='pt').to(self.device) # type: ignore
                outputs = self.model(**batch_dict) # type: ignore
                embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']) # type: ignore
                normalized_embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach().numpy()
                all_embeddings.append(normalized_embeddings)
            return np.concatenate(all_embeddings, axis=0)

# Workaround for modern BERT - transformers version incompatible with LLM2Vec
# class LLM2VecLlama3_7B_InstructSupervisedEncoder(LLM2VecLLMEncoder):
#     def __init__(self, max_input_length: int, **kwargs) -> None:
#         pass
#         
# class LLM2VecLlama3_1_7B_InstructSupervisedEncoder(LLM2VecLLMEncoder):
#     def __init__(self, max_input_length: int, **kwargs) -> None:
#         pass
#         
# class LLM2VecLlama2_Sheared_1_3B_SupervisedEncoder(LLM2VecLLMEncoder):
#     def __init__(self, max_input_length: int, **kwargs) -> None:
#         pass
# 
# class LLM2VecMistral_7B_InstructSupervisedEncoder(LLM2VecLLMEncoder):
#     def __init__(self, max_input_length: int, **kwargs) -> None:
#         pass

class LLM2VecLlama3_7B_InstructSupervisedEncoder(LLM2VecLLMEncoder):
    
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

class LLM2VecLlama3_1_7B_InstructSupervisedEncoder(LLM2VecLLMEncoder):
    
    def __init__(self, max_input_length: int, **kwargs) -> None:
        super().__init__(embedding_size=4096, model_max_input_length=128000, max_input_length=max_input_length)
        # peft_model_name_or_path = "/mntp-supervised/Meta-Llama-3.1-8B-Instruct_1000_mntp_steps/E5_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-64_l-512_bidirectional-True_e-3_s-42_w-300_lr-0.0002_lora_r-16/checkpoint-1000"
        peft_model_name_or_path = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised"
        if 'custom_path' in kwargs:
            # TODO: Added reproducibility path
            model_path = "/home/sthe14/llm2vec-repro/output"
            peft_model_name_or_path = model_path + kwargs['custom_path']
        # Changed this to updated loading instructions from https://huggingface.co/McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised
        self.model = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp",
            trust_remote_code=True,
            peft_model_name_or_path=peft_model_name_or_path,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            max_length=self.max_input_length,
            doc_max_length=self.max_input_length,
        )

class LLM2VecLlama2_Sheared_1_3B_SupervisedEncoder(LLM2VecLLMEncoder):

    def __init__(self, max_input_length: int, **kwargs) -> None:
        super().__init__(embedding_size=2048, model_max_input_length=4096, max_input_length=max_input_length)
        self.model = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            max_length=self.max_input_length,
            doc_max_length=self.max_input_length,
        )

class LLM2VecMistral_7B_InstructSupervisedEncoder(LLM2VecLLMEncoder):
    
    def __init__(self, max_input_length: int, **kwargs) -> None:
        super().__init__(embedding_size=4096, model_max_input_length=32768, max_input_length=max_input_length)
        self.model = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised",
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
        # Enable multi-gpu support
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

class LLM2VecLlama3_1_7B_InstructSupervisedChunkedEncoder(LLM2VecLlama3_1_7B_InstructSupervisedEncoder):
    
    def __init__(self, max_input_length: int, **kwargs) -> None:
        super().__init__(max_input_length=max_input_length, **kwargs)
        
    def _encode(self, inputs: List, **kwargs) -> NDArray[Any]:
        # Use multiples of this base input length to determine the max number of chunks, e.g. for 2k chunks use max number of 2
        BASE_INPUT_LENGTH = 4096
        max_chunks = BASE_INPUT_LENGTH // self.max_input_length
        # To save memory, shorten texts to BASE_INPUT_LENGTH * 8 characters as a very loose upper bound for the number of tokens
        inputs = [text[:BASE_INPUT_LENGTH * 8] for text in inputs]
        
        # Create chunks of the inputs before calling the superclass encode method
        num_inputs = len(inputs)
        print(f"Creating chunks for {num_inputs} inputs of size {self.max_input_length} (max_chunks: {max_chunks}).")
        # NOTE: Must use self.model.tokenizer instead of self.tokenizer
        inputs, chunk_counts = self.get_chunked_dataset(inputs, self.model.tokenizer, max_chunks=max_chunks)
        
        print(f"Encoding {len(inputs)} chunks.")
        all_embeddings = super()._encode(inputs)
        
        # Average chunk embeddings for each original text
        all_embeddings = self.get_averaged_chunks(all_embeddings, chunk_counts)
        assert len(all_embeddings) == num_inputs
            
        return all_embeddings

class GTEQwen2_7B_InstructChunkedEncoder(GTEQwen2_7B_InstructEncoder):
    
    def __init__(self, max_input_length: int, **kwargs) -> None:
        super().__init__(max_input_length=max_input_length, **kwargs)
        
    def _encode(self, inputs: List, **kwargs) -> NDArray[Any]:
        # Use multiples of this base input length to determine the max number of chunks, e.g. for 2k chunks use max number of 2
        BASE_INPUT_LENGTH = 4096
        max_chunks = BASE_INPUT_LENGTH // self.max_input_length
        # To save memory, shorten texts to BASE_INPUT_LENGTH * 8 characters as a very loose upper bound for the number of tokens
        inputs = [text[:BASE_INPUT_LENGTH * 8] for text in inputs]
        
        # Create chunks of the inputs before calling the superclass encode method
        num_inputs = len(inputs)
        print(f"Creating chunks for {num_inputs} inputs of size {self.max_input_length} (max_chunks: {max_chunks}).")
        inputs, chunk_counts = self.get_chunked_dataset(inputs, self.tokenizer, max_chunks=max_chunks)
        
        print(f"Encoding {len(inputs)} chunks.")
        all_embeddings = super()._encode(inputs)
        
        # Average chunk embeddings for each original text
        all_embeddings = self.get_averaged_chunks(all_embeddings, chunk_counts)
        assert len(all_embeddings) == num_inputs
            
        return all_embeddings
               
class GTEQwen2_1_5B_InstructEncoder(Qwen2LLMEncoder):
    
    def __init__(self, max_input_length: int, **kwargs) -> None:
        super().__init__(embedding_size=1536, model_max_input_length=128000, max_input_length=max_input_length) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-1.5B-instruct', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-1.5B-instruct', trust_remote_code=True, torch_dtype=torch.float16).to(self.device)  
        
        # Enable multi-gpu support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs.")
            self.model = torch.nn.DataParallel(self.model)
        
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

class BertEncoder(BERTLLMEncoder):
    
    def __init__(self, max_input_length: int, bert_identifier: str, embedding_size: int, model_max_input_length: int, **kwargs) -> None:
        # use variable bert_identifier, embedding_size, model_max_input_length to allow for different BERT models
        super().__init__(embedding_size=embedding_size, model_max_input_length=model_max_input_length, max_input_length=max_input_length)  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(bert_identifier)
        self.model = AutoModel.from_pretrained(bert_identifier).to(self.device)

        # Enable multi-gpu support
        self.model = AutoModel.from_pretrained(bert_identifier).to(self.device)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs.")
            self.model = torch.nn.DataParallel(self.model)

    def _encode(self, inputs: List, **kwargs) -> NDArray[Any]:
        # Use multiples of this base input length to determine the max number of chunks, e.g. for 2k chunks use max number of 2
        BASE_INPUT_LENGTH = 4096
        max_chunks = BASE_INPUT_LENGTH // self.max_input_length
        # To save memory, shorten texts to BASE_INPUT_LENGTH * 8 characters as a very loose upper bound for the number of tokens
        inputs = [text[:BASE_INPUT_LENGTH * 8] for text in inputs]
        
        # Create chunks of the inputs before calling the superclass encode method
        num_inputs = len(inputs)
        print(f"Creating chunks for {num_inputs} inputs of size {self.max_input_length} (max_chunks: {max_chunks}).")
        inputs, chunk_counts = self.get_chunked_dataset(inputs, self.tokenizer, max_chunks=max_chunks)
        
        print(f"Encoding {len(inputs)} chunks.")
        gpu_factor = 1 # if torch.cuda.device_count() <= 1 else int(torch.cuda.device_count() / 2)  # Divide by 2 to ensure not too large batch per GPU
        dataloader = DataLoader(TextsDataset(inputs), batch_size=self.batch_size * gpu_factor, shuffle=False, collate_fn=lambda batch: batch)
        
        all_embeddings_list = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding Chunks"):
                inputs_dict = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_input_length, return_tensors='pt')
                inputs_dict = {k: v.to(self.device) for k, v in inputs_dict.items()}
                outputs = self.model(**inputs_dict, output_hidden_states=True)
                # Get average of all hidden states in the last hidden layer
                # Shown to be superior to cls token or max (https://arxiv.org/pdf/1908.10084)
                # For implementation see: https://github.com/autoliuweijie/BERT-whitening-pytorch/blob/b5cfbd606bd19fc3b3adf9e074dc0bfd830ef597/all_utils.py#L33
                # Want to reproduce Jiang et al. Health system-scale language models are all-purpose prediction engines 2023. However, unclear what MLM classification head exactly means.
                last_avg_embedding = outputs.hidden_states[-1].mean(dim=1)
                all_embeddings_list.append(last_avg_embedding.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings_list, axis=0)

        all_embeddings = self.get_averaged_chunks(all_embeddings, chunk_counts)
        assert len(all_embeddings) == num_inputs

        return all_embeddings
        
#         # Old routine: 
#         dataset = dataset.map(self.get_last_avg_embedding, batched=True, batch_size=self.batch_size)
#         all_embeddings = np.array(dataset['embedding'])
#         
#         # Average chunk embeddings for each original text
#         all_embeddings = self.get_averaged_chunks(all_embeddings, chunk_counts)
#         assert len(all_embeddings) == num_inputs
#             
#         return all_embeddings

class TextEncoder:
    def __init__(self, encoder: LLMEncoder):
        self.encoder = encoder
        
    def _store_or_check_fingerprint(self, inputs: List, cache_dir: str) -> None:
        fingerprint_file = os.path.join(cache_dir, "cache_fingerprint.txt")
        
        # Generate fingerprint
        hasher = hashlib.sha256()
        for input in inputs:
            hasher.update((str(input)).encode('utf-8'))
        fingerprint = str(len(inputs)) + '-' + hasher.hexdigest()
        
        # Check for existing fingerprint
        if os.path.exists(fingerprint_file):
            with open(fingerprint_file, "r") as f:
                existing_fingerprint = f.read().strip()
            if existing_fingerprint != fingerprint:
                raise ValueError("Cache fingerprint does not match. Data inconsistency detected.")
        else:
            with open(fingerprint_file, "w") as f:
                f.write(fingerprint)

    def _get_cache_files(self, cache_dir: str) -> List[str]:
        file_names = os.listdir(cache_dir)
        return [f for f in file_names if f.startswith('cache_') and f.endswith('.pkl')]
    
    def _delete_all_cache_files(self, cache_dir: str) -> None:
        cache_files = self._get_cache_files(cache_dir)
        for cache_file in cache_files:
            os.remove(os.path.join(cache_dir, cache_file))
    
    def encode_texts(self, instructions: List[str], texts: List[str], cache_dir: str) -> NDArray[Any]:
        if all([instruction is None or len(instruction) == 0 for instruction in instructions]):
            inputs = texts
        else:
            inputs = [self.encoder.add_instruction(instruction, text) for instruction, text in zip(instructions, texts)]
        
        # Old variant: process with duplicates
        return self.encoder._encode(inputs)
        
        # # Performance improvement: Remove exact duplicates and restore them after encoding
        # # Careful: inputs are lists of strings, so we need to convert them to tuples for hashing
        # def serialize_input(input):
        #     return input if isinstance(input, str) else tuple(input)
        
        # # Store original indices of inputs
        # input_to_indices = defaultdict(list)
        # for i, input in enumerate(inputs):
        #     input_to_indices[serialize_input(input)].append(i)
        #     
        # # Deduplicate inputs while preserving the first occurrence and encode them
        # unique_inputs = list(input_to_indices.keys())
        
        # # Performance improvement: Process in batches and store intermediate results
        # batch_size = 600576 # 131072 # 65536 / 131072 (around 10 batches for full) / 2097152
        # current_index = 0
        
        # # Store or check fingerprint
        # self._store_or_check_fingerprint(unique_inputs, cache_dir)
        
        # # Load cached intermediate results of format cache_{start_index}.pkl
        # cache_files = self._get_cache_files(cache_dir)
        # if len(cache_files) > 0:
        #     max_start_indices = max([int(f.split('_')[1].split('.')[0]) for f in cache_files])
        #     current_index = max_start_indices
        #     cache_file = os.path.join(cache_dir, f"cache_{max_start_indices}.pkl")
        #     with open(cache_file, "rb") as f:
        #         unique_embeddings = pickle.load(f)
        #         if isinstance(unique_embeddings, np.ndarray):
        #             unique_embeddings = unique_embeddings.tolist()
        #     print(f"Loaded {len(unique_embeddings)} cached embeddings.")
        # else:
        #     unique_embeddings = []
        #     print("No cache files found.")
        #     
        # # Create embeddings
        # for start in range(current_index, len(unique_inputs), batch_size):
        #     print(f"Processing batch {start // batch_size + 1} of {len(unique_inputs) // batch_size + 1}")
        #     batch = unique_inputs[start:start + batch_size]
        #     batch_embeddings = self.encoder._encode(batch)
        #     unique_embeddings.extend(batch_embeddings.tolist())
        #     
        #     # Delete all old cache files
        #     self._delete_all_cache_files(cache_dir)
        #         
        #     # Save intermediate results
        #     cache_file = os.path.join(cache_dir, f"cache_{len(unique_embeddings)}.pkl")
        #     with open(cache_file, "wb") as f:
        #         pickle.dump(unique_embeddings, f)
        #     print(f"Saved {len(unique_embeddings)} embeddings to {cache_file}")
        #        
        # # Delete all old cache files
        # self._delete_all_cache_files(cache_dir)
        
        # # Restore deduplicated embeddings to original order
        # embeddings = [None] * len(inputs)
        # for i, input in enumerate(unique_inputs):
        #     for j in input_to_indices[input]:
        #         embeddings[j] = unique_embeddings[i]
        # assert all(embedding is not None for embedding in embeddings)
        # return np.array(embeddings)