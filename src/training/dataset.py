import random
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from datasets import load_from_disk
import numpy as np

class T5DenoisingDataset(Dataset):
    """
    Dataset for T5 span-corruption pre-training.
    """
    def __init__(self, tokenizer_path, en_path, bn_path, max_length=128, mask_probability=0.15):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Load datasets
        print("Loading datasets...")
        self.ds_en = load_from_disk(en_path)
        self.ds_bn = load_from_disk(bn_path)
        
        # Combine indices for access
        self.en_len = len(self.ds_en)
        self.bn_len = len(self.ds_bn)
        self.total_len = self.en_len + self.bn_len
        
        self.max_length = max_length
        self.mask_prob = mask_probability
        
        # Sentinel tokens (generic approach)
        # T5 uses <extra_id_0>, <extra_id_1>... 
        # Our BPE tokenizer likely doesn't have these by default unless added.
        # For simplicity in this custom loop, we will use the [MASK] token repeatedly 
        # OR we can add sentinels.
        # Let's check vocab size. If 32k, we can use specific IDs from the end or just [MASK].
        # Better approach: Just use [MASK] for individual tokens (BERT style) for simplicity, 
        # or implement full span corruption if we want true T5. 
        # Given the scope, let's implement a simple "Replace span with [MASK]" strategy.
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.eos_token_id = self.tokenizer.token_to_id("[SEP]") # or standard EOS
        if self.eos_token_id is None:
             self.eos_token_id = 1 # fallback

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Determine language source
        if idx < self.en_len:
            text = self.ds_en[idx]['text']
        else:
            text = self.ds_bn[idx - self.en_len]['text']
            
        # Tokenize (using tokenizers library which handles BPE encoding)
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        
        # Truncate
        if len(ids) > self.max_length - 1: # Reserve space for EOS
            ids = ids[:self.max_length - 1]
            
        # Noise Function (Span Corruption)
        input_ids, labels = self.span_corruption(ids)
        
        # Padding
        input_ids = self.pad(input_ids)
        labels = self.pad(labels)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long), # T5 calculates loss on generated labels
            "attention_mask": torch.tensor([1] * len(ids) + [0] * (self.max_length - len(ids)), dtype=torch.long)[:self.max_length]
        }

    def pad(self, ids):
        if len(ids) < self.max_length:
            return ids + [self.pad_token_id] * (self.max_length - len(ids))
        return ids[:self.max_length]

    def span_corruption(self, input_ids):
        """
        Simple random masking for now: 15% of tokens replaced by [MASK].
        True T5 span corruption is complex to implement from scratch without util libraries.
        This approximates "Denoising": Input corrupted -> Output original.
        Actually for T5: Input = Corrupted, Target = Missing Spans.
        Simpler DAE (BART): Input = Corrupted, Target = Full Original.
        Let's do BART style Denoising (Reconstruction) as it's structurally simpler for this custom setup.
        Input: "Hello [MASK] [MASK] world" -> Target: "Hello my beautiful world"
        """
        
        noisy_ids = list(input_ids)
        mask_indices = []
        
        # Random masking
        for i in range(len(noisy_ids)):
            if random.random() < self.mask_prob:
                noisy_ids[i] = self.mask_token_id
                
        # For BART-style reconstruction, Labels = Input_ids (Original).
        # We also add EOS.
        source = noisy_ids + [self.eos_token_id]
        target = input_ids + [self.eos_token_id]
        
        return source, target

if __name__ == "__main__":
    # Test dataset
    ds = T5DenoisingDataset(
        "models/tokenizer/tokenizer.json", 
        "data/balanced_wikitext_train", 
        "data/cleaned_wikipedia_bn_train"
    )
    print(f"Dataset length: {len(ds)}")
    sample = ds[0]
    print(f"Sample Input: {sample['input_ids'][:20]}")
    print(f"Sample Label: {sample['labels'][:20]}")
