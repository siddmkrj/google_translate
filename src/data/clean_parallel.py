import argparse
import hashlib
import os
import re
from typing import Set, Tuple, List, Dict

from datasets import load_from_disk, Dataset
import fasttext
from tqdm import tqdm

# Import shared utilities
# Assumes running as module: python -m src.data.clean_parallel
# Or if src is in path.
from src.data import normalize_text, clean_characters, get_language

def count_numbers(text):
    """
    Counts sequences of digits in text.
    """
    return len(re.findall(r'\d+', text))

class ParallelCleaner:
    def __init__(self, src_lang: str, tgt_lang: str, lid_model_path: str, min_len: int = 2, max_len: int = 150, ratio_threshold: float = 3.0):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.min_len = min_len
        self.max_len = max_len
        self.ratio_threshold = ratio_threshold
        
        print(f"Loading LID model from {lid_model_path}...")
        self.lid_model = fasttext.load_model(lid_model_path)
        self.seen_hashes: Set[str] = set()

    def is_valid_length(self, src_text: str, tgt_text: str) -> bool:
        src_len = len(src_text.split())
        tgt_len = len(tgt_text.split())
        
        if src_len < self.min_len or tgt_len < self.min_len:
            return False
        if src_len > self.max_len or tgt_len > self.max_len:
            return False
            
        # Check ratio (length difference shouldn't be too extreme)
        if src_len == 0 or tgt_len == 0: return False # avoid div/0
        ratio = max(src_len, tgt_len) / min(src_len, tgt_len)
        if ratio > self.ratio_threshold:
            return False
            
        return True

    def check_consistency(self, src_text: str, tgt_text: str) -> bool:
        # Number Check
        src_nums = count_numbers(src_text)
        tgt_nums = count_numbers(tgt_text)
        if src_nums != tgt_nums:
            return False
            
        return True

    def process_batch(self, batch):
        cleaned_src_texts = []
        cleaned_tgt_texts = []
        is_valid_mask = []
        
        # Determine keys for source and target
        # Typical HF datasets have 'translation': {'en': '...', 'bn': '...'}
        # But some might be flat. We'll handle 'translation' dict column primarily.
        
        # If 'translation' column exists and is a list of dicts
        translations = batch.get('translation', None)
        
        # Fallback if flat columns (e.g. 'en', 'bn') - adjust based on actual input
        # Assuming standard 'translation' dict format for now as per `ingest_parallel.py`
        
        for i, item in enumerate(translations):
            if not isinstance(item, dict):
                 # Skip invalid format
                 cleaned_src_texts.append("")
                 cleaned_tgt_texts.append("")
                 is_valid_mask.append(False)
                 continue

            src_raw = item.get(self.src_lang, "")
            tgt_raw = item.get(self.tgt_lang, "")
            
            # 1. Normalize
            src_norm = normalize_text(src_raw)
            tgt_norm = normalize_text(tgt_raw)
            
            # 2. Clean Characters
            src_clean = clean_characters(src_norm, self.src_lang)
            tgt_clean = clean_characters(tgt_norm, self.tgt_lang)
            
            # 3. Validation: Empty
            if not src_clean or not tgt_clean:
                cleaned_src_texts.append("")
                cleaned_tgt_texts.append("")
                is_valid_mask.append(False)
                continue
                
            # 4. Validation: Deduplication
            # Hash combined source and target
            pair_hash = hashlib.md5((src_clean + "\t" + tgt_clean).encode('utf-8')).hexdigest()
            if pair_hash in self.seen_hashes:
                cleaned_src_texts.append("")
                cleaned_tgt_texts.append("")
                is_valid_mask.append(False)
                continue
            
            # 5. Validation: Length / Ratio
            if not self.is_valid_length(src_clean, tgt_clean):
                 cleaned_src_texts.append("")
                 cleaned_tgt_texts.append("")
                 is_valid_mask.append(False)
                 continue

            # 6. Validation: Language ID
            # Only check if verify strict is on, or just check source for stricter control
            # Checking both doubles inference time.
            # Let's check source is src_lang OR 'en' (if src is en)
            # and target is tgt_lang OR 'other' (if confidence low)
            # For strictness:
            src_pred, src_score = get_language(src_clean, self.lid_model)
            tgt_pred, tgt_score = get_language(tgt_clean, self.lid_model)
            
            # If high confidence and wrong language, reject
            # Threshold 0.5
            if src_score > 0.5 and src_pred != self.src_lang:
                 cleaned_src_texts.append("")
                 cleaned_tgt_texts.append("")
                 is_valid_mask.append(False)
                 continue
            if tgt_score > 0.5 and tgt_pred != self.tgt_lang:
                 cleaned_src_texts.append("")
                 cleaned_tgt_texts.append("")
                 is_valid_mask.append(False)
                 continue
                 
            # 7. Validation: Consistency (Numbers)
            if not self.check_consistency(src_clean, tgt_clean):
                 cleaned_src_texts.append("")
                 cleaned_tgt_texts.append("")
                 is_valid_mask.append(False)
                 continue
            
            # If passed all checks
            self.seen_hashes.add(pair_hash)
            cleaned_src_texts.append(src_clean)
            cleaned_tgt_texts.append(tgt_clean)
            is_valid_mask.append(True)
            
        return {
            "cleaned_translation": [
                {self.src_lang: s, self.tgt_lang: t} if v else None 
                for s, t, v in zip(cleaned_src_texts, cleaned_tgt_texts, is_valid_mask)
            ],
            "valid": is_valid_mask
        }

def clean_parallel_dataset(input_path, output_path, src_lang, tgt_lang, lid_model_path):
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)
    print(f"Original size: {len(dataset)}")
    
    cleaner = ParallelCleaner(src_lang, tgt_lang, lid_model_path)
    
    print("Cleaning dataset...")
    # Map to clean and mark valid
    # We use batched=True for speed
    dataset = dataset.map(cleaner.process_batch, batched=True, batch_size=1000)
    
    # Filter valid
    dataset = dataset.filter(lambda example: example['valid'])
    
    # Restore structure
    # The 'cleaned_translation' contains the good dicts, but we need to remove the old 'translation' and rename
    # OR just update 'translation'
    
    def restore_structure(example):
        return {"translation": example["cleaned_translation"]}
    
    dataset = dataset.map(restore_structure, remove_columns=["cleaned_translation", "valid", "translation"])
    # Note: verify if columns match what we want. 
    # The map above removes "translation" (old) and puts new one in.
    
    print(f"Cleaned size: {len(dataset)}")
    print(f"Saving to {output_path}...")
    dataset.save_to_disk(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to input dataset disk folder")
    parser.add_argument("--output_path", required=True, help="Path to save cleaned dataset")
    parser.add_argument("--src", required=True, help="Source language code")
    parser.add_argument("--tgt", required=True, help="Target language code")
    parser.add_argument("--model_path", default="models/lid.176.bin", help="Path to fasttext model")
    
    args = parser.parse_args()
    
    clean_parallel_dataset(args.input_path, args.output_path, args.src, args.tgt, args.model_path)
