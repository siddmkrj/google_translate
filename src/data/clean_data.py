import argparse
import os
import re
import hashlib
import unicodedata
from datasets import load_from_disk
import fasttext

def normalize_text(text):
    """
    Normalizes text using NFKC and strips whitespace.
    """
    if not text:
        return ""
    return unicodedata.normalize('NFKC', text).strip()

def clean_characters(text, language):
    """
    Removes unsupported characters based on language.
    """
    if not text:
        return ""
        
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove Emojis (basic range) and other non-printable characters could be handled here.
    # For now, we will be strict about allowed characters.

    if language == 'en':
        # Keep Latin, numbers, punctuation, whitespace
        # This is a broad regex, can be tightened.
        # \p{L} is letter, but python re doesn't support unicode properties well without regex module.
        # We'll use strict ranges for now.
        # Latin: a-zA-Z, common european accents might be needed but wikitext usually is standard.
        # Let's simple check: Allow ASCII + common punctuation.
        # Actually for "General English" we usually want to keep it simple.
        # Remove anything that isn't roughly standard text.
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"\-()\[\];:]', ' ', text)
        
    elif language == 'bn':
        # Keep Bengali, numbers, punctuation, whitespace
        # Bengali Unicode block: 0980–09FF
        # Also keep standard English punctuation as it's often used in Bengali strings.
        text = re.sub(r'[^\u0980-\u09FFa-zA-Z0-9\s.,!?\'"\-()\[\];:]', ' ', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_language(text, model):
    """
    Predicts language of text using fasttext model.
    Returns 'en', 'bn', or 'other'.
    """
    text = text.replace('\n', ' ')
    if not text.strip():
        return 'other'
        
    labels, scores = model.predict(text)
    label = labels[0].replace('__label__', '')
    score = scores[0]
    
    return label, score

def clean_dataset(dataset_path, output_path, lang, model_path):
    """
    Loads, cleans, and saves the dataset.
    """
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    print(f"Original size: {len(dataset)}")
    
    # Load FastText model
    print(f"Loading ID model from {model_path}...")
    model = fasttext.load_model(model_path)
    
    seen_hashes = set()
    
    def process_batch(batch):
        cleaned_texts = []
        is_valid_mask = []
        
        for text in batch['text']:
            # 1. Normalize
            text = normalize_text(text)
            
            # 2. Character Clean
            text = clean_characters(text, lang)
            
            if not text or len(text) < 10: # Minimum length filter
                cleaned_texts.append("")
                is_valid_mask.append(False)
                continue

            # 3. Deduplication (Exact hash)
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            if text_hash in seen_hashes:
                cleaned_texts.append("")
                is_valid_mask.append(False)
                continue
            seen_hashes.add(text_hash)
            
            # 4. Language ID
            pred_lang, score = get_language(text, model)
            if pred_lang != lang or score < 0.5: # Threshold
                 cleaned_texts.append("")
                 is_valid_mask.append(False)
                 continue
                 
            cleaned_texts.append(text)
            is_valid_mask.append(True)
            
        return {"text": cleaned_texts, "valid": is_valid_mask}

    print("Cleaning dataset...")
    # Map to clean and mark valid
    dataset = dataset.map(process_batch, batched=True, batch_size=1000)
    
    # Filter valid
    dataset = dataset.filter(lambda example: example['valid'])
    # Remove 'valid' column, keep 'text' (and others if present, but we largely care about text)
    dataset = dataset.remove_columns(['valid'])
    
    print(f"Cleaned size: {len(dataset)}")
    print(f"Saving to {output_path}...")
    dataset.save_to_disk(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="Path to input dataset disk folder")
    parser.add_argument("--output_path", required=True, help="Path to save cleaned dataset")
    parser.add_argument("--lang", required=True, choices=['en', 'bn'], help="Expected language")
    parser.add_argument("--model_path", default="models/lid.176.bin", help="Path to fasttext model")
    
    args = parser.parse_args()
    
    clean_dataset(args.dataset_path, args.output_path, args.lang, args.model_path)
