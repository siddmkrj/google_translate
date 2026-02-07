import os
import shutil
import sys
from datasets import Dataset
import pandas as pd

# Ensure src is in path
sys.path.append(os.getcwd())

from src.data.clean_parallel import clean_parallel_dataset

def test_cleanup():
    # Setup paths
    input_path = "data/test_input"
    output_path = "data/test_output"
    model_path = "models/lid.176.bin"
    
    # Cleanup previous runs
    if os.path.exists(input_path): shutil.rmtree(input_path)
    if os.path.exists(output_path): shutil.rmtree(output_path)
    
    # Create dummy data
    data = [
        # 1. Valid
        {"translation": {"en": "Hello world", "bn": "ওহে বিশ্ব"}},
        
        # 2. Duplicate (Exact match to 1)
        {"translation": {"en": "Hello world", "bn": "ওহে বিশ্ব"}},
        
        # 3. Noise (Length Ratio)
        {"translation": {"en": "Hi", "bn": "This is a very long sentence that should definitely be removed because the ratio is way off"}},
        
        # 4. Noise (Empty)
        {"translation": {"en": "", "bn": "Something"}},
        
        # 5. Consistency (Numbers)
        {"translation": {"en": "I have 3 cats", "bn": "আমার চারটি বিড়াল আছে"}}, # 3 vs 4 (mismatched digits 3 vs nothing, assuming 'চারটি' is word 4)
        # Note: 'চারটি' is word, '4' is digit. My number counter only counts digits \d+.
        # So "3" count is 1. "চারটি" count is 0. 1 != 0. Removed.
        
        # 6. Valid with numbers
        {"translation": {"en": "I have 2 dogs", "bn": "আমার 2 টি কুকুর আছে"}},
    ]
    
    print("Creating input dataset...")
    # Convert to standard format
    dataset = Dataset.from_list(data)
    dataset.save_to_disk(input_path)
    
    print("Running cleanup...")
    clean_parallel_dataset(input_path, output_path, src_lang='en', tgt_lang='bn', lid_model_path=model_path)
    
    print("Verifying output...")
    cleaned_dataset = Dataset.load_from_disk(output_path)
    
    print(f"Input size: {len(data)}")
    print(f"Output size: {len(cleaned_dataset)}")
    
    cleaned_data = [item['translation'] for item in cleaned_dataset]
    for item in cleaned_data:
        print(f"Kept: {item}")
        
    # Expectations
    # 1. Kept
    # 2. Removed (Duplicate)
    # 3. Removed (Ratio)
    # 4. Removed (Empty)
    # 5. Removed (Number mismatch: 1 vs 0)
    # 6. Kept
    
    assert len(cleaned_dataset) == 2, f"Expected 2 items, got {len(cleaned_dataset)}"
    assert cleaned_data[0]['en'] == "Hello world"
    assert cleaned_data[1]['en'] == "I have 2 dogs"
    
    print("TEST PASSED!")
    
    # Cleanup
    shutil.rmtree(input_path)
    shutil.rmtree(output_path)

if __name__ == "__main__":
    test_cleanup()
