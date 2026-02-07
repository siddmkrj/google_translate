import os
from datasets import load_from_disk
import sys

def inspect_dataset(path, name, num_samples=3):
    print(f"\n{'='*60}")
    print(f"📦 Inspecting: {name}")
    print(f"   Path: {path}")
    print(f"{'-'*60}")

    if not os.path.exists(path):
        print(f"❌ Dataset NOT found at {path}")
        return

    try:
        ds = load_from_disk(path)
        print(f"✅ Loaded successfully.")
        print(f"   Total Samples: {len(ds)}")
        print(f"{'-'*60}")
        
        # Select samples
        samples = ds.select(range(min(len(ds), num_samples)))
        
        for i, sample in enumerate(samples):
            print(f"\n🔹 Sample {i+1}:")
            for key, value in sample.items():
                val_str = str(value)
                # Truncate for console display
                if len(val_str) > 300:
                    val_str = val_str[:300] + "... [TRUNCATED]"
                print(f"   {key}: {val_str}")
                
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")

def main():
    print("running data visualization...")
    
    # 1. English
    inspect_dataset("data/wikitext_wikitext-2-raw-v1_train", "Raw Wikitext (English)")
    inspect_dataset("data/cleaned_wikitext_train", "Cleaned Wikitext (English)")

    # 2. Bengali
    inspect_dataset("data/wikimedia_wikipedia_20231101.bn_train", "Raw Wikipedia (Bengali)")
    inspect_dataset("data/cleaned_wikipedia_bn_train", "Cleaned Wikipedia (Bengali)")

    # 3. Parallel
    inspect_dataset("data/csebuetnlp_banglanmt_parallel", "BanglaNMT Parallel Data")
    
    # 4. Balanced (if exists)
    if os.path.exists("data/balanced_wikitext_train"):
        inspect_dataset("data/balanced_wikitext_train", "Balanced Wikitext (English)")

if __name__ == "__main__":
    # Ensure we are running from project root or handle relative paths
    # Assuming running from project root as per standard instructions
    if not os.path.exists("data"):
        print("⚠️  'data' directory not found. Please run this script from the project root.")
        print(f"   Current CWD: {os.getcwd()}")
    else:
        main()
