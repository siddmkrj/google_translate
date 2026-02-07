import argparse
import os
from datasets import load_dataset, Dataset

def ingest_parallel_data(dataset_name, config_name=None, split='train', output_dir='data', src_lang='en', tgt_lang='bn', max_samples=None):
    """
    Ingests a parallel dataset from Hugging Face and saves it locally.
    Expects dataset to have 'translation' column or similar structure.
    """
    print(f"Loading parallel dataset: {dataset_name} (config: {config_name}, split: {split})")
    
    try:
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
            
        print(f"Dataset loaded. Original Size: {len(dataset)} examples.")

        if max_samples and max_samples < len(dataset):
            print(f"Limiting to {max_samples} samples.")
            dataset = dataset.select(range(max_samples))
            
        # Verify structure (usually 'translation': {'en': '...', 'bn': '...'})
        sample = dataset[0]
        print(f"Sample structure: {sample}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        safe_name = dataset_name.replace("/", "_")
        if config_name:
            safe_name += f"_{config_name}"
        safe_name += "_parallel"
        
        output_path = os.path.join(output_dir, safe_name)
        print(f"Saving dataset to {output_path}...")
        dataset.save_to_disk(output_path)
        print("Dataset saved successfully.")
        
    except Exception as e:
        print(f"Error ingesting dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest parallel datasets from Hugging Face.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--config", type=str, default=None, help="Dataset configuration")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()

    ingest_parallel_data(args.dataset, args.config, args.split, args.output_dir, max_samples=args.max_samples)
