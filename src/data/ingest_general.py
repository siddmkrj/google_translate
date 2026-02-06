import argparse
import os
from datasets import load_dataset, Dataset

def ingest_data(dataset_name, config_name=None, split='train', output_dir='data', max_samples=None, streaming=False, trust_remote_code=False):
    """
    Ingests a dataset from Hugging Face and saves it locally.
    """
    print(f"Loading dataset: {dataset_name} (config: {config_name}, split: {split}, streaming: {streaming}, trust_remote_code: {trust_remote_code})")
    
    try:
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split=split, streaming=streaming, trust_remote_code=trust_remote_code)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming, trust_remote_code=trust_remote_code)
            
        if streaming:
            print("Streaming mode enabled.")
            if max_samples:
                print(f"Taking first {max_samples} samples from stream...")
                dataset_head = dataset.take(max_samples)
                # Convert iterable dataset to standard dataset for saving
                dataset = Dataset.from_generator(lambda: (row for row in dataset_head))
            else:
                 # If streaming but no max_samples, we iterate all (dangerous if huge) or user intends to process stream. 
                 # But sticking to the goal of "saving to disk", we must materialize it.
                 print("Warning: Streaming enabled but no max_samples set. Attempting to materialize entire stream (could be slow)...")
                 dataset = Dataset.from_generator(lambda: (row for row in dataset))
                 
            print(f"Materialized dataset size: {len(dataset)} examples.")
        else:
            print(f"Dataset loaded. Original Size: {len(dataset)} examples.")
            if max_samples and max_samples < len(dataset):
                print(f"Limiting to {max_samples} samples.")
                dataset = dataset.select(range(max_samples))
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save dataset
        # Construct a safe filename
        safe_name = dataset_name.replace("/", "_")
        if config_name:
            safe_name += f"_{config_name}"
        safe_name += f"_{split}"
        
        save_path = os.path.join(output_dir, safe_name)
        print(f"Saving dataset to {save_path}...")
        dataset.save_to_disk(save_path)
        print("Dataset saved successfully.")
        
    except Exception as e:
        print(f"Error ingesting dataset: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest datasets from Hugging Face.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset on Hugging Face (e.g., 'wikitext')")
    parser.add_argument("--config", type=str, default=None, help="Dataset configuration (e.g., 'wikitext-2-raw-v1')")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save the dataset")

    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to ingest")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (recommended for large datasets)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code (required for some datasets like mc4)")

    args = parser.parse_args()

    ingest_data(args.dataset, args.config, args.split, args.output_dir, args.max_samples, args.streaming, args.trust_remote_code)
