import argparse
from datasets import load_from_disk
import os

def balance_dataset(input_path, output_path, num_samples, seed=42):
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)
    current_len = len(dataset)
    print(f"Current size: {current_len}")
    
    if current_len <= num_samples:
        print(f"Dataset already smaller or equal to {num_samples}. No downsampling needed (copying to output).")
        dataset.save_to_disk(output_path)
        return

    print(f"Downsampling to {num_samples} samples (seed={seed})...")
    # Shuffle and select
    downsampled_dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    
    print(f"Saving to {output_path}...")
    downsampled_dataset.save_to_disk(output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Input dataset path")
    parser.add_argument("--output_path", required=True, help="Output dataset path")
    parser.add_argument("--num_samples", type=int, required=True, help="Target number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    balance_dataset(args.input_path, args.output_path, args.num_samples, args.seed)
