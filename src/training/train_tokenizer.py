import argparse
import os
from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

def train_tokenizer(en_path, bn_path, output_dir, vocab_size=32000):
    print(f"Loading datasets...")
    ds_en = load_from_disk(en_path)
    ds_bn = load_from_disk(bn_path)
    
    print(f"English samples: {len(ds_en)}")
    print(f"Bengali samples: {len(ds_bn)}")
    
    # Iterator for training
    def batch_iterator():
        batch_size = 1000
        for i in range(0, len(ds_en), batch_size):
            yield ds_en[i : i + batch_size]["text"]
        for i in range(0, len(ds_bn), batch_size):
            yield ds_bn[i : i + batch_size]["text"]

    # Initialize Tokenizer (BPE)
    print("Initializing BPE Tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # Pre-normalization (create clean splits)
    # Using specific pre-tokenizers isn't strictly necessary if data is well cleaned, 
    # but generic whitespace splitting is good for BPE.
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    print("Training tokenizer (this may take a moment)...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # Post-processor (optional, but good for BERT/RoBERTa style)
    # For T5/Seq2Seq, standard might be simpler. Let's stick to simple BPE for now.
    
    # Decoder
    tokenizer.decoder = decoders.ByteLevel() # or BPEDecoder
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "tokenizer.json")
    print(f"Saving tokenizer to {save_path}...")
    tokenizer.save(save_path)
    print("Tokenizer saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_path", default="data/cleaned_wikitext_train", help="Path to English dataset")
    parser.add_argument("--bn_path", default="data/cleaned_wikipedia_bn_train", help="Path to Bengali dataset")
    parser.add_argument("--output_dir", default="models/tokenizer", help="Output directory")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    
    args = parser.parse_args()
    
    train_tokenizer(args.en_path, args.bn_path, args.output_dir, args.vocab_size)
