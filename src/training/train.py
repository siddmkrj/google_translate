import argparse
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from src.training.model import get_model
from src.training.dataset import T5DenoisingDataset
from tokenizers import Tokenizer

def train(output_dir="models/checkpoints", tokenizer_path="models/tokenizer/tokenizer.json", en_path="data/cleaned_wikitext_train", bn_path="data/cleaned_wikipedia_bn_train", epochs=3, batch_size=8, lr=5e-4):
    
    # 1. Dataset
    print("Initializing Dataset...")
    dataset = T5DenoisingDataset(tokenizer_path, en_path, bn_path)
    
    # 2. Model
    print("Initializing Model...")
    # Vocab size from tokenizer
    tokenizer_obj = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer_obj.get_vocab_size()
    model = get_model(vocab_size=vocab_size)
    
    # 3. Data Collator
    # For T5, we usually pad dynamically.
    # Since our dataset already returns tensors, we can use the standard collator 
    # but we need to ensure padding token is set correctly.
    # Our tokenizer wrapper is raw `tokenizers`, not `transformers.PreTrainedTokenizer`.
    # So we might need a custom collator or just rely on default if we pad in dataset (which we did).
    # Actually, `default_data_collator` stacks tensors. Our dataset outputs fixed length (padded).
    # So default collator is fine.
    
    # 4. Training Arguments
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        save_steps=500,
        logging_steps=100,
        prediction_loss_only=True,
        remove_unused_columns=False, # Custom dataset keys
        fp16=torch.cuda.is_available(), # Use Mixed Precision if CUDA available (MPS fp16 support varies, standard auto-detection is better)
    )
    
    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )
    
    print("Starting Training...")
    trainer.train()
    
    print("Training Complete. Saving Final Model...")
    trainer.save_model(output_dir + "/final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tokenizer_path", default="models/tokenizer/tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    
    parser.add_argument("--en_path", default="data/balanced_wikitext_train", help="Path to English dataset")
    parser.add_argument("--bn_path", default="data/cleaned_wikipedia_bn_train", help="Path to Bengali dataset")
    
    args = parser.parse_args()
    train(output_dir=args.output_dir, tokenizer_path=args.tokenizer_path, en_path=args.en_path, bn_path=args.bn_path, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
