import argparse
import os
from typing import Optional

import torch
from datasets import load_from_disk
from tokenizers import Tokenizer
from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration

from src.training.dataset import T5TranslationDataset


def _find_hf_checkpoints(search_root: str = "models") -> list[str]:
    """
    Best-effort discovery of local Hugging Face checkpoints.
    We consider a directory a checkpoint if it contains `config.json`.
    """
    candidates: list[str] = []
    if not os.path.isdir(search_root):
        return candidates

    for dirpath, _, filenames in os.walk(search_root):
        if "config.json" in filenames:
            candidates.append(dirpath)
    candidates.sort()
    return candidates


def finetune(
    parallel_path: str = "data/cleaned_banglanmt_parallel",
    src_lang: str = "en",
    tgt_lang: str = "bn",
    init_model_dir: str = "models/checkpoints/final",
    output_dir: str = "models/finetuned_en_bn",
    tokenizer_path: str = "models/tokenizer/tokenizer.json",
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 3e-4,
    max_input_length: int = 128,
    max_target_length: int = 128,
    val_size: float = 0.01,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    prefix: str = "translate English to Bengali: ",
):
    if not os.path.isdir(init_model_dir):
        found = _find_hf_checkpoints("models")
        if found:
            # If `final/` wasn't produced (e.g. interrupted run), fall back to latest checkpoint.
            latest = found[-1]
            print(
                f"WARNING: init_model_dir not found: {init_model_dir}\n"
                f"Falling back to latest detected checkpoint: {latest}"
            )
            init_model_dir = latest
        else:
            raise FileNotFoundError(
                f"init_model_dir not found: {init_model_dir}. "
                "Run pre-training first (src.training.train) or pass --init_model_dir.\n\n"
                "No local Hugging Face checkpoints were found under `models/`.\n"
                "Create one by running pre-training first, e.g.:\n"
                "  venv/bin/python -m src.training.train --output_dir models/checkpoints\n"
                "Then fine-tune with:\n"
                "  --init_model_dir models/checkpoints/final"
            )

    print("Loading tokenizer...")
    tok = Tokenizer.from_file(tokenizer_path)
    pad_token_id = tok.token_to_id("[PAD]")
    eos_token_id = tok.token_to_id("[SEP]")
    if pad_token_id is None:
        raise ValueError("Tokenizer is missing required special token [PAD].")
    if eos_token_id is None:
        raise ValueError("Tokenizer is missing required special token [SEP] (used as EOS).")

    print(f"Loading initial model from {init_model_dir} ...")
    model = T5ForConditionalGeneration.from_pretrained(init_model_dir)
    model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = eos_token_id
    model.config.decoder_start_token_id = pad_token_id

    print(f"Loading cleaned parallel dataset from {parallel_path} ...")
    ds = load_from_disk(parallel_path)
    print(f"Parallel examples: {len(ds)}")

    if val_size <= 0 or val_size >= 1:
        raise ValueError("--val_size must be in (0, 1).")

    print(f"Creating train/val split (val_size={val_size}, seed={seed}) ...")
    split = ds.train_test_split(test_size=val_size, seed=seed, shuffle=True)
    ds_train = split["train"]
    ds_val = split["test"]

    if max_train_samples is not None:
        ds_train = ds_train.select(range(min(max_train_samples, len(ds_train))))
    if max_eval_samples is not None:
        ds_val = ds_val.select(range(min(max_eval_samples, len(ds_val))))

    print(f"Train examples: {len(ds_train)} | Val examples: {len(ds_val)}")

    train_dataset = T5TranslationDataset(
        tokenizer_path=tokenizer_path,
        parallel_path=parallel_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        prefix=prefix,
        dataset=ds_train,
    )
    eval_dataset = T5TranslationDataset(
        tokenizer_path=tokenizer_path,
        parallel_path=parallel_path,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        prefix=prefix,
        dataset=ds_val,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        prediction_loss_only=True,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Fine-tuning complete. Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune on cleaned parallel translation data.")
    parser.add_argument("--parallel_path", default="data/cleaned_banglanmt_parallel")
    parser.add_argument("--src_lang", default="en")
    parser.add_argument("--tgt_lang", default="bn")
    parser.add_argument("--init_model_dir", default="models/checkpoints/final")
    parser.add_argument("--output_dir", default="models/finetuned_en_bn")
    parser.add_argument("--tokenizer_path", default="models/tokenizer/tokenizer.json")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--val_size", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--prefix", default="translate English to Bengali: ")

    a = parser.parse_args()
    finetune(
        parallel_path=a.parallel_path,
        src_lang=a.src_lang,
        tgt_lang=a.tgt_lang,
        init_model_dir=a.init_model_dir,
        output_dir=a.output_dir,
        tokenizer_path=a.tokenizer_path,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        max_input_length=a.max_input_length,
        max_target_length=a.max_target_length,
        val_size=a.val_size,
        seed=a.seed,
        max_train_samples=a.max_train_samples,
        max_eval_samples=a.max_eval_samples,
        prefix=a.prefix,
    )

