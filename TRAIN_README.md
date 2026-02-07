# Training Pipeline Documentation

This document tracks the training process for the English-Bengali Translator.

## Quick Start (Runbook)

### 1. Train Tokenizer
First, you must create the shared vocabulary. This script merges the En/Bn data and trains a BPE tokenizer.

```bash
venv/bin/python src/training/train_tokenizer.py
```
*Output*: `models/tokenizer/tokenizer.json`

### 2. Run Pre-training
Once the tokenizer is ready, start the pre-training loop (Denoising objective).

```bash
# Basic run with defaults (3 epochs, batch size 8)
venv/bin/python -m src.training.train

# Custom configuration (e.g. more epochs)
venv/bin/python -m src.training.train --epochs 5 --batch_size 16 --output_dir models/custom_ckpt
```
*Output*: Checkpoints in `models/checkpoints/`

---

## Phases Detail

### 1. Tokenizer Training
- **Goal**: Create a shared vocabulary for both English and Bengali.
- **Algorithm**: Byte-Pair Encoding (BPE).
- **Vocab Size**: 32,000 (standard for multilingual models).
- **Status**: **Done**

### 2. Pre-training (Monolingual)
- **Goal**: Train the model to understand language structure using Denoising Auto-Encoding (BART/mBART style).
- **Data**: `cleaned_wikitext_train` (En) + `cleaned_wikipedia_bn_train` (Bn).
- **Status**: **Ready**

### 3. Translation Training (Parallel)
- **Goal**: Fine-tune on parallel corpora (e.g., IndicTrans2) for translation.
- **Status**: *Future*

## Artifacts
- `models/tokenizer/`: Saved tokenizer files (`tokenizer.json`, vocab).
- `models/checkpoints/`: Model checkpoints.
