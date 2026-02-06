# Training Pipeline Documentation

This document tracks the training process for the English-Bengali Translator.

## Phases

### 1. Tokenizer Training
- **Goal**: Create a shared vocabulary for both English and Bengali.
- **Algorithm**: Byte-Pair Encoding (BPE).
- **Vocab Size**: 32,000 (standard for multilingual models).
- **Status**: **Done** (Saved to `models/tokenizer/`)

### 2. Pre-training (Monolingual)
- **Goal**: Train the model to understand language structure using Denoising Auto-Encoding (BART/mBART style).
- **Data**: `balanced_wikitext_train` (En) + `cleaned_wikipedia_bn_train` (Bn).
- **Status**: **Ready**

#### How to Start Pre-training
Run the training script from the project root:

```bash
# Basic run with defaults (3 epochs, batch size 8)
venv/bin/python -m src.training.train

# Custom configuration
venv/bin/python -m src.training.train --epochs 5 --batch_size 16 --output_dir models/custom_ckpt
```

**Note**: Ensure you use `python -m src.training.train` to avoid import errors.

### 3. Translation Training (Parallel)
- **Goal**: Fine-tune on parallel corpora (e.g., IndicTrans2) for translation.
- **Status**: *Future*

## Artifacts
- `models/tokenizer/`: Saved tokenizer files (`tokenizer.json`, vocab).
- `models/checkpoints/`: Model checkpoints.
