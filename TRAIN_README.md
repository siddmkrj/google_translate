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
- **Status**: *Not Started*

### 3. Translation Training (Parallel)
- **Goal**: Fine-tune on parallel corpora (e.g., IndicTrans2) for translation.
- **Status**: *Future*

## Artifacts
- `models/tokenizer/`: Saved tokenizer files (`tokenizer.json`, vocab).
- `models/checkpoints/`: Model checkpoints.
