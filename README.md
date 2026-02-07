# GenAI Bengali <-> English Translator

A high-performance, End-to-End Generative AI project for translating between English and Bengali using Transformer architectures and FastAPI.

## Project Scope
This project involves the full lifecycle of a GenAI project:
1.  **Data Engineering**: Collection, cleaning, pre-training (general data), and fine-tuning (specialized data) dataset preparation.
2.  **Model Development**: 
    -   Language detection.
    -   Pre-training a Transformer model (T5-Small) from scratch.
    -   Fine-tuning on high-quality parallel corpuses.
3.  **Inference Engine**:
    -   FastAPI-based serving layer (Planned).

## Tech Stack
-   **Core**: Python, PyTorch
-   **Model Architecture**: Transformers (T5 Encoder-Decoder)
-   **Data Processing**: Hugging Face Datasets, Pandas, FastText
-   **Training**: Hugging Face Transformers / Accelerate / Tokenizers

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 1.5 Download Language ID Model
```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -O models/lid.176.bin
```

### 2. Run Data Pipeline
**Ingest & Clean Monolingual Data**:
```bash
# 1. Ingest Data (English & Bengali)
# English (Wikitext)
venv/bin/python -m src.data.ingest_general --dataset wikitext --config wikitext-2-raw-v1 --split train

# Bengali (Wikipedia)
venv/bin/python -m src.data.ingest_general --dataset wikimedia/wikipedia --config 20231101.bn --split train

# 2. Clean Data
# English
venv/bin/python -m src.data.clean_data --dataset_path data/wikitext_wikitext-2-raw-v1_train --output_path data/cleaned_wikitext_train --lang en

# Bengali
venv/bin/python -m src.data.clean_data --dataset_path data/wikimedia_wikipedia_20231101.bn_train --output_path data/cleaned_wikipedia_bn_train --lang bn
```

**Ingest Parallel Data**:
```bash
venv/bin/python -m src.data.ingest_parallel --dataset csebuetnlp/banglanmt --split train --max_samples 50000

# Clean Parallel Data
venv/bin/python -m src.data.clean_parallel --input_path data/csebuetnlp_banglanmt_parallel --output_path data/cleaned_banglanmt_parallel --src en --tgt bn --model_path models/lid.176.bin
```

Note: if the on-disk parallel dataset is stored in a WebDataset-style schema (e.g. a `jsonl` **bytes** column plus `__key__` / `__url__`), `clean_parallel` will automatically expand it into a standard `translation` dataset before cleaning.

### 3. Run Training Pipeline
**Train Tokenizer (BPE)**:
```bash
venv/bin/python -m src.training.train_tokenizer
```

**Verify Tokenizer**:
```bash
venv/bin/python -m src.data.test_tokenizer
```

**Run Pre-training (Denoising Objective)**:
```bash
# Basic run (3 epochs, batch size 8)
venv/bin/python -m src.training.train --en_path data/cleaned_wikitext_train --bn_path data/cleaned_wikipedia_bn_train

# Custom run
venv/bin/python -m src.training.train --epochs 5 --batch_size 16 --output_dir models/custom_ckpt --en_path data/cleaned_wikitext_train --bn_path data/cleaned_wikipedia_bn_train
```

Pre-training saves a Hugging Face model checkpoint to `<output_dir>/final`.
**Run Fine-tuning (Translation, en → bn)**:
```bash
# Fine-tune starting from the pre-training checkpoint saved at models/checkpoints/final
venv/bin/python -m src.training.finetune \
  --parallel_path data/cleaned_banglanmt_parallel \
  --src_lang en \
  --tgt_lang bn \
  --init_model_dir models/checkpoints/final \
  --output_dir models/finetuned_en_bn \
  --epochs 1 \
  --batch_size 8

# Smoke test (small subset)
venv/bin/python -m src.training.finetune \
  --parallel_path data/cleaned_banglanmt_parallel \
  --src_lang en \
  --tgt_lang bn \
  --init_model_dir models/checkpoints/final \
  --output_dir models/finetuned_en_bn_smoke \
  --epochs 1 \
  --batch_size 8 \
  --max_train_samples 2000 \
  --max_eval_samples 200
```

Fine-tuning saves a Hugging Face model checkpoint to `<output_dir>/final`.

Note: `--init_model_dir` must point to an existing Hugging Face checkpoint directory (a folder containing `config.json` + model weights).
If you trained pre-training with a different `--output_dir`, set `--init_model_dir` to that `<output_dir>/final` folder.
If you haven’t run pre-training yet, run:
`venv/bin/python -m src.training.train --output_dir models/checkpoints` (which produces `models/checkpoints/final`).
---

## 📂 Data Pipeline Details

### Directory Structure
- `data/wikitext_...`: Raw English data.
- `data/wikimedia_...`: Raw Bengali data.
- `data/cleaned_wikitext_train`: Cleaned English data combined.
- `data/cleaned_wikipedia_bn_train`: Cleaned Bengali data.
- `data/csebuetnlp_banglanmt_...`: Parallel English-Bengali data.
- `data/cleaned_banglanmt_parallel`: Cleaned parallel corpus used for fine-tuning.
- `models/checkpoints/final`: Pre-training output checkpoint (default).
- `models/finetuned_en_bn/final`: Fine-tuned translation checkpoint (example output).

### Processing Steps
1.  **Ingestion**: Streaming from Hugging Face (`wikitext`, `wikipedia`, `banglanmt`).
2.  **Cleaning**:
    -   **Normalization**: Unicode NFKC.
    -   **Language ID**: FastText (`lid.176.bin`) to filter correct language.
    -   **Deduplication**: MD5 hash-based.
    -   **Filtering**: Script validation (Latin for En, Bengali for Bn).
    -   **Parallel Data**: Consistency checks (numbers, length ratio) and strict language mismatch filtering.

### Visualization
Check `src/data/visualize_data.py` to inspect the datasets interactively in the console.

```bash
venv/bin/python -m src.data.visualize_data
```

---

## 🧠 Training Pipeline Details

### 1. Tokenizer
-   **Algorithm**: Byte-Pair Encoding (BPE).
-   **Vocab Size**: 32,000.
-   **Shared Vocabulary**: Trained on combined En + Bn corpora.

### 2. Pre-training (Current Phase)
-   **Objective**: Denoising Auto-Encoder (Span Corruption/Reconstruction).
-   **Model**: T5-Small (~60M params).
-   **Data**: Cleaned Monolingual Corpora.
-   **Status**: Ready & Verified.

### 3. Fine-tuning (Translation)
-   **Objective**: Machine Translation (Seq2Seq).
-   **Data**: Parallel Corpora (`banglanmt`).
-   **Entrypoint**: `venv/bin/python -m src.training.finetune` (loads `data/cleaned_banglanmt_parallel` and fine-tunes from `models/checkpoints/final`).
-   **Local dev tip**: Use `--max_train_samples` / `--max_eval_samples` for a quick smoke test before running full fine-tuning on the full cleaned dataset.

---

## Roadmap
- [x] **Initialization**: Repo & Environment.
- [x] **Data Pipeline**: Ingestion, Cleaning, Parallel Data.
- [x] **Training Setup**: Tokenizer, Model Arch, Pre-training Loop.
- [x] **Fine-tuning**: Train on Translation task.
- [ ] **Evaluation**: BLEU/METEOR scores.
- [ ] **Deployment**: FastAPI Service & Docker.
