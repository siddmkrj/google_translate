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
venv/bin/python src/data/ingest_general.py --dataset wikitext --config wikitext-2-raw-v1 --split train

# Bengali (Wikipedia)
venv/bin/python src/data/ingest_general.py --dataset wikimedia/wikipedia --config 20231101.bn --split train

# 2. Clean Data
# English
venv/bin/python src/data/clean_data.py --dataset_path data/wikitext_wikitext-2-raw-v1_train --output_path data/cleaned_wikitext_train --lang en

# Bengali
venv/bin/python src/data/clean_data.py --dataset_path data/wikimedia_wikipedia_20231101.bn_train --output_path data/cleaned_wikipedia_bn_train --lang bn
```

**Ingest Parallel Data**:
```bash
venv/bin/python src/data/ingest_parallel.py --dataset csebuetnlp/banglanmt --split train --max_samples 50000
```

### 3. Run Training Pipeline
**Train Tokenizer (BPE)**:
```bash
venv/bin/python src/training/train_tokenizer.py
```

**Verify Tokenizer**:
```bash
venv/bin/python src/data/test_tokenizer.py
```

**Run Pre-training (Denoising Objective)**:
```bash
# Basic run (3 epochs, batch size 8)
venv/bin/python -m src.training.train --en_path data/cleaned_wikitext_train --bn_path data/cleaned_wikipedia_bn_train

# Custom run
venv/bin/python -m src.training.train --epochs 5 --batch_size 16 --output_dir models/custom_ckpt --en_path data/cleaned_wikitext_train --bn_path data/cleaned_wikipedia_bn_train
```

---

## 📂 Data Pipeline Details

### Directory Structure
- `data/wikitext_...`: Raw English data.
- `data/wikimedia_...`: Raw Bengali data.
- `data/cleaned_wikitext_train`: Cleaned English data combined.
- `data/cleaned_wikipedia_bn_train`: Cleaned Bengali data.
- `data/csebuetnlp_banglanmt_...`: Parallel English-Bengali data.

### Processing Steps
1.  **Ingestion**: Streaming from Hugging Face (`wikitext`, `wikipedia`, `banglanmt`).
2.  **Cleaning**:
    -   **Normalization**: Unicode NFKC.
    -   **Language ID**: FastText (`lid.176.bin`) to filter correct language.
    -   **Deduplication**: MD5 hash-based.
    -   **Filtering**: Script validation (Latin for En, Bengali for Bn).

### Visualization
Check `src/data/visualize_data.py` to inspect the datasets interactively in the console.

```bash
venv/bin/python src/data/visualize_data.py
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

### 3. Fine-tuning (Next Phase)
-   **Objective**: Machine Translation (Seq2Seq).
-   **Data**: Parallel Corpora (`banglanmt`).

---

## Roadmap
- [x] **Initialization**: Repo & Environment.
- [x] **Data Pipeline**: Ingestion, Cleaning, Parallel Data.
- [x] **Training Setup**: Tokenizer, Model Arch, Pre-training Loop.
- [ ] **Fine-tuning**: Train on Translation task.
- [ ] **Evaluation**: BLEU/METEOR scores.
- [ ] **Deployment**: FastAPI Service & Docker.
