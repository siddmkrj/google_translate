# GenAI Bengali <-> English Translator

A high-performance, End-to-End Generative AI project for translating between English and Bengali using Transformer architectures and FastAPI.

## Project Scope
This project aims to build a robust machine translation system capable of handling high Queries Per Second (QPS). It involves the full lifecycle of a GenAI project:
1.  **Data Engineering**: Collection, cleaning, pre-training (general data), and fine-tuning (specialized data) dataset preparation.
2.  **Model Development**: 
    -   Language detection.
    -   Pre-training a Transformer model from scratch or fine-tuning existing checkpoints.
    -   Fine-tuning on high-quality parallel corpuses.
3.  **Inference Engine**:
    -   FastAPI-based serving layer.
    -   Optimization for high concurrency and low latency.
    -   Batching and caching strategies.

## Documentation
Detailed documentation for each phase is available:

-   **Data Pipeline**: [DATA_README.md](DATA_README.md)
    -   Describes Ingestion, Cleaning (FastText), and Balancing (En-Bn) steps.
-   **Training Pipeline**: [TRAIN_README.md](TRAIN_README.md)
    -   Runbook for Tokenizer Training and Pre-training Loop (Denoising).

## Tech Stack
-   **Core**: Python, PyTorch
-   **Model Architecture**: Transformers (T5 Encoder-Decoder)
-   **Serving**: FastAPI, Uvicorn
-   **Data Processing**: Hugging Face Datasets, Pandas
-   **Training**: Hugging Face Transformers / Accelerate / Tokenizers

## Roadmap

### 1. Project Initialization
- [x] Repository setup
- [x] Environment configuration

### 2. Data Pipeline
- [x] Data ingestion scripts (Wikitext En, Wikipedia Bn)
- [x] Language detection integration (FastText)
- [x] Cleaning and Tokenization pipelines
- [x] Corpus Balancing

### 3. Training
- [x] Tokenizer Training (BPE)
- [x] Model Architecture Setup (T5-Small)
- [x] Pre-training loop (Denoising Objective)
- [ ] Fine-tuning loop on high-quality English-Bengali pairs
- [ ] Evaluation metrics (BLEU, METEOR, chrF)

### 4. Deployment
- [ ] FastAPI service implementation
- [ ] Deployment using Docker
- [ ] Load testing and Optimization

## Getting Started

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Data Pipeline
See [DATA_README.md](DATA_README.md) for details.
```bash
venv/bin/python src/data/ingest_general.py ...
venv/bin/python src/data/clean_data.py ...
venv/bin/python src/data/balance_data.py ...
```

### 3. Run Training
See [TRAIN_README.md](TRAIN_README.md) for full runbook.
```bash
# 1. Train Tokenizer
venv/bin/python src/training/train_tokenizer.py

# 2. Run Pre-training Loop
venv/bin/python -m src.training.train
```
