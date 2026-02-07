# Data Pipeline Documentation

This directory contains the datasets and documentation for the English-Bengali translation project.

## Directory Structure
- `wikitext_wikitext-2-raw-v1_train`: Raw English data (Wikitext).
- `wikimedia_wikipedia_20231101.bn_train`: Raw Bengali data (Wikipedia).
- `cleaned_wikitext_train`: Cleaned English data.
- `cleaned_wikipedia_bn_train`: Cleaned Bengali data.

## Pipeline Steps

### 1. Ingestion
Script: `src/data/ingest_general.py`
- **English**: Ingested full `wikitext` training set (~36k samples).
- **Bengali**: Streamed and ingested 10k samples from `wikimedia/wikipedia`.

### 2. Cleaning
Script: `src/data/clean_data.py`
- **Normalization**: Unicode NFKC.
- **Filtering**:
    - **English**: Kept Latin script, basic punctuation. Removed garbage/short lines.
    - **Bengali**: Kept Bengali Unicode range.
- **Deduplication**: Exact hash-based.
- **Results**:
    - En: 36,718 -> 19,328 samples.
    - Bn: 10,000 -> 9,983 samples.

## Current Status (Ready for Training)
| Language | Dataset Name | Path | Samples |
| :--- | :--- | :--- | :--- |
| **English** | Wikitext (Cleaned) | `data/cleaned_wikitext_train` | **19,328** |
| **Bengali** | Wikipedia (Cleaned) | `data/cleaned_wikipedia_bn_train` | **9,983** |

## Usage
To load the data:
```python
from datasets import load_from_disk
ds_en = load_from_disk('data/cleaned_wikitext_train')
ds_bn = load_from_disk('data/cleaned_wikipedia_bn_train')
```