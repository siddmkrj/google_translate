# Data Pipeline Summary

## 1. Data Ingestion
**Script**: `src/data/ingest_general.py`
- Supports loading from Hugging Face, streaming, and local saving.

**Datasets Ingested**:
1.  **English**: 
    - Source: `wikitext` (config: `wikitext-2-raw-v1`)
    - Split: `train` (full)
    - Path: `data/wikitext_wikitext-2-raw-v1_train`
    - Initial Size: ~36,718 samples

2.  **Bengali**:
    - Source: `wikimedia/wikipedia` (config: `20231101.bn`)
    - Split: `train` (subset)
    - Path: `data/wikimedia_wikipedia_20231101.bn_train`
    - Initial Size: 10,000 samples (streamed)

## 2. Data Cleaning
**Script**: `src/data/clean_data.py`
- **Normalization**: Unicode NFKC, whitespace stripping.
- **Language ID**: Uses `fasttext` (model: `lid.176.bin`).
- **Deduplication**: MD5 hash-based exact deduplication.
- **Filtering**: Strict regex for Latin (En) and Bengali (Bn) scripts.

**Cleaning Results**:
1.  **English (Wikitext)**:
    - Path: `data/cleaned_wikitext_train`
    - Count: 36,718 -> **19,328**
    - *Note*: High reduction due to short lines/headers in raw wikitext.

2.  **Bengali (Wikipedia)**:
    - Path: `data/cleaned_wikipedia_bn_train`
    - Count: 10,000 -> **9,983**
    - *Note*: High quality retention.

## 3. Balancing
Script: `src/data/balance_data.py`
- **Objective**: Equalize corpus size for better initial alignment.
- **Action**: Downsampled English `cleaned_wikitext_train` to 10,000 samples.
- **Output**: `balanced_wikitext_train`.

## Current Status (Ready for Training)
| Language | Dataset Name | Path | Samples |
| :--- | :--- | :--- | :--- |
| **English** | Wikitext (Balanced) | `data/balanced_wikitext_train` | **10,000** |
| **Bengali** | Wikipedia (Cleaned) | `data/cleaned_wikipedia_bn_train` | **9,983** |

## Usage
To load the balanced data:
```python
from datasets import load_from_disk
ds_en = load_from_disk('data/balanced_wikitext_train')
ds_bn = load_from_disk('data/cleaned_wikipedia_bn_train')
```