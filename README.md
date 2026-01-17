# DNA Sequence Classifier

End-to-end machine learning pipeline for classifying DNA sequences as Promoter, Enhancer, or Non-regulatory regions. 
Built with PyTorch CNN, k-mer feature engineering, and comprehensive data preprocessing.

## Overview

This project implements a complete ML pipeline for **DNA sequence classification**:

- **Input**: Raw DNA sequences
- **Features**: k-mer frequency vectors (k=1 to k=5)
- **Models**: CNN (PyTorch), Baseline ML models
- **Output**: Classification as Promoter, Enhancer, or Non-regulatory

**Key Results**: CNN model achieves **68.6% accuracy** on held-out test data.
## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/bleedblack1/dna-sequence-classifier.git
cd dna-sequence-classifier
pip install -r requirements.txt

```
### 2. Run Streamlit App
```bash
streamlit run deployment/streamlit_app.py
```
### Open http://localhost:8501

### 3. Predict Example Sequence
```example text
ATGCGATAGCTAGCTCGTAGCTAG
```
```Project structure
DNA-CLASSIFIER/
├── data/
│   ├── raw/                 # Original CSVs from Ensembl
│   ├── processed/           # .npy files (kmer_k1.npy, labels.npy)
│   └── external/            # Downloaded references
├── data_collection/         # Download + preprocess raw data
├── data_preparation/        # Train/val/test splits
├── deployment/              # Streamlit app + FastAPI
│   ├── streamlit_app.py     # Main UI
│   ├── app.py               # FastAPI backend
│   └── predict.py           # PredictionPipeline
├── feature_engineering/     # k-mer feature extraction (k=1-5)
├── modeling/                # CNN training + evaluation
│   └── scripts/train_cnn_pytorch.py
└── results/                 # Plots + metrics (confusion_matrix.png)

```
### Hugging face live demo : https://huggingface.co/spaces/NISHANT-INDIA/DNA-SEQUENCE-CLASSIFIER






