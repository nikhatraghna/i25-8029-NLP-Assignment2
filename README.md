# DS5007 NLP Assignment 2

**Repository:** i25-8029-NLP-Assignment2  
**Course:** DS5007 — Natural Language Processing

This repository contains the implementation, experiments, evaluation results, and report for NLP Assignment 2.

## Assignment components

1. **Word embeddings:** TF-IDF, PPMI, and a custom Skip-gram Word2Vec implementation, followed by embedding evaluation.
2. **Sequence labeling:** POS tagging and named entity recognition (NER), including lexicon/gazetteer features and a BiLSTM-based sequence-labeling model with CRF/Viterbi decoding.
3. **Topic classification:** A custom Transformer encoder for five-class text classification, with a BiLSTM baseline and evaluation visualizations.

## Repository structure

```text
notebook/       Main Colab/Jupyter notebook
data/           Assignment datasets and processed data
src/            Reusable implementation modules
embeddings/     Generated word embeddings
models/         Trained model checkpoints
results/        Metrics, tables, and plots
report/         Final 2–3 page PDF report
```

## Environment

The experiments are developed and executed in Google Colab. See [requirements.txt](requirements.txt) for the Python dependencies.

## Reproducibility

Run the notebook cells from top to bottom. Dataset files should be placed in the `data/` directory. Do not commit private credentials, temporary Colab files, or large generated artifacts unless they are required for submission.

## Status

Project structure initialized. Implementation and dataset details will be added incrementally with documented Git commits.
