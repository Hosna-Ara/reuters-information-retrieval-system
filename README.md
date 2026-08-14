# Reuters Information Retrieval System

## Overview

The Reuters Information Retrieval System is a Python/NLTK application developed for KIT719 Project 1 by Group 6 and presented as a portfolio project. It searches the NLTK Reuters corpus of 10,788 documents using NLP preprocessing, an inverted index, TF-IDF cosine similarity, and BM25 ranking. The system also supports spelling correction, conservative corpus-aware WordNet query expansion, a Streamlit interface, and automated evaluation.

## Key Features

- Document tokenisation, normalisation, stopword removal, and lemmatisation
- Inverted indexing of the Reuters corpus
- TF-IDF cosine-similarity ranking
- BM25 ranking
- Optional spelling correction
- Optional corpus-aware WordNet query expansion
- Ranked search results with document details
- Standard retrieval evaluation metrics and latency measurement
- Interactive Streamlit user interface

## Project Structure

- `app.py` — Streamlit search interface
- `src/` — corpus loading, preprocessing, indexing, ranking, and query-processing modules
- `evaluation/` — evaluation queries, relevance data, metrics, and evaluation runner
- `experiments/` — scripts for generating experimental summaries and figures
- `results/` — generated evaluation results, tables, and figures
- `tests/` — automated test suite
- `scripts/setup_nltk.py` — NLTK resource setup script

## Requirements

- Python 3.12 recommended
- Dependencies installed from `requirements.txt`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## NLTK Resources

```bash
python scripts/setup_nltk.py
```

This downloads the required `reuters`, `punkt_tab`, `stopwords`, and `wordnet` resources.

## Run the Application

```bash
streamlit run app.py
```

The interface accepts a raw search query and provides controls for BM25 or TF-IDF ranking, the top 5, 10, or 20 results, optional spelling correction, and optional query expansion.

## Example Queries

- `oil prices middle east`
- `international trade deficit`
- `coffee production exports`
- `corporate acquisition takeover`
- `cofee producion`

The final query demonstrates the optional spelling-correction feature.

## Testing

```bash
python -m pytest -v
```

The automated tests cover preprocessing, indexing, ranking, query and search processing, spelling correction, query expansion, and evaluation metrics.

## Evaluation

Reuters category labels provide reproducible binary relevance judgments, and evaluation queries are stored in `evaluation/queries.csv`. The evaluation compares five retrieval configurations using Precision@5, Precision@10, Recall@10, F1@10, Average Precision / MAP, Reciprocal Rank / MRR, nDCG@10, and search latency.

Run the evaluation:

```bash
python -m evaluation.evaluation
```

Generate the configuration summary table and figures:

```bash
python experiments/ranking_experiment.py
```

Generated outputs are written to:

- `results/evaluation_results.csv`
- `results/tables/configuration_summary.csv`
- `results/figures/`

## Reproducibility

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/setup_nltk.py
python -m pytest
python -m evaluation.evaluation
python experiments/ranking_experiment.py
streamlit run app.py
```

## Technology

- Python
- NLTK
- Streamlit
- Matplotlib
- pytest

## Academic Integrity / AI Use

Generative AI tools were used as learning and development aids. AI use is documented in `AI_USAGE_LOG.md`.
