# Fake News Detection by GNN

## Introduction
This project explores fake news classification with graph neural networks on social/context graphs.

Core objective:
- Classify news into **Fake** and **Real** categories using GNN-based learning over interaction-style graph data.

Reported highlights:
- Constructed user-news interaction graphs using NetworkX and PyTorch Geometric based on UPFD dataset.
- Implemented GraphSAGE with an MLP encoder pipeline and a Graph Attention Network style variant for graph representation learning.
- Used BERT embeddings to capture user preferences and contextual signals, with **Binary Cross Entropy loss (BCEWithLogitsLoss)**.
- Achieved **85% test accuracy** (reported project result) for Fake/Real classification.

## Project Structure
- `fakenews.ipynb`
  - Compact baseline pipeline.
  - Uses UPFD (`politifact`, `feature='bert'`) with train/test loaders.
  - Model: 2-layer GraphSAGE + GlobalAttention pooling + linear head.
  - Saved output includes test score: `0.832579185520362`.

- `GNN_Fake_News.ipynb`
  - More structured end-to-end training workflow.
  - Includes package setup, reproducible hyperparameters, train/eval loops, metric tracking (accuracy/precision/recall/F1), checkpoint saving.
  - Uses UPFD train/val/test splits and feature normalization.
  - Contains a BERT text embedder class with SHA256-based embedding cache (`./bert_cache`).

- `gnn-graph.ipynb`
  - Graph construction/analytics exploration with NetworkX.
  - Builds a weighted co-author graph from arXiv metadata (first 50k records).
  - Filters to multi-author papers, then computes structural stats and visualizes top connected nodes.

## Requirements and Installation
Suggested environment:
- Python 3.10+
- PyTorch
- PyTorch Geometric and companion packages

Install (as used in notebook workflow):

```bash
pip install torch
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
pip install transformers scikit-learn tqdm networkx matplotlib pandas numpy
```

Notes:
- In notebook cells, `TORCH` and `CUDA` are inferred automatically from your local torch install.
- Dataset used in experiments: `UPFD` (`politifact`, BERT feature mode).

## Approach
1. Data and graph setup
- Load UPFD graph data from PyTorch Geometric datasets.
- Use `feature='bert'` for node features, and `NormalizeFeatures()` in the structured notebook pipeline.
- Batch graph samples via `torch_geometric.loader.DataLoader`.

2. Representation learning
- GraphSAGE encoders (`SAGEConv`) are used for neighborhood aggregation.
- Baseline notebook applies attention pooling (`GlobalAttention`) to emphasize informative nodes.
- Structured notebook applies global mean pooling followed by an MLP classifier head (Linear + BatchNorm + ReLU + Dropout + Linear).

3. Text/context features
- BERT integration exists through:
  - UPFD BERT feature mode (`feature='bert'`).
  - A separate `BertTextEmbedder` class (transformers) with cache support for reusable embeddings.

4. Optimization and objective
- Binary classification objective with `torch.nn.BCEWithLogitsLoss`.
- Optimizers seen across notebooks:
  - `Adam` (baseline).
  - `AdamW` (structured pipeline).

5. Evaluation
- Metrics: Accuracy, Precision, Recall, and F1.
- Structured notebook saves the best model by validation F1 and evaluates on test set.

## Important Details Extracted from Notebooks
- `fakenews.ipynb`
  - Input dimension explicitly set to `768` (BERT feature size).
  - Architecture print confirms `SAGEConv(768, 32)` -> `SAGEConv(32, 32)` + `GlobalAttention(gate_nn=Linear(32,1))`.
  - Training loop runs up to 59 epochs in current code cell.
  - Saved notebook output test score: `0.832579185520362`.

- `GNN_Fake_News.ipynb`
  - Hyperparameters: `N_EPOCHS=50`, `BATCH_SIZE=128`, `LR=1e-3`, `WEIGHT_DECAY=1e-4`, `HIDDEN_DIM=128`, `NUM_SAGE_LAYERS=2`, `DROPOUT=0.1`.
  - Uses `seed=42` for reproducibility.
  - Includes fallback handling for transformers availability.
  - Logged run in saved output shows `Feature dim: 768` 

- `gnn-graph.ipynb`
  - Reads arXiv metadata (`arxiv-metadata-oai-snapshot.json`) and caps ingest at 50k lines.
  - After filtering, printed: `Total papers: 28677`.
  - Built graph statistics:
    - `Total authors: 78086`
    - `Total co-author links: 519004`
    - `Density: 0.00017023933322221502`
    - `Average clustering coefficient: 0.8051187592970495`
  - Top collaborators by degree include: Sarkar S., Gehrels N., Giorgi M., Delgado C., Pimenta M., and others.

## Challenges
- Class imbalance / hard validation behavior
  - In the structured notebook output, validation F1 remains at 0.0 across many epochs, indicating poor threshold behavior or class prediction collapse.

- Model consistency across notebooks
  - Different notebook variants use different pooling/classifier choices, making direct run-to-run comparison difficult.

- Dataset/experiment switching
  - A "gossipcop alternate dataset" section exists, but the shown code still points to `politifact`, so experiment labeling and code need tighter alignment.

- API evolution
  - Baseline notebook logs a deprecation warning for `GlobalAttention`; migration to the newer PyG recommendation would improve maintainability.

## Tech Stack Used
- Python
- PyTorch
- PyTorch Geometric
- transformers (BERT)
- scikit-learn
- NetworkX
- NumPy, pandas
- matplotlib
- tqdm

## Results
- Project-reported headline: **85% test accuracy** for Fake vs Real news classification.
- Saved notebook snapshot values currently visible in this repo include:
  - `fakenews.ipynb`: `0.8326` accuracy-like score from `test(test_loader)`.

This indicates multiple experiment states/checkpoints in notebooks; consolidating final reproducible runs into one tracked experiment script is a recommended next step.
