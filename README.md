# D-GRaNDe

**Rank Correlation Meets Gaussian Degree Normalization in GNNs for Image Classification**

Official source code for the paper accepted at **SIBGRAPI 2026** (39th Conference on Graphics, Patterns and Images).

D-GRaNDe is a unified framework for **semi-supervised image classification** that integrates two rank-based ideas into a Graph Neural Network (GNN):

- **DGCG** (Density-Guided Correlation Graph) — builds the input graph from the *rank correlation* between image similarity rankings, automatically selecting the correlation threshold to keep the graph sparse (target density `η ∈ [0.03, 0.04]`).
- **GRaNDe** (Gaussian Rank-based Neighborhood Degree) — replaces the standard node-degree normalization in message passing with a measure that combines the neighbor count with the Gaussian-weighted distances to those neighbors.

Using only **10% of labeled data**, D-GRaNDe outperforms its individual components and several traditional and recent baselines.

Project page: <https://dgrande.lucasvalem.com>

## Authors

- **Rafael Mendonça Duarte** — `rmduarte@usp.br` 
- **Gabriel Maia Brito** — `gabrielmaiab@usp.br` 
- **Lucas Pascotti Valem** — `lucas@icmc.usp.br`

Institute of Mathematics and Computer Science (ICMC), University of São Paulo (USP), São Carlos, Brazil.

## Repository structure

```
D-GRanDe/
├── data/Features-Labels-Lists/   # Precomputed features and label lists (Flowers17 included)
├── src/
│   ├── main.py            # Entry point: global config (ExperimentConfig + CONTROL PANEL) + battery
│   ├── data_loader.py     # Dataset paths and feature/label loading
│   ├── graph_builder.py   # Ranked lists, rank-correlation measures, DGCG threshold search
│   ├── gcn_model.py       # SGC / APPNP models and the GCNClassifier trainer
│   ├── grande_models.py   # GRaNDe score and GRaNDe-normalized SGC/APPNP layers
│   └── utils.py           # Cross-validation folds and metric helpers
├── results/               # Output CSVs (created on first run)
└── requirements.txt
```

## Installation

Requires **Python 3.11+** and a working **PyTorch 2.6 / PyTorch Geometric 2.6** setup.

```bash
# 1. Create and activate an environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install the base dependencies
pip install -r requirements.txt

# 3. Install the PyG extension wheels (torch-scatter and torch-sparse)
#    These are NOT on plain PyPI — they must come from the PyG index.
#    Replace cu124 with your CUDA version (or use +cpu for a CPU-only build).
pip install torch-scatter torch-sparse \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

The code runs on **GPU when available** (it prints which device it uses) and falls back to CPU.

## Data

The features, labels and ranked lists for **Flowers17** (ResNet152, SENet154 and ViT-B16)
are already included under `data/Features-Labels-Lists/`, so the example below runs out of
the box. The Pets and CUB200 datasets used in the paper are not bundled here; add their
feature files following the naming in `src/data_loader.py` to reproduce those tables.

## Running the simple example (Flowers + ResNet)

The default configuration in `src/main.py` is already set to the flagship setup:
**Flowers17 + ResNet152 + DGCG (RBO) + APPNP + GRaNDe (σ = 0.2)**.

```bash
cd src
python main.py
```

This trains under the paper's protocol (10-fold cross-validation, 200 epochs, 5 repetitions)
and appends the results to `results/results_batch_<timestamp>.csv`. The expected accuracy for
this setup is **≈ 87.8%** (Table III in the paper).

### Configuring other experiments

All configuration lives at the top of `src/main.py`:

- **`ExperimentConfig`** — fixed hyperparameters (`L=200`, `top_k=40`, `epochs=200`,
  `n_repetitions=5`). Learning rate and `k_graph` are set per dataset automatically.
- **CONTROL PANEL** — the lists you edit to sweep experiments:

  ```python
  DATASETS = ['flowers']                 # 'flowers', 'cub200', 'pets', 'corel'
  FEATURE_EXTRACTORS = ['resnet']        # 'resnet', 'senet', 'vit'
  GRAPH_TYPES = ['dgcg']                 # 'knn', 'rec', 'dgcg'
  DGCG_CORRELATIONS = ['rbo']            # 'rbo', 'jaccard_max', 'jaccard_median', 'jaccardk'
  GCN_TYPES = ['appnp_grande']           # 'sgc_degree', 'sgc_grande', 'appnp_degree', 'appnp_grande'
  SIGMA_VALUES = [0.2]                   # GRaNDe Gaussian bandwidth
  N_FOLDS_OPTIONS = [10]
  ```

  Each list is swept combinatorially, so adding values runs more experiments in one batch.

### Understanding the output CSV

Each row is one experiment combination. Two accuracy aggregations are reported:

| Column | Meaning |
| --- | --- |
| `accuracy_mean`, `accuracy_std_per_repetition` | Mean and std over the **per-repetition** means (std reflects variability *between repetitions*). This is the metric reported in the paper. |
| `accuracy_mean_per_fold`, `accuracy_std_per_fold` | Mean and std over **all individual folds** pooled together (std reflects variability *between folds*). |
| `f1_*`, `precision_*`, `recall_*` | Macro-averaged metrics, averaged per repetition. |
| `density`, `threshold` | Final DGCG graph density and the correlation threshold selected by the density search. |

## Citation

```bibtex
@inproceedings{duarte2026dgrande,
  title     = {Rank Correlation Meets Gaussian Degree Normalization in GNNs for Image Classification},
  author    = {Duarte, Rafael Mendon{\c{c}}a and Brito, Gabriel Maia and Valem, Lucas Pascotti},
  booktitle = {2026 39th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)},
  year      = {2026}
}
```

## Acknowledgments

This work was supported by the São Paulo Research Foundation – FAPESP
(grants #2026/16921-8, #2026/13387-0, and #2025/10602-5), the University of São Paulo
(PRPI Ordinance No. 1032, New Faculty Support Program), and the Institute of Mathematics
and Computer Science (ICMC).
