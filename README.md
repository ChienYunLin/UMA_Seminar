# User Mention Link Prediction with Relational GNNs

Predicting which users a Twitter user will mention next, framed as a temporal link prediction task on a heterogeneous relational graph. Built on top of [RelBench](https://relbench.stanford.edu/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), with a [RelGNN](https://github.com/snap-stanford/RelGNN)-style attention architecture and heuristic baselines for comparison.

## Overview

Given a snapshot of tweets, users, mentions, and replies up to time *t*, the task is: for each source user, predict the top-*10* users they will mention in the next 30 days. The graph is constructed automatically from primary-key/foreign-key relationships via RelBench, and the model learns over heterogeneous node types (`users`, `tweets`, `mention_rel`, `replies` and `reply_mention_rel`).

Two model families are supported:

- **RelGNN** — heterogeneous graph attention with *dim-dim* and *dim-fact-dim* atomic routes derived from the schema (see `atomic_routes.py`, `relgnn_conv.py`, `relgnn_nn.py`).
- **HeteroGraphSAGE** — standard baseline from `relbench.modeling.nn`, toggled via `is_relgnn: False` in the config.

Two non-learned baselines are included for sanity-checking:

- **Global Popularity** — predict the most-mentioned users overall (up to the cutoff).
- **Past Visit** — predict the users this source user has mentioned most often in the past, padded with global popularity.

## Repository layout

```
.
├── src/                                       # Core pipeline modules
│   ├── dataset.py                             # TweetMentionDatasetBase (RelBench Dataset)
│   ├── task.py                                # UserMentionTaskBase (link prediction task)
│   ├── model.py                               # Encoder + temporal encoder + RelGNN/SAGE + MLP head
│   ├── relgnn_nn.py                           # RelGNN stack (multi-layer hetero conv + LayerNorm)
│   ├── relgnn_hetero_conv.py                  # Hetero wrapper, dispatches per atomic-route type
│   ├── relgnn_conv.py                         # Single-relation conv (TransformerConv + SAGE)
│   ├── atomic_routes.py                       # Schema → atomic routes (dim-dim, dim-fact-dim)
│   ├── trainer.py                             # Train / eval / embedding extraction loop
│   ├── runner.py                              # Multi-seed ExperimentRunner
│   ├── baseline_evaluator.py                  # Global-popularity & past-visit baselines
│   ├── logging_utils.py                       # File + stdout logger
│   ├── config.py                              # Shared paths & split timestamps
│   └── utils.py
├── scripts/                                   # Entry points
│   ├── analyze_split.ipynb                    # Pick train/val/test timestamps
│   ├── mention_link_prediction.py             # Run user-mention variant
│   └── run_baselines.py                       # Run popularity / past-visit baselines
├── doc/
│   └── preprocessed_schema.md                 # Table schemas & ER diagram
├── data/                                      # Input parquet files
└── results/                                   # Model outputs
```

## Data

The pipeline expects the following parquet files in `./data/`:

| File | Description |
|---|---|
| `users_2019.parquet` | User attributes (verified flag, description embedding, …) |
| `tweets_2019.parquet` | Tweets with timestamps, topics, emotions, text embeddings |
| `mention_rel.parquet` | Tweet → mentioned-user edges |
| `replies_2019.parquet` | Reply tweets (optional, used by the +reply variant) |
| `reply_mention_rel.parquet` | Reply → mentioned-user edges (optional) |

Splits are temporal:

- **Train**: everything before `2019-09-01`
- **Validation**: `2019-09-01` → `2019-11-01`
- **Test**: `2019-11-01` onwards

The 30-day prediction window and split timestamps are defined in `dataset.py` and `task.py`.

## Installation

```bash
pip install torch torch-geometric torch-frame relbench pandas numpy tqdm
```

A CUDA-capable GPU is strongly recommended — neighbor sampling and attention over the full mention graph are expensive on CPU.

## Running the GNN experiment

Two entry points are provided:

```bash
cd scripts
python mention_link_prediction.py             # users + tweets + mentions
python mention_link_prediction_with_reply.py  # adds replies + reply_mention_rel
```

This runs five seeds (42, 123, 456, 789, 1024) with the configuration block at the top of the script. For each run it saves:

- `last_epoch_model.pt`, `best_map_model.pt` — model checkpoints
- `*_val_pred.npy`, `*_test_pred.npy` — top-*k* predictions
- `results.json` — per-run and aggregated metrics (precision, recall, MAP @ k=10)
- `experiment_<timestamp>.log` — full training log

Key hyperparameters (edit `CONFIG` in the script):

| Key | Default | Notes |
|---|---|---|
| `is_relgnn` | `True` | Set `False` to fall back to HeteroGraphSAGE |
| `num_heads` | `1` | Attention/prediction heads in RelGNN |
| `num_layers` | `4` | GNN depth |
| `num_neighbors` | `[128, 128, 128, 128]` | Per-layer neighbor sampling fan-out |
| `channels` | `128` | Hidden dimension |
| `epochs` | `20` | |
| `batch_size` | `512` | |
| `lr` | `0.001` | |
| `eval_k` | `10` | Top-k for precision / recall / MAP |
| `timedelta_days` | `30` | Prediction horizon |

## Running the baselines

```bash
cd scripts
python run_baselines.py
```

Outputs `baseline_metrics.json` plus `.npy` prediction arrays for both baselines on val and test.

## Metrics

All evaluation uses RelBench's link-prediction metrics at *k* = 10:

- `link_prediction_precision`
- `link_prediction_recall`
- `link_prediction_map`

The runner reports both the **last-epoch** model and the model that achieved the **best validation MAP**, aggregated as mean ± std over the five seeds.

## How the graph is built

1. `TweetMentionDataset.make_db()` wraps the parquet tables as `relbench.base.Table` objects with PK/FK declarations.
2. `make_pkey_fkey_graph` (RelBench) materializes a `HeteroData` graph plus column-type stats.
3. `get_atomic_routes` walks the edge types and produces:
   - **dim-dim** routes for direct PK/FK joins,
   - **dim-fact-dim** routes whenever a fact table (`f2p_*` edges) connects two dimensions through a shared key.
4. `RelGNN_HeteroConv` dispatches each route to a `RelGNNConv`, which uses `TransformerConv` for attention and `SAGEConv` to aggregate the fact-side representation in the dim-fact-dim case.

## Citations

FEY, Matthias, et al. Relational deep learning: Graph representation learning on relational databases. arXiv preprint arXiv:2312.04615, 2023.

ROBINSON, Joshua, et al. Relbench: A benchmark for deep learning on relational databases. Advances in Neural Information Processing Systems, 2024, 37: 21330-21341.

CHEN, Tianlang; KANATSOULIS, Charilaos; LESKOVEC, Jure. Relgnn: Composite message passing for relational deep learning. arXiv preprint arXiv:2502.06784, 2025.