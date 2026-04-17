# User Mention Link Prediction on Climate Twitter (2019)

A temporal graph neural network pipeline for predicting **user-to-user mention links** on a 2019 climate-focused Twitter dataset, built on top of [RelBench](https://relbench.stanford.edu/) and PyTorch Geometric. The project compares a HeteroGraphSAGE model against popularity and past-visit baselines, and includes analyses of community structure, mention behavior, and permutation tests.

## Task

Given a source user and a timestamp, predict the top-*k* users they will mention in the next 30 days. Evaluation uses Precision@10, Recall@10, and MAP@10.

Two model variants are supported:

| Variant | Script | Graph |
|---|---|---|
| User-Mention | `scripts/mention_link_prediction.py` | same core tables, 3 layers |
| User-Mention + Reply | `scripts/mention_link_prediction_with_reply.py` | adds `replies` and `reply_mention_rel` tables, 4 layers |

## Repository structure

```
.
├── src/                                  # Core pipeline modules
│   ├── dataset.py                        # TweetMentionDatasetBase (RelBench Dataset)
│   ├── task.py                           # UserMentionTaskBase (link prediction task)
│   ├── model.py                          # HeteroGraphSAGE + temporal + shallow embeddings
│   ├── trainer.py                        # Train / eval / embedding extraction loop
│   ├── runner.py                         # Multi-seed ExperimentRunner
│   ├── baseline_evaluator.py             # Global-popularity & past-visit baselines
│   ├── logging_utils.py                  # File + stdout logger
│   ├── config.py                         # Shared paths & split timestamps
│   └── utils.py
├── scripts/                              # Entry points
│   ├── analyze_split.ipynb               # Pick train/val/test timestamps
│   ├── mention_link_prediction.py        # Run user-mention variant
│   ├── mention_link_prediction_with_reply.py  # Run user-mention + reply variant
│   └── run_baselines.py                  # Run popularity / past-visit baselines
├── analysis/                             # Post-hoc analysis notebooks
│   ├── across_community_analysis.ipynb
│   ├── mention_behavior_analysis.ipynb
│   └── permutation_analaysis.ipynb
├── doc/
│   ├── preprocessed_schema.md            # Table schemas & ER diagram
│   └── tweet_mention_link_prediction_output_manual.md  # Output file reference
├── data/                                 # Input parquet files
└── results/                              # Model outputs 
```

## Data

Expected files in `./data/` (see `doc/preprocessed_schema.md` for full schemas):

- `users_2019.parquet` — user dimension table with `verified`, `actor_type`, and a 384-dim SentenceTransformer `description_embedding`.
- `tweets_2019.parquet` — tweet fact table with `text_embedding`, engagement counts, sentiment/emotion/topic/theme categoricals, and 11 binary emotion flags.
- `mention_rel.parquet` — bridge table linking tweets to mentioned users.
- `replies_2019.parquet`, `reply_mention_rel.parquet` — only needed for the reply variant.

Temporal split (set in `src/dataset.py` and `src/config.py`):

- **Train:** before `2019-09-01` (~70% of tweets, ~64% of mentions)
- **Val:** `2019-09-01` – `2019-11-01` (~15%)
- **Test:** from `2019-11-01` (~15%)

## Installation

```bash
pip install torch torch-geometric torch-frame relbench pandas numpy tqdm
```

The code was developed against PyTorch 2.x with CUDA. A GPU is strongly recommended.

## Usage

**1. Choose split timestamps** — open `scripts/analyze_split.ipynb` to inspect tweet/mention density over time and confirm the quantile-based split.

**2. Train the GNN** — each script is self-contained with its own `CONFIG` dict at the top:

```bash
python scripts/mention_link_prediction.py
python scripts/mention_link_prediction_with_reply.py
```

Default hyperparameters: lr `1e-3`, 20 epochs, batch size 512, 128 channels, `[128, 128, 128]` neighbor sampling, 5 seeds (`[42, 123, 456, 789, 1024]`).

**3. Run baselines**:

```bash
python scripts/run_baselines.py
```

This evaluates two non-learned baselines:
- **Global popularity** — always predict the top-*k* most-mentioned users.
- **Past visit** — for each source user, predict users they mentioned most frequently before the evaluation timestamp, padded with globally popular users.

## Outputs

Each run writes to the configured `output_dir` (e.g. `results/user_mention_link_prediction/`):

- `results.json` — aggregated metrics across seeds
- `run_{1..5}/{model_state_dict.pt, val_pred.npy, test_pred.npy}` — per-seed artifacts
- `user_embeddings.npy` / `user_ids.npy`, `tweet_embeddings.npy` / `tweet_ids.npy` — 128-dim GNN embeddings with companion ID arrays
- `user_metadata.parquet`, `tweet_metadata.parquet` — join keys for embeddings
- `baselines/` — popularity and past-visit predictions + metrics

Prediction arrays have shape `[N_samples, 10]` and are row-aligned with `task._get_table(split).df`. See `doc/tweet_mention_link_prediction_output_manual.md` for full recipes on mapping predictions and embeddings back to the original tweets and users.

## Analysis notebooks

- **`across_community_analysis.ipynb`** — breaks down model performance across Leiden-detected user communities.
- **`mention_behavior_analysis.ipynb`** — examines per-user mention patterns and how they relate to prediction accuracy.
- **`permutation_analaysis.ipynb`** — compares metrics on original vs. feature-permuted runs to quantify the contribution of node features.

All three notebooks read from `src/config.py`, which defines `RESULT_DIRS` for the `{original, permuted} × {tweet_mention, user_mention, user_mention_reply}` conditions and guards missing directories via `available_result_dirs()`.

## Key design notes

- **RelBench integration** — `TweetMentionDatasetBase` and `UserMentionTaskBase` subclass RelBench's `Dataset` and `RecommendationTask` so the pipeline reuses RelBench's temporal sampling, graph materialization, and link-prediction metrics.
- **Feature encoding** — categorical, numerical, and 384-dim text/description embeddings are all handled by `torch_frame` stypes; see `build_col_to_stype_dict` in each script.
- **Model** — `HeteroEncoder` + `HeteroTemporalEncoder` + `HeteroGraphSAGE` + shallow per-node-type embeddings + ID-aware MLP head (`src/model.py`).
- **Reproducibility** — `ExperimentRunner` trains with 5 fixed seeds and aggregates metrics; caches are cleared at the start of each run.
