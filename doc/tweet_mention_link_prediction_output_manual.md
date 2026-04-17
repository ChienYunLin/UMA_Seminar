# Tweet Mention Link Prediction — Output Files Manual

## Output Directory Structure

```
results/mention_link_prediction/
├── results.json                    # Aggregated metrics across all runs
├── tweet_embeddings.npy            # GNN embeddings for all tweets [N_tweets, 128]
├── tweet_ids.npy                   # Corresponding tweet_idx values [N_tweets]
├── user_embeddings.npy             # GNN embeddings for all users [N_users, 128]
├── user_ids.npy                    # Corresponding user_idx values [N_users]
├── tweet_metadata.parquet          # Tweet attributes (topic, sentiment, etc.)
├── user_metadata.parquet           # User attributes (username, actor_type, etc.)
├── baselines/
│   ├── past_visit_val_pred.npy     # Past-visit baseline predictions on val [N_val, 10]
│   ├── past_visit_test_pred.npy    # Past-visit baseline predictions on test [N_test, 10]
│   └── past_visit_metrics.json     # Past-visit precision/recall/MAP
├── run_1/
│   ├── model_state_dict.pt         # Saved model weights (last epoch)
│   ├── val_pred.npy                # GNN predictions on val [N_val, 10]
│   └── test_pred.npy               # GNN predictions on test [N_test, 10]
├── run_2/ ...
└── run_5/ ...
```

## Key Concepts

**Prediction arrays** (`val_pred.npy`, `test_pred.npy`, `past_visit_val_pred.npy`, etc.) are 2D arrays of shape `[N_samples, 10]`. Each row contains the top-10 predicted `user_idx` values for that sample. The rows are aligned 1:1 with the val/test table DataFrames — row `i` of the prediction corresponds to row `i` of `val_table.df` or `test_table.df`.

**Embedding arrays** (`tweet_embeddings.npy`, `user_embeddings.npy`) are 2D arrays of shape `[N, 128]`. The companion `_ids.npy` files contain the `tweet_idx` or `user_idx` for each row. These IDs are the join keys to the metadata parquet files.

---

## 1. Mapping Embeddings to Original Tweets and Users

```python
import numpy as np
import pandas as pd

# Load embeddings and IDs
tweet_emb = np.load("tweet_embeddings.npy")
tweet_ids = np.load("tweet_ids.npy")
user_emb = np.load("user_embeddings.npy")
user_ids = np.load("user_ids.npy")

# Load metadata
tweet_meta = pd.read_parquet("tweet_metadata.parquet")
user_meta = pd.read_parquet("user_metadata.parquet")

# Build DataFrames with embeddings
tweet_emb_df = pd.DataFrame({"tweet_idx": tweet_ids})
tweet_emb_df["embedding"] = list(tweet_emb)  # each row is a 128-dim vector

user_emb_df = pd.DataFrame({"user_idx": user_ids})
user_emb_df["embedding"] = list(user_emb)

# Join with metadata
tweets_with_emb = tweet_meta.merge(tweet_emb_df, on="tweet_idx", how="inner")
users_with_emb = user_meta.merge(user_emb_df, on="user_idx", how="inner")

print(f"Tweets with embeddings: {len(tweets_with_emb)}")
print(f"Users with embeddings:  {len(users_with_emb)}")
```

**Note:** Not every tweet/user in the metadata may have an embedding — only those that appeared as seed nodes in the NeighborLoader get embeddings. Use `how="inner"` to keep only matched rows, or `how="left"` to see which ones are missing.

---

## 2. Mapping Predictions Back to Val/Test Tweets

The prediction arrays are row-aligned with the val/test table DataFrames. You need to reconstruct these tables to get the mapping.

```python
# Reconstruct the val/test tables (same logic as the pipeline)
from mention_link_prediction_pipeline import (
    TweetMentionDataset, TweetMentionTask, CONFIG
)

dataset = TweetMentionDataset(data_dir=CONFIG["data_dir"], cache_dir=CONFIG["cache_dir"])
task = TweetMentionTask(dataset=dataset)

val_table = task._get_table("val")
test_table = task._get_table("test")

val_df = val_table.df.reset_index(drop=True)
test_df = test_table.df.reset_index(drop=True)
```

`val_df` has columns: `tweet_idx`, `user_idx` (list of ground-truth mentioned users), `timestamp`.

### 2a. GNN Predictions

```python
# Load GNN predictions (e.g., from run 1)
val_pred = np.load("run_1/val_pred.npy")    # shape [N_val, 10]
test_pred = np.load("run_1/test_pred.npy")  # shape [N_test, 10]

# Attach predictions to val DataFrame
val_df["gnn_top10"] = list(val_pred)

# Each row now has:
#   tweet_idx:  the tweet being evaluated
#   user_idx:   ground-truth list of mentioned users
#   timestamp:  the evaluation timestamp
#   gnn_top10:  array of 10 predicted user_idx values
```

### 2b. Past-Visit Baseline Predictions

```python
pv_val_pred = np.load("baselines/past_visit_val_pred.npy")
pv_test_pred = np.load("baselines/past_visit_test_pred.npy")

val_df["past_visit_top10"] = list(pv_val_pred)
```

### 2c. Resolving Predicted User IDs to Usernames

```python
user_meta = pd.read_parquet("user_metadata.parquet")
user_lookup = user_meta.set_index("user_idx")["username"].to_dict()

# For a single row
row = val_df.iloc[0]
print(f"Tweet: {row['tweet_idx']}")
print(f"Ground truth users: {[user_lookup.get(u, u) for u in row['user_idx']]}")
print(f"GNN predicted users: {[user_lookup.get(u, u) for u in row['gnn_top10']]}")
print(f"Past-visit predicted: {[user_lookup.get(u, u) for u in row['past_visit_top10']]}")
```

---

## 3. Per-Sample Metric Computation

```python
def average_precision_at_k(predicted, ground_truth, k=10):
    """Compute AP@k for a single sample."""
    gt_set = set(ground_truth)
    hits = 0
    score = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in gt_set:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(gt_set), k) if gt_set else 0.0

def precision_at_k(predicted, ground_truth, k=10):
    gt_set = set(ground_truth)
    return len(set(predicted[:k]) & gt_set) / k

def recall_at_k(predicted, ground_truth, k=10):
    gt_set = set(ground_truth)
    return len(set(predicted[:k]) & gt_set) / len(gt_set) if gt_set else 0.0

# Compute per-sample AP for GNN
val_df["gnn_ap"] = [
    average_precision_at_k(pred, gt)
    for pred, gt in zip(val_df["gnn_top10"], val_df["user_idx"])
]

val_df["pv_ap"] = [
    average_precision_at_k(pred, gt)
    for pred, gt in zip(val_df["past_visit_top10"], val_df["user_idx"])
]

print(f"GNN MAP@10:        {val_df['gnn_ap'].mean():.6f}")
print(f"Past-visit MAP@10: {val_df['pv_ap'].mean():.6f}")
```
