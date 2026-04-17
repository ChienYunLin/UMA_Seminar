from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm
import pandas as pd

from relbench.base import Database, Table

if TYPE_CHECKING:
    from task import UserMentionTaskBase


class BaselineEvaluator:
    """Encapsulates all baseline prediction strategies."""

    def __init__(self, db: Database, task: "UserMentionTaskBase"):
        self.task = task
        self._mentions = db.table_dict["mention_rel"].df
        self._tweets = db.table_dict["tweets"].df

    def _mention_ids_up_to(self, cutoff: pd.Timestamp) -> pd.Index:
        """Return tweet_idx values for tweets created on or before *cutoff*."""
        return self._tweets.loc[
            self._tweets["created_at"] <= cutoff, "tweet_idx"
        ]

    def _compute_global_top_k(self, cutoff: pd.Timestamp) -> np.ndarray:
        """Most-mentioned users using only data up to *cutoff*."""
        valid_tweet_ids = self._mention_ids_up_to(cutoff)
        past_mentions = self._mentions[
            self._mentions["tweet_idx"].isin(valid_tweet_ids)
        ]
        top_k = (
            past_mentions["user_idx"]
            .value_counts()
            .index[: self.task.eval_k]
            .values
        )
        if len(top_k) < self.task.eval_k:
            top_k = np.pad(top_k, (0, self.task.eval_k - len(top_k)))
        return top_k

    def global_popularity(self, table: Table, cutoff: pd.Timestamp) -> np.ndarray:
        """Always predict the most-mentioned users globally."""
        global_top_k = self._compute_global_top_k(cutoff)
        return np.tile(global_top_k, (len(table.df), 1))

    def past_visit(self, table: Table, cutoff: pd.Timestamp) -> np.ndarray:
        """For each src user, predict the users they mentioned most in the past."""
        global_top_k = self._compute_global_top_k(cutoff)

        preds = []
        for _, row in tqdm(table.df.iterrows(), total=len(table.df), desc="Past visit baseline"):
            ts, src_user = row["timestamp"], row["src_user_idx"]
            # Past tweets by this user
            past_tweets = self._tweets[
                (self._tweets["user_idx"] == src_user) & (self._tweets["created_at"] <= ts)
            ]["tweet_idx"]
            # Count who this user mentioned in the past
            past_mentions = self._mentions[
                self._mentions["tweet_idx"].isin(past_tweets)
            ]["user_idx"].value_counts()

            top_k = past_mentions.index[: self.task.eval_k].values
            if len(top_k) < self.task.eval_k:
                remaining = self.task.eval_k - len(top_k)
                filler = [u for u in global_top_k if u not in top_k][:remaining]
                top_k = np.concatenate([top_k, filler])
            if len(top_k) < self.task.eval_k:
                top_k = np.pad(top_k, (0, self.task.eval_k - len(top_k)))
            preds.append(top_k)

        return np.stack(preds)