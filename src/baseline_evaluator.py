import logging
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from relbench.base import Database, Table

if TYPE_CHECKING:
    from task import UserMentionTaskBase


class BaselineEvaluator:
    """Encapsulates all baseline prediction strategies."""

    def __init__(self, db: Database, task: "UserMentionTaskBase"):
        self.task = task
        self._mentions = db.table_dict["mention_rel"].df
        self._tweets = db.table_dict["tweets"].df
        self._global_top_k = self._compute_global_top_k()

    def _compute_global_top_k(self) -> np.ndarray:
        top_k = self._mentions["user_idx"].value_counts().index[: self.task.eval_k].values
        # Pad if fewer than eval_k
        if len(top_k) < self.task.eval_k:
            top_k = np.pad(top_k, (0, self.task.eval_k - len(top_k)))
        return top_k

    def global_popularity(self, table: Table) -> np.ndarray:
        """Always predict the most-mentioned users globally."""
        return np.tile(self._global_top_k, (len(table.df), 1))

    def past_visit(self, table: Table, logger: logging.Logger) -> np.ndarray:
        """For each src user, predict the users they mentioned most in the past."""
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
                filler = [u for u in self._global_top_k if u not in top_k][:remaining]
                top_k = np.concatenate([top_k, filler])
            if len(top_k) < self.task.eval_k:
                top_k = np.pad(top_k, (0, self.task.eval_k - len(top_k)))
            preds.append(top_k)

        return np.stack(preds)