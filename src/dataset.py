import os
from typing import Dict

import pandas as pd

from relbench.base import Database, Dataset, Table


class TweetMentionDataset(Dataset):
    """Base dataset that loads core tweet/user/mention tables.

    Subclasses override ``make_db`` to add extra tables (e.g. replies).
    """

    val_timestamp = pd.Timestamp("2019-09-01")
    test_timestamp = pd.Timestamp("2019-11-01")

    def __init__(self,
                 data_dir: str,
                 users_table_name: str,
                 tweets_table_name: str,
                 mentions_table_name: str,
                 **kwargs
        ):
        self.data_dir = data_dir
        self.users_table_name = users_table_name
        self.tweets_table_name = tweets_table_name
        self.mentions_table_name = mentions_table_name

        super().__init__(**kwargs)

    def _load_core_tables(self) -> Dict[str, pd.DataFrame]:
        """Load and preprocess the three tables shared by all variants."""
        # Load tables
        users = pd.read_parquet(os.path.join(self.data_dir, self.users_table_name))
        tweets = pd.read_parquet(os.path.join(self.data_dir, self.tweets_table_name))
        mention_rel = pd.read_parquet(os.path.join(self.data_dir, self.mentions_table_name))
        replies = pd.read_parquet(os.path.join(self.data_dir, "replies_2019.parquet"))
        reply_mention_rel = pd.read_parquet(os.path.join(self.data_dir, "reply_mention_rel.parquet"))

        # Preprocess data
        users["verified"] = users["verified"].astype(int)
        users["missing_description"] = users["missing_description"].astype(int)

        for col in [c for c in tweets.columns if c.startswith("emotion_")]:
            tweets[col] = tweets[col].astype(int)

        return {"users": users, "tweets": tweets, "mention_rel": mention_rel, "replies": replies, "reply_mention_rel": reply_mention_rel}

    def _build_core_relbench_tables(self, raw: Dict[str, pd.DataFrame]) -> Dict[str, Table]:
        """Convert raw DataFrames to RelBench Table objects for the core tables."""
        return {
            "users": Table(
                df=pd.DataFrame(raw["users"]),
                fkey_col_to_pkey_table={},
                pkey_col="user_idx",
                time_col=None,
            ),
            "tweets": Table(
                df=pd.DataFrame(raw["tweets"]),
                fkey_col_to_pkey_table={"user_idx": "users"},
                pkey_col="tweet_idx",
                time_col="created_at",
            ),
            "mention_rel": Table(
                df=pd.DataFrame(raw["mention_rel"]),
                fkey_col_to_pkey_table={"user_idx": "users", "tweet_idx": "tweets"},
                pkey_col="mention_idx",
                time_col="created_at",
            ),
            "replies": Table(
                df=pd.DataFrame(raw["replies"]),
                fkey_col_to_pkey_table={"tweet_idx": "tweets"},
                pkey_col="reply_idx",
                time_col="created_at",
            ),
            "reply_mention_rel": Table(
                df=pd.DataFrame(raw["reply_mention_rel"]),
                fkey_col_to_pkey_table={"reply_idx": "replies", "user_idx": "users"},
                pkey_col="idx",
                time_col="created_at",
            )
        }

    def make_db(self) -> Database:
        raw = self._load_core_tables()
        tables = self._build_core_relbench_tables(raw)
        return Database(tables)