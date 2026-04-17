import pandas as pd

from relbench.base import Database, RecommendationTask, Table, TaskType
from relbench.metrics import (
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
)


class UserMentionTaskBase(RecommendationTask):
    """Shared task logic: predict which users a source user will mention next.
    Subclasses supply ``timedelta`` and ``eval_k`` (Via a CONFIG dict).
    """

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "src_user_idx"
    src_entity_table = "users"
    dst_entity_col = "dst_user_idx"
    dst_entity_table = "users"
    time_col = "timestamp"
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]

    def make_table(self, db: Database, timestamps, **kwargs) -> Table:
        tweets = db.table_dict["tweets"].df
        mentions = db.table_dict["mention_rel"].df

        rows = []
        for ts in timestamps:
            future_tweets = tweets[
                (tweets["created_at"] > ts)
                & (tweets["created_at"] <= ts + self.timedelta)
            ]
            future_mentions = mentions[
                mentions["tweet_idx"].isin(future_tweets["tweet_idx"])
            ]
            mention_with_author = future_mentions.merge(
                future_tweets[["tweet_idx", "user_idx"]].rename(
                    columns={"user_idx": "src_user_idx"}
                ),
                on="tweet_idx",
            )
            mention_with_author = mention_with_author.rename(
                columns={"user_idx": "dst_user_idx"}
            )
            mention_with_author = mention_with_author[
                mention_with_author["src_user_idx"] != mention_with_author["dst_user_idx"]
            ]
            grouped = (
                mention_with_author.groupby("src_user_idx")["dst_user_idx"]
                .apply(lambda x: list(x.unique()))
                .reset_index()
            )
            grouped["timestamp"] = ts
            rows.append(grouped)

        if not rows or all(len(r) == 0 for r in rows):
            df = pd.DataFrame(columns=["timestamp", "src_user_idx", "dst_user_idx"])
        else:
            df = pd.concat(rows, ignore_index=True)

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )

    def _get_table(self, split: str) -> Table:
        db = self.dataset.get_db(upto_test_timestamp=(split != "test"))

        if split == "test":
            start = self.dataset.test_timestamp
            end = db.max_timestamp - self.timedelta
            timestamps = pd.date_range(start=start, end=end, freq=self.timedelta)
        elif split == "val":
            start = self.dataset.val_timestamp
            end = self.dataset.test_timestamp - self.timedelta
            timestamps = pd.date_range(start=start, end=end, freq=self.timedelta)
        else:
            start = self.dataset.val_timestamp - self.timedelta
            end = db.min_timestamp
            timestamps = pd.date_range(start=start, end=end, freq=-self.timedelta)

        table = self.make_table(db, timestamps)
        if split != "test":
            table = self.filter_dangling_entities(table)
        return table