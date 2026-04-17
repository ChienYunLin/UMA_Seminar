"""
User-to-User Mention Link Prediction (with reply nodes)
========================================================
Unique to this variant:
- output_dir: results/user_mention_link_prediction_with_reply_2019
- num_layers = 4
- Extra tables: replies, reply_mention_rel
"""

import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "..", "src")
sys.path.insert(0, os.path.normpath(src_path))

import shutil
import pandas as pd
from torch_frame import stype

from dataset import TweetMentionDataset
from runner import ExperimentRunner
from task import UserMentionTaskBase

from relbench.base import Database, Table
from relbench.modeling.utils import get_stype_proposal


CONFIG = {
    "data_dir": "./data",
    # ====input data====
    "users_table_name": "users_2019.parquet",
    "tweets_table_name": "tweets_2019.parquet",
    "mentions_table_name" : "mention_rel.parquet",
    # ====output data====
    "output_dir": "./results/user_mention_link_prediction_with_reply",
    "cache_dir": "./cache/user_mention_with_reply",
    "graph_cache_dir": "./user_mention_with_reply_materialized_cache",
    # ====rel-gnn or not===
    "is_relgnn": False,
    # ====hyper-parameters====
    "lr": 0.001,
    "epochs": 20,
    "batch_size": 512,
    "channels": 128,
    "out_channels": 1,
    "num_layers": 4,
    "num_neighbors": [128, 128, 128, 128],
    "aggr": "sum",
    "norm": "batch_norm",
    "temporal_strategy": "uniform",
    "eval_k": 10,
    "timedelta_days": 30,
    # ====experiment runs====
    "num_runs": 5,
    "seeds": [42, 123, 456, 789, 1024],
}


class UserMentionTask(UserMentionTaskBase):
    timedelta = pd.Timedelta(days=CONFIG["timedelta_days"])
    eval_k = CONFIG["eval_k"]


def build_col_to_stype_dict(db_full) -> dict:
    col_to_stype_dict = get_stype_proposal(db_full)

    col_to_stype_dict["users"]["verified"] = stype.numerical
    col_to_stype_dict["users"]["missing_description"] = stype.numerical
    col_to_stype_dict["users"]["description_embedding"] = stype.embedding
    col_to_stype_dict["users"].pop("username", None)

    col_to_stype_dict["tweets"]["dominant_emotion"] = stype.categorical
    col_to_stype_dict["tweets"]["topic"] = stype.categorical
    col_to_stype_dict["tweets"]["theme"] = stype.categorical
    for col in [c for c in col_to_stype_dict["tweets"] if c.startswith("emotion_")]:
        col_to_stype_dict["tweets"][col] = stype.numerical
    col_to_stype_dict["tweets"]["text_embedding"] = stype.embedding
    col_to_stype_dict["tweets"].pop("conversation_id", None)
    col_to_stype_dict["tweets"].pop("created_at", None)

    col_to_stype_dict["replies"].pop("reply_id", None)
    col_to_stype_dict["replies"].pop("created_at", None)

    col_to_stype_dict["mention_rel"].pop("created_at", None)
    col_to_stype_dict["reply_mention_rel"].pop("created_at", None)

    return col_to_stype_dict


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    shutil.rmtree(CONFIG["cache_dir"], ignore_errors=True)
    shutil.rmtree(CONFIG["graph_cache_dir"], ignore_errors=True)

    dataset = TweetMentionDataset(
        data_dir=CONFIG["data_dir"],
        cache_dir=CONFIG["cache_dir"],
        users_table_name=CONFIG["users_table_name"],
        tweets_table_name=CONFIG["tweets_table_name"],
        mentions_table_name=CONFIG["mentions_table_name"],
    )
    task = UserMentionTask(dataset=dataset)
    if task.cache_dir:
        shutil.rmtree(task.cache_dir, ignore_errors=True)
    task.cache_dir = None

    db_full = dataset.get_db(upto_test_timestamp=False)
    col_to_stype_dict = build_col_to_stype_dict(db_full)

    ExperimentRunner(
        config=CONFIG,
        dataset=dataset,
        task=task,
        db_full=db_full,
        col_to_stype_dict=col_to_stype_dict,
        experiment_title="User-to-User Mention Link Prediction Experiment With Reply Node",
    ).run()


if __name__ == "__main__":
    main()