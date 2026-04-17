"""
Baseline Runner
===============
Evaluates global-popularity and past-visit baselines on val and test splits.

Run:
    python run_baselines.py
"""

import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "..", "src")
sys.path.insert(0, os.path.normpath(src_path))

import json
import shutil
import numpy as np
import pandas as pd

from baseline_evaluator import BaselineEvaluator
from dataset import TweetMentionDatasetBase
from logging_utils import setup_logging
from task import UserMentionTaskBase

from relbench.base import Database
from relbench.metrics import (
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
)


CONFIG = {
    "data_dir": "./data",
    # ====input data====
    "users_table_name": "users_2019.parquet",
    "tweets_table_name": "tweets_2019.parquet",
    "mentions_table_name" : "mention_rel.parquet",
    # ====output data====
    "output_dir": "./results/user_mention_link_prediction",
    "cache_dir": "./cache/user_mention",
    # ====hyper-parameters====
    "eval_k": 10,
    "timedelta_days": 30,
}

class TweetMentionDataset(TweetMentionDatasetBase):
    def make_db(self) -> Database:
        raw = self._load_core_tables()
        tables = self._build_core_relbench_tables(raw)
        return Database(tables)

class UserMentionTask(UserMentionTaskBase):
    timedelta = pd.Timedelta(days=CONFIG["timedelta_days"])
    eval_k = CONFIG["eval_k"]
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map]

class BaselineRunner:
    def __init__(self):
        self.output_dir = CONFIG["output_dir"]
        self.logger = setup_logging(output_dir=self.output_dir, logger_name= "mention_link_pred_baseline")

        self.logger.info("Loading dataset...")
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

        self.val_timestamp = dataset.val_timestamp
        self.test_timestamp = dataset.test_timestamp
        self.task = task
        self.db_full = dataset.get_db(upto_test_timestamp=False)

        self.logger.info("Building val/test tables...")
        self.val_table  = task._get_table("val")
        self.test_table = task._get_table("test")
        self.logger.info(f"Val rows:  {len(self.val_table.df):,}")
        self.logger.info(f"Test rows: {len(self.test_table.df):,}")

    def run(self):
        logger  = self.logger
        task    = self.task
        out_dir = self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        evaluator = BaselineEvaluator(self.db_full, task)
        results = {}

        # ── Global Popularity ─────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Baseline: Global Popularity")
        logger.info("=" * 60)

        val_pop_pred  = evaluator.global_popularity(self.val_table, self.val_timestamp)
        test_pop_pred = evaluator.global_popularity(self.test_table, self.test_timestamp)

        val_pop_metrics  = task.evaluate(val_pop_pred,  self.val_table)
        test_pop_metrics = task.evaluate(test_pop_pred, self.test_table)

        logger.info(f"  Val:  {val_pop_metrics}")
        logger.info(f"  Test: {test_pop_metrics}")

        np.save(os.path.join(out_dir, "global_popularity_val_pred.npy"),  val_pop_pred)
        np.save(os.path.join(out_dir, "global_popularity_test_pred.npy"), test_pop_pred)

        results["global_popularity"] = {
            "val":  {k: float(v) for k, v in val_pop_metrics.items()},
            "test": {k: float(v) for k, v in test_pop_metrics.items()},
        }

        # ── Past Visit ────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("Baseline: Past Visit")
        logger.info("=" * 60)

        val_pv_pred  = evaluator.past_visit(self.val_table,  self.val_timestamp)
        test_pv_pred = evaluator.past_visit(self.test_table, self.test_timestamp)

        val_pv_metrics  = task.evaluate(val_pv_pred,  self.val_table)
        test_pv_metrics = task.evaluate(test_pv_pred, self.test_table)

        logger.info(f"  Val:  {val_pv_metrics}")
        logger.info(f"  Test: {test_pv_metrics}")

        np.save(os.path.join(out_dir, "past_visit_val_pred.npy"),  val_pv_pred)
        np.save(os.path.join(out_dir, "past_visit_test_pred.npy"), test_pv_pred)

        results["past_visit"] = {
            "val":  {k: float(v) for k, v in val_pv_metrics.items()},
            "test": {k: float(v) for k, v in test_pv_metrics.items()},
        }

        # ── Save combined metrics ─────────────────────────────────────
        with open(os.path.join(out_dir, "baseline_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"All baseline results saved to {out_dir}/")
        logger.info("=" * 60)


if __name__ == "__main__":
    BaselineRunner().run()