import copy
import json
import math
import os
import time
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything

from relbench.modeling.graph import get_link_train_table_input, make_pkey_fkey_graph
from relbench.modeling.loader import SparseTensor

from dataset import TweetMentionDataset
from logging_utils import setup_logging
from model import Model
from task import UserMentionTaskBase
from trainer import Trainer


METRIC_NAMES = [
    "link_prediction_precision",
    "link_prediction_recall",
    "link_prediction_map",
]


class ExperimentRunner:
    """Orchestrates the full multi-run training pipeline.

    Parameters
    ----------
    config:
        Experiment configuration dict (paths, hyperparameters, seeds, …).
    dataset:
        A ``TweetMentionDatasetBase`` instance.
    task:
        A ``UserMentionTaskBase`` instance.
    db_full:
        The full database (upto_test_timestamp=False), already fetched by the caller.
    col_to_stype_dict:
        Column-type mapping, already customised by the caller.
    experiment_title:
        Short string printed in the log header.
    """

    def __init__(
        self,
        config: Dict,
        dataset: TweetMentionDataset,
        task: UserMentionTaskBase,
        db_full,
        col_to_stype_dict: Dict,
        experiment_title: str = "Mention Link Prediction Experiment",
    ):
        self.config = config
        self.dataset = dataset
        self.task = task
        self.db_full = db_full
        self.col_to_stype_dict = col_to_stype_dict
        self.experiment_title = experiment_title

        self.output_dir = config["output_dir"]
        self.logger = setup_logging(self.output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        cfg = self.config
        logger = self.logger

        logger.info("=" * 70)
        logger.info(self.experiment_title)
        logger.info("=" * 70)
        logger.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")
        logger.info(f"Device: {self.device}")

        # ── Tables ────────────────────────────────────────────────────
        logger.info("Building train/val/test tables...")
        train_table = self.task._get_table("train")
        val_table   = self.task._get_table("val")
        test_table  = self.task._get_table("test")
        logger.info(f"Train rows: {len(train_table.df):,}")
        logger.info(f"Val rows:   {len(val_table.df):,}")
        logger.info(f"Test rows:  {len(test_table.df):,}")

        # ── Graph ─────────────────────────────────────────────────────
        logger.info("Building heterogeneous graph...")
        db_full = self.db_full
        data, col_stats_dict = make_pkey_fkey_graph(
            db_full,
            col_to_stype_dict=self.col_to_stype_dict,
            text_embedder_cfg=None,
            cache_dir=cfg["graph_cache_dir"],
        )
        logger.info(f"Graph: {data}")

        # ── Data Loaders ──────────────────────────────────────────────
        logger.info("Building data loaders...")
        loader_dict, dst_nodes_dict = self._build_loaders(data, train_table, val_table, test_table)
        entity_table = self.task.src_entity_table
        dst_table_name = self.task.dst_entity_table
        train_sparse_tensor = SparseTensor(dst_nodes_dict["train"][1])

        # ── Multi-Run Training ────────────────────────────────────────
        logger.info("=" * 70)
        logger.info(f"Starting {cfg['num_runs']} training runs...")

        all_run_results = []
        model = None  # will be set during runs

        for run_idx, seed in enumerate(cfg["seeds"]):
            logger.info(f"\n{'-' * 50}")
            logger.info(f"Run {run_idx + 1}/{cfg['num_runs']} (seed={seed})")
            logger.info(f"{'-' * 50}")
            seed_everything(seed)

            model = Model(
                data=data,
                col_stats_dict=col_stats_dict,
                num_layers=cfg["num_layers"],
                channels=cfg["channels"],
                out_channels=cfg["out_channels"],
                aggr=cfg["aggr"],
                norm=cfg["norm"],
                is_relgnn=cfg["is_relgnn"],
            ).to(self.device)

            trainer = Trainer(model, self.device, self.output_dir)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

            run_result, last_state_dict, best_map_state_dict = self._train_run(
                trainer, optimizer, loader_dict, train_sparse_tensor,
                entity_table, dst_table_name, val_table, test_table, run_idx, seed,
            )
            all_run_results.append(run_result)

            run_dir = os.path.join(self.output_dir, f"run_{run_idx + 1}")
            os.makedirs(run_dir, exist_ok=True)
            self._save_run_artefacts(run_dir, run_result, last_state_dict, best_map_state_dict)

        # ── Aggregation & Summary ─────────────────────────────────────
        summary = self._aggregate_results(all_run_results)

        final_results = {
            "config": {
                k: str(v) if not isinstance(v, (int, float, str, list, dict, bool)) else v
                for k, v in cfg.items()
            },
            "gnn_summary": summary,
            "all_runs": all_run_results,
        }
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(final_results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {results_path}")

        logger.info("=" * 70)
        logger.info("Experiment complete!")
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_loaders(self, data, train_table, val_table, test_table):
        cfg = self.config
        loader_dict, dst_nodes_dict = {}, {}
        for split, table in [("train", train_table), ("val", val_table), ("test", test_table)]:
            table_input = get_link_train_table_input(table, self.task)
            dst_nodes_dict[split] = table_input.dst_nodes
            loader_dict[split] = NeighborLoader(
                data,
                num_neighbors=cfg["num_neighbors"],
                time_attr="time",
                input_nodes=table_input.src_nodes,
                input_time=table_input.src_time,
                subgraph_type="bidirectional",
                batch_size=cfg["batch_size"],
                temporal_strategy=cfg["temporal_strategy"],
                shuffle=(split == "train"),
                num_workers=0,
                persistent_workers=False,
            )
        return loader_dict, dst_nodes_dict

    def _train_run(
        self, trainer, optimizer, loader_dict, train_sparse_tensor,
        entity_table, dst_table, val_table, test_table, run_idx, seed,
    ):
        cfg = self.config
        logger = self.logger

        best_val_metric = -math.inf
        best_map_state_dict = None
        best_map_epoch = 0
        epoch_logs = []
        train_loss = 0.0

        for epoch in range(1, cfg["epochs"] + 1):
            epoch_start = time.time()

            train_loss = trainer.train_epoch(
                loader_dict["train"], optimizer, train_sparse_tensor,
                entity_table, dst_table,
            )
            val_pred = trainer.evaluate(
                loader_dict["val"], entity_table, dst_table,
                self.task.num_dst_nodes, self.task.eval_k,
            )
            val_metrics = self.task.evaluate(val_pred, val_table)
            epoch_time = time.time() - epoch_start

            epoch_logs.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_metrics": {k: float(v) for k, v in val_metrics.items()},
                "epoch_time_sec": epoch_time,
            })
            logger.info(
                f"  Epoch {epoch:02d} | Loss: {train_loss:.6f} | "
                f"Val MAP: {val_metrics['link_prediction_map']:.6f} | "
                f"Val Prec: {val_metrics['link_prediction_precision']:.6f} | "
                f"Val Rec: {val_metrics['link_prediction_recall']:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            if val_metrics["link_prediction_map"] > best_val_metric:
                best_val_metric = val_metrics["link_prediction_map"]
                best_map_state_dict = copy.deepcopy(trainer.model.state_dict())
                best_map_epoch = epoch

        last_state_dict = copy.deepcopy(trainer.model.state_dict())

        # Evaluate last epoch
        val_pred_last = trainer.evaluate(
            loader_dict["val"], entity_table, dst_table,
            self.task.num_dst_nodes, self.task.eval_k,
        )
        val_metrics_last = self.task.evaluate(val_pred_last, val_table)
        test_pred_last = trainer.evaluate(
            loader_dict["test"], entity_table, dst_table,
            self.task.num_dst_nodes, self.task.eval_k,
        )
        test_metrics_last = self.task.evaluate(test_pred_last, test_table)

        # Evaluate best MAP
        trainer.model.load_state_dict(best_map_state_dict)
        val_pred_best = trainer.evaluate(
            loader_dict["val"], entity_table, dst_table,
            self.task.num_dst_nodes, self.task.eval_k,
        )
        val_metrics_best = self.task.evaluate(val_pred_best, val_table)
        test_pred_best = trainer.evaluate(
            loader_dict["test"], entity_table, dst_table,
            self.task.num_dst_nodes, self.task.eval_k,
        )
        test_metrics_best = self.task.evaluate(test_pred_best, test_table)

        logger.info(f"  Last epoch val:  {val_metrics_last}")
        logger.info(f"  Last epoch test: {test_metrics_last}")
        logger.info(f"  Best MAP (epoch {best_map_epoch}) val:  {val_metrics_best}")
        logger.info(f"  Best MAP (epoch {best_map_epoch}) test: {test_metrics_best}")

        run_result = {
            "run": run_idx + 1,
            "seed": seed,
            "final_epoch": cfg["epochs"],
            "final_train_loss": train_loss,
            "best_map_epoch": best_map_epoch,
            "last_epoch_val_metrics":  {k: float(v) for k, v in val_metrics_last.items()},
            "last_epoch_test_metrics": {k: float(v) for k, v in test_metrics_last.items()},
            "best_map_val_metrics":    {k: float(v) for k, v in val_metrics_best.items()},
            "best_map_test_metrics":   {k: float(v) for k, v in test_metrics_best.items()},
            "epoch_logs": epoch_logs,
            "_preds": {
                "last_val": val_pred_last,  "last_test": test_pred_last,
                "best_val": val_pred_best,  "best_test": test_pred_best,
            },
        }
        return run_result, last_state_dict, best_map_state_dict

    def _save_run_artefacts(self, run_dir, run_result, last_state_dict, best_map_state_dict):
        preds = run_result.pop("_preds")
        torch.save(last_state_dict,     os.path.join(run_dir, "last_epoch_model.pt"))
        torch.save(best_map_state_dict, os.path.join(run_dir, "best_map_model.pt"))
        np.save(os.path.join(run_dir, "last_epoch_val_pred.npy"),  preds["last_val"])
        np.save(os.path.join(run_dir, "last_epoch_test_pred.npy"), preds["last_test"])
        np.save(os.path.join(run_dir, "best_map_val_pred.npy"),    preds["best_val"])
        np.save(os.path.join(run_dir, "best_map_test_pred.npy"),   preds["best_test"])

    def _aggregate_results(self, all_run_results: List[Dict]) -> Dict:
        logger = self.logger
        logger.info("\n" + "=" * 70)
        logger.info(f"Aggregated Results (mean ± std over {len(all_run_results)} runs)")
        logger.info("=" * 70)

        summary = {}
        for model_type in ["last_epoch", "best_map"]:
            summary[model_type] = {}
            for split in ["val", "test"]:
                summary[model_type][split] = {}
                for metric in METRIC_NAMES:
                    values = [r[f"{model_type}_{split}_metrics"][metric] for r in all_run_results]
                    mean, std = float(np.mean(values)), float(np.std(values))
                    summary[model_type][split][metric] = {"mean": mean, "std": std}
                    logger.info(
                        f"  GNN ({model_type}) {split:5s} | {metric}: {mean:.6f} ± {std:.6f}"
                    )
        return summary