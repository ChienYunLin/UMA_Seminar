from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from relbench.modeling.loader import SparseTensor

from model import Model
import logging


class Trainer:
    """Handles one training epoch and evaluation/embedding extraction passes."""

    def __init__(self, model: Model, device: torch.device, output_dir: str):
        self.model = model
        self.device = device
        self.logger = logging.getLogger("mention_link_pred")

    def train_epoch(
        self,
        loader: NeighborLoader,
        optimizer: torch.optim.Optimizer,
        train_sparse_tensor: SparseTensor,
        entity_table: str,
        dst_table: str,
    ) -> float:
        self.model.train()
        loss_accum = count_accum = 0

        for batch in tqdm(loader, desc="Training", leave=False):
            batch = batch.to(self.device)
            batch_size = batch[entity_table].batch_size

            out = self.model.forward_dst_readout(batch, entity_table, dst_table)
            out = out.squeeze(-1)

            input_id = batch[entity_table].input_id.cpu()
            src_batch, dst_index = train_sparse_tensor[input_id]
            src_batch, dst_index = src_batch.to(self.device), dst_index.to(self.device)

            target = torch.isin(
                batch[dst_table].batch + batch_size * batch[dst_table].n_id,
                src_batch + batch_size * dst_index,
            ).float()

            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(out, target)
            loss.backward()
            optimizer.step()

            loss_accum += loss.detach().item() * out.size(0)
            count_accum += out.size(0)

        return loss_accum / count_accum

    @torch.no_grad()
    def evaluate(
        self,
        loader: NeighborLoader,
        entity_table: str,
        dst_table: str,
        num_dst_nodes: int,
        eval_k: int,
    ) -> np.ndarray:
        self.model.eval()
        pred_list = []

        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(self.device)
            batch_size = batch[entity_table].batch_size

            out = self.model.forward_dst_readout(batch, entity_table, dst_table)
            out = out.squeeze(-1)

            scores = torch.zeros(batch_size, num_dst_nodes, device=out.device)
            scores[batch[dst_table].batch, batch[dst_table].n_id] = torch.sigmoid(out)

            if not pred_list:
                self.logger.info(
                    f"  Users scored per sample: "
                    f"{(scores > 0).sum(dim=1).float().mean():.0f} / {num_dst_nodes}"
                )

            _, pred_mini = torch.topk(scores, k=eval_k, dim=1)
            pred_list.append(pred_mini)

        return torch.cat(pred_list, dim=0).cpu().numpy()