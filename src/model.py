from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder


class Model(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
        shallow_list: List[NodeType] = None,
        id_awareness: bool = False,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(channels, out_channels=out_channels, norm=norm, num_layers=1)
        self.embedding_dict = ModuleDict(
            {node: Embedding(data.num_nodes_dict[node], channels) for node in shallow_list}
        )
        self.id_awareness_emb = torch.nn.Embedding(1, channels) if id_awareness else None
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        if self.id_awareness_emb is not None:
            self.id_awareness_emb.reset_parameters()

    def _encode(
        self,
        batch: HeteroData,
        entity_table: NodeType,
        batch_size: int,
        apply_id_awareness: bool = False,
    ) -> Tuple[Dict, Tensor]:
        """Shared encode path: encoder -> temporal -> [id_awareness] -> shallow embeddings.
        Returns (x_dict, seed_time).
        """
        x_dict = self.encoder(batch.tf_dict)
        seed_time = batch[entity_table].seed_time

        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict, batch.batch_dict)
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        if apply_id_awareness:
            x_dict[entity_table][:batch_size] += self.id_awareness_emb.weight

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)
        return x_dict, seed_time


    def embed(self, batch: HeteroData, entity_table: NodeType) -> Tensor:
        """GNN embeddings for seed nodes (with id_awareness, no head).
        Used for embedding extraction after training.
        """
        batch_size = batch[entity_table].batch_size
        x_dict, _ = self._encode(batch, entity_table, batch_size, apply_id_awareness=True)
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        return x_dict[entity_table][:batch_size]

    def forward_dst_readout(
        self, batch: HeteroData, entity_table: NodeType, dst_table: NodeType
    ) -> Tensor:
        """Score all dst nodes in the subgraph (with id_awareness on src seeds)."""
        batch_size = batch[entity_table].batch_size
        x_dict, _ = self._encode(batch, entity_table, batch_size, apply_id_awareness=True)
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        return self.head(x_dict[dst_table]).flatten()