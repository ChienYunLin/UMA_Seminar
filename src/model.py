from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import Embedding, ModuleDict
from torch_frame.data.stats import StatType
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP
from torch_geometric.typing import NodeType

from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from atomic_routes import get_atomic_routes
from relgnn_nn import RelGNN


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
        is_relgnn: bool,
        num_heads: int,
        shallow_list: List[NodeType] = [],
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

        if is_relgnn:
            atomic_routes_list = get_atomic_routes(data.edge_types)
            self.gnn = RelGNN(
                node_types=data.node_types,
                edge_types=atomic_routes_list,
                channels=channels,
                aggr=aggr,
                num_model_layers=num_layers,
                num_heads=num_heads,  # Number of prediction heads
            )
        else:
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
        self.id_awareness_emb = torch.nn.Embedding(1, channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()
        for embedding in self.embedding_dict.values():
            torch.nn.init.normal_(embedding.weight, std=0.1)
        self.id_awareness_emb.reset_parameters()

    def forward_dst_readout(
        self, batch: HeteroData, entity_table: NodeType, dst_table: NodeType
    ) -> Tensor:
        """Score all dst nodes in the subgraph (with id_awareness on src seeds)."""
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)
        # Add ID-awareness to the root node
        x_dict[entity_table][: seed_time.size(0)] += self.id_awareness_emb.weight

        rel_time_dict = self.temporal_encoder(
            seed_time, batch.time_dict, batch.batch_dict
        )

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        for node_type, embedding in self.embedding_dict.items():
            x_dict[node_type] = x_dict[node_type] + embedding(batch[node_type].n_id)

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[dst_table])