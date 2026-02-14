"""GNN that outputs per-qubit LLR corrections for BP decoding.

Architecture:
  - Typed message passing on the heterogeneous Tanner graph
  - GRU cell for node state updates across MP layers
  - Final readout MLP on data-qubit nodes → scalar LLR correction per qubit
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_correction(channel_llr: torch.Tensor, gnn_output, mode: str) -> torch.Tensor:
    """Apply GNN correction to channel LLR based on correction mode.

    Args:
        channel_llr: (B, n) or (n,) channel LLR values
        gnn_output: for 'additive'/'multiplicative': (B, n) or (n,) tensor
                    for 'both': tuple of (additive, multiplicative) tensors
        mode: 'additive', 'multiplicative', or 'both'

    Returns:
        corrected_llr: same shape as channel_llr
    """
    if mode == "additive":
        return channel_llr + gnn_output
    elif mode == "multiplicative":
        return channel_llr * torch.exp(gnn_output)
    elif mode == "both":
        additive, multiplicative = gnn_output
        return channel_llr * torch.exp(multiplicative) + additive
    else:
        raise ValueError(f"Unknown correction mode: {mode}")


class TypedMessagePassingLayer(nn.Module):
    """Single message-passing layer with edge-type-specific transforms."""

    def __init__(self, hidden_dim: int, edge_types: int, dropout: float = 0.1):
        super().__init__()
        self.edge_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(edge_types)
        ])
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:          (num_nodes, hidden_dim) node embeddings
            edge_index: (2, num_edges) int64
            edge_type:  (num_edges,) int64
        Returns:
            x_new: (num_nodes, hidden_dim) updated node embeddings
        """
        num_nodes = x.shape[0]
        device = x.device

        src, dst = edge_index[0], edge_index[1]

        # Compute messages per edge type
        agg = torch.zeros(num_nodes, x.shape[1], device=device)
        for etype, mlp in enumerate(self.edge_mlps):
            mask = edge_type == etype
            if not mask.any():
                continue
            src_e = src[mask]
            dst_e = dst[mask]
            msg_input = torch.cat([x[src_e], x[dst_e]], dim=1)
            msgs = mlp(msg_input)
            agg.scatter_add_(0, dst_e.unsqueeze(1).expand_as(msgs), msgs)

        # GRU update
        x_new = self.gru(agg, x)
        return x_new


class TannerGNN(nn.Module):
    """GNN on the Tanner graph that outputs per-qubit LLR corrections.

    Supports three correction modes:
      - additive (default): corrected = channel_llr + gnn_output
      - multiplicative: corrected = channel_llr * exp(gnn_output)
      - both: outputs 2 values per qubit, combines additive + multiplicative
    """

    def __init__(
        self,
        node_feat_dim: int = 4,
        hidden_dim: int = 64,
        num_mp_layers: int = 3,
        edge_types: int = 2,
        dropout: float = 0.1,
        correction_mode: str = "additive",
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.correction_mode = correction_mode

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            TypedMessagePassingLayer(hidden_dim, edge_types, dropout)
            for _ in range(num_mp_layers)
        ])

        # Readout MLP: only applied to data-qubit nodes
        readout_dim = 2 if correction_mode == "both" else 1
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, readout_dim),
        )

    def forward(self, data) -> torch.Tensor:
        """
        Args:
            data: torch_geometric Data with fields:
                x:          (num_nodes, node_feat_dim) node features
                edge_index: (2, num_edges) graph edges
                edge_type:  (num_edges,) edge types
                node_type:  (num_nodes,) node types (0=data, 1=X-check, 2=Z-check)

        Returns:
            If correction_mode is 'additive' or 'multiplicative':
                llr_correction: (n_data,) per-data-qubit scalar
            If correction_mode is 'both':
                (additive, multiplicative): tuple of (n_data,) tensors
        """
        x = self.input_proj(data.x)

        for mp_layer in self.mp_layers:
            x = mp_layer(x, data.edge_index, data.edge_type)

        # Extract data qubit nodes (node_type == 0) and apply readout
        data_mask = data.node_type == 0
        data_embeddings = x[data_mask]  # (n_data, hidden_dim)

        if self.correction_mode == "both":
            out = self.readout(data_embeddings)  # (n_data, 2)
            return out[:, 0], out[:, 1]  # additive, multiplicative
        else:
            return self.readout(data_embeddings).squeeze(-1)  # (n_data,)
