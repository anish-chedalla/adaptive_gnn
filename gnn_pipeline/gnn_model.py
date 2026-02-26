"""GNN models for QLDPC decoding.

TannerGNN outputs per-qubit LLR corrections for BP decoding.
Supports FiLM conditioning for drift-adaptive decoding.
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
        return channel_llr * torch.exp(gnn_output.clamp(-5.0, 5.0))
    elif mode == "both":
        additive, multiplicative = gnn_output
        return channel_llr * torch.exp(multiplicative.clamp(-5.0, 5.0)) + additive
    else:
        raise ValueError(f"Unknown correction mode: {mode}")


class FiLMGenerator(nn.Module):
    """Feature-wise Linear Modulation generator.

    Takes noise features (e.g. per-shot p_value) and produces
    per-layer (gamma, beta) modulation parameters for the GNN.

    Gamma is initialized near 1.0 and beta near 0.0 so the model
    starts as an identity modulation (un-conditioned behavior).
    """

    def __init__(self, noise_feat_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # MLP: noise features -> per-layer gamma and beta
        out_dim = num_layers * 2 * hidden_dim  # gamma + beta for each layer
        self.mlp = nn.Sequential(
            nn.Linear(noise_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

        # Initialize last layer so gamma ~ 1.0, beta ~ 0.0
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            bias = torch.zeros(out_dim)
            # Set gamma biases to 1.0, beta biases to 0.0
            for layer_idx in range(num_layers):
                gamma_start = layer_idx * 2 * hidden_dim
                bias[gamma_start:gamma_start + hidden_dim] = 1.0
                # beta part stays at 0.0
            self.mlp[-1].bias.copy_(bias)

    def forward(self, noise_features: torch.Tensor):
        """
        Args:
            noise_features: (B, noise_feat_dim) noise conditioning input

        Returns:
            list of (gamma, beta) tuples, one per MP layer.
            gamma, beta: (B, hidden_dim) each.
        """
        out = self.mlp(noise_features)  # (B, num_layers * 2 * hidden_dim)

        film_params = []
        for layer_idx in range(self.num_layers):
            start = layer_idx * 2 * self.hidden_dim
            gamma = out[:, start:start + self.hidden_dim]
            beta = out[:, start + self.hidden_dim:start + 2 * self.hidden_dim]
            film_params.append((gamma, beta))

        return film_params


class TypedMessagePassingLayer(nn.Module):
    """Single message-passing layer with edge-type-specific transforms.

    Supports optional residual connections, layer normalization,
    attention, and FiLM conditioning.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_types: int,
        dropout: float = 0.1,
        use_residual: bool = False,
        use_layer_norm: bool = False,
        use_attention: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_attention = use_attention
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
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None

        # Attention: per-edge scalar score from src||dst embeddings
        if use_attention:
            self.attn_linears = nn.ModuleList([
                nn.Linear(2 * hidden_dim, 1)
                for _ in range(edge_types)
            ])

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        film_params=None,
    ) -> torch.Tensor:
        """
        Args:
            x:          (num_nodes, hidden_dim) node embeddings
            edge_index: (2, num_edges) int64
            edge_type:  (num_edges,) int64
            film_params: optional (gamma, beta) tuple, each (num_nodes, hidden_dim)
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

            # Optional attention weighting
            if self.use_attention:
                attn_score = torch.sigmoid(self.attn_linears[etype](msg_input))  # (E_type, 1)
                msgs = msgs * attn_score

            agg.scatter_add_(0, dst_e.unsqueeze(1).expand_as(msgs), msgs)

        # FiLM modulation: scale and shift aggregated messages
        if film_params is not None:
            gamma, beta = film_params
            agg = gamma * agg + beta

        # GRU update
        x_new = self.gru(agg, x)

        # Optional residual connection (same dim, no new params)
        if self.use_residual:
            x_new = x_new + x

        # Optional layer normalization
        if self.layer_norm is not None:
            x_new = self.layer_norm(x_new)

        return x_new


class TannerGNN(nn.Module):
    """GNN on the Tanner graph that outputs per-qubit LLR corrections.

    Supports three correction modes:
      - additive (default): corrected = channel_llr + gnn_output
      - multiplicative: corrected = channel_llr * exp(gnn_output)
      - both: outputs 2 values per qubit, combines additive + multiplicative

    Supports FiLM conditioning for drift-adaptive decoding:
      - When use_film=True, a FiLMGenerator produces per-layer modulation
        parameters from per-shot noise features (e.g. p_value).

    Supports interleaved GNN-BP:
      - When bp_marginal is provided in data.x as 5th feature, the GNN
        sees the intermediate BP decoding state for mid-iteration correction.
      - node_feat_dim=5: [channel_llr/syndrome, is_data, is_x_check, is_z_check, bp_marginal]

    Supports multi-class Pauli output (Astra-style):
      - When output_mode='multiclass', outputs 4-class Pauli probabilities (I,X,Z,Y)
        per qubit instead of scalar LLR corrections. This is the Astra architecture.
      - When output_mode='correction' (default), outputs scalar LLR corrections.

    Supports transfer learning:
      - freeze_backbone() freezes all layers except readout for fine-tuning
      - replace_readout() swaps the readout head (e.g. for different code size)
    """

    def __init__(
        self,
        node_feat_dim: int = 4,
        hidden_dim: int = 64,
        num_mp_layers: int = 3,
        edge_types: int = 2,
        dropout: float = 0.1,
        correction_mode: str = "additive",
        use_residual: bool = False,
        use_layer_norm: bool = False,
        use_attention: bool = False,
        use_film: bool = False,
        noise_feat_dim: int = 1,
        output_mode: str = "correction",
        standardize_input: bool = False,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.num_mp_layers = num_mp_layers
        self.correction_mode = correction_mode
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.use_attention = use_attention
        self.use_film = use_film
        self.noise_feat_dim = noise_feat_dim
        self.output_mode = output_mode
        self.standardize_input = standardize_input

        # Input projection with normalization for mixed-scale features
        # (LLR values ~20, indicator bits ~1, bp_marginal ~0-1)
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            TypedMessagePassingLayer(
                hidden_dim, edge_types, dropout,
                use_residual=use_residual,
                use_layer_norm=use_layer_norm,
                use_attention=use_attention,
            )
            for _ in range(num_mp_layers)
        ])

        # FiLM generator for noise-adaptive conditioning
        if use_film:
            self.film_gen = FiLMGenerator(noise_feat_dim, hidden_dim, num_mp_layers)

        # Readout MLP: only applied to data-qubit nodes
        if output_mode == "multiclass":
            # 4-class Pauli output: I, X, Z, Y (Astra-style)
            readout_dim = 4
        elif correction_mode == "both":
            readout_dim = 2
        else:
            readout_dim = 1

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, readout_dim),
        )

    def freeze_backbone(self):
        """Freeze all parameters except the readout head.

        Useful for transfer learning: train on one code, then fine-tune
        only the readout on a different code (potentially different size).
        """
        for param in self.input_proj.parameters():
            param.requires_grad = False
        for param in self.mp_layers.parameters():
            param.requires_grad = False
        if self.use_film and hasattr(self, 'film_gen'):
            for param in self.film_gen.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all parameters (reverse freeze_backbone)."""
        for param in self.parameters():
            param.requires_grad = True

    def replace_readout(self, new_output_dim: int, dropout: float = 0.1):
        """Replace the readout head for a new task/code size.

        This enables transfer learning across different code sizes:
        train backbone on small codes, then replace readout for larger codes.

        Args:
            new_output_dim: output dimension for the new readout head
            dropout: dropout rate for the new readout
        """
        self.readout = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, new_output_dim),
        )

    def forward(self, data, return_per_iter: bool = False):
        """
        Args:
            data: torch_geometric Data with fields:
                x:          (num_nodes, node_feat_dim) node features
                edge_index: (2, num_edges) graph edges
                edge_type:  (num_edges,) edge types
                node_type:  (num_nodes,) node types (0=data, 1=X-check, 2=Z-check)
                p_value:    (B,) per-shot noise level (optional, for FiLM)
                batch:      (num_nodes,) sample index per node (from DataLoader)
            return_per_iter: if True, return list of readout outputs from each
                MP layer (Astra-inspired per-iteration supervision).

        Returns:
            If return_per_iter is False (default):
                If correction_mode is 'additive' or 'multiplicative':
                    llr_correction: (n_data,) per-data-qubit scalar
                If correction_mode is 'both':
                    (additive, multiplicative): tuple of (n_data,) tensors
            If return_per_iter is True:
                list of T readout outputs (one per MP layer), each in the
                same format as the non-per-iter return.
        """
        raw_x = data.x
        if self.standardize_input:
            # Per-batch zero-mean unit-variance standardization of the LLR feature
            # (channel 0). This stabilises training across mixed-p datasets where
            # LLR ranges from ~3 (p=0.05) to ~20 (p=0.001).
            llr_col = raw_x[:, 0]
            llr_mean = llr_col.mean()
            llr_std = llr_col.std() + 1e-6
            raw_x = raw_x.clone()
            raw_x[:, 0] = (llr_col - llr_mean) / llr_std

        x = self.input_proj(raw_x)

        # Compute FiLM parameters if enabled
        film_params_list = None
        if self.use_film and hasattr(data, 'p_value'):
            p_val = data.p_value
            if p_val.dim() == 0:
                p_val = p_val.unsqueeze(0)
            noise_feat = p_val.unsqueeze(-1) if p_val.dim() == 1 else p_val  # (B, 1)
            noise_feat = noise_feat.to(x.device)
            film_params_list = self.film_gen(noise_feat)  # list of (gamma, beta)

            # Expand from (B, hidden_dim) to (num_nodes, hidden_dim) using batch vector
            batch_vec = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(
                x.shape[0], dtype=torch.long, device=x.device
            )
            expanded_film = []
            for gamma, beta in film_params_list:
                expanded_film.append((gamma[batch_vec], beta[batch_vec]))
            film_params_list = expanded_film

        data_mask = data.node_type == 0
        per_iter_outputs = [] if return_per_iter else None

        for layer_idx, mp_layer in enumerate(self.mp_layers):
            fp = film_params_list[layer_idx] if film_params_list is not None else None
            x = mp_layer(x, data.edge_index, data.edge_type, film_params=fp)

            if return_per_iter:
                data_embeddings = x[data_mask]
                per_iter_outputs.append(self._apply_readout(data_embeddings))

        if return_per_iter:
            return per_iter_outputs

        # Extract data qubit nodes (node_type == 0) and apply readout
        data_embeddings = x[data_mask]  # (n_data, hidden_dim)
        return self._apply_readout(data_embeddings)

    def _apply_readout(self, data_embeddings: torch.Tensor):
        """Apply readout head to data-node embeddings."""
        if self.output_mode == "multiclass":
            logits = self.readout(data_embeddings)  # (n_data, 4)
            return F.softmax(logits, dim=-1)
        elif self.correction_mode == "both":
            out = self.readout(data_embeddings)  # (n_data, 2)
            return out[:, 0], out[:, 1]
        else:
            return self.readout(data_embeddings).squeeze(-1)  # (n_data,)
