"""Min-sum Belief Propagation decoder for CSS QLDPC codes.

Operates in LLR (log-likelihood ratio) domain.
Differentiable via PyTorch for end-to-end GNN training.

CSS handling: two independent decoder instances are used externally —
one with Hz (decodes X errors from Z-syndrome) and
one with Hx (decodes Z errors from X-syndrome).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class MinSumBPDecoder(nn.Module):
    """Min-sum BP decoder with optional normalization factor.

    Args:
        pcm: (m, n) binary parity-check matrix
        max_iter: maximum BP iterations
        alpha: min-sum normalization factor (0 < alpha <= 1)
        clamp_llr: clamp LLR values to [-clamp_llr, clamp_llr]
    """

    def __init__(
        self,
        pcm: np.ndarray,
        max_iter: int = 50,
        alpha: float = 0.8,
        clamp_llr: float = 20.0,
    ):
        super().__init__()
        self.max_iter = max_iter
        self.alpha = alpha
        self.clamp_llr = clamp_llr

        pcm = np.asarray(pcm, dtype=np.uint8)
        assert pcm.ndim == 2
        self.num_checks, self.num_vars = pcm.shape

        # Extract edge structure from PCM
        check_indices, var_indices = np.nonzero(pcm)
        self.num_edges = len(check_indices)

        self.register_buffer("check_idx", torch.from_numpy(check_indices.astype(np.int64)))
        self.register_buffer("var_idx", torch.from_numpy(var_indices.astype(np.int64)))

        # Build adjacency lists for efficient message passing
        # For each check c: list of edge indices where check_idx == c
        # For each var v: list of edge indices where var_idx == v
        check_adj: List[List[int]] = [[] for _ in range(self.num_checks)]
        var_adj: List[List[int]] = [[] for _ in range(self.num_vars)]
        for e_idx in range(self.num_edges):
            c = int(check_indices[e_idx])
            v = int(var_indices[e_idx])
            check_adj[c].append(e_idx)
            var_adj[v].append(e_idx)

        # Pad to rectangular arrays for vectorized operations
        max_check_deg = max(len(adj) for adj in check_adj) if check_adj else 1
        max_var_deg = max(len(adj) for adj in var_adj) if var_adj else 1

        check_adj_arr = np.zeros((self.num_checks, max_check_deg), dtype=np.int64)
        check_adj_mask = np.zeros((self.num_checks, max_check_deg), dtype=np.float32)
        for c, adj in enumerate(check_adj):
            for j, e in enumerate(adj):
                check_adj_arr[c, j] = e
                check_adj_mask[c, j] = 1.0

        var_adj_arr = np.zeros((self.num_vars, max_var_deg), dtype=np.int64)
        var_adj_mask = np.zeros((self.num_vars, max_var_deg), dtype=np.float32)
        for v, adj in enumerate(var_adj):
            for j, e in enumerate(adj):
                var_adj_arr[v, j] = e
                var_adj_mask[v, j] = 1.0

        self.register_buffer("check_adj", torch.from_numpy(check_adj_arr))
        self.register_buffer("check_adj_mask", torch.from_numpy(check_adj_mask))
        self.register_buffer("var_adj", torch.from_numpy(var_adj_arr))
        self.register_buffer("var_adj_mask", torch.from_numpy(var_adj_mask))

    def forward(
        self,
        syndrome: torch.Tensor,
        channel_llr: torch.Tensor,
        max_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run min-sum BP decoding.

        Args:
            syndrome:    (batch, m) float — syndrome bits (0 or 1)
            channel_llr: (batch, n) float — per-qubit channel LLR
            max_iter:    override max iterations

        Returns:
            marginals:     (batch, n) float — soft error probabilities in [0, 1]
            hard_decision: (batch, n) int   — binary error estimate
            converged:     (batch,) bool    — whether syndrome is satisfied
        """
        iters = max_iter if max_iter is not None else self.max_iter
        B = syndrome.shape[0]
        device = syndrome.device

        # Convert syndrome to +1/-1 sign convention: s_c = (-1)^syndrome[c]
        s_sign = 1.0 - 2.0 * syndrome  # (B, m)

        # Initialize messages: check-to-var (CTV) = 0
        ctv = torch.zeros(B, self.num_edges, device=device)

        for _ in range(iters):
            # --- Variable-to-check (VTC) ---
            # mu_{v->c} = channel_llr[v] + sum_{c' != c} ctv[c'->v]
            # Compute total incoming CTV per variable
            # var_adj: (n, max_var_deg) indexing into ctv
            ctv_gathered = ctv[:, self.var_adj]  # (B, n, max_var_deg)
            ctv_masked = ctv_gathered * self.var_adj_mask.unsqueeze(0)  # zero out padding
            var_total = ctv_masked.sum(dim=2)  # (B, n) total CTV per var

            # VTC for edge e = channel_llr[var_idx[e]] + var_total[var_idx[e]] - ctv[e]
            vtc = channel_llr[:, self.var_idx] + var_total[:, self.var_idx] - ctv  # (B, num_edges)
            vtc = vtc.clamp(-self.clamp_llr, self.clamp_llr)

            # --- Check-to-variable (CTV): min-sum ---
            # For each check c, gather VTC messages from neighbors
            vtc_gathered = vtc[:, self.check_adj]  # (B, m, max_check_deg)

            # Sign product: product of signs excluding current edge
            signs = torch.sign(vtc_gathered)
            # Replace zeros with +1 for sign computation
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            signs_masked = signs * self.check_adj_mask.unsqueeze(0) + (1.0 - self.check_adj_mask.unsqueeze(0))

            # Include syndrome sign
            total_sign = s_sign.unsqueeze(2) * signs_masked.prod(dim=2, keepdim=True)  # (B, m, 1)
            # Exclude-self sign: total_sign / sign_of_self
            excl_sign = total_sign / signs_masked  # (B, m, max_check_deg)

            # Minimum absolute value excluding current edge
            abs_vtc = vtc_gathered.abs()
            # Set padded positions to a large value so they don't affect min
            abs_vtc_masked = abs_vtc + (1.0 - self.check_adj_mask.unsqueeze(0)) * 1e6

            # For each edge in the check, compute min of all OTHER edges
            # Sort to find the two smallest
            sorted_abs, _ = abs_vtc_masked.sort(dim=2)
            min1 = sorted_abs[:, :, 0:1]  # (B, m, 1) smallest
            min2 = sorted_abs[:, :, 1:2] if sorted_abs.shape[2] > 1 else min1

            # If current edge has the minimum, use second minimum; else use first minimum
            is_min = (abs_vtc_masked - min1).abs() < 1e-9
            excl_min = torch.where(is_min, min2.expand_as(abs_vtc_masked), min1.expand_as(abs_vtc_masked))

            # CTV message: alpha * excl_sign * excl_min
            ctv_new = self.alpha * excl_sign * excl_min
            ctv_new = ctv_new * self.check_adj_mask.unsqueeze(0)  # zero padding

            # Scatter back: ctv_new is (B, m, max_check_deg) indexed by check_adj
            ctv = torch.zeros_like(ctv)
            check_adj_expanded = self.check_adj.unsqueeze(0).expand(B, -1, -1)
            ctv.scatter_(1,
                         check_adj_expanded.reshape(B, -1),
                         ctv_new.reshape(B, -1))

        # Final marginals
        ctv_gathered = ctv[:, self.var_adj]
        ctv_masked = ctv_gathered * self.var_adj_mask.unsqueeze(0)
        var_total = ctv_masked.sum(dim=2)
        total_llr = channel_llr + var_total  # (B, n)

        marginals = torch.sigmoid(-total_llr)  # P(error) = sigma(-LLR)
        hard_decision = (marginals > 0.5).long()

        # Check convergence: does hard_decision satisfy syndrome?
        # syndrome_hat[c] = XOR of hard_decision[v] for v in support(check c)
        hd_float = hard_decision.float()
        syndrome_hat = torch.zeros(B, self.num_checks, device=device)
        for e in range(self.num_edges):
            c = self.check_idx[e]
            v = self.var_idx[e]
            syndrome_hat[:, c] = syndrome_hat[:, c] + hd_float[:, v]
        syndrome_hat = syndrome_hat % 2
        converged = (syndrome_hat - syndrome).abs().sum(dim=1) < 0.5  # (B,)

        return marginals, hard_decision, converged
