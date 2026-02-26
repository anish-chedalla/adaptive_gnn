"""Min-sum Belief Propagation decoder for CSS QLDPC codes.

Operates in LLR (log-likelihood ratio) domain.
Differentiable via PyTorch for end-to-end GNN training.

CSS handling: two independent decoder instances are used externally --
one with Hz (decodes X errors from Z-syndrome) and
one with Hx (decodes Z errors from X-syndrome).

Supports Neural BP mode: per-iteration learned scalar weights for
channel LLR, VTC messages, and CTV messages (Nachmani et al. approach).
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
        neural_bp: if True, add per-iteration learned weights (Neural BP)
    """

    def __init__(
        self,
        pcm: np.ndarray,
        max_iter: int = 50,
        alpha: float = 0.8,
        clamp_llr: float = 20.0,
        neural_bp: bool = False,
    ):
        super().__init__()
        self.max_iter = max_iter
        self._alpha_init = alpha
        self.clamp_llr = clamp_llr
        self.neural_bp = neural_bp
        # Default: fixed alpha. Use make_alpha_learnable() to make it a parameter.
        self.alpha = alpha

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

        # Sparse PCM for fast syndrome check (replaces Python edge loop)
        pcm_sparse = torch.from_numpy(pcm.astype(np.float32))
        self.register_buffer("pcm_dense", pcm_sparse)  # (m, n)

        # Neural BP: per-iteration learned scalar weights
        # Weights are stored as raw parameters and bounded via softplus in _bp_weight()
        # to ensure they stay in (0, ~2) range, preventing divergence.
        if neural_bp:
            self.w_ch = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in range(max_iter)
            ])
            self.w_vtc = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in range(max_iter)
            ])
            self.w_ctv = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in range(max_iter)
            ])

    @staticmethod
    def _bp_weight(raw: torch.Tensor) -> torch.Tensor:
        """Bound a raw neural BP weight parameter to positive range via softplus.

        softplus(0) = ln(2) ~ 0.693.  Dividing by ln(2) normalises so that
        init-at-zero maps to exactly 1.0 (identity scaling for standard BP).
        Range: (0, +inf) in theory, but values stay near 1 with normal training.
        """
        return torch.nn.functional.softplus(raw) / 0.6931471805599453

    def make_alpha_learnable(self) -> nn.Parameter:
        """Convert fixed alpha to a learnable nn.Parameter.

        Returns the parameter so it can be included in optimizer param groups.
        """
        self.alpha_param = nn.Parameter(torch.tensor(self._alpha_init))
        self.alpha = None  # Signal to use alpha_param in forward
        return self.alpha_param

    @property
    def effective_alpha(self) -> torch.Tensor:
        """Get the current alpha value (fixed float or learnable parameter)."""
        if self.alpha is not None:
            return self.alpha
        # Learnable alpha: clamp to (0, 1) via sigmoid
        return torch.sigmoid(self.alpha_param)

    def forward(
        self,
        syndrome: torch.Tensor,
        channel_llr: torch.Tensor,
        max_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run min-sum BP decoding.

        Args:
            syndrome:    (batch, m) float -- syndrome bits (0 or 1)
            channel_llr: (batch, n) float -- per-qubit channel LLR
            max_iter:    override max iterations

        Returns:
            marginals:     (batch, n) float -- soft error probabilities in [0, 1]
            hard_decision: (batch, n) int   -- binary error estimate
            converged:     (batch,) bool    -- whether syndrome is satisfied
        """
        iters = max_iter if max_iter is not None else self.max_iter
        B = syndrome.shape[0]
        device = syndrome.device

        # Convert syndrome to +1/-1 sign convention: s_c = (-1)^syndrome[c]
        s_sign = 1.0 - 2.0 * syndrome  # (B, m)

        # Initialize messages: check-to-var (CTV) = 0
        ctv = torch.zeros(B, self.num_edges, device=device)

        for it in range(iters):
            # --- Variable-to-check (VTC) ---
            # mu_{v->c} = channel_llr[v] + sum_{c' != c} ctv[c'->v]
            # Compute total incoming CTV per variable
            # var_adj: (n, max_var_deg) indexing into ctv
            ctv_gathered = ctv[:, self.var_adj]  # (B, n, max_var_deg)
            ctv_masked = ctv_gathered * self.var_adj_mask.unsqueeze(0)  # zero out padding
            var_total = ctv_masked.sum(dim=2)  # (B, n) total CTV per var

            # VTC for edge e = channel_llr[var_idx[e]] + var_total[var_idx[e]] - ctv[e]
            if self.neural_bp and it < len(self.w_ch):
                # Neural BP: bounded per-iteration channel and message weights
                vtc = (self._bp_weight(self.w_ch[it]) * channel_llr[:, self.var_idx]
                       + self._bp_weight(self.w_vtc[it]) * (var_total[:, self.var_idx] - ctv))
            else:
                vtc = channel_llr[:, self.var_idx] + var_total[:, self.var_idx] - ctv
            vtc = vtc.clamp(-self.clamp_llr, self.clamp_llr)

            # --- Check-to-variable (CTV): min-sum with TopK-2 ---
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
            # Safe division: signs_masked is +/-1 for real edges and +1 for padding,
            # but guard against any zero values to prevent NaN in gradients.
            safe_signs = torch.where(signs_masked.abs() > 0.5, signs_masked, torch.ones_like(signs_masked))
            excl_sign = total_sign * safe_signs  # Equivalent to total_sign / sign when sign is +/-1  # (B, m, max_check_deg)

            # Minimum absolute value excluding current edge
            abs_vtc = vtc_gathered.abs()
            # Set padded positions to a large value so they don't affect min
            abs_vtc_masked = abs_vtc + (1.0 - self.check_adj_mask.unsqueeze(0)) * 1e6

            # TopK-2: only need top-2 smallest values (O(n) via topk vs O(n log n) sort)
            k = min(2, abs_vtc_masked.shape[2])
            topk_vals, _ = abs_vtc_masked.topk(k, dim=2, largest=False)
            min1 = topk_vals[:, :, 0:1]  # (B, m, 1) smallest
            min2 = topk_vals[:, :, 1:2] if k > 1 else min1

            # If current edge has the minimum, use second minimum; else use first minimum
            is_min = (abs_vtc_masked - min1).abs() < 1e-9
            excl_min = torch.where(is_min, min2.expand_as(abs_vtc_masked), min1.expand_as(abs_vtc_masked))

            # CTV message: alpha * excl_sign * excl_min
            if self.neural_bp and it < len(self.w_ctv):
                ctv_new = self.effective_alpha * self._bp_weight(self.w_ctv[it]) * excl_sign * excl_min
            else:
                ctv_new = self.effective_alpha * excl_sign * excl_min
            ctv_new = ctv_new * self.check_adj_mask.unsqueeze(0)  # zero padding

            # Scatter back: ctv_new is (B, m, max_check_deg) indexed by check_adj
            ctv = torch.zeros_like(ctv)
            check_adj_expanded = self.check_adj.unsqueeze(0).expand(B, -1, -1)
            ctv.scatter_(1,
                         check_adj_expanded.reshape(B, -1),
                         ctv_new.reshape(B, -1))

            # Early termination: check if all samples converged
            # In eval mode: check every iteration for fastest exit.
            # In training mode: check every 3 iterations (gradient-aware);
            #   breaking early is safe because the gradient graph through the
            #   converged iterations is still intact and provides correct signal.
            if it < iters - 1:
                check_now = (not self.training) or (it > 0 and it % 3 == 0)
                if check_now:
                    with torch.no_grad():
                        ctv_g = ctv[:, self.var_adj]
                        ctv_m = ctv_g * self.var_adj_mask.unsqueeze(0)
                        vt = ctv_m.sum(dim=2)
                        hd = ((channel_llr + vt) < 0).float()
                        syn_hat = torch.fmod(torch.round(hd @ self.pcm_dense.t()), 2.0)
                        if (syn_hat - syndrome).abs().sum(dim=1).max() < 0.5:
                            break  # All converged, skip remaining iterations

        # Final marginals
        ctv_gathered = ctv[:, self.var_adj]
        ctv_masked = ctv_gathered * self.var_adj_mask.unsqueeze(0)
        var_total = ctv_masked.sum(dim=2)
        total_llr = channel_llr + var_total  # (B, n)

        marginals = torch.sigmoid(-total_llr)  # P(error) = sigma(-LLR)
        hard_decision = (marginals > 0.5).long()

        # Check convergence via vectorized matmul (replaces Python edge loop)
        hd_float = hard_decision.float()
        # Use fmod(round()) for safe integer mod-2 on floats
        syndrome_hat = torch.fmod(torch.round(hd_float @ self.pcm_dense.t()), 2.0)  # (B, m)
        converged = (syndrome_hat - syndrome).abs().sum(dim=1) < 0.5  # (B,)

        return marginals, hard_decision, converged

    def forward_stages(
        self,
        syndrome: torch.Tensor,
        channel_llr: torch.Tensor,
        stage_iters=5,
        num_stages: int = 2,
        correction_fn=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run BP in stages with optional corrections between stages.

        This enables interleaved GNN-BP: the GNN can correct LLRs
        between BP stages based on the intermediate decoding state.

        Args:
            syndrome:      (batch, m) float -- syndrome bits
            channel_llr:   (batch, n) float -- initial channel LLR
            stage_iters:   iterations per stage (int for uniform, or list of ints
                          for per-stage counts, e.g. [10, 90] for 2 stages)
            num_stages:    number of stages (ignored if stage_iters is a list)
            correction_fn: callable(marginals, channel_llr, stage_idx) -> corrected_llr
                          Called between stages with current soft marginals.
                          Should return corrected channel_llr for next stage.
                          If None, no inter-stage corrections are applied.

        Returns:
            marginals, hard_decision, converged (same as forward)
        """
        B = syndrome.shape[0]
        device = syndrome.device
        s_sign = 1.0 - 2.0 * syndrome

        ctv = torch.zeros(B, self.num_edges, device=device)
        current_llr = channel_llr

        # Support per-stage iter counts
        if isinstance(stage_iters, (list, tuple)):
            iters_per_stage = list(stage_iters)
            num_stages = len(iters_per_stage)
        else:
            iters_per_stage = [stage_iters] * num_stages

        total_it = 0
        for stage in range(num_stages):
            iters_this_stage = iters_per_stage[stage]

            for it_local in range(iters_this_stage):
                it = total_it  # global iteration index for neural_bp weights

                # --- VTC ---
                ctv_gathered = ctv[:, self.var_adj]
                ctv_masked = ctv_gathered * self.var_adj_mask.unsqueeze(0)
                var_total = ctv_masked.sum(dim=2)

                if self.neural_bp and it < len(self.w_ch):
                    vtc = (self._bp_weight(self.w_ch[it]) * current_llr[:, self.var_idx]
                           + self._bp_weight(self.w_vtc[it]) * (var_total[:, self.var_idx] - ctv))
                else:
                    vtc = current_llr[:, self.var_idx] + var_total[:, self.var_idx] - ctv
                vtc = vtc.clamp(-self.clamp_llr, self.clamp_llr)

                # --- CTV with TopK-2 ---
                vtc_gathered = vtc[:, self.check_adj]
                signs = torch.sign(vtc_gathered)
                signs = torch.where(signs == 0, torch.ones_like(signs), signs)
                signs_masked = signs * self.check_adj_mask.unsqueeze(0) + (1.0 - self.check_adj_mask.unsqueeze(0))

                total_sign = s_sign.unsqueeze(2) * signs_masked.prod(dim=2, keepdim=True)
                # Safe division: signs_masked is +/-1 for real edges and +1 for padding,
                # but guard against any zero values to prevent NaN in gradients.
                safe_signs = torch.where(signs_masked.abs() > 0.5, signs_masked, torch.ones_like(signs_masked))
                excl_sign = total_sign * safe_signs  # Equivalent to total_sign / sign when sign is +/-1

                abs_vtc = vtc_gathered.abs()
                abs_vtc_masked = abs_vtc + (1.0 - self.check_adj_mask.unsqueeze(0)) * 1e6
                k = min(2, abs_vtc_masked.shape[2])
                topk_vals, _ = abs_vtc_masked.topk(k, dim=2, largest=False)
                min1 = topk_vals[:, :, 0:1]
                min2 = topk_vals[:, :, 1:2] if k > 1 else min1
                is_min = (abs_vtc_masked - min1).abs() < 1e-9
                excl_min = torch.where(is_min, min2.expand_as(abs_vtc_masked), min1.expand_as(abs_vtc_masked))

                if self.neural_bp and it < len(self.w_ctv):
                    ctv_new = self.effective_alpha * self._bp_weight(self.w_ctv[it]) * excl_sign * excl_min
                else:
                    ctv_new = self.effective_alpha * excl_sign * excl_min
                ctv_new = ctv_new * self.check_adj_mask.unsqueeze(0)

                ctv = torch.zeros_like(ctv)
                check_adj_expanded = self.check_adj.unsqueeze(0).expand(B, -1, -1)
                ctv.scatter_(1, check_adj_expanded.reshape(B, -1), ctv_new.reshape(B, -1))

                total_it += 1

            # Inter-stage correction: compute marginals and let GNN correct
            if correction_fn is not None and stage < num_stages - 1:
                ctv_g = ctv[:, self.var_adj]
                ctv_m = ctv_g * self.var_adj_mask.unsqueeze(0)
                vt = ctv_m.sum(dim=2)
                mid_total_llr = current_llr + vt
                mid_marginals = torch.sigmoid(-mid_total_llr)  # (B, n)
                current_llr = correction_fn(mid_marginals, current_llr, stage)

        # Final marginals
        ctv_gathered = ctv[:, self.var_adj]
        ctv_masked = ctv_gathered * self.var_adj_mask.unsqueeze(0)
        var_total = ctv_masked.sum(dim=2)
        total_llr = current_llr + var_total

        marginals = torch.sigmoid(-total_llr)
        hard_decision = (marginals > 0.5).long()

        hd_float = hard_decision.float()
        syndrome_hat = torch.fmod(torch.round(hd_float @ self.pcm_dense.t()), 2.0)
        converged = (syndrome_hat - syndrome).abs().sum(dim=1) < 0.5

        return marginals, hard_decision, converged
