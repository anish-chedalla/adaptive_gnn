"""Evaluate BP decoder, GNN-assisted BP, and BP-OSD on test data.

Supports two modes:
  - code_capacity (default): Separate CSS decoding with hx/hz on synthetic data
  - circuit_level: DEM-based decoding on Stim circuit-level data

Compares logical error rates (LER) with Wilson score confidence intervals.

Usage:
    # Code-capacity mode (default)
    python -m gnn_pipeline.evaluate --test_npz "data/pure_code_cap.npz" --out_dir "runs/eval"

    # Code-capacity with GNN and BP-OSD
    python -m gnn_pipeline.evaluate --test_npz "data/test.npz" --gnn_model "runs/model/best_model.pt" --bposd --out_dir "runs/eval"

    # Circuit-level mode
    python -m gnn_pipeline.evaluate --test_npz "data/circuit.npz" --mode circuit_level --out_dir "runs/eval_dem"

    # Circuit-level with BP-OSD
    python -m gnn_pipeline.evaluate --test_npz "data/circuit.npz" --mode circuit_level --bposd --out_dir "runs/eval_dem"
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from gnn_pipeline.bp_decoder import MinSumBPDecoder
from gnn_pipeline.dataset import _load_npz, _parse_meta
from gnn_pipeline.gnn_model import TannerGNN, apply_correction
from gnn_pipeline.tanner_graph import build_tanner_graph


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def mcnemar_test(
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
) -> Tuple[float, float]:
    """McNemar's test for paired binary outcomes (with continuity correction).

    Tests whether two decoders have significantly different logical error rates
    on the same set of shots. More appropriate than unpaired proportion tests
    because decoder outcomes on the same shot are correlated.

    Args:
        outcomes_a: (N,) bool array — True = logical error for decoder A
        outcomes_b: (N,) bool array — True = logical error for decoder B

    Returns:
        (chi2, p_value): McNemar's test statistic and two-sided p-value.
        Returns (0.0, 1.0) if the test is degenerate (n_01 + n_10 == 0).
    """
    a = outcomes_a.astype(bool)
    b = outcomes_b.astype(bool)

    # Discordant pairs
    n_01 = int(np.sum(~a & b))  # A correct, B wrong
    n_10 = int(np.sum(a & ~b))  # A wrong, B correct

    n_disc = n_01 + n_10
    if n_disc == 0:
        return 0.0, 1.0

    # McNemar's chi-squared with continuity correction
    chi2 = (abs(n_01 - n_10) - 1) ** 2 / n_disc
    # p-value from chi-squared distribution with 1 df
    # Use survival function: P(X > chi2) for chi2(df=1)
    # chi2 CDF via regularized incomplete gamma: P = 1 - gammainc(0.5, chi2/2)
    # For chi2(df=1): p = erfc(sqrt(chi2/2))
    p_value = _chi2_sf_1df(chi2)
    return float(chi2), float(p_value)


def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi-squared distribution with 1 degree of freedom.

    P(X > x) = erfc(sqrt(x/2)) for chi2(df=1).
    Uses math.erfc for accuracy without scipy dependency.
    """
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def pairwise_significance(
    decoder_outcomes: Dict[str, np.ndarray],
    reference: str = "BP (stale LLR)",
) -> List[dict]:
    """Run McNemar's test between each decoder and a reference decoder.

    Args:
        decoder_outcomes: {decoder_name: (N,) bool array of logical errors}
        reference: name of the reference decoder (default: plain BP)

    Returns:
        List of dicts with keys: decoder_a, decoder_b, chi2, p_value, significant
    """
    if reference not in decoder_outcomes:
        return []

    ref = decoder_outcomes[reference]
    results = []
    for name, outcomes in decoder_outcomes.items():
        if name == reference:
            continue
        chi2, pval = mcnemar_test(ref, outcomes)
        results.append({
            "decoder_a": reference,
            "decoder_b": name,
            "chi2": round(chi2, 4),
            "p_value": pval,
            "significant": pval < 0.05,
        })
    return results


def bootstrap_ler_difference_ci(
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for LER_A - LER_B.

    Computes a percentile bootstrap CI for the *difference* in logical error
    rate between two decoders on the same data.  A negative difference means
    decoder B has lower (better) LER.

    Args:
        outcomes_a: (N,) bool — True when decoder A made a logical error
        outcomes_b: (N,) bool — True when decoder B made a logical error
        n_bootstrap: number of bootstrap resamples (default 2000)
        confidence: CI level (default 0.95 for 95% CI)
        seed: random seed for reproducibility

    Returns:
        (mean_diff, ci_low, ci_high) — mean difference and CI bounds
    """
    rng = np.random.default_rng(seed)
    n = len(outcomes_a)
    if n == 0:
        return 0.0, 0.0, 0.0
    a = outcomes_a.astype(np.float64)
    b = outcomes_b.astype(np.float64)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.percentile(diffs, 100 * alpha))
    ci_high = float(np.percentile(diffs, 100 * (1.0 - alpha)))
    return float(diffs.mean()), ci_low, ci_high


def wilson_score_interval_binom(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute Wilson score interval for binomial proportion."""
    if trials == 0:
        return (0.0, 1.0)

    from scipy import stats

    z = stats.norm.ppf((1 + confidence) / 2.0)
    z_sq = z * z

    p_hat = successes / trials
    denom = 1.0 + z_sq / trials
    center = (p_hat + z_sq / (2.0 * trials)) / denom
    margin = z * np.sqrt(p_hat * (1.0 - p_hat) / trials + z_sq / (4.0 * trials**2)) / denom

    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
    )


# ---------------------------------------------------------------------------
# Vectorized Batch Construction
# ---------------------------------------------------------------------------

def _build_eval_batch(
    all_x_syn_chunk: np.ndarray,
    all_z_syn_chunk: np.ndarray,
    avg_llr: float,
    n: int,
    mx: int,
    mz: int,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    node_type: torch.Tensor,
    p_vals_chunk: np.ndarray,
    device: torch.device,
) -> Batch:
    """Build a batched GNN input without per-sample Python loops.

    Constructs the feature tensor, replicates the graph structure, and
    returns a ready-to-use ``Batch`` on ``device``.

    Args:
        all_x_syn_chunk: (B, mx) X-syndrome values
        all_z_syn_chunk: (B, mz) Z-syndrome values
        avg_llr: average channel LLR (scalar, broadcast to all data qubits)
        n, mx, mz: code dimensions
        edge_index: (2, E) edge indices for a single graph
        edge_type: (E,) edge types for a single graph
        node_type: (num_nodes,) node types for a single graph
        p_vals_chunk: (B,) per-shot noise estimate
        device: target device

    Returns:
        ``Batch`` object with all fields set, on ``device``.
    """
    B = all_x_syn_chunk.shape[0]
    num_nodes = n + mx + mz
    E = edge_index.shape[1]

    # --- Vectorized feature tensor: (B, num_nodes, 4) ---
    x_3d = torch.zeros(B, num_nodes, 4, dtype=torch.float32)
    x_3d[:, :n, 0] = avg_llr
    x_3d[:, :n, 1] = 1.0
    x_3d[:, n:n + mx, 0] = torch.from_numpy(all_x_syn_chunk).float()
    x_3d[:, n:n + mx, 2] = 1.0
    x_3d[:, n + mx:, 0] = torch.from_numpy(all_z_syn_chunk).float()
    x_3d[:, n + mx:, 3] = 1.0

    # --- Replicate graph with per-sample offsets ---
    ei_cpu = edge_index.cpu()
    offsets = torch.arange(B, dtype=torch.long).unsqueeze(1) * num_nodes  # (B, 1)
    batch_ei = (
        ei_cpu.unsqueeze(0).expand(B, 2, E) + offsets.unsqueeze(1)
    ).reshape(2, -1)

    batch_et = edge_type.cpu().repeat(B)
    batch_nt = node_type.cpu().repeat(B)
    batch_vec = torch.arange(B, dtype=torch.long).repeat_interleave(num_nodes)
    batch_ch_llr = torch.full((B * n,), avg_llr, dtype=torch.float32)
    batch_p = torch.from_numpy(p_vals_chunk.astype(np.float32))

    batch = Batch(
        x=x_3d.reshape(-1, 4),
        edge_index=batch_ei,
        edge_type=batch_et,
        node_type=batch_nt,
        channel_llr=batch_ch_llr,
        p_value=batch_p,
        batch=batch_vec,
    )
    batch._num_graphs = B
    return batch.to(device)


# ---------------------------------------------------------------------------
# Code-Capacity Decoders
# ---------------------------------------------------------------------------

def run_css_bp_decoder(
    x_syndrome: np.ndarray,
    z_syndrome: np.ndarray,
    hx: np.ndarray,
    hz: np.ndarray,
    llr_z: float,
    llr_x: float,
    n: int,
    device: torch.device,
    max_iter: int = 100,
    dec_z_prebuilt: Optional[MinSumBPDecoder] = None,
    dec_x_prebuilt: Optional[MinSumBPDecoder] = None,
) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    """Run separate CSS BP decoding.

    CSS decoding splits into two independent problems:
      - hx @ z_error = x_syndrome  (X-checks detect Z-errors)
      - hz @ x_error = z_syndrome  (Z-checks detect X-errors)

    Args:
        x_syndrome: (mx,) X-check syndrome
        z_syndrome: (mz,) Z-check syndrome
        hx: (mx, n) X parity check matrix
        hz: (mz, n) Z parity check matrix
        llr_z: channel LLR for Z-error probability
        llr_x: channel LLR for X-error probability
        n: number of data qubits
        device: torch device
        max_iter: BP iterations
        dec_z_prebuilt: optional pre-built Z-error decoder (reuse across shots)
        dec_x_prebuilt: optional pre-built X-error decoder (reuse across shots)

    Returns:
        (z_errors, x_errors, z_converged, x_converged)
    """
    # Decode Z-errors using hx
    if x_syndrome.sum() > 0:
        dec_z = dec_z_prebuilt if dec_z_prebuilt is not None else \
            MinSumBPDecoder(hx, max_iter=max_iter, alpha=0.8, clamp_llr=20.0).to(device)
        syn_t = torch.from_numpy(x_syndrome[np.newaxis, :]).float().to(device)
        llr_t = torch.full((1, n), llr_z, dtype=torch.float32, device=device)
        _, hard_z, conv_z = dec_z(syn_t, llr_t)
        z_errors = hard_z[0].cpu().numpy()
        z_conv = bool(conv_z[0])
    else:
        z_errors = np.zeros(n, dtype=np.int64)
        z_conv = True

    # Decode X-errors using hz
    if z_syndrome.sum() > 0:
        dec_x = dec_x_prebuilt if dec_x_prebuilt is not None else \
            MinSumBPDecoder(hz, max_iter=max_iter, alpha=0.8, clamp_llr=20.0).to(device)
        syn_t = torch.from_numpy(z_syndrome[np.newaxis, :]).float().to(device)
        llr_t = torch.full((1, n), llr_x, dtype=torch.float32, device=device)
        _, hard_x, conv_x = dec_x(syn_t, llr_t)
        x_errors = hard_x[0].cpu().numpy()
        x_conv = bool(conv_x[0])
    else:
        x_errors = np.zeros(n, dtype=np.int64)
        x_conv = True

    return z_errors, x_errors, z_conv, x_conv


def run_gnn_css_bp_decoder(
    x_syndrome: np.ndarray,
    z_syndrome: np.ndarray,
    hx: np.ndarray,
    hz: np.ndarray,
    llr_z: float,
    llr_x: float,
    n: int,
    gnn_model: nn.Module,
    device: torch.device,
    max_iter: int = 100,
    correction_mode: str = "additive",
    dec_z_prebuilt: Optional[MinSumBPDecoder] = None,
    dec_x_prebuilt: Optional[MinSumBPDecoder] = None,
    tanner_graph: Optional[Tuple] = None,
    p_value: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    """Run GNN-assisted separate CSS BP decoding.

    The GNN provides per-qubit LLR corrections before BP runs.

    Args:
        dec_z_prebuilt: optional pre-built Z-error decoder (reuse across shots)
        dec_x_prebuilt: optional pre-built X-error decoder (reuse across shots)
        tanner_graph: optional pre-built (node_type_t, edge_index_t, edge_type_t)
                      tensors on device (reuse across shots)
    """
    mx, mz = hx.shape[0], hz.shape[0]

    # Use cached Tanner graph or build fresh
    if tanner_graph is not None:
        node_type_t, edge_index_t, edge_type_t = tanner_graph
    else:
        node_type_np, edge_index_np, edge_type_np = build_tanner_graph(hx, hz)
        node_type_t = torch.from_numpy(node_type_np).long().to(device)
        edge_index_t = torch.from_numpy(edge_index_np).long().to(device)
        edge_type_t = torch.from_numpy(edge_type_np).long().to(device)

    # Node features: [channel_llr, is_data, is_x_check, is_z_check]
    num_nodes = n + mx + mz
    x_feat = torch.zeros(num_nodes, 4, dtype=torch.float32)
    avg_llr = (llr_z + llr_x) / 2.0
    x_feat[:n, 0] = avg_llr
    x_feat[:n, 1] = 1.0
    x_feat[n:n+mx, 0] = torch.from_numpy(x_syndrome).float()
    x_feat[n:n+mx, 2] = 1.0
    x_feat[n+mx:, 0] = torch.from_numpy(z_syndrome).float()
    x_feat[n+mx:, 3] = 1.0

    data_obj = Data(
        x=x_feat.to(device),
        edge_index=edge_index_t,
        edge_type=edge_type_t,
        node_type=node_type_t,
        channel_llr=torch.full((n,), avg_llr, dtype=torch.float32).to(device),
    )
    # Attach p_value for FiLM conditioning (if available)
    if p_value is not None:
        data_obj.p_value = torch.tensor(float(p_value), dtype=torch.float32).to(device)

    # Get GNN corrections and apply based on mode
    with torch.no_grad():
        gnn_model.eval()
        gnn_out = gnn_model(data_obj)

    llr_z_t = torch.full((n,), llr_z, dtype=torch.float32, device=device)
    llr_x_t = torch.full((n,), llr_x, dtype=torch.float32, device=device)

    if correction_mode == "both":
        add_corr, mul_corr = gnn_out
        add_corr = torch.clamp(add_corr, -20.0, 20.0)
        mul_corr = torch.clamp(mul_corr, -5.0, 5.0)
        gnn_out = (add_corr, mul_corr)
    else:
        gnn_out = torch.clamp(gnn_out, -20.0, 20.0)

    corrected_llr_z = apply_correction(llr_z_t, gnn_out, correction_mode).cpu().numpy()
    corrected_llr_x = apply_correction(llr_x_t, gnn_out, correction_mode).cpu().numpy()

    # Decode Z-errors using hx with corrected LLRs
    if x_syndrome.sum() > 0:
        dec_z = dec_z_prebuilt if dec_z_prebuilt is not None else \
            MinSumBPDecoder(hx, max_iter=max_iter, alpha=0.8, clamp_llr=20.0).to(device)
        syn_t = torch.from_numpy(x_syndrome[np.newaxis, :]).float().to(device)
        llr_t = torch.from_numpy(corrected_llr_z[np.newaxis, :]).float().to(device)
        _, hard_z, conv_z = dec_z(syn_t, llr_t)
        z_errors = hard_z[0].cpu().numpy()
        z_conv = bool(conv_z[0])
    else:
        z_errors = np.zeros(n, dtype=np.int64)
        z_conv = True

    # Decode X-errors using hz with corrected LLRs
    if z_syndrome.sum() > 0:
        dec_x = dec_x_prebuilt if dec_x_prebuilt is not None else \
            MinSumBPDecoder(hz, max_iter=max_iter, alpha=0.8, clamp_llr=20.0).to(device)
        syn_t = torch.from_numpy(z_syndrome[np.newaxis, :]).float().to(device)
        llr_t = torch.from_numpy(corrected_llr_x[np.newaxis, :]).float().to(device)
        _, hard_x, conv_x = dec_x(syn_t, llr_t)
        x_errors = hard_x[0].cpu().numpy()
        x_conv = bool(conv_x[0])
    else:
        x_errors = np.zeros(n, dtype=np.int64)
        x_conv = True

    return z_errors, x_errors, z_conv, x_conv


def _check_logical_error(
    z_errors: np.ndarray,
    x_errors: np.ndarray,
    lx: np.ndarray,
    lz: np.ndarray,
    observable: np.ndarray,
) -> bool:
    """Check if decoded correction results in a logical error.

    Z-errors can flip X-type logical operators (lx).
    X-errors can flip Z-type logical operators (lz).
    """
    obs_from_z = (lx @ z_errors) % 2
    obs_from_x = (lz @ x_errors) % 2
    predicted_obs = np.concatenate([obs_from_z, obs_from_x])

    n_obs = min(len(predicted_obs), len(observable))
    return bool(np.any(predicted_obs[:n_obs] != observable[:n_obs]))


# ---------------------------------------------------------------------------
# Evaluation Loops
# ---------------------------------------------------------------------------

def _check_logical_errors_batch(
    z_errors: np.ndarray,
    x_errors: np.ndarray,
    lx: np.ndarray,
    lz: np.ndarray,
    observables: np.ndarray,
) -> np.ndarray:
    """Vectorized logical error check for a batch of shots.

    Args:
        z_errors: (B, n) decoded Z-error estimates
        x_errors: (B, n) decoded X-error estimates
        lx: (k_x, n) X-type logical operators
        lz: (k_z, n) Z-type logical operators
        observables: (B, num_obs) actual observable flips

    Returns:
        (B,) bool array — True if logical error occurred
    """
    obs_from_z = (z_errors @ lx.T) % 2  # (B, k_x)
    obs_from_x = (x_errors @ lz.T) % 2  # (B, k_z)
    predicted = np.concatenate([obs_from_z, obs_from_x], axis=1)  # (B, k_x + k_z)
    n_obs = min(predicted.shape[1], observables.shape[1])
    return np.any(predicted[:, :n_obs] != observables[:, :n_obs], axis=1)


def evaluate_code_capacity(
    npz_data: dict,
    meta: dict,
    device: torch.device,
    gnn_model: Optional[nn.Module] = None,
    use_bposd: bool = False,
    use_mwpm: bool = False,
    use_bplsd: bool = False,
    use_belieffind: bool = False,
    correction_mode: str = "additive",
    lsd_order: int = 0,
    lsd_method: str = "LSD_CS",
    osd_order: int = 0,
    checkpoint: Optional[dict] = None,
) -> dict:
    """Evaluate decoders in code-capacity mode (separate CSS decoding).

    Uses batched BP decoding for massive parallelism on CPU/GPU.
    BP-OSD and MWPM use multiprocessing for CPU parallelism.

    Returns:
        Dictionary with results for each decoder.
    """
    syndromes = npz_data["syndromes"].astype(np.float32)
    observables = npz_data["observables"].astype(np.float32)
    hx = npz_data["hx"].astype(np.uint8)
    hz = npz_data["hz"].astype(np.uint8)
    lx = npz_data["lx"].astype(np.uint8)
    lz = npz_data["lz"].astype(np.uint8)

    mx, n = hx.shape
    mz = hz.shape[0]
    shots = syndromes.shape[0]
    num_detectors = syndromes.shape[1]
    total_checks = mx + mz

    print(f"Code: n={n}, mx={mx}, mz={mz}, shots={shots}")
    print(f"Detectors per shot: {num_detectors}, checks: {total_checks}")

    # Compute per-Pauli LLRs from noise parameters
    p = float(meta.get("p", 0.01))
    eta = float(meta.get("eta", 1.0))

    pz = p * eta / (eta + 1)
    px = p / (eta + 1)

    pz_clamped = max(min(pz, 1.0 - 1e-7), 1e-7)
    px_clamped = max(min(px, 1.0 - 1e-7), 1e-7)

    llr_z = float(math.log((1.0 - pz_clamped) / pz_clamped))
    llr_x = float(math.log((1.0 - px_clamped) / px_clamped))

    # Per-shot LLRs for drifting noise (uses p_values when available)
    has_per_shot_p = "p_values" in npz_data
    if has_per_shot_p:
        p_vals = npz_data["p_values"].astype(np.float64)
        p_vals = np.clip(p_vals, 1e-7, 1.0 - 1e-7)
        pz_vals = p_vals * eta / (eta + 1)
        px_vals = p_vals / (eta + 1)
        pz_vals = np.clip(pz_vals, 1e-7, 1.0 - 1e-7)
        px_vals = np.clip(px_vals, 1e-7, 1.0 - 1e-7)
        llr_z_per_shot = np.log((1.0 - pz_vals) / pz_vals).astype(np.float32)  # (shots,)
        llr_x_per_shot = np.log((1.0 - px_vals) / px_vals).astype(np.float32)  # (shots,)
        print(f"Noise: p_base={p}, eta={eta}, per-shot p range=[{p_vals.min():.4f}, {p_vals.max():.4f}]")
        print(f"Per-shot LLR_z range=[{llr_z_per_shot.min():.4f}, {llr_z_per_shot.max():.4f}]")
        print(f"Per-shot LLR_x range=[{llr_x_per_shot.min():.4f}, {llr_x_per_shot.max():.4f}]")
    else:
        llr_z_per_shot = None
        llr_x_per_shot = None
        print(f"Noise: p={p}, eta={eta}, pz={pz:.6f}, px={px:.6f}")
        print(f"LLR_z={llr_z:.4f}, LLR_x={llr_x:.4f}")

    # Stale LLR from base p (what a non-adaptive decoder would use)
    llr_z_stale = llr_z
    llr_x_stale = llr_x

    # Compute number of detector rounds
    if total_checks > 0 and num_detectors >= total_checks:
        num_rounds = num_detectors // total_checks
    else:
        num_rounds = 1

    # ---------- Preprocess ALL syndromes in bulk (vectorized) ----------
    if num_rounds > 1:
        syn_3d = syndromes.reshape(shots, num_rounds, total_checks)
        collapsed = syn_3d.sum(axis=1) % 2
    else:
        collapsed = syndromes[:, :total_checks]

    collapsed = collapsed.astype(np.float32)
    all_x_syn = collapsed[:, :mx]   # (shots, mx)
    all_z_syn = collapsed[:, mx:]   # (shots, mz)

    # ---------- Batched BP decoding (stale LLR — doesn't know about drift) ----------
    print("\nBatched BP decoding (stale LLR)...")
    t_start = time.time()

    dec_z_pre = MinSumBPDecoder(hx, max_iter=100, alpha=0.8, clamp_llr=20.0).to(device)
    dec_x_pre = MinSumBPDecoder(hz, max_iter=100, alpha=0.8, clamp_llr=20.0).to(device)

    # Process in chunks to control memory (e.g. 512 shots per chunk)
    CHUNK = 512
    bp_hard_z_all = []
    bp_hard_x_all = []
    bp_conv_all = []

    for start in range(0, shots, CHUNK):
        end = min(start + CHUNK, shots)
        x_syn_t = torch.from_numpy(all_x_syn[start:end]).float().to(device)
        z_syn_t = torch.from_numpy(all_z_syn[start:end]).float().to(device)
        B_chunk = end - start

        llr_z_t = torch.full((B_chunk, n), llr_z_stale, dtype=torch.float32, device=device)
        llr_x_t = torch.full((B_chunk, n), llr_x_stale, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, hard_z, conv_z = dec_z_pre(x_syn_t, llr_z_t)
            _, hard_x, conv_x = dec_x_pre(z_syn_t, llr_x_t)

        bp_hard_z_all.append(hard_z.cpu().numpy())
        bp_hard_x_all.append(hard_x.cpu().numpy())
        bp_conv_all.append((conv_z & conv_x).cpu().numpy())

        if end % max(1, shots // 5) < CHUNK:
            elapsed = time.time() - t_start
            print(f"  BP: {end}/{shots} shots ({end / elapsed:.0f} shots/s)")

    bp_z_errors = np.concatenate(bp_hard_z_all, axis=0)  # (shots, n)
    bp_x_errors = np.concatenate(bp_hard_x_all, axis=0)  # (shots, n)
    bp_conv = np.concatenate(bp_conv_all, axis=0)         # (shots,)

    bp_logical = _check_logical_errors_batch(bp_z_errors, bp_x_errors, lx, lz, observables)
    bp_errors = int(bp_logical.sum())
    bp_converged = int(bp_conv.sum())

    bp_elapsed = time.time() - t_start
    print(f"  BP done: {shots / bp_elapsed:.0f} shots/s")

    # ---------- Oracle BP (true per-shot LLR — upper bound) ----------
    oracle_bp_errors = 0
    oracle_bp_converged = 0
    if has_per_shot_p:
        print("\nBatched Oracle BP decoding (true per-shot LLR)...")
        t_oracle = time.time()

        oracle_hard_z_all = []
        oracle_hard_x_all = []
        oracle_conv_all = []

        for start in range(0, shots, CHUNK):
            end = min(start + CHUNK, shots)
            x_syn_t = torch.from_numpy(all_x_syn[start:end]).float().to(device)
            z_syn_t = torch.from_numpy(all_z_syn[start:end]).float().to(device)
            B_chunk = end - start

            # Per-shot LLRs: broadcast (B,) -> (B, n)
            llr_z_t = torch.from_numpy(
                np.broadcast_to(llr_z_per_shot[start:end, None], (B_chunk, n)).copy()
            ).float().to(device)
            llr_x_t = torch.from_numpy(
                np.broadcast_to(llr_x_per_shot[start:end, None], (B_chunk, n)).copy()
            ).float().to(device)

            with torch.no_grad():
                _, hard_z, conv_z = dec_z_pre(x_syn_t, llr_z_t)
                _, hard_x, conv_x = dec_x_pre(z_syn_t, llr_x_t)

            oracle_hard_z_all.append(hard_z.cpu().numpy())
            oracle_hard_x_all.append(hard_x.cpu().numpy())
            oracle_conv_all.append((conv_z & conv_x).cpu().numpy())

            if end % max(1, shots // 5) < CHUNK:
                elapsed = time.time() - t_oracle
                print(f"  Oracle BP: {end}/{shots} shots ({end / elapsed:.0f} shots/s)")

        oracle_z_errors = np.concatenate(oracle_hard_z_all, axis=0)
        oracle_x_errors = np.concatenate(oracle_hard_x_all, axis=0)
        oracle_conv = np.concatenate(oracle_conv_all, axis=0)

        oracle_logical = _check_logical_errors_batch(oracle_z_errors, oracle_x_errors, lx, lz, observables)
        oracle_bp_errors = int(oracle_logical.sum())
        oracle_bp_converged = int(oracle_conv.sum())
        print(f"  Oracle BP done: {shots / (time.time() - t_oracle):.0f} shots/s")

    # ---------- Batched GNN-BP decoding ----------
    gnn_bp_errors = 0
    gnn_bp_converged = 0
    interleaved_bp_errors = 0
    interleaved_bp_converged = 0
    has_neural_bp = checkpoint is not None and checkpoint.get("neural_bp", False)
    has_interleaved = has_neural_bp  # interleaved requires neural BP weights

    if gnn_model is not None:
        print("\nBatched GNN-BP decoding (stale LLR + GNN correction)...")
        t_gnn = time.time()

        # Build Neural BP decoders if checkpoint has trained weights
        ckpt_bp_iters = checkpoint.get("bp_iters", 10) if checkpoint else 10
        # Use 100 total iters: first ckpt_bp_iters use learned weights, rest use vanilla BP
        eval_bp_iters = 100
        if has_neural_bp:
            print(f"  Loading Neural BP weights (trained={ckpt_bp_iters} iters, eval={eval_bp_iters} iters)...")
            gnn_dec_z = MinSumBPDecoder(
                hx, max_iter=eval_bp_iters, alpha=0.8, clamp_llr=20.0, neural_bp=True
            ).to(device)
            gnn_dec_x = MinSumBPDecoder(
                hz, max_iter=eval_bp_iters, alpha=0.8, clamp_llr=20.0, neural_bp=True
            ).to(device)
            # Load trained weights (covers first ckpt_bp_iters iterations)
            # Remaining iterations automatically fall back to vanilla BP
            gnn_dec_z.load_state_dict(checkpoint["dec_z_state_dict"], strict=False)
            gnn_dec_x.load_state_dict(checkpoint["dec_x_state_dict"], strict=False)
            gnn_dec_z.eval()
            gnn_dec_x.eval()
            print(f"  Neural BP: learned weights for iters 1-{ckpt_bp_iters}, vanilla BP for iters {ckpt_bp_iters+1}-{eval_bp_iters}")
        else:
            gnn_dec_z = dec_z_pre
            gnn_dec_x = dec_x_pre

        # Batched GNN evaluation: build Tanner graph once, batch all shots
        node_type_np, edge_index_np, edge_type_np = build_tanner_graph(hx, hz)
        node_type_t = torch.from_numpy(node_type_np).long().to(device)
        edge_index_t = torch.from_numpy(edge_index_np).long().to(device)
        edge_type_t = torch.from_numpy(edge_type_np).long().to(device)
        num_nodes = n + mx + mz

        gnn_hard_z_all = []
        gnn_hard_x_all = []
        gnn_conv_all = []

        avg_llr = (llr_z_stale + llr_x_stale) / 2.0

        # Pre-compute p_value array for vectorized batch construction
        if has_per_shot_p:
            p_vals_arr = p_vals.astype(np.float32)
        else:
            p_vals_arr = np.full(shots, float(p), dtype=np.float32)

        for start in range(0, shots, CHUNK):
            end = min(start + CHUNK, shots)
            B_chunk = end - start

            # Vectorized batch construction (no per-shot Python loop)
            batch_data = _build_eval_batch(
                all_x_syn[start:end], all_z_syn[start:end],
                avg_llr, n, mx, mz,
                edge_index_t, edge_type_t, node_type_t,
                p_vals_arr[start:end], device,
            )

            with torch.no_grad():
                gnn_model.eval()
                gnn_out = gnn_model(batch_data)

            # Reshape GNN output from flat to (B, n) and apply corrections
            if correction_mode == "both":
                add_c, mul_c = gnn_out
                add_c = torch.clamp(add_c, -20.0, 20.0).view(B_chunk, n)
                mul_c = torch.clamp(mul_c, -5.0, 5.0).view(B_chunk, n)
                gnn_out_batched = (add_c, mul_c)
            else:
                gnn_out_batched = torch.clamp(gnn_out, -20.0, 20.0).view(B_chunk, n)

            llr_z_batch = torch.full((B_chunk, n), llr_z_stale, dtype=torch.float32, device=device)
            llr_x_batch = torch.full((B_chunk, n), llr_x_stale, dtype=torch.float32, device=device)
            corrections_z_t = apply_correction(llr_z_batch, gnn_out_batched, correction_mode)
            corrections_x_t = apply_correction(llr_x_batch, gnn_out_batched, correction_mode)

            # Batched BP with corrected LLRs (using Neural BP decoders if available)
            x_syn_t = torch.from_numpy(all_x_syn[start:end]).float().to(device)
            z_syn_t = torch.from_numpy(all_z_syn[start:end]).float().to(device)

            with torch.no_grad():
                _, hard_z, conv_z = gnn_dec_z(x_syn_t, corrections_z_t)
                _, hard_x, conv_x = gnn_dec_x(z_syn_t, corrections_x_t)

            gnn_hard_z_all.append(hard_z.cpu().numpy())
            gnn_hard_x_all.append(hard_x.cpu().numpy())
            gnn_conv_all.append((conv_z & conv_x).cpu().numpy())

            if end % max(1, shots // 5) < CHUNK:
                elapsed = time.time() - t_gnn
                print(f"  GNN-BP: {end}/{shots} shots ({end / elapsed:.0f} shots/s)")

        gnn_z_errors = np.concatenate(gnn_hard_z_all, axis=0)
        gnn_x_errors = np.concatenate(gnn_hard_x_all, axis=0)
        gnn_conv = np.concatenate(gnn_conv_all, axis=0)

        gnn_logical = _check_logical_errors_batch(gnn_z_errors, gnn_x_errors, lx, lz, observables)
        gnn_bp_errors = int(gnn_logical.sum())
        gnn_bp_converged = int(gnn_conv.sum())
        print(f"  GNN-BP done: {shots / (time.time() - t_gnn):.0f} shots/s")

    # ---------- Interleaved GNN-BP decoding ----------
    if gnn_model is not None and has_interleaved:
        print("\nInterleaved GNN-BP decoding (GNN corrects mid-BP)...")
        t_inter = time.time()

        # Build Neural BP decoders for interleaved mode with extended iterations
        # Stage 1: neural BP iters, then GNN correction, then Stage 2: vanilla BP iters
        stage1_iters = ckpt_bp_iters  # learned weights
        stage2_iters = 90             # vanilla BP continuation
        num_stages = 2
        total_inter_iters = stage1_iters + stage2_iters
        print(f"  Stage 1: {stage1_iters} neural iters, Stage 2: {stage2_iters} vanilla iters = {total_inter_iters} total")

        inter_dec_z = MinSumBPDecoder(
            hx, max_iter=total_inter_iters, alpha=0.8, clamp_llr=20.0, neural_bp=True
        ).to(device)
        inter_dec_x = MinSumBPDecoder(
            hz, max_iter=total_inter_iters, alpha=0.8, clamp_llr=20.0, neural_bp=True
        ).to(device)
        inter_dec_z.load_state_dict(checkpoint["dec_z_state_dict"], strict=False)
        inter_dec_x.load_state_dict(checkpoint["dec_x_state_dict"], strict=False)
        inter_dec_z.eval()
        inter_dec_x.eval()

        inter_hard_z_all = []
        inter_hard_x_all = []
        inter_conv_all = []

        for start in range(0, shots, CHUNK):
            end = min(start + CHUNK, shots)
            B_chunk = end - start

            x_syn_t = torch.from_numpy(all_x_syn[start:end]).float().to(device)
            z_syn_t = torch.from_numpy(all_z_syn[start:end]).float().to(device)
            llr_z_t = torch.full((B_chunk, n), llr_z_stale, dtype=torch.float32, device=device)
            llr_x_t = torch.full((B_chunk, n), llr_x_stale, dtype=torch.float32, device=device)

            # Correction function: runs GNN on each sample's current BP state
            def _make_correction_fn(syn_x_chunk, syn_z_chunk, shot_offset):
                def correction_fn(mid_marginals, current_llr, stage_idx):
                    """GNN sees current BP marginals and corrects LLRs."""
                    B_c = mid_marginals.shape[0]
                    corrected = current_llr.clone()
                    for i in range(B_c):
                        si = shot_offset + i
                        # Build features with BP marginal as 5th feature
                        x_feat = torch.zeros(num_nodes, 4, dtype=torch.float32)
                        avg_llr = (llr_z_stale + llr_x_stale) / 2.0
                        x_feat[:n, 0] = avg_llr
                        x_feat[:n, 1] = 1.0
                        x_feat[n:n+mx, 0] = syn_x_chunk[i]
                        x_feat[n:n+mx, 2] = 1.0
                        x_feat[n+mx:, 0] = syn_z_chunk[i]
                        x_feat[n+mx:, 3] = 1.0

                        data_obj = Data(
                            x=x_feat.to(device),
                            edge_index=edge_index_t,
                            edge_type=edge_type_t,
                            node_type=node_type_t,
                            channel_llr=torch.full((n,), avg_llr, dtype=torch.float32, device=device),
                        )
                        if has_per_shot_p:
                            data_obj.p_value = torch.tensor(float(p_vals[si]), dtype=torch.float32).to(device)
                        else:
                            data_obj.p_value = torch.tensor(float(p), dtype=torch.float32).to(device)

                        with torch.no_grad():
                            gnn_out = gnn_model(data_obj)

                        if correction_mode == "both":
                            add_c, mul_c = gnn_out
                            add_c = torch.clamp(add_c, -20.0, 20.0)
                            mul_c = torch.clamp(mul_c, -5.0, 5.0)
                            gnn_out_batched = (add_c.unsqueeze(0), mul_c.unsqueeze(0))
                        else:
                            gnn_out_batched = torch.clamp(gnn_out, -20.0, 20.0).unsqueeze(0)

                        corrected[i] = apply_correction(current_llr[i:i+1], gnn_out_batched, correction_mode).squeeze(0)
                    return corrected
                return correction_fn

            with torch.no_grad():
                # Z-errors: hx @ z_error = x_syndrome
                # Stage 1: neural BP iters, then GNN correction, then Stage 2: vanilla BP
                _, hard_z, conv_z = inter_dec_z.forward_stages(
                    x_syn_t, llr_z_t,
                    stage_iters=[stage1_iters, stage2_iters],
                    correction_fn=_make_correction_fn(x_syn_t, z_syn_t, start),
                )
                # X-errors: hz @ x_error = z_syndrome
                _, hard_x, conv_x = inter_dec_x.forward_stages(
                    z_syn_t, llr_x_t,
                    stage_iters=[stage1_iters, stage2_iters],
                    correction_fn=_make_correction_fn(x_syn_t, z_syn_t, start),
                )

            inter_hard_z_all.append(hard_z.cpu().numpy())
            inter_hard_x_all.append(hard_x.cpu().numpy())
            inter_conv_all.append((conv_z & conv_x).cpu().numpy())

            if end % max(1, shots // 5) < CHUNK:
                elapsed = time.time() - t_inter
                print(f"  Interleaved: {end}/{shots} shots ({end / elapsed:.0f} shots/s)")

        inter_z_errors = np.concatenate(inter_hard_z_all, axis=0)
        inter_x_errors = np.concatenate(inter_hard_x_all, axis=0)
        inter_conv = np.concatenate(inter_conv_all, axis=0)

        inter_logical = _check_logical_errors_batch(inter_z_errors, inter_x_errors, lx, lz, observables)
        interleaved_bp_errors = int(inter_logical.sum())
        interleaved_bp_converged = int(inter_conv.sum())
        print(f"  Interleaved done: {shots / (time.time() - t_inter):.0f} shots/s")

    # ---------- BP-OSD (parallel with threads) ----------
    # Use ThreadPoolExecutor: BP-OSD and MWPM are C-extension-backed
    # (ldpc, pymatching) and release the GIL, so threads give true
    # parallelism. ProcessPoolExecutor fails on Windows (can't pickle
    # local functions with spawn).
    bposd_errors = 0
    bposd_available = False
    if use_bposd:
        try:
            from gnn_pipeline.bposd_decoder import run_css_bposd_decoder
            bposd_available = True
        except ImportError as e:
            print(f"Warning: BP-OSD not available: {e}")

    if bposd_available:
        import concurrent.futures
        import os

        print("BP-OSD decoding (parallel)...")
        t_bposd = time.time()
        n_workers = max(1, os.cpu_count() - 1)

        def _bposd_shot(idx):
            z_e, x_e = run_css_bposd_decoder(
                all_x_syn[idx], all_z_syn[idx], hx, hz,
                error_rate_z=pz, error_rate_x=px,
            )
            return _check_logical_error(z_e, x_e, lx, lz, observables[idx])

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            results_bposd = list(pool.map(_bposd_shot, range(shots)))

        bposd_errors = sum(results_bposd)
        print(f"  BP-OSD done: {shots / (time.time() - t_bposd):.0f} shots/s ({n_workers} workers)")

    # ---------- MWPM (parallel with threads) ----------
    mwpm_errors = 0
    mwpm_available = False
    if use_mwpm:
        try:
            from gnn_pipeline.matching_decoder import build_mwpm_css, run_mwpm_css
            matcher_z, matcher_x, emap_z, emap_x = build_mwpm_css(hx, hz, pz, px)
            mwpm_available = True
        except ImportError as e:
            print(f"Warning: MWPM not available: {e}")

    if mwpm_available:
        import concurrent.futures
        import os

        print("MWPM decoding (parallel)...")
        t_mwpm = time.time()
        n_workers = max(1, os.cpu_count() - 1)

        def _mwpm_shot(idx):
            z_e, x_e = run_mwpm_css(
                all_x_syn[idx], all_z_syn[idx], matcher_z, matcher_x, n,
                edge_map_z=emap_z, edge_map_x=emap_x,
            )
            return _check_logical_error(z_e, x_e, lx, lz, observables[idx])

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            results_mwpm = list(pool.map(_mwpm_shot, range(shots)))

        mwpm_errors = sum(results_mwpm)
        print(f"  MWPM done: {shots / (time.time() - t_mwpm):.0f} shots/s ({n_workers} workers)")

    # ---------- BP-LSD (parallel with threads) ----------
    bplsd_errors = 0
    bplsd_available = False
    if use_bplsd:
        try:
            from gnn_pipeline.bplsd_decoder import run_css_bplsd_decoder
            bplsd_available = True
        except ImportError as e:
            print(f"Warning: BP-LSD not available: {e}")

    if bplsd_available:
        import concurrent.futures
        import os

        print(f"BP-LSD decoding (parallel, method={lsd_method}, order={lsd_order})...")
        t_bplsd = time.time()
        n_workers = max(1, os.cpu_count() - 1)

        def _bplsd_shot(idx):
            z_e, x_e = run_css_bplsd_decoder(
                all_x_syn[idx], all_z_syn[idx], hx, hz,
                error_rate_z=pz, error_rate_x=px,
                lsd_order=lsd_order, lsd_method=lsd_method,
            )
            return _check_logical_error(z_e, x_e, lx, lz, observables[idx])

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            results_bplsd = list(pool.map(_bplsd_shot, range(shots)))

        bplsd_errors = sum(results_bplsd)
        print(f"  BP-LSD done: {shots / (time.time() - t_bplsd):.0f} shots/s ({n_workers} workers)")

    # ---------- BeliefFind (parallel with threads) ----------
    bf_errors = 0
    bf_available = False
    if use_belieffind:
        try:
            from gnn_pipeline.bplsd_decoder import run_css_belief_find_decoder
            bf_available = True
        except ImportError as e:
            print(f"Warning: BeliefFind not available: {e}")

    if bf_available:
        import concurrent.futures
        import os

        print("BeliefFind decoding (parallel)...")
        t_bf = time.time()
        n_workers = max(1, os.cpu_count() - 1)

        def _bf_shot(idx):
            z_e, x_e = run_css_belief_find_decoder(
                all_x_syn[idx], all_z_syn[idx], hx, hz,
                error_rate_z=pz, error_rate_x=px,
            )
            return _check_logical_error(z_e, x_e, lx, lz, observables[idx])

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            results_bf = list(pool.map(_bf_shot, range(shots)))

        bf_errors = sum(results_bf)
        print(f"  BeliefFind done: {shots / (time.time() - t_bf):.0f} shots/s ({n_workers} workers)")

    # ---------- GNN + BP-LSD (the novel combination) ----------
    gnn_bplsd_errors = 0
    gnn_bplsd_available = False
    if gnn_model is not None and use_bplsd:
        try:
            from gnn_pipeline.bplsd_decoder import run_css_bplsd_with_llr
            gnn_bplsd_available = True
        except ImportError:
            pass

    if gnn_bplsd_available:
        import concurrent.futures
        import os

        print("GNN + BP-LSD decoding (GNN corrections -> BP-LSD post-process)...")
        t_gnn_lsd = time.time()
        n_workers = max(1, os.cpu_count() - 1)

        # Batched GNN corrections (reuse cached Tanner graph from GNN-BP above)
        all_corr_llr_z = np.zeros((shots, n), dtype=np.float32)
        all_corr_llr_x = np.zeros((shots, n), dtype=np.float32)
        avg_llr_lsd = (llr_z_stale + llr_x_stale) / 2.0

        for start_lsd in range(0, shots, CHUNK):
            end_lsd = min(start_lsd + CHUNK, shots)
            B_lsd = end_lsd - start_lsd

            # Vectorized batch construction
            batch_lsd = _build_eval_batch(
                all_x_syn[start_lsd:end_lsd], all_z_syn[start_lsd:end_lsd],
                avg_llr_lsd, n, mx, mz,
                edge_index_t, edge_type_t, node_type_t,
                p_vals_arr[start_lsd:end_lsd], device,
            )

            with torch.no_grad():
                gnn_model.eval()
                gnn_out_lsd = gnn_model(batch_lsd)

            if correction_mode == "both":
                add_c, mul_c = gnn_out_lsd
                add_c = torch.clamp(add_c, -20.0, 20.0).view(B_lsd, n)
                mul_c = torch.clamp(mul_c, -5.0, 5.0).view(B_lsd, n)
                gnn_out_lsd = (add_c, mul_c)
            else:
                gnn_out_lsd = torch.clamp(gnn_out_lsd, -20.0, 20.0).view(B_lsd, n)

            llr_z_b = torch.full((B_lsd, n), llr_z_stale, dtype=torch.float32, device=device)
            llr_x_b = torch.full((B_lsd, n), llr_x_stale, dtype=torch.float32, device=device)
            all_corr_llr_z[start_lsd:end_lsd] = apply_correction(llr_z_b, gnn_out_lsd, correction_mode).cpu().numpy()
            all_corr_llr_x[start_lsd:end_lsd] = apply_correction(llr_x_b, gnn_out_lsd, correction_mode).cpu().numpy()

            if end_lsd % max(1, shots // 5) < CHUNK:
                elapsed_gnn = time.time() - t_gnn_lsd
                print(f"  GNN corrections: {end_lsd}/{shots} ({end_lsd/elapsed_gnn:.0f} shots/s)")

        # Now run BP-LSD with per-qubit corrected LLRs (parallel)
        print("  Running BP-LSD with GNN-corrected LLRs...")
        t_lsd_part = time.time()

        def _gnn_bplsd_shot(idx):
            z_e, x_e = run_css_bplsd_with_llr(
                all_x_syn[idx], all_z_syn[idx], hx, hz,
                per_qubit_llr_z=all_corr_llr_z[idx],
                per_qubit_llr_x=all_corr_llr_x[idx],
                lsd_order=lsd_order, lsd_method=lsd_method,
            )
            return _check_logical_error(z_e, x_e, lx, lz, observables[idx])

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            results_gnn_bplsd = list(pool.map(_gnn_bplsd_shot, range(shots)))

        gnn_bplsd_errors = sum(results_gnn_bplsd)
        print(f"  GNN+BP-LSD done: {shots / (time.time() - t_gnn_lsd):.0f} shots/s total")

    # ---------- GNN + BP-OSD (novel combination) ----------
    gnn_bposd_errors = 0
    gnn_bposd_available = False
    if gnn_model is not None and use_bposd:
        try:
            from gnn_pipeline.bposd_decoder import run_css_bposd_with_llr
            gnn_bposd_available = True
        except ImportError:
            pass

    if gnn_bposd_available:
        import concurrent.futures
        import os

        print("GNN + BP-OSD decoding (GNN corrections -> BP-OSD post-process)...")
        t_gnn_osd = time.time()
        n_workers = max(1, os.cpu_count() - 1)

        # Batched GNN corrections (reuse cached Tanner graph)
        all_corr_llr_z_osd = np.zeros((shots, n), dtype=np.float32)
        all_corr_llr_x_osd = np.zeros((shots, n), dtype=np.float32)
        avg_llr_osd = (llr_z_stale + llr_x_stale) / 2.0

        for start_osd in range(0, shots, CHUNK):
            end_osd = min(start_osd + CHUNK, shots)
            B_osd = end_osd - start_osd

            # Vectorized batch construction
            batch_osd = _build_eval_batch(
                all_x_syn[start_osd:end_osd], all_z_syn[start_osd:end_osd],
                avg_llr_osd, n, mx, mz,
                edge_index_t, edge_type_t, node_type_t,
                p_vals_arr[start_osd:end_osd], device,
            )

            with torch.no_grad():
                gnn_model.eval()
                gnn_out_osd = gnn_model(batch_osd)

            if correction_mode == "both":
                add_c, mul_c = gnn_out_osd
                add_c = torch.clamp(add_c, -20.0, 20.0).view(B_osd, n)
                mul_c = torch.clamp(mul_c, -5.0, 5.0).view(B_osd, n)
                gnn_out_osd = (add_c, mul_c)
            else:
                gnn_out_osd = torch.clamp(gnn_out_osd, -20.0, 20.0).view(B_osd, n)

            llr_z_b = torch.full((B_osd, n), llr_z_stale, dtype=torch.float32, device=device)
            llr_x_b = torch.full((B_osd, n), llr_x_stale, dtype=torch.float32, device=device)
            all_corr_llr_z_osd[start_osd:end_osd] = apply_correction(llr_z_b, gnn_out_osd, correction_mode).cpu().numpy()
            all_corr_llr_x_osd[start_osd:end_osd] = apply_correction(llr_x_b, gnn_out_osd, correction_mode).cpu().numpy()

        # Now run BP-OSD with per-qubit corrected LLRs (parallel)
        print("  Running BP-OSD with GNN-corrected LLRs...")

        def _gnn_bposd_shot(idx):
            z_e, x_e = run_css_bposd_with_llr(
                all_x_syn[idx], all_z_syn[idx], hx, hz,
                per_qubit_llr_z=all_corr_llr_z_osd[idx],
                per_qubit_llr_x=all_corr_llr_x_osd[idx],
                osd_order=osd_order,
            )
            return _check_logical_error(z_e, x_e, lx, lz, observables[idx])

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            results_gnn_bposd = list(pool.map(_gnn_bposd_shot, range(shots)))

        gnn_bposd_errors = sum(results_gnn_bposd)
        print(f"  GNN+BP-OSD done: {shots / (time.time() - t_gnn_osd):.0f} shots/s total")

    elapsed = time.time() - t_start

    # ---------- Report results ----------
    bp_ler = bp_errors / shots if shots > 0 else 0.0
    bp_ler_low, bp_ler_high = wilson_score_interval_binom(bp_errors, shots)

    print(f"\n=== Results ({elapsed:.1f}s) ===")
    print(f"BP Decoder (stale LLR, separate CSS, batched):")
    print(f"  Logical Error Rate: {bp_ler:.6f}")
    print(f"  95% CI: [{bp_ler_low:.6f}, {bp_ler_high:.6f}]")
    print(f"  Errors: {bp_errors}/{shots}")
    print(f"  Convergence: {bp_converged}/{shots} ({100*bp_converged/shots:.1f}%)")

    results = {
        "mode": "code_capacity",
        "code": {"n": int(n), "mx": int(mx), "mz": int(mz)},
        "noise": {"p": p, "eta": eta, "pz": pz, "px": px},
        "has_per_shot_p": has_per_shot_p,
        "test_shots": shots,
        "elapsed_seconds": round(elapsed, 1),
        "bp": {
            "errors": int(bp_errors),
            "ler": float(bp_ler),
            "ler_ci_low": float(bp_ler_low),
            "ler_ci_high": float(bp_ler_high),
            "convergence_rate": float(bp_converged / shots),
        },
    }

    if has_per_shot_p:
        oracle_ler = oracle_bp_errors / shots if shots > 0 else 0.0
        oracle_ler_low, oracle_ler_high = wilson_score_interval_binom(oracle_bp_errors, shots)

        print(f"\nOracle BP (true per-shot LLR, upper bound):")
        print(f"  Logical Error Rate: {oracle_ler:.6f}")
        print(f"  95% CI: [{oracle_ler_low:.6f}, {oracle_ler_high:.6f}]")
        print(f"  Errors: {oracle_bp_errors}/{shots}")
        print(f"  Convergence: {oracle_bp_converged}/{shots} ({100*oracle_bp_converged/shots:.1f}%)")

        oracle_improv = (bp_ler - oracle_ler) / bp_ler if bp_ler > 0 else 0.0
        print(f"  Improvement over stale BP: {100*oracle_improv:.1f}%")

        results["oracle_bp"] = {
            "errors": int(oracle_bp_errors),
            "ler": float(oracle_ler),
            "ler_ci_low": float(oracle_ler_low),
            "ler_ci_high": float(oracle_ler_high),
            "convergence_rate": float(oracle_bp_converged / shots),
        }

    if gnn_model is not None:
        gnn_bp_ler = gnn_bp_errors / shots if shots > 0 else 0.0
        gnn_bp_ler_low, gnn_bp_ler_high = wilson_score_interval_binom(gnn_bp_errors, shots)

        print(f"\nGNN-Assisted BP (separate CSS, batched):")
        print(f"  Logical Error Rate: {gnn_bp_ler:.6f}")
        print(f"  95% CI: [{gnn_bp_ler_low:.6f}, {gnn_bp_ler_high:.6f}]")
        print(f"  Errors: {gnn_bp_errors}/{shots}")
        print(f"  Convergence: {gnn_bp_converged}/{shots} ({100*gnn_bp_converged/shots:.1f}%)")

        improvement = (bp_ler - gnn_bp_ler) / bp_ler if bp_ler > 0 else 0.0
        print(f"  Improvement over stale BP: {100*improvement:.1f}%")

        results["gnn_bp"] = {
            "errors": int(gnn_bp_errors),
            "ler": float(gnn_bp_ler),
            "ler_ci_low": float(gnn_bp_ler_low),
            "ler_ci_high": float(gnn_bp_ler_high),
            "convergence_rate": float(gnn_bp_converged / shots),
        }
        results["improvement_pct"] = float(100 * improvement)

    if gnn_model is not None and has_interleaved:
        inter_ler = interleaved_bp_errors / shots if shots > 0 else 0.0
        inter_ler_low, inter_ler_high = wilson_score_interval_binom(interleaved_bp_errors, shots)

        print(f"\nInterleaved GNN-BP (GNN corrects mid-BP, Neural BP weights):")
        print(f"  Logical Error Rate: {inter_ler:.6f}")
        print(f"  95% CI: [{inter_ler_low:.6f}, {inter_ler_high:.6f}]")
        print(f"  Errors: {interleaved_bp_errors}/{shots}")
        print(f"  Convergence: {interleaved_bp_converged}/{shots} ({100*interleaved_bp_converged/shots:.1f}%)")

        inter_improvement = (bp_ler - inter_ler) / bp_ler if bp_ler > 0 else 0.0
        print(f"  Improvement over stale BP: {100*inter_improvement:.1f}%")

        results["interleaved_gnn_bp"] = {
            "errors": int(interleaved_bp_errors),
            "ler": float(inter_ler),
            "ler_ci_low": float(inter_ler_low),
            "ler_ci_high": float(inter_ler_high),
            "convergence_rate": float(interleaved_bp_converged / shots),
        }

    if bposd_available:
        bposd_ler = bposd_errors / shots if shots > 0 else 0.0
        bposd_ler_low, bposd_ler_high = wilson_score_interval_binom(bposd_errors, shots)

        print(f"\nBP-OSD Decoder (separate CSS, parallel):")
        print(f"  Logical Error Rate: {bposd_ler:.6f}")
        print(f"  95% CI: [{bposd_ler_low:.6f}, {bposd_ler_high:.6f}]")
        print(f"  Errors: {bposd_errors}/{shots}")

        bp_vs_bposd = (bp_ler - bposd_ler) / bp_ler if bp_ler > 0 else 0.0
        print(f"  Improvement over BP: {100*bp_vs_bposd:.1f}%")

        results["bposd"] = {
            "errors": int(bposd_errors),
            "ler": float(bposd_ler),
            "ler_ci_low": float(bposd_ler_low),
            "ler_ci_high": float(bposd_ler_high),
        }

    if mwpm_available:
        mwpm_ler = mwpm_errors / shots if shots > 0 else 0.0
        mwpm_ler_low, mwpm_ler_high = wilson_score_interval_binom(mwpm_errors, shots)

        print(f"\nMWPM Decoder (separate CSS, parallel, approximate for LDPC):")
        print(f"  Logical Error Rate: {mwpm_ler:.6f}")
        print(f"  95% CI: [{mwpm_ler_low:.6f}, {mwpm_ler_high:.6f}]")
        print(f"  Errors: {mwpm_errors}/{shots}")

        bp_vs_mwpm = (bp_ler - mwpm_ler) / bp_ler if bp_ler > 0 else 0.0
        print(f"  Improvement over BP: {100*bp_vs_mwpm:.1f}%")

        results["mwpm"] = {
            "errors": int(mwpm_errors),
            "ler": float(mwpm_ler),
            "ler_ci_low": float(mwpm_ler_low),
            "ler_ci_high": float(mwpm_ler_high),
        }

    if bplsd_available:
        bplsd_ler = bplsd_errors / shots if shots > 0 else 0.0
        bplsd_ler_low, bplsd_ler_high = wilson_score_interval_binom(bplsd_errors, shots)

        print(f"\nBP-LSD Decoder ({lsd_method}, order={lsd_order}):")
        print(f"  Logical Error Rate: {bplsd_ler:.6f}")
        print(f"  95% CI: [{bplsd_ler_low:.6f}, {bplsd_ler_high:.6f}]")
        print(f"  Errors: {bplsd_errors}/{shots}")

        bp_vs_bplsd = (bp_ler - bplsd_ler) / bp_ler if bp_ler > 0 else 0.0
        print(f"  Improvement over BP: {100*bp_vs_bplsd:.1f}%")

        results["bplsd"] = {
            "errors": int(bplsd_errors),
            "ler": float(bplsd_ler),
            "ler_ci_low": float(bplsd_ler_low),
            "ler_ci_high": float(bplsd_ler_high),
        }

    if bf_available:
        bf_ler = bf_errors / shots if shots > 0 else 0.0
        bf_ler_low, bf_ler_high = wilson_score_interval_binom(bf_errors, shots)

        print(f"\nBeliefFind Decoder (BP + Union Find):")
        print(f"  Logical Error Rate: {bf_ler:.6f}")
        print(f"  95% CI: [{bf_ler_low:.6f}, {bf_ler_high:.6f}]")
        print(f"  Errors: {bf_errors}/{shots}")

        bp_vs_bf = (bp_ler - bf_ler) / bp_ler if bp_ler > 0 else 0.0
        print(f"  Improvement over BP: {100*bp_vs_bf:.1f}%")

        results["belieffind"] = {
            "errors": int(bf_errors),
            "ler": float(bf_ler),
            "ler_ci_low": float(bf_ler_low),
            "ler_ci_high": float(bf_ler_high),
        }

    if gnn_bplsd_available:
        gnn_bplsd_ler = gnn_bplsd_errors / shots if shots > 0 else 0.0
        gnn_bplsd_ler_low, gnn_bplsd_ler_high = wilson_score_interval_binom(gnn_bplsd_errors, shots)

        print(f"\nGNN + BP-LSD Decoder (GNN-corrected LLRs -> BP-LSD):")
        print(f"  Logical Error Rate: {gnn_bplsd_ler:.6f}")
        print(f"  95% CI: [{gnn_bplsd_ler_low:.6f}, {gnn_bplsd_ler_high:.6f}]")
        print(f"  Errors: {gnn_bplsd_errors}/{shots}")

        bp_vs_gnn_bplsd = (bp_ler - gnn_bplsd_ler) / bp_ler if bp_ler > 0 else 0.0
        print(f"  Improvement over BP: {100*bp_vs_gnn_bplsd:.1f}%")

        if bplsd_available:
            lsd_vs_gnn_lsd = (bplsd_ler - gnn_bplsd_ler) / bplsd_ler if bplsd_ler > 0 else 0.0
            print(f"  Improvement over plain BP-LSD: {100*lsd_vs_gnn_lsd:.1f}%")

        results["gnn_bplsd"] = {
            "errors": int(gnn_bplsd_errors),
            "ler": float(gnn_bplsd_ler),
            "ler_ci_low": float(gnn_bplsd_ler_low),
            "ler_ci_high": float(gnn_bplsd_ler_high),
        }

    if gnn_bposd_available:
        gnn_bposd_ler = gnn_bposd_errors / shots if shots > 0 else 0.0
        gnn_bposd_ler_low, gnn_bposd_ler_high = wilson_score_interval_binom(gnn_bposd_errors, shots)

        print(f"\nGNN + BP-OSD Decoder (GNN-corrected LLRs -> BP-OSD):")
        print(f"  Logical Error Rate: {gnn_bposd_ler:.6f}")
        print(f"  95% CI: [{gnn_bposd_ler_low:.6f}, {gnn_bposd_ler_high:.6f}]")
        print(f"  Errors: {gnn_bposd_errors}/{shots}")

        bp_vs_gnn_bposd = (bp_ler - gnn_bposd_ler) / bp_ler if bp_ler > 0 else 0.0
        print(f"  Improvement over BP: {100*bp_vs_gnn_bposd:.1f}%")

        if bposd_available:
            osd_vs_gnn_osd = (bposd_ler - gnn_bposd_ler) / bposd_ler if bposd_ler > 0 else 0.0
            print(f"  Improvement over plain BP-OSD: {100*osd_vs_gnn_osd:.1f}%")

        results["gnn_bposd"] = {
            "errors": int(gnn_bposd_errors),
            "ler": float(gnn_bposd_ler),
            "ler_ci_low": float(gnn_bposd_ler_low),
            "ler_ci_high": float(gnn_bposd_ler_high),
        }

    # ---------- Collect per-shot outcomes for statistical testing ----------
    decoder_outcomes = {"BP (stale LLR)": bp_logical}
    if has_per_shot_p:
        decoder_outcomes["Oracle BP"] = oracle_logical
    if gnn_model is not None:
        decoder_outcomes["GNN-BP"] = gnn_logical
    if gnn_model is not None and has_interleaved:
        decoder_outcomes["Interleaved GNN-BP"] = inter_logical
    if bposd_available:
        decoder_outcomes["BP-OSD"] = np.array(results_bposd, dtype=bool)
    if bplsd_available:
        decoder_outcomes["BP-LSD"] = np.array(results_bplsd, dtype=bool)
    if bf_available:
        decoder_outcomes["BeliefFind"] = np.array(results_bf, dtype=bool)
    if gnn_bplsd_available:
        decoder_outcomes["GNN + BP-LSD"] = np.array(results_gnn_bplsd, dtype=bool)
    if gnn_bposd_available:
        decoder_outcomes["GNN + BP-OSD"] = np.array(results_gnn_bposd, dtype=bool)
    if mwpm_available:
        decoder_outcomes["MWPM"] = np.array(results_mwpm, dtype=bool)

    # Run McNemar's test (each decoder vs BP baseline)
    stat_tests = pairwise_significance(decoder_outcomes, reference="BP (stale LLR)")
    pval_lookup = {t["decoder_b"]: t["p_value"] for t in stat_tests}

    # ---------- Sorted Comparison Table ----------
    decoder_table = []
    decoder_table.append(("BP (stale LLR)", bp_ler))
    if has_per_shot_p:
        oracle_ler_val = oracle_bp_errors / shots if shots > 0 else 0.0
        decoder_table.append(("Oracle BP", oracle_ler_val))
    if gnn_model is not None:
        gnn_bp_ler_val = gnn_bp_errors / shots if shots > 0 else 0.0
        decoder_table.append(("GNN-BP", gnn_bp_ler_val))
    if bposd_available:
        decoder_table.append(("BP-OSD", bposd_ler))
    if bplsd_available:
        decoder_table.append(("BP-LSD", bplsd_ler))
    if bf_available:
        decoder_table.append(("BeliefFind", bf_ler))
    if gnn_bplsd_available:
        decoder_table.append(("GNN + BP-LSD", gnn_bplsd_ler))
    if gnn_bposd_available:
        decoder_table.append(("GNN + BP-OSD", gnn_bposd_ler))
    if gnn_model is not None and has_interleaved:
        inter_ler_val = interleaved_bp_errors / shots if shots > 0 else 0.0
        decoder_table.append(("Interleaved GNN-BP", inter_ler_val))
    if mwpm_available:
        decoder_table.append(("MWPM", mwpm_ler))

    decoder_table.sort(key=lambda x: x[1])

    print(f"\n{'='*60}")
    print(f"  Decoder Comparison (sorted by LER)")
    print(f"{'='*60}")
    for rank, (name, ler_val) in enumerate(decoder_table, 1):
        improv = (bp_ler - ler_val) / bp_ler * 100 if bp_ler > 0 else 0.0
        tag = " (best)" if rank == 1 else ""
        pval = pval_lookup.get(name)
        if pval is not None:
            sig = "*" if pval < 0.05 else " "
            pstr = f"  p={pval:.4f}{sig}" if pval >= 0.0001 else f"  p<0.0001*"
        else:
            pstr = "  (ref)     "
        print(f"  {rank}. {name:20s}  LER={ler_val:.6f}  ({improv:+.1f}% vs BP){pstr}{tag}")
    print(f"{'='*60}")
    print(f"  McNemar's test vs BP baseline (* = p < 0.05)")

    # Store statistical tests in results
    if stat_tests:
        results["statistical_tests"] = stat_tests

    return results


def evaluate_circuit_level(
    npz_data: dict,
    meta: dict,
    device: torch.device,
    use_bposd: bool = False,
    gnn_model: Optional[nn.Module] = None,
    correction_mode: str = "additive",
    checkpoint: Optional[dict] = None,
) -> dict:
    """Evaluate decoders in circuit-level mode (DEM-based decoding).

    Supports: plain BP, BP-OSD, and GNN-assisted BP on DEM.

    Returns:
        Dictionary with results for each decoder.
    """
    from gnn_pipeline.dem_decoder import extract_dem_pcm, build_dem_bp_decoder, run_dem_bp_decoder
    from gnn_pipeline.tanner_graph import build_dem_tanner_graph

    syndromes = npz_data["syndromes"].astype(np.float32)
    observables = npz_data["observables"].astype(np.float32)

    shots = syndromes.shape[0]
    num_detectors = syndromes.shape[1]
    num_obs = observables.shape[1]

    # Extract circuit text from NPZ
    if "circuit" not in npz_data:
        print("ERROR: NPZ does not contain 'circuit' key. Cannot use circuit_level mode.")
        print("  Use --mode code_capacity for data generated with generate_codecap.py")
        return {"error": "no_circuit_in_npz"}

    circuit_bytes = npz_data["circuit"]
    if hasattr(circuit_bytes, 'tobytes'):
        circuit_text = circuit_bytes.tobytes().decode("utf-8")
    else:
        circuit_text = str(circuit_bytes)

    print(f"Circuit loaded: {len(circuit_text)} chars")
    print(f"Shots: {shots}, Detectors: {num_detectors}, Observables: {num_obs}")

    # Extract DEM PCM (done once)
    print("Extracting DEM PCM...")
    dem_pcm, error_probs, obs_matrix = extract_dem_pcm(circuit_text)
    num_errors = dem_pcm.shape[1]
    print(f"DEM PCM: ({dem_pcm.shape[0]} detectors x {num_errors} errors)")
    print(f"Error probs range: [{error_probs.min():.6f}, {error_probs.max():.6f}]")
    print(f"Observable matrix: ({obs_matrix.shape[0]} errors x {obs_matrix.shape[1]} obs)")

    # Pre-compute channel LLRs
    p_clamped = np.clip(error_probs, 1e-10, 1.0 - 1e-10)
    prior_llr = np.log((1.0 - p_clamped) / p_clamped).astype(np.float32)

    # Optionally load BP-OSD for DEM
    bposd_dem_available = False
    if use_bposd:
        try:
            from gnn_pipeline.bposd_decoder import run_dem_bposd_decoder
            bposd_dem_available = True
            print("BP-OSD (DEM) decoder enabled")
        except ImportError as e:
            print(f"Warning: BP-OSD not available: {e}")

    # Pre-build BP decoder (expensive -- do once, reuse for all shots)
    print("Building DEM BP decoder...")
    dem_decoder, dem_channel_llr = build_dem_bp_decoder(
        dem_pcm, error_probs, device, max_iter=100
    )
    print("  Done.")

    # --- GNN-BP setup (if GNN model provided) ---
    has_gnn = gnn_model is not None
    has_neural_bp = checkpoint is not None and checkpoint.get("neural_bp", False)
    gnn_dec = None

    if has_gnn:
        print("\nSetting up GNN-BP for circuit-level...")

        # Build DEM Tanner graph for GNN
        node_type_np, edge_index_np, edge_type_np = build_dem_tanner_graph(dem_pcm)
        node_type_t = torch.from_numpy(node_type_np).long().to(device)
        edge_index_t = torch.from_numpy(edge_index_np).long().to(device)
        edge_type_t = torch.from_numpy(edge_type_np).long().to(device)
        prior_llr_t = torch.from_numpy(prior_llr).float().to(device)
        num_nodes = num_errors + num_detectors

        # Build Neural BP decoder if checkpoint has trained weights
        ckpt_bp_iters = checkpoint.get("bp_iters", 10) if checkpoint else 10
        eval_bp_iters = 100  # extend to full 100 iterations for eval
        if has_neural_bp:
            print(f"  Neural BP: trained={ckpt_bp_iters} iters, eval={eval_bp_iters} iters")
            gnn_dec = MinSumBPDecoder(
                dem_pcm, max_iter=eval_bp_iters, alpha=0.8, clamp_llr=20.0, neural_bp=True
            ).to(device)
            gnn_dec.load_state_dict(checkpoint["decoder_state_dict"], strict=False)
            gnn_dec.eval()
        else:
            gnn_dec = dem_decoder  # reuse plain BP decoder

        p_val = float(meta.get("p", 0.005))

    # Run evaluation
    print("\nEvaluating circuit-level decoding...")
    t_start = time.time()

    bp_errors = 0
    bp_converged = 0
    bposd_errors = 0
    bposd_converged = 0
    gnn_bp_errors = 0
    gnn_bp_converged = 0

    # Per-shot outcome arrays for statistical testing
    bp_logical_all = []
    gnn_logical_all = []

    CHUNK = 512  # Batch size for batched BP

    # --- Batched plain BP on DEM ---
    print("Batched BP decoding on DEM...")
    t_bp = time.time()
    for start in range(0, shots, CHUNK):
        end = min(start + CHUNK, shots)
        B_chunk = end - start

        syn_t = torch.from_numpy(syndromes[start:end]).float().to(device)
        llr_t = torch.from_numpy(
            np.broadcast_to(prior_llr[None, :], (B_chunk, num_errors)).copy()
        ).float().to(device)

        with torch.no_grad():
            _, hard, converged = dem_decoder(syn_t, llr_t)

        hard_np = hard.cpu().numpy()  # (B, num_errors)
        conv_np = converged.cpu().numpy()

        # Map to observables
        pred_obs = (hard_np @ obs_matrix) % 2  # (B, num_obs)
        obs_actual = observables[start:end].astype(np.int64)
        n_obs_check = min(pred_obs.shape[1], obs_actual.shape[1])
        logical_err = np.any(pred_obs[:, :n_obs_check] != obs_actual[:, :n_obs_check], axis=1)

        bp_logical_all.append(logical_err)
        bp_errors += int(logical_err.sum())
        bp_converged += int(conv_np.sum())

        if end % max(1, shots // 5) < CHUNK:
            elapsed = time.time() - t_bp
            print(f"  BP: {end}/{shots} shots ({end / elapsed:.0f} shots/s)")

    bp_logical_arr = np.concatenate(bp_logical_all)
    print(f"  BP done: {shots / (time.time() - t_bp):.0f} shots/s")

    # --- Batched GNN-BP on DEM ---
    if has_gnn:
        print("\nGNN-BP decoding on DEM...")
        t_gnn = time.time()

        for start in range(0, shots, CHUNK):
            end = min(start + CHUNK, shots)
            B_chunk = end - start

            # Compute per-shot GNN corrections
            corrections = np.zeros((B_chunk, num_errors), dtype=np.float32)
            for i in range(B_chunk):
                si = start + i
                det_events = syndromes[si]

                # Build node features
                x_feat = torch.zeros(num_nodes, 4, dtype=torch.float32)
                x_feat[:num_errors, 0] = prior_llr_t
                x_feat[:num_errors, 1] = 1.0
                x_feat[num_errors:, 0] = torch.from_numpy(det_events).float()
                x_feat[num_errors:, 2] = 1.0

                data_obj = Data(
                    x=x_feat.to(device),
                    edge_index=edge_index_t,
                    edge_type=edge_type_t,
                    node_type=node_type_t,
                    channel_llr=prior_llr_t.clone(),
                )
                data_obj.p_value = torch.tensor(p_val, dtype=torch.float32).to(device)

                with torch.no_grad():
                    gnn_model.eval()
                    gnn_out = gnn_model(data_obj)

                llr_base = prior_llr_t.clone()
                if correction_mode == "both":
                    add_c, mul_c = gnn_out
                    add_c = torch.clamp(add_c, -20.0, 20.0)
                    mul_c = torch.clamp(mul_c, -5.0, 5.0)
                    gnn_out = (add_c, mul_c)
                else:
                    gnn_out = torch.clamp(gnn_out, -20.0, 20.0)

                corrections[i] = apply_correction(llr_base, gnn_out, correction_mode).cpu().numpy()

            # Batched BP with corrected LLRs
            syn_t = torch.from_numpy(syndromes[start:end]).float().to(device)
            corr_t = torch.from_numpy(corrections).float().to(device)

            with torch.no_grad():
                _, hard, converged = gnn_dec(syn_t, corr_t)

            hard_np = hard.cpu().numpy()
            conv_np = converged.cpu().numpy()

            pred_obs = (hard_np @ obs_matrix) % 2
            obs_actual = observables[start:end].astype(np.int64)
            n_obs_check = min(pred_obs.shape[1], obs_actual.shape[1])
            logical_err = np.any(pred_obs[:, :n_obs_check] != obs_actual[:, :n_obs_check], axis=1)

            gnn_logical_all.append(logical_err)
            gnn_bp_errors += int(logical_err.sum())
            gnn_bp_converged += int(conv_np.sum())

            if end % max(1, shots // 5) < CHUNK:
                elapsed = time.time() - t_gnn
                print(f"  GNN-BP: {end}/{shots} shots ({end / elapsed:.0f} shots/s)")

        print(f"  GNN-BP done: {shots / (time.time() - t_gnn):.0f} shots/s")

    # --- BP-OSD on DEM (per-shot, parallelized) ---
    if bposd_dem_available:
        import concurrent.futures
        import os

        print("\nBP-OSD decoding on DEM (parallel)...")
        t_bposd = time.time()
        n_workers = max(1, os.cpu_count() - 1)

        def _bposd_shot(idx):
            det_row = syndromes[idx].astype(np.int64)
            pred_obs_bposd, conv_bposd = run_dem_bposd_decoder(
                det_row, dem_pcm, error_probs, obs_matrix
            )
            observable = observables[idx].astype(np.int64)
            n_obs_check = min(len(pred_obs_bposd), len(observable))
            logical_err = bool(np.any(pred_obs_bposd[:n_obs_check] != observable[:n_obs_check]))
            return logical_err, conv_bposd

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            results_bposd = list(pool.map(_bposd_shot, range(shots)))

        bposd_errors = sum(r[0] for r in results_bposd)
        bposd_converged = sum(r[1] for r in results_bposd)
        print(f"  BP-OSD done: {shots / (time.time() - t_bposd):.0f} shots/s ({n_workers} workers)")

    elapsed = time.time() - t_start

    # --- Report results ---
    bp_ler = bp_errors / shots if shots > 0 else 0.0
    bp_ler_low, bp_ler_high = wilson_score_interval_binom(bp_errors, shots)

    print(f"\n=== Circuit-Level Results ({elapsed:.1f}s) ===")
    print(f"BP Decoder (DEM-based, batched):")
    print(f"  Logical Error Rate: {bp_ler:.6f}")
    print(f"  95% CI: [{bp_ler_low:.6f}, {bp_ler_high:.6f}]")
    print(f"  Errors: {bp_errors}/{shots}")
    print(f"  Convergence: {bp_converged}/{shots} ({100*bp_converged/shots:.1f}%)")

    results = {
        "mode": "circuit_level",
        "dem": {
            "num_detectors": int(dem_pcm.shape[0]),
            "num_errors": int(num_errors),
            "num_observables": int(obs_matrix.shape[1]),
        },
        "test_shots": shots,
        "elapsed_seconds": round(elapsed, 1),
        "bp_dem": {
            "errors": int(bp_errors),
            "ler": float(bp_ler),
            "ler_ci_low": float(bp_ler_low),
            "ler_ci_high": float(bp_ler_high),
            "convergence_rate": float(bp_converged / shots),
        },
    }

    if has_gnn:
        gnn_ler = gnn_bp_errors / shots if shots > 0 else 0.0
        gnn_ler_low, gnn_ler_high = wilson_score_interval_binom(gnn_bp_errors, shots)
        improvement = (bp_ler - gnn_ler) / bp_ler if bp_ler > 0 else 0.0

        print(f"\nGNN-BP Decoder (DEM-based, Neural BP={has_neural_bp}):")
        print(f"  Logical Error Rate: {gnn_ler:.6f}")
        print(f"  95% CI: [{gnn_ler_low:.6f}, {gnn_ler_high:.6f}]")
        print(f"  Errors: {gnn_bp_errors}/{shots}")
        print(f"  Convergence: {gnn_bp_converged}/{shots} ({100*gnn_bp_converged/shots:.1f}%)")
        print(f"  Improvement over BP: {100*improvement:.1f}%")

        results["gnn_bp_dem"] = {
            "errors": int(gnn_bp_errors),
            "ler": float(gnn_ler),
            "ler_ci_low": float(gnn_ler_low),
            "ler_ci_high": float(gnn_ler_high),
            "convergence_rate": float(gnn_bp_converged / shots),
        }

    if bposd_dem_available:
        bposd_ler = bposd_errors / shots if shots > 0 else 0.0
        bposd_ler_low, bposd_ler_high = wilson_score_interval_binom(bposd_errors, shots)

        print(f"\nBP-OSD Decoder (DEM-based):")
        print(f"  Logical Error Rate: {bposd_ler:.6f}")
        print(f"  95% CI: [{bposd_ler_low:.6f}, {bposd_ler_high:.6f}]")
        print(f"  Errors: {bposd_errors}/{shots}")
        print(f"  Convergence: {bposd_converged}/{shots} ({100*bposd_converged/shots:.1f}%)")

        bp_vs_bposd = (bp_ler - bposd_ler) / bp_ler if bp_ler > 0 else 0.0
        print(f"  Improvement over BP: {100*bp_vs_bposd:.1f}%")

        results["bposd_dem"] = {
            "errors": int(bposd_errors),
            "ler": float(bposd_ler),
            "ler_ci_low": float(bposd_ler_low),
            "ler_ci_high": float(bposd_ler_high),
            "convergence_rate": float(bposd_converged / shots),
        }

    # --- Collect per-shot outcomes for statistical testing ---
    decoder_outcomes = {"BP (DEM)": bp_logical_arr}
    if has_gnn and gnn_logical_all:
        decoder_outcomes["GNN-BP (DEM)"] = np.concatenate(gnn_logical_all)
    if bposd_dem_available:
        decoder_outcomes["BP-OSD (DEM)"] = np.array([r[0] for r in results_bposd], dtype=bool)

    stat_tests = pairwise_significance(decoder_outcomes, reference="BP (DEM)")
    pval_lookup = {t["decoder_b"]: t["p_value"] for t in stat_tests}

    # --- Comparison table ---
    decoder_table = [("BP (DEM)", bp_ler)]
    if has_gnn:
        decoder_table.append(("GNN-BP (DEM)", gnn_ler))
    if bposd_dem_available:
        decoder_table.append(("BP-OSD (DEM)", bposd_ler))

    decoder_table.sort(key=lambda x: x[1])
    print(f"\n{'='*60}")
    print(f"  Circuit-Level Decoder Comparison (sorted by LER)")
    print(f"{'='*60}")
    for rank, (name, ler_val) in enumerate(decoder_table, 1):
        improv = (bp_ler - ler_val) / bp_ler * 100 if bp_ler > 0 else 0.0
        tag = " (best)" if rank == 1 else ""
        pval = pval_lookup.get(name)
        if pval is not None:
            sig = "*" if pval < 0.05 else " "
            pstr = f"  p={pval:.4f}{sig}" if pval >= 0.0001 else f"  p<0.0001*"
        else:
            pstr = "  (ref)     "
        print(f"  {rank}. {name:20s}  LER={ler_val:.6f}  ({improv:+.1f}% vs BP){pstr}{tag}")
    print(f"{'='*60}")
    print(f"  McNemar's test vs BP baseline (* = p < 0.05)")

    if stat_tests:
        results["statistical_tests"] = stat_tests

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate BP vs GNN-assisted BP vs BP-OSD")
    parser.add_argument("--test_npz", type=str, required=True,
                        help="Path to test data NPZ file")
    parser.add_argument("--gnn_model", type=str, required=False,
                        help="Path to trained GNN model (code_capacity or circuit_level)")
    parser.add_argument("--mode", type=str, default="code_capacity",
                        choices=["code_capacity", "circuit_level"],
                        help="Decoding mode: code_capacity (separate CSS) or circuit_level (DEM)")
    parser.add_argument("--bposd", action="store_true",
                        help="Enable BP-OSD baseline decoder")
    parser.add_argument("--mwpm", action="store_true",
                        help="Enable MWPM baseline decoder (approximate for LDPC codes)")
    parser.add_argument("--bplsd", action="store_true",
                        help="Enable BP-LSD decoder (stronger than BP-OSD for LDPC codes)")
    parser.add_argument("--belieffind", action="store_true",
                        help="Enable BeliefFind (BP + Union Find) decoder")
    parser.add_argument("--lsd_order", type=int, default=0,
                        help="LSD order for BP-LSD decoder (0=fastest, higher=stronger)")
    parser.add_argument("--lsd_method", type=str, default="LSD_CS",
                        choices=["LSD_0", "LSD_E", "LSD_CS"],
                        help="LSD method for BP-LSD decoder")
    parser.add_argument("--osd_order", type=int, default=0,
                        help="OSD order for BP-OSD decoder (0=fastest, higher=stronger)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for results")

    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Mode: {args.mode}")

    # Load test data
    print(f"Loading test data from {args.test_npz}...")
    npz_data = _load_npz(args.test_npz)
    meta = _parse_meta(npz_data)

    # Load GNN model if provided
    gnn_model_inst = None
    gnn_correction_mode = "additive"
    gnn_checkpoint = None
    if args.gnn_model:
        print(f"Loading GNN model from {args.gnn_model}...")
        checkpoint = torch.load(args.gnn_model, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            gnn_checkpoint = checkpoint
            gnn_correction_mode = checkpoint.get("correction_mode", "additive")
            gnn_use_film = checkpoint.get("use_film", False)
            gnn_noise_feat_dim = checkpoint.get("noise_feat_dim", 1)
            # DEM models use edge_types=1, code-capacity uses edge_types=2
            gnn_edge_types = checkpoint.get("edge_types", 2)
            gnn_model_inst = TannerGNN(
                hidden_dim=checkpoint.get("hidden_dim", 64),
                num_mp_layers=checkpoint.get("num_mp_layers", 3),
                edge_types=gnn_edge_types,
                correction_mode=gnn_correction_mode,
                use_residual=checkpoint.get("use_residual", False),
                use_layer_norm=checkpoint.get("use_layer_norm", False),
                use_attention=checkpoint.get("use_attention", False),
                use_film=gnn_use_film,
                noise_feat_dim=gnn_noise_feat_dim,
                standardize_input=checkpoint.get("standardize_input", False),
            )
            gnn_model_inst.load_state_dict(checkpoint["model_state_dict"])
            ckpt_mode = checkpoint.get("mode", "code_capacity")
            film_str = f", film={gnn_use_film}" if gnn_use_film else ""
            print(f"  Loaded checkpoint (mode={ckpt_mode}, hidden_dim={checkpoint.get('hidden_dim', 64)}, "
                  f"num_mp_layers={checkpoint.get('num_mp_layers', 3)}, "
                  f"edge_types={gnn_edge_types}, "
                  f"correction_mode={gnn_correction_mode}, "
                  f"attention={checkpoint.get('use_attention', False)}{film_str})")
        else:
            gnn_model_inst = TannerGNN()
            gnn_model_inst.load_state_dict(checkpoint)
        gnn_model_inst = gnn_model_inst.to(device)
        gnn_model_inst.eval()

    # Run evaluation
    if args.mode == "code_capacity":
        results = evaluate_code_capacity(
            npz_data, meta, device,
            gnn_model=gnn_model_inst,
            use_bposd=args.bposd,
            use_mwpm=args.mwpm,
            use_bplsd=args.bplsd,
            use_belieffind=args.belieffind,
            correction_mode=gnn_correction_mode,
            lsd_order=args.lsd_order,
            lsd_method=args.lsd_method,
            osd_order=args.osd_order,
            checkpoint=gnn_checkpoint,
        )
    elif args.mode == "circuit_level":
        results = evaluate_circuit_level(
            npz_data, meta, device,
            use_bposd=args.bposd,
            gnn_model=gnn_model_inst,
            correction_mode=gnn_correction_mode,
            checkpoint=gnn_checkpoint,
        )
    else:
        print(f"Unknown mode: {args.mode}")
        return 1

    # Save results
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
