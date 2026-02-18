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
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from gnn_pipeline.bp_decoder import MinSumBPDecoder
from gnn_pipeline.dataset import _load_npz, _parse_meta
from gnn_pipeline.gnn_model import TannerGNN, apply_correction
from gnn_pipeline.tanner_graph import build_tanner_graph


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

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
    correction_mode: str = "additive",
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

    print(f"Noise: p={p}, eta={eta}, pz={pz:.6f}, px={px:.6f}")
    print(f"LLR_z={llr_z:.4f}, LLR_x={llr_x:.4f}")

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

    # ---------- Batched BP decoding ----------
    print("\nBatched BP decoding...")
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

        llr_z_t = torch.full((B_chunk, n), llr_z, dtype=torch.float32, device=device)
        llr_x_t = torch.full((B_chunk, n), llr_x, dtype=torch.float32, device=device)

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

    # ---------- Batched GNN-BP decoding ----------
    gnn_bp_errors = 0
    gnn_bp_converged = 0
    if gnn_model is not None:
        print("Batched GNN-BP decoding...")
        t_gnn = time.time()

        # GNN runs per-shot (graph-level), but BP is batched after corrections
        node_type_np, edge_index_np, edge_type_np = build_tanner_graph(hx, hz)
        node_type_t = torch.from_numpy(node_type_np).long().to(device)
        edge_index_t = torch.from_numpy(edge_index_np).long().to(device)
        edge_type_t = torch.from_numpy(edge_type_np).long().to(device)
        num_nodes = n + mx + mz

        gnn_hard_z_all = []
        gnn_hard_x_all = []
        gnn_conv_all = []

        for start in range(0, shots, CHUNK):
            end = min(start + CHUNK, shots)
            B_chunk = end - start

            # Build per-shot GNN corrections
            corrections_z = np.zeros((B_chunk, n), dtype=np.float32)
            corrections_x = np.zeros((B_chunk, n), dtype=np.float32)

            for i in range(B_chunk):
                si = start + i
                x_feat = torch.zeros(num_nodes, 4, dtype=torch.float32)
                avg_llr = (llr_z + llr_x) / 2.0
                x_feat[:n, 0] = avg_llr
                x_feat[:n, 1] = 1.0
                x_feat[n:n+mx, 0] = torch.from_numpy(all_x_syn[si]).float()
                x_feat[n:n+mx, 2] = 1.0
                x_feat[n+mx:, 0] = torch.from_numpy(all_z_syn[si]).float()
                x_feat[n+mx:, 3] = 1.0

                data_obj = Data(
                    x=x_feat.to(device),
                    edge_index=edge_index_t,
                    edge_type=edge_type_t,
                    node_type=node_type_t,
                    channel_llr=torch.full((n,), avg_llr, dtype=torch.float32, device=device),
                )

                with torch.no_grad():
                    gnn_model.eval()
                    gnn_out = gnn_model(data_obj)

                llr_z_s = torch.full((n,), llr_z, dtype=torch.float32, device=device)
                llr_x_s = torch.full((n,), llr_x, dtype=torch.float32, device=device)

                if correction_mode == "both":
                    add_c, mul_c = gnn_out
                    add_c = torch.clamp(add_c, -20.0, 20.0)
                    mul_c = torch.clamp(mul_c, -5.0, 5.0)
                    gnn_out = (add_c, mul_c)
                else:
                    gnn_out = torch.clamp(gnn_out, -20.0, 20.0)

                corrections_z[i] = apply_correction(llr_z_s, gnn_out, correction_mode).cpu().numpy()
                corrections_x[i] = apply_correction(llr_x_s, gnn_out, correction_mode).cpu().numpy()

            # Batched BP with corrected LLRs
            x_syn_t = torch.from_numpy(all_x_syn[start:end]).float().to(device)
            z_syn_t = torch.from_numpy(all_z_syn[start:end]).float().to(device)
            corr_z_t = torch.from_numpy(corrections_z).float().to(device)
            corr_x_t = torch.from_numpy(corrections_x).float().to(device)

            with torch.no_grad():
                _, hard_z, conv_z = dec_z_pre(x_syn_t, corr_z_t)
                _, hard_x, conv_x = dec_x_pre(z_syn_t, corr_x_t)

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

    elapsed = time.time() - t_start

    # ---------- Report results ----------
    bp_ler = bp_errors / shots if shots > 0 else 0.0
    bp_ler_low, bp_ler_high = wilson_score_interval_binom(bp_errors, shots)

    print(f"\n=== Results ({elapsed:.1f}s) ===")
    print(f"BP Decoder (separate CSS, batched):")
    print(f"  Logical Error Rate: {bp_ler:.6f}")
    print(f"  95% CI: [{bp_ler_low:.6f}, {bp_ler_high:.6f}]")
    print(f"  Errors: {bp_errors}/{shots}")
    print(f"  Convergence: {bp_converged}/{shots} ({100*bp_converged/shots:.1f}%)")

    results = {
        "mode": "code_capacity",
        "code": {"n": int(n), "mx": int(mx), "mz": int(mz)},
        "noise": {"p": p, "eta": eta, "pz": pz, "px": px},
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

    if gnn_model is not None:
        gnn_bp_ler = gnn_bp_errors / shots if shots > 0 else 0.0
        gnn_bp_ler_low, gnn_bp_ler_high = wilson_score_interval_binom(gnn_bp_errors, shots)

        print(f"\nGNN-Assisted BP (separate CSS, batched):")
        print(f"  Logical Error Rate: {gnn_bp_ler:.6f}")
        print(f"  95% CI: [{gnn_bp_ler_low:.6f}, {gnn_bp_ler_high:.6f}]")
        print(f"  Errors: {gnn_bp_errors}/{shots}")
        print(f"  Convergence: {gnn_bp_converged}/{shots} ({100*gnn_bp_converged/shots:.1f}%)")

        improvement = (bp_ler - gnn_bp_ler) / bp_ler if bp_ler > 0 else 0.0
        print(f"\n  Improvement over BP: {100*improvement:.1f}%")

        results["gnn_bp"] = {
            "errors": int(gnn_bp_errors),
            "ler": float(gnn_bp_ler),
            "ler_ci_low": float(gnn_bp_ler_low),
            "ler_ci_high": float(gnn_bp_ler_high),
            "convergence_rate": float(gnn_bp_converged / shots),
        }
        results["improvement_pct"] = float(100 * improvement)

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

    return results


def evaluate_circuit_level(
    npz_data: dict,
    meta: dict,
    device: torch.device,
    use_bposd: bool = False,
) -> dict:
    """Evaluate decoders in circuit-level mode (DEM-based decoding).

    Returns:
        Dictionary with results for each decoder.
    """
    from gnn_pipeline.dem_decoder import extract_dem_pcm, build_dem_bp_decoder, run_dem_bp_decoder

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

    # Optionally load BP-OSD for DEM
    bposd_dem_available = False
    if use_bposd:
        try:
            from gnn_pipeline.bposd_decoder import run_dem_bposd_decoder
            bposd_dem_available = True
            print("BP-OSD (DEM) decoder enabled")
        except ImportError as e:
            print(f"Warning: BP-OSD not available: {e}")

    # Pre-build BP decoder (expensive — do once, reuse for all shots)
    print("Building DEM BP decoder...")
    dem_decoder, dem_channel_llr = build_dem_bp_decoder(
        dem_pcm, error_probs, device, max_iter=100
    )
    print("  Done.")

    # Run evaluation
    print("\nEvaluating circuit-level decoding...")
    t_start = time.time()

    bp_errors = 0
    bp_converged = 0
    bposd_errors = 0
    bposd_converged = 0

    for shot_idx in range(shots):
        det_row = syndromes[shot_idx].astype(np.int64)
        observable = observables[shot_idx].astype(np.int64)

        # --- BP on DEM ---
        pred_obs_bp, conv_bp = run_dem_bp_decoder(
            det_row, dem_pcm, error_probs, obs_matrix, device, max_iter=100,
            decoder=dem_decoder, channel_llr=dem_channel_llr,
        )
        bp_converged += int(conv_bp)

        # Check logical error: compare predicted observable flips vs actual
        n_obs_check = min(len(pred_obs_bp), len(observable))
        bp_logical_err = bool(np.any(pred_obs_bp[:n_obs_check] != observable[:n_obs_check]))
        bp_errors += int(bp_logical_err)

        # --- BP-OSD on DEM ---
        if bposd_dem_available:
            pred_obs_bposd, conv_bposd = run_dem_bposd_decoder(
                det_row, dem_pcm, error_probs, obs_matrix
            )
            bposd_converged += int(conv_bposd)
            bposd_logical_err = bool(np.any(
                pred_obs_bposd[:n_obs_check] != observable[:n_obs_check]
            ))
            bposd_errors += int(bposd_logical_err)

        if (shot_idx + 1) % max(1, shots // 10) == 0:
            elapsed = time.time() - t_start
            rate = (shot_idx + 1) / elapsed
            print(f"  Processed {shot_idx + 1}/{shots} shots ({rate:.1f} shots/s)")

    elapsed = time.time() - t_start

    # Compute LER with CIs
    bp_ler = bp_errors / shots if shots > 0 else 0.0
    bp_ler_low, bp_ler_high = wilson_score_interval_binom(bp_errors, shots)

    print(f"\n=== Circuit-Level Results ({elapsed:.1f}s) ===")
    print(f"BP Decoder (DEM-based):")
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
                        help="Path to trained GNN model (code_capacity mode only)")
    parser.add_argument("--mode", type=str, default="code_capacity",
                        choices=["code_capacity", "circuit_level"],
                        help="Decoding mode: code_capacity (separate CSS) or circuit_level (DEM)")
    parser.add_argument("--bposd", action="store_true",
                        help="Enable BP-OSD baseline decoder")
    parser.add_argument("--mwpm", action="store_true",
                        help="Enable MWPM baseline decoder (approximate for LDPC codes)")
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

    # Load GNN model if provided (code_capacity mode only)
    gnn_model_inst = None
    gnn_correction_mode = "additive"
    if args.gnn_model:
        if args.mode == "circuit_level":
            print("Warning: GNN model not supported in circuit_level mode (ignored)")
        else:
            print(f"Loading GNN model from {args.gnn_model}...")
            checkpoint = torch.load(args.gnn_model, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                gnn_correction_mode = checkpoint.get("correction_mode", "additive")
                gnn_model_inst = TannerGNN(
                    hidden_dim=checkpoint.get("hidden_dim", 64),
                    num_mp_layers=checkpoint.get("num_mp_layers", 3),
                    correction_mode=gnn_correction_mode,
                )
                gnn_model_inst.load_state_dict(checkpoint["model_state_dict"])
                print(f"  Loaded checkpoint (hidden_dim={checkpoint.get('hidden_dim', 64)}, "
                      f"num_mp_layers={checkpoint.get('num_mp_layers', 3)}, "
                      f"correction_mode={gnn_correction_mode})")
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
            correction_mode=gnn_correction_mode,
        )
    elif args.mode == "circuit_level":
        results = evaluate_circuit_level(
            npz_data, meta, device,
            use_bposd=args.bposd,
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
