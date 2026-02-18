"""Threshold sweep: LER vs physical error rate for multiple decoders.

Generates code-capacity data at each error rate (static and drifting),
runs BP, GNN-BP, and BP-OSD decoders, and produces publication-quality
threshold plots with Wilson score confidence intervals.

Supports multiple drift models (sine, OU, RTN) and code sizes.

Usage:
    # Quick validation (3 points, 1000 shots)
    python -m gnn_pipeline.threshold_sweep --p_min 0.01 --p_max 0.04 --num_points 3 --shots 1000 --eta 20 --drift_amp 0.015 --drift_period 500 --bposd --out_dir runs/threshold_quick

    # Full sweep with multiple drift models
    python -m gnn_pipeline.threshold_sweep --p_min 0.005 --p_max 0.06 --num_points 8 --shots 5000 --eta 20 --drift_models sine,ou,rtn --drift_amp 0.02 --gnn_model runs/model.pt --bposd --out_dir runs/threshold_full

    # Sweep with specific code
    python -m gnn_pipeline.threshold_sweep --code 144_12_12 --p_min 0.01 --p_max 0.04 --num_points 4 --shots 2000 --eta 20 --out_dir runs/sweep_144
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from codes.codes_q import create_bivariate_bicycle_codes
from codes.code_registry import get_code_params, list_codes
from gnn_pipeline.evaluate import (
    _check_logical_error,
    wilson_score_interval_binom,
)
from gnn_pipeline.generate_codecap import generate_code_capacity_data
from gnn_pipeline.gnn_model import TannerGNN


def _decode_all_shots(
    syndromes: np.ndarray,
    observables: np.ndarray,
    hx: np.ndarray,
    hz: np.ndarray,
    lx: np.ndarray,
    lz: np.ndarray,
    p: float,
    eta: float,
    device: torch.device,
    gnn_model: Optional[torch.nn.Module] = None,
    use_bposd: bool = False,
    use_mwpm: bool = False,
) -> dict:
    """Decode all shots and return results for each decoder.

    Uses batched BP decoding for CPU/GPU parallelism and multiprocessing
    for BP-OSD/MWPM.

    Returns:
        Dictionary with keys like 'bp', 'gnn_bp', 'bposd', each containing
        'errors', 'convergences', 'shots'.
    """
    from gnn_pipeline.bp_decoder import MinSumBPDecoder
    from gnn_pipeline.tanner_graph import build_tanner_graph
    from gnn_pipeline.evaluate import _check_logical_errors_batch
    from gnn_pipeline.gnn_model import apply_correction
    from torch_geometric.data import Data

    mx, n = hx.shape
    mz = hz.shape[0]
    shots = syndromes.shape[0]

    # Per-Pauli rates and LLRs
    pz = p * eta / (eta + 1)
    px = p / (eta + 1)
    pz_c = max(min(pz, 1.0 - 1e-7), 1e-7)
    px_c = max(min(px, 1.0 - 1e-7), 1e-7)
    llr_z = float(math.log((1.0 - pz_c) / pz_c))
    llr_x = float(math.log((1.0 - px_c) / px_c))

    # Split syndromes into X and Z parts (vectorized)
    all_x_syn = syndromes[:, :mx].astype(np.float32)
    all_z_syn = syndromes[:, mx:mx+mz].astype(np.float32)

    # Pre-build BP decoders (reused across all shots -- major speedup)
    dec_z_pre = MinSumBPDecoder(hx, max_iter=100, alpha=0.8, clamp_llr=20.0).to(device)
    dec_x_pre = MinSumBPDecoder(hz, max_iter=100, alpha=0.8, clamp_llr=20.0).to(device)

    # ---------- Batched BP ----------
    CHUNK = 512
    bp_hard_z_all = []
    bp_hard_x_all = []
    bp_conv_all = []

    for start in range(0, shots, CHUNK):
        end = min(start + CHUNK, shots)
        B_chunk = end - start
        x_syn_t = torch.from_numpy(all_x_syn[start:end]).float().to(device)
        z_syn_t = torch.from_numpy(all_z_syn[start:end]).float().to(device)
        llr_z_t = torch.full((B_chunk, n), llr_z, dtype=torch.float32, device=device)
        llr_x_t = torch.full((B_chunk, n), llr_x, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, hard_z, conv_z = dec_z_pre(x_syn_t, llr_z_t)
            _, hard_x, conv_x = dec_x_pre(z_syn_t, llr_x_t)

        bp_hard_z_all.append(hard_z.cpu().numpy())
        bp_hard_x_all.append(hard_x.cpu().numpy())
        bp_conv_all.append((conv_z & conv_x).cpu().numpy())

    bp_z_errors = np.concatenate(bp_hard_z_all, axis=0)
    bp_x_errors = np.concatenate(bp_hard_x_all, axis=0)
    bp_conv = np.concatenate(bp_conv_all, axis=0)

    bp_logical = _check_logical_errors_batch(
        bp_z_errors, bp_x_errors, lx, lz, observables.astype(np.float32),
    )
    bp_errors = int(bp_logical.sum())
    bp_converged = int(bp_conv.sum())

    # ---------- Batched GNN-BP ----------
    gnn_errors = 0
    gnn_converged = 0
    if gnn_model is not None:
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

                correction_mode = getattr(gnn_model, "correction_mode", "additive")
                if correction_mode == "both":
                    add_c, mul_c = gnn_out
                    add_c = torch.clamp(add_c, -20.0, 20.0)
                    mul_c = torch.clamp(mul_c, -5.0, 5.0)
                    gnn_out = (add_c, mul_c)
                else:
                    gnn_out = torch.clamp(gnn_out, -20.0, 20.0)

                corrections_z[i] = apply_correction(llr_z_s, gnn_out, correction_mode).cpu().numpy()
                corrections_x[i] = apply_correction(llr_x_s, gnn_out, correction_mode).cpu().numpy()

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

        gnn_z_errors = np.concatenate(gnn_hard_z_all, axis=0)
        gnn_x_errors = np.concatenate(gnn_hard_x_all, axis=0)
        gnn_conv = np.concatenate(gnn_conv_all, axis=0)

        gnn_logical = _check_logical_errors_batch(
            gnn_z_errors, gnn_x_errors, lx, lz, observables.astype(np.float32),
        )
        gnn_errors = int(gnn_logical.sum())
        gnn_converged = int(gnn_conv.sum())

    # ---------- BP-OSD (parallel with threads) ----------
    # ThreadPoolExecutor: BP-OSD/MWPM are C-extension-backed and release
    # the GIL, so threads give true parallelism. Avoids Windows pickle
    # issues with ProcessPoolExecutor.
    bposd_errors = 0
    bposd_available = False
    if use_bposd:
        try:
            from gnn_pipeline.bposd_decoder import run_css_bposd_decoder
            bposd_available = True
        except ImportError:
            pass

    if bposd_available:
        import concurrent.futures
        import os

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

    # ---------- MWPM (parallel with threads) ----------
    mwpm_errors = 0
    mwpm_available = False
    if use_mwpm:
        try:
            from gnn_pipeline.matching_decoder import build_mwpm_css, run_mwpm_css
            m_z, m_x, emap_z, emap_x = build_mwpm_css(hx, hz, pz, px)
            mwpm_available = True
        except ImportError:
            pass

    if mwpm_available:
        import concurrent.futures
        import os

        n_workers = max(1, os.cpu_count() - 1)

        def _mwpm_shot(idx):
            z_e, x_e = run_mwpm_css(
                all_x_syn[idx], all_z_syn[idx], m_z, m_x, n,
                edge_map_z=emap_z, edge_map_x=emap_x,
            )
            return _check_logical_error(z_e, x_e, lx, lz, observables[idx])

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            results_mwpm = list(pool.map(_mwpm_shot, range(shots)))
        mwpm_errors = sum(results_mwpm)

    # ---------- Build result dict ----------
    result = {
        "bp": {
            "errors": bp_errors,
            "converged": bp_converged,
            "shots": shots,
        }
    }
    if gnn_model is not None:
        result["gnn_bp"] = {
            "errors": gnn_errors,
            "converged": gnn_converged,
            "shots": shots,
        }
    if bposd_available:
        result["bposd"] = {
            "errors": bposd_errors,
            "shots": shots,
        }
    if mwpm_available:
        result["mwpm"] = {
            "errors": mwpm_errors,
            "shots": shots,
        }

    return result


# Line style mapping for different drift types
_DRIFT_STYLES = {
    "static": {"linestyle": "-", "marker_offset": 0},
    "drift_sine": {"linestyle": "--", "marker_offset": 0},
    "drift_ou": {"linestyle": "-.", "marker_offset": 0},
    "drift_rtn": {"linestyle": ":", "marker_offset": 0},
}


def _make_plots(
    all_results: list,
    out_dir: pathlib.Path,
    has_gnn: bool,
    has_bposd: bool,
    has_mwpm: bool = False,
):
    """Generate threshold plots from sweep results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group results by noise_type
    noise_types = sorted(set(r["noise_type"] for r in all_results))
    grouped = {nt: [r for r in all_results if r["noise_type"] == nt] for nt in noise_types}

    # Individual plots per noise type
    for noise_type, results_list in grouped.items():
        if not results_list:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        p_vals = [r["p"] for r in results_list]

        # BP
        bp_ler = [r["bp"]["ler"] for r in results_list]
        bp_lo = [r["bp"]["ci_low"] for r in results_list]
        bp_hi = [r["bp"]["ci_high"] for r in results_list]
        bp_err_lo = [max(0.0, l - lo) for l, lo in zip(bp_ler, bp_lo)]
        bp_err_hi = [max(0.0, hi - l) for l, hi in zip(bp_ler, bp_hi)]
        ax.errorbar(p_vals, bp_ler, yerr=[bp_err_lo, bp_err_hi],
                     marker='o', capsize=4, label='BP', linewidth=2)

        # GNN-BP
        if has_gnn and "gnn_bp" in results_list[0]:
            gnn_ler = [r["gnn_bp"]["ler"] for r in results_list]
            gnn_lo = [r["gnn_bp"]["ci_low"] for r in results_list]
            gnn_hi = [r["gnn_bp"]["ci_high"] for r in results_list]
            gnn_err_lo = [max(0.0, l - lo) for l, lo in zip(gnn_ler, gnn_lo)]
            gnn_err_hi = [max(0.0, hi - l) for l, hi in zip(gnn_ler, gnn_hi)]
            ax.errorbar(p_vals, gnn_ler, yerr=[gnn_err_lo, gnn_err_hi],
                         marker='s', capsize=4, label='GNN-BP', linewidth=2)

        # BP-OSD
        if has_bposd and "bposd" in results_list[0]:
            bposd_ler = [r["bposd"]["ler"] for r in results_list]
            bposd_lo = [r["bposd"]["ci_low"] for r in results_list]
            bposd_hi = [r["bposd"]["ci_high"] for r in results_list]
            bposd_err_lo = [max(0.0, l - lo) for l, lo in zip(bposd_ler, bposd_lo)]
            bposd_err_hi = [max(0.0, hi - l) for l, hi in zip(bposd_ler, bposd_hi)]
            ax.errorbar(p_vals, bposd_ler, yerr=[bposd_err_lo, bposd_err_hi],
                         marker='^', capsize=4, label='BP-OSD', linewidth=2)

        # MWPM
        if has_mwpm and "mwpm" in results_list[0]:
            mwpm_ler = [r["mwpm"]["ler"] for r in results_list]
            mwpm_lo = [r["mwpm"]["ci_low"] for r in results_list]
            mwpm_hi = [r["mwpm"]["ci_high"] for r in results_list]
            mwpm_err_lo = [max(0.0, l - lo) for l, lo in zip(mwpm_ler, mwpm_lo)]
            mwpm_err_hi = [max(0.0, hi - l) for l, hi in zip(mwpm_ler, mwpm_hi)]
            ax.errorbar(p_vals, mwpm_ler, yerr=[mwpm_err_lo, mwpm_err_hi],
                         marker='D', capsize=4, label='MWPM', linewidth=2)

        label = noise_type.replace("_", " ").title()
        ax.set_xlabel("Physical error rate p", fontsize=13)
        ax.set_ylabel("Logical error rate (LER)", fontsize=13)
        ax.set_title(f"Threshold: {label} (code-capacity)", fontsize=14)
        ax.set_yscale("log")
        ax.legend(fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(left=0)

        fig.tight_layout()
        fname = f"threshold_{noise_type}.png"
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")

    # Convergence plot (all noise types)
    if noise_types:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for noise_type, results_list in grouped.items():
            if not results_list:
                continue
            style = _DRIFT_STYLES.get(noise_type, {"linestyle": "-"})
            p_vals = [r["p"] for r in results_list]
            bp_conv = [r["bp"]["convergence"] for r in results_list]
            label_nt = noise_type.replace("_", " ")
            ax.plot(p_vals, bp_conv, marker='o', linestyle=style["linestyle"],
                    label=f'BP ({label_nt})', linewidth=2)

            if has_gnn and "gnn_bp" in results_list[0]:
                gnn_conv = [r["gnn_bp"]["convergence"] for r in results_list]
                ax.plot(p_vals, gnn_conv, marker='s', linestyle=style["linestyle"],
                        label=f'GNN-BP ({label_nt})', linewidth=2)

        ax.set_xlabel("Physical error rate p", fontsize=13)
        ax.set_ylabel("BP Convergence rate", fontsize=13)
        ax.set_title("Decoder Convergence vs Error Rate", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(out_dir / "convergence_plot.png", dpi=150)
        plt.close(fig)
        print(f"  Saved convergence_plot.png")

    # Multi-drift comparison plot (if multiple drift types)
    drift_types = [nt for nt in noise_types if nt != "static"]
    if len(drift_types) >= 2:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        markers = ['o', 's', '^', 'D', 'v']
        colors = plt.cm.tab10.colors

        for idx, noise_type in enumerate(noise_types):
            results_list = grouped[noise_type]
            if not results_list:
                continue
            style = _DRIFT_STYLES.get(noise_type, {"linestyle": "-"})
            p_vals = [r["p"] for r in results_list]
            label_nt = noise_type.replace("_", " ")

            # BP
            bp_ler = [r["bp"]["ler"] for r in results_list]
            ax.plot(p_vals, bp_ler, marker=markers[idx % len(markers)],
                    linestyle=style["linestyle"], color=colors[idx * 2 % len(colors)],
                    label=f'BP ({label_nt})', linewidth=2)

            # GNN-BP
            if has_gnn and "gnn_bp" in results_list[0]:
                gnn_ler = [r["gnn_bp"]["ler"] for r in results_list]
                ax.plot(p_vals, gnn_ler, marker=markers[idx % len(markers)],
                        linestyle=style["linestyle"], color=colors[(idx * 2 + 1) % len(colors)],
                        label=f'GNN-BP ({label_nt})', linewidth=2, markerfacecolor='none')

        ax.set_xlabel("Physical error rate p", fontsize=13)
        ax.set_ylabel("Logical error rate (LER)", fontsize=13)
        ax.set_title("Multi-Drift Comparison", fontsize=14)
        ax.set_yscale("log")
        ax.legend(fontsize=10, ncol=2)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(left=0)

        fig.tight_layout()
        fig.savefig(out_dir / "threshold_drift_comparison.png", dpi=150)
        plt.close(fig)
        print(f"  Saved threshold_drift_comparison.png")

    # Static vs single drift side-by-side (backward compat)
    static = grouped.get("static", [])
    drift_sine = grouped.get("drift_sine", grouped.get("drift", []))
    if static and drift_sine:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for ax, results_list, title in [
            (ax1, static, "Static Noise"),
            (ax2, drift_sine, "Drifting Noise (sine)"),
        ]:
            p_vals = [r["p"] for r in results_list]

            bp_ler = [r["bp"]["ler"] for r in results_list]
            ax.plot(p_vals, bp_ler, marker='o', label='BP', linewidth=2)

            if has_gnn and "gnn_bp" in results_list[0]:
                gnn_ler = [r["gnn_bp"]["ler"] for r in results_list]
                ax.plot(p_vals, gnn_ler, marker='s', label='GNN-BP', linewidth=2)

            if has_bposd and "bposd" in results_list[0]:
                bposd_ler = [r["bposd"]["ler"] for r in results_list]
                ax.plot(p_vals, bposd_ler, marker='^', label='BP-OSD', linewidth=2)

            if has_mwpm and "mwpm" in results_list[0]:
                mwpm_ler = [r["mwpm"]["ler"] for r in results_list]
                ax.plot(p_vals, mwpm_ler, marker='D', label='MWPM', linewidth=2)

            ax.set_xlabel("Physical error rate p", fontsize=13)
            ax.set_title(title, fontsize=14)
            ax.set_yscale("log")
            ax.legend(fontsize=11)
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xlim(left=0)

        ax1.set_ylabel("Logical error rate (LER)", fontsize=13)
        fig.suptitle("Static vs Drifting Noise Comparison", fontsize=15, y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / "threshold_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved threshold_comparison.png")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Threshold sweep: LER vs p")
    parser.add_argument("--code", type=str, default="72_12_6",
                        choices=list_codes(),
                        help="Code to use (default: 72_12_6)")
    parser.add_argument("--p_min", type=float, default=0.005, help="Minimum error rate")
    parser.add_argument("--p_max", type=float, default=0.06, help="Maximum error rate")
    parser.add_argument("--num_points", type=int, default=8, help="Number of p values")
    parser.add_argument("--shots", type=int, default=5000, help="Shots per (p, noise) point")
    parser.add_argument("--eta", type=float, default=20.0, help="Z-bias ratio")
    parser.add_argument("--drift_model", type=str, default=None,
                        choices=["none", "sine", "ou", "rtn"],
                        help="Single drift model to sweep")
    parser.add_argument("--drift_models", type=str, default=None,
                        help="Comma-separated drift models to sweep (e.g. sine,ou,rtn)")
    parser.add_argument("--drift_amp", type=float, default=0.0,
                        help="Drift amplitude (0 = skip drifting noise sweep)")
    parser.add_argument("--drift_period", type=int, default=500, help="Drift period in shots")
    parser.add_argument("--ou_theta", type=float, default=0.1, help="OU mean-reversion rate")
    parser.add_argument("--ou_sigma", type=float, default=0.005, help="OU volatility")
    parser.add_argument("--rtn_delta", type=float, default=0.01, help="RTN half-distance")
    parser.add_argument("--rtn_switch", type=float, default=0.005, help="RTN switch prob")
    parser.add_argument("--gnn_model", type=str, default=None,
                        help="Path to trained GNN model")
    parser.add_argument("--bposd", action="store_true", help="Enable BP-OSD baseline")
    parser.add_argument("--mwpm", action="store_true",
                        help="Enable MWPM baseline (approximate for LDPC codes)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")

    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build code
    code_params = get_code_params(args.code)
    print(f"Building [[{args.code}]] bivariate bicycle code...")
    css, _, _ = create_bivariate_bicycle_codes(**code_params)
    hx = np.array(css.hx, dtype=np.uint8)
    hz = np.array(css.hz, dtype=np.uint8)
    lx = np.array(css.lx, dtype=np.uint8)
    lz = np.array(css.lz, dtype=np.uint8)
    n, mx, mz = hx.shape[1], hx.shape[0], hz.shape[0]
    print(f"Code: n={n}, mx={mx}, mz={mz}, k={lx.shape[0]}")

    # Load GNN model if provided
    gnn_model = None
    if args.gnn_model:
        print(f"Loading GNN model from {args.gnn_model}...")
        checkpoint = torch.load(args.gnn_model, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            gnn_model = TannerGNN(
                hidden_dim=checkpoint.get("hidden_dim", 64),
                num_mp_layers=checkpoint.get("num_mp_layers", 3),
            )
            gnn_model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded checkpoint (hidden_dim={checkpoint.get('hidden_dim', 64)}, "
                  f"num_mp_layers={checkpoint.get('num_mp_layers', 3)})")
        else:
            gnn_model = TannerGNN()
            gnn_model.load_state_dict(checkpoint)
        gnn_model = gnn_model.to(device)
        gnn_model.eval()

    p_values = np.linspace(args.p_min, args.p_max, args.num_points)

    # Determine noise types to sweep
    noise_configs = [("static", "none")]  # always include static

    if args.drift_models:
        # Comma-separated list of drift models
        for dm in args.drift_models.split(","):
            dm = dm.strip()
            if dm and dm != "none":
                noise_configs.append((f"drift_{dm}", dm))
    elif args.drift_model and args.drift_model != "none":
        noise_configs.append((f"drift_{args.drift_model}", args.drift_model))
    elif args.drift_amp > 0:
        # Backward compat: drift_amp without drift_model defaults to sine
        noise_configs.append(("drift_sine", "sine"))

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    csv_rows = []

    # Load partial results for resume support
    partial_path = out_dir / "results_partial.json"
    completed_keys = set()
    if partial_path.exists():
        try:
            with open(partial_path) as f:
                partial_data = json.load(f)
            all_results = partial_data.get("results", [])
            csv_rows = partial_data.get("csv_rows", [])
            for r in all_results:
                completed_keys.add((r["p"], r["noise_type"]))
            print(f"Resuming: {len(completed_keys)} points already completed")
        except (json.JSONDecodeError, KeyError):
            print("Warning: corrupt partial results, starting fresh")
            all_results = []
            csv_rows = []

    total_combos = len(p_values) * len(noise_configs)
    combo_idx = 0
    t_total_start = time.time()

    for noise_label, drift_model in noise_configs:
        for p_val in p_values:
            combo_idx += 1

            # Skip already-completed points (resume support)
            if (float(p_val), noise_label) in completed_keys:
                print(f"\n[{combo_idx}/{total_combos}] p={p_val:.4f}, noise={noise_label} -- SKIPPED (already done)")
                continue

            if drift_model == "none":
                drift_str = "none"
            elif drift_model == "sine":
                drift_str = f"sine(amp={args.drift_amp})"
            elif drift_model == "ou":
                drift_str = f"ou(theta={args.ou_theta}, sigma={args.ou_sigma})"
            elif drift_model == "rtn":
                drift_str = f"rtn(delta={args.rtn_delta}, switch={args.rtn_switch})"
            else:
                drift_str = drift_model

            print(f"\n[{combo_idx}/{total_combos}] p={p_val:.4f}, noise={noise_label}, drift={drift_str}")

            # Generate data
            t0 = time.time()
            data = generate_code_capacity_data(
                hx, hz, lx, lz,
                shots=args.shots,
                p_base=p_val,
                eta=args.eta,
                drift_model=drift_model,
                drift_amp=args.drift_amp,
                drift_period=args.drift_period,
                ou_theta=args.ou_theta,
                ou_sigma=args.ou_sigma,
                rtn_delta=args.rtn_delta,
                rtn_switch=args.rtn_switch,
                seed=args.seed + combo_idx,
            )
            syndromes = data["syndromes"]
            observables = data["observables"]

            # Decode
            decoded = _decode_all_shots(
                syndromes, observables, hx, hz, lx, lz,
                p=p_val, eta=args.eta, device=device,
                gnn_model=gnn_model, use_bposd=args.bposd,
                use_mwpm=args.mwpm,
            )
            elapsed = time.time() - t0

            # Compute LER + CI for each decoder
            point_result = {
                "p": float(p_val),
                "noise_type": noise_label,
                "drift_model": drift_model,
                "shots": args.shots,
                "elapsed_s": round(elapsed, 1),
            }

            for dec_name, dec_data in decoded.items():
                errs = dec_data["errors"]
                shots = dec_data["shots"]
                ler = errs / shots if shots > 0 else 0.0
                ci_lo, ci_hi = wilson_score_interval_binom(errs, shots)

                dec_result = {
                    "errors": errs,
                    "ler": float(ler),
                    "ci_low": float(ci_lo),
                    "ci_high": float(ci_hi),
                }
                if "converged" in dec_data:
                    conv_rate = dec_data["converged"] / shots if shots > 0 else 0.0
                    dec_result["convergence"] = float(conv_rate)

                point_result[dec_name] = dec_result

                csv_rows.append({
                    "p": f"{p_val:.4f}",
                    "noise_type": noise_label,
                    "drift_model": drift_model,
                    "decoder": dec_name,
                    "ler": f"{ler:.6f}",
                    "ci_low": f"{ci_lo:.6f}",
                    "ci_high": f"{ci_hi:.6f}",
                    "errors": errs,
                    "shots": shots,
                    "convergence": f"{dec_result.get('convergence', 'N/A')}",
                })

            all_results.append(point_result)

            # Save partial results after each point (resume support)
            partial_data = {
                "results": all_results,
                "csv_rows": csv_rows,
            }
            with open(partial_path, "w") as f:
                json.dump(partial_data, f, indent=2)

            # Print summary
            for dec_name in decoded:
                r = point_result[dec_name]
                conv_str = f", conv={r['convergence']:.1%}" if "convergence" in r else ""
                print(f"  {dec_name}: LER={r['ler']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]{conv_str}")
            print(f"  ({elapsed:.1f}s)")

    total_elapsed = time.time() - t_total_start
    print(f"\nTotal sweep time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Save JSON
    results_json = {
        "config": {
            "code": args.code,
            "p_min": args.p_min,
            "p_max": args.p_max,
            "num_points": args.num_points,
            "shots": args.shots,
            "eta": args.eta,
            "drift_model": args.drift_model,
            "drift_models": args.drift_models,
            "drift_amp": args.drift_amp,
            "drift_period": args.drift_period,
            "ou_theta": args.ou_theta,
            "ou_sigma": args.ou_sigma,
            "rtn_delta": args.rtn_delta,
            "rtn_switch": args.rtn_switch,
            "gnn_model": args.gnn_model,
            "bposd": args.bposd,
            "mwpm": args.mwpm,
        },
        "code": {"n": int(n), "mx": int(mx), "mz": int(mz), "k": int(lx.shape[0])},
        "results": all_results,
        "total_elapsed_s": round(total_elapsed, 1),
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved {json_path}")

    # Save CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "p", "noise_type", "drift_model", "decoder", "ler", "ci_low", "ci_high",
            "errors", "shots", "convergence",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {csv_path}")

    # Generate plots
    print("\nGenerating plots...")
    _make_plots(
        all_results, out_dir,
        has_gnn=(gnn_model is not None),
        has_bposd=args.bposd,
        has_mwpm=args.mwpm,
    )

    # Clean up partial file after successful completion
    if partial_path.exists():
        partial_path.unlink()

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
