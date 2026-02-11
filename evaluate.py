"""Evaluate BP decoder and GNN-assisted BP on test data.

Compares logical error rates (LER) with Wilson score confidence intervals.

Usage:
    python -m gnn_pipeline.evaluate --test_npz "data/toy.npz" --gnn_model "runs/selfsup_test/best_model.pt" --out_dir "runs/eval_test"
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from gnn_pipeline.bp_decoder import MinSumBPDecoder
from gnn_pipeline.dataset import _load_npz, _parse_meta
from gnn_pipeline.gnn_model import TannerGNN
from gnn_pipeline.tanner_graph import build_tanner_graph


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


def run_bp_decoder(
    syndrome: np.ndarray,
    pcm: np.ndarray,
    channel_llr: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, bool]:
    """Run BP decoder on a single syndrome.

    Args:
        syndrome: detector syndrome (m,) where m = mx+mz
        pcm: combined parity check matrix vstack([hx, hz]), shape (m, n)
        channel_llr: initial channel LLRs (n,)
        device: torch device

    Returns:
        (qubit_errors, converged)
    """
    decoder = MinSumBPDecoder(pcm, max_iter=20, alpha=0.8, clamp_llr=20.0)
    decoder = decoder.to(device)

    syn_t = torch.from_numpy(syndrome[np.newaxis, :]).float().to(device)
    llr_t = torch.from_numpy(channel_llr[np.newaxis, :]).float().to(device)

    marginals, hard_decision, converged = decoder(syn_t, llr_t)

    return hard_decision[0].cpu().numpy(), bool(converged[0])


def run_gnn_bp_decoder(
    syndrome: np.ndarray,
    hx: np.ndarray,
    hz: np.ndarray,
    pcm: np.ndarray,
    channel_llr: np.ndarray,
    gnn_model: nn.Module,
    device: torch.device,
) -> Tuple[np.ndarray, bool]:
    """Run GNN-assisted BP decoder.

    Args:
        syndrome: detector syndrome
        hx, hz: parity check matrices (for Tanner graph)
        pcm: combined PCM vstack([hx, hz])
        channel_llr: initial channel LLRs
        gnn_model: trained GNN model
        device: torch device

    Returns:
        (qubit_errors, converged)
    """
    decoder = MinSumBPDecoder(pcm, max_iter=20, alpha=0.8, clamp_llr=20.0)
    decoder = decoder.to(device)

    # Build Tanner graph
    node_type_np, edge_index_np, edge_type_np = build_tanner_graph(hx, hz)
    node_type_t = torch.from_numpy(node_type_np).long()
    edge_index_t = torch.from_numpy(edge_index_np).long()
    edge_type_t = torch.from_numpy(edge_type_np).long()

    n_data = int((node_type_np == 0).sum())
    mx, mz = hx.shape[0], hz.shape[0]

    # Build node features
    x_feat = torch.zeros(n_data + mx + mz, 4, dtype=torch.float32)
    x_feat[:n_data, 0] = torch.from_numpy(channel_llr).float()
    x_feat[:n_data, 1] = 1.0
    x_feat[n_data:n_data+mx, 2] = 1.0
    x_feat[n_data+mx:, 3] = 1.0

    # Create Data object
    data_obj = Data(
        x=x_feat.to(device),
        edge_index=edge_index_t.to(device),
        edge_type=edge_type_t.to(device),
        node_type=node_type_t.to(device),
        channel_llr=torch.from_numpy(channel_llr).float().to(device),
    )

    # Get GNN corrections
    with torch.no_grad():
        gnn_model.eval()
        llr_corrections = gnn_model(data_obj)
    llr_corrections = torch.clamp(llr_corrections, -20.0, 20.0)

    corrected_llr = (
        torch.from_numpy(channel_llr).float().to(device) + llr_corrections
    )

    # Run decoder with corrected LLRs
    syn_t = torch.from_numpy(syndrome[np.newaxis, :]).float().to(device)

    marginals, hard_decision, converged = decoder(syn_t, corrected_llr[np.newaxis, :])

    return hard_decision[0].cpu().numpy(), bool(converged[0])


def _check_logical_error(
    qubit_errors: np.ndarray,
    lx: np.ndarray,
    lz: np.ndarray,
    observable: np.ndarray,
) -> bool:
    """Check if a decoded correction results in a logical error.

    Compares the correction's effect on logical operators against the
    actual observable flips from stim.
    """
    correction_obs_x = (lx @ qubit_errors) % 2
    correction_obs_z = (lz @ qubit_errors) % 2
    predicted_obs = np.concatenate([correction_obs_x, correction_obs_z])

    # Compare with actual observable flips (trim to matching length)
    n_obs = min(len(predicted_obs), len(observable))
    return bool(np.any(predicted_obs[:n_obs] != observable[:n_obs]))


def main(argv: List[str] | None = None) -> int:
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate BP vs GNN-assisted BP")
    parser.add_argument("--test_npz", type=str, required=True)
    parser.add_argument("--gnn_model", type=str, required=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    print(f"Loading test data from {args.test_npz}...")
    npz_data = _load_npz(args.test_npz)
    meta = _parse_meta(npz_data)

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

    # Combined PCM for BP
    pcm = np.vstack([hx, hz])

    print(f"Code: n={n}, mx={mx}, mz={mz}, shots={shots}")
    print(f"Detectors per shot: {num_detectors}, checks: {total_checks}")

    # Channel LLR
    p = float(meta.get("p", 0.01))
    p_clamped = max(min(p, 1.0 - 1e-7), 1e-7)
    channel_llr_val = float(math.log((1.0 - p_clamped) / p_clamped))
    channel_llr = np.full(n, channel_llr_val, dtype=np.float32)

    # Compute number of detector rounds
    if total_checks > 0:
        num_rounds = num_detectors // total_checks
    else:
        num_rounds = 1

    # Load GNN if provided
    gnn_model_inst = None
    if args.gnn_model:
        print(f"Loading GNN model from {args.gnn_model}...")
        gnn_model_inst = TannerGNN()  # use default params
        gnn_model_inst.load_state_dict(
            torch.load(args.gnn_model, map_location=device, weights_only=True)
        )
        gnn_model_inst = gnn_model_inst.to(device)
        gnn_model_inst.eval()

    # Run evaluation
    print("\nEvaluating...")

    bp_errors = 0
    gnn_bp_errors = 0
    bp_converged = 0
    gnn_bp_converged = 0

    for shot_idx in range(shots):
        det_row = syndromes[shot_idx]  # (num_detectors,)
        observable = observables[shot_idx]

        # Use last round's cumulative syndrome for code-capacity-like decoding
        if num_rounds > 1:
            det_3d = det_row.reshape(num_rounds, total_checks)
            # Cumulative XOR across rounds to get effective syndrome
            syndrome = det_3d.sum(axis=0) % 2
        else:
            syndrome = det_row[:total_checks]

        syndrome = syndrome.astype(np.float32)

        # BP
        qubit_errors, converged = run_bp_decoder(syndrome, pcm, channel_llr, device)
        bp_converged += int(converged)
        bp_logical_error = _check_logical_error(qubit_errors, lx, lz, observable)
        bp_errors += int(bp_logical_error)

        # GNN-BP
        if gnn_model_inst:
            qubit_errors, converged = run_gnn_bp_decoder(
                syndrome, hx, hz, pcm, channel_llr, gnn_model_inst, device
            )
            gnn_bp_converged += int(converged)
            gnn_bp_logical_error = _check_logical_error(qubit_errors, lx, lz, observable)
            gnn_bp_errors += int(gnn_bp_logical_error)

        if (shot_idx + 1) % max(1, shots // 10) == 0:
            print(f"  Processed {shot_idx + 1}/{shots} shots")

    # Compute LER with confidence intervals
    bp_ler = bp_errors / shots if shots > 0 else 0.0
    bp_ler_low, bp_ler_high = wilson_score_interval_binom(bp_errors, shots)

    print(f"\n=== Results ===")
    print(f"BP Decoder:")
    print(f"  Logical Error Rate: {bp_ler:.6f}")
    print(f"  95% CI: [{bp_ler_low:.6f}, {bp_ler_high:.6f}]")
    print(f"  Convergence: {bp_converged}/{shots} ({100*bp_converged/shots:.1f}%)")

    if gnn_model_inst:
        gnn_bp_ler = gnn_bp_errors / shots if shots > 0 else 0.0
        gnn_bp_ler_low, gnn_bp_ler_high = wilson_score_interval_binom(gnn_bp_errors, shots)

        print(f"\nGNN-Assisted BP:")
        print(f"  Logical Error Rate: {gnn_bp_ler:.6f}")
        print(f"  95% CI: [{gnn_bp_ler_low:.6f}, {gnn_bp_ler_high:.6f}]")
        print(f"  Convergence: {gnn_bp_converged}/{shots} ({100*gnn_bp_converged/shots:.1f}%)")

        improvement = (bp_ler - gnn_bp_ler) / bp_ler if bp_ler > 0 else 0.0
        print(f"\nImprovement: {100*improvement:.1f}%")

    # Save results
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "code": {"n": int(n), "mx": int(mx), "mz": int(mz)},
        "test_shots": shots,
        "bp": {
            "errors": int(bp_errors),
            "ler": float(bp_ler),
            "ler_ci_low": float(bp_ler_low),
            "ler_ci_high": float(bp_ler_high),
            "convergence_rate": float(bp_converged / shots),
        },
    }

    if gnn_model_inst:
        results["gnn_bp"] = {
            "errors": int(gnn_bp_errors),
            "ler": float(gnn_bp_ler),
            "ler_ci_low": float(gnn_bp_ler_low),
            "ler_ci_high": float(gnn_bp_ler_high),
            "convergence_rate": float(gnn_bp_converged / shots),
        }
        results["improvement_pct"] = float(100 * improvement)

    results_path = out_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
