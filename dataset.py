"""Convert NPZ syndrome data to torch_geometric Data objects with sliding windows.

Each sample is a windowed view of W consecutive detector rounds, with the
next round's syndrome as the self-supervised target.
"""
from __future__ import annotations

import glob
import json
import math
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from gnn_pipeline.tanner_graph import build_tanner_graph


def _load_npz(path: str) -> dict:
    """Load an NPZ file and return contents as a dict."""
    d = np.load(path, allow_pickle=True)
    result = {}
    for key in d.files:
        result[key] = d[key]
    return result


def _parse_meta(npz_data: dict) -> dict:
    """Extract metadata from NPZ, handling both old and new formats."""
    if "meta" in npz_data:
        raw = npz_data["meta"]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, bytes):
            return json.loads(raw.decode("utf-8"))
        if isinstance(raw, str):
            return json.loads(raw)
    # Fallback: reconstruct from available data
    hx = npz_data.get("hx", np.zeros((0, 0)))
    hz = npz_data.get("hz", np.zeros((0, 0)))
    return {
        "n": int(hx.shape[1]) if hx.ndim == 2 else 0,
        "mx": int(hx.shape[0]) if hx.ndim == 2 else 0,
        "mz": int(hz.shape[0]) if hz.ndim == 2 else 0,
        "p": 0.01,
        "eta": 20.0,
    }


def build_graph_dataset(
    npz_paths: List[str],
    W: int = 4,
    mode: str = "selfsup",
    seed: int = 42,
) -> Tuple[List[Data], List[Data], List[Data], dict]:
    """Build train/val/test datasets from NPZ files.

    Args:
        npz_paths: list of paths to .npz files
        W: window width (number of detector rounds per sample)
        mode: "selfsup" or "supervised"
        seed: random seed for deterministic splitting

    Returns:
        (train_data, val_data, test_data, meta_dict)
    """
    all_samples: List[Data] = []

    for npz_path in npz_paths:
        npz_data = _load_npz(npz_path)
        meta = _parse_meta(npz_data)

        syndromes = npz_data["syndromes"].astype(np.float32)  # (shots, num_detectors)
        observables = npz_data["observables"].astype(np.float32)  # (shots, k)
        hx = npz_data["hx"].astype(np.uint8)
        hz = npz_data["hz"].astype(np.uint8)

        mx, n = hx.shape
        mz = hz.shape[0]
        total_checks = mx + mz
        shots = syndromes.shape[0]
        num_detectors = syndromes.shape[1]

        # Determine number of detector rounds
        if total_checks > 0:
            num_det_rounds = num_detectors // total_checks
        else:
            continue

        if num_det_rounds < W + 1:
            # Not enough rounds for windowing with a target
            # Use what we have
            W_eff = max(1, num_det_rounds - 1)
        else:
            W_eff = W

        # Build Tanner graph (shared for all samples from this code)
        node_type_np, edge_index_np, edge_type_np = build_tanner_graph(hx, hz)
        node_type_t = torch.from_numpy(node_type_np)
        edge_index_t = torch.from_numpy(edge_index_np)
        edge_type_t = torch.from_numpy(edge_type_np)

        num_nodes = n + mx + mz

        # Channel LLR from noise parameters
        p = float(meta.get("p", 0.01))
        p_clamped = max(min(p, 1.0 - 1e-7), 1e-7)
        channel_llr_val = float(math.log((1.0 - p_clamped) / p_clamped))
        channel_llr = torch.full((n,), channel_llr_val, dtype=torch.float32)

        # Reshape syndromes into (shots, num_det_rounds, total_checks)
        syn_3d = syndromes.reshape(shots, num_det_rounds, total_checks)

        for shot_idx in range(shots):
            num_windows = max(1, num_det_rounds - W_eff)
            for w_start in range(num_windows):
                w_end = w_start + W_eff
                if w_end > num_det_rounds:
                    break

                window = syn_3d[shot_idx, w_start:w_end, :]  # (W_eff, total_checks)

                # Target: next round's syndrome (or last round if at boundary)
                if w_end < num_det_rounds:
                    target_syn = syn_3d[shot_idx, w_end, :]  # (total_checks,)
                else:
                    target_syn = syn_3d[shot_idx, w_end - 1, :]

                # Build node features
                x_feat = torch.zeros(num_nodes, 4, dtype=torch.float32)

                # Data qubits: [channel_llr, 1, 0, 0]
                x_feat[:n, 0] = channel_llr
                x_feat[:n, 1] = 1.0

                # X-checks: [mean_syndrome, 0, 1, 0]
                window_t = torch.from_numpy(window)
                x_feat[n:n+mx, 0] = window_t[:, :mx].mean(dim=0)
                x_feat[n:n+mx, 2] = 1.0

                # Z-checks: [mean_syndrome, 0, 0, 1]
                x_feat[n+mx:, 0] = window_t[:, mx:].mean(dim=0)
                x_feat[n+mx:, 3] = 1.0

                data_obj = Data(
                    x=x_feat,
                    edge_index=edge_index_t.clone(),
                    edge_type=edge_type_t.clone(),
                    node_type=node_type_t.clone(),
                    channel_llr=channel_llr.clone(),
                    window_syndromes=window_t,
                    target_syndrome=torch.from_numpy(target_syn.copy()).float(),
                    observable=torch.from_numpy(observables[shot_idx].copy()).float(),
                    sample_idx=torch.tensor(len(all_samples), dtype=torch.long),
                )
                all_samples.append(data_obj)

    # Deterministic split
    rng = torch.Generator().manual_seed(seed)
    n_total = len(all_samples)
    indices = torch.randperm(n_total, generator=rng).tolist()

    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_data = [all_samples[i] for i in train_idx]
    val_data = [all_samples[i] for i in val_idx]
    test_data = [all_samples[i] for i in test_idx]

    meta_out = {
        "W": W,
        "mode": mode,
        "seed": seed,
        "num_total": n_total,
        "num_train": len(train_data),
        "num_val": len(val_data),
        "num_test": len(test_data),
        "source_files": [str(p) for p in npz_paths],
    }

    return train_data, val_data, test_data, meta_out
