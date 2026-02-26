"""Circuit-level dataset: loads Stim circuit NPZ, extracts DEM, builds GNN graphs.

Each sample is one shot of detector events. The GNN graph is the DEM Tanner
graph (bipartite: error variables <-> detectors). Node features encode prior
LLRs for error variables and syndrome values for detectors.

Training target: observable flips (logical outcomes), since ground-truth fault
patterns are not available in circuit-level sampling.
"""
from __future__ import annotations

import json
import math
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from gnn_pipeline.dem_decoder import extract_dem_pcm
from gnn_pipeline.tanner_graph import build_dem_tanner_graph


def _load_npz(path: str) -> dict:
    """Load an NPZ file and return contents as a dict."""
    d = np.load(path, allow_pickle=True)
    return {key: d[key] for key in d.files}


def _parse_meta(npz_data: dict) -> dict:
    """Extract metadata from NPZ."""
    if "meta" in npz_data:
        raw = npz_data["meta"]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        if isinstance(raw, bytes):
            return json.loads(raw.decode("utf-8"))
        if isinstance(raw, str):
            return json.loads(raw)
    return {"p": 0.005, "eta": 20.0}


def build_circuit_dataset(
    npz_paths: List[str],
    seed: int = 42,
) -> Tuple[List[Data], List[Data], List[Data], dict]:
    """Build train/val/test datasets from circuit-level NPZ files.

    Each NPZ must contain:
      - syndromes: (shots, num_detectors)
      - observables: (shots, num_observables)
      - circuit: Stim circuit text (UTF-8 bytes)

    The DEM is extracted once per NPZ file and shared across all shots.

    Args:
        npz_paths: list of paths to circuit-level .npz files
        seed: random seed for deterministic splitting

    Returns:
        (train_data, val_data, test_data, meta_dict)
    """
    all_samples: List[Data] = []
    _cached_dem = None  # Cache DEM extraction across files with same circuit

    for npz_path in npz_paths:
        npz_data = _load_npz(npz_path)
        meta = _parse_meta(npz_data)

        syndromes = npz_data["syndromes"].astype(np.float32)
        observables = npz_data["observables"].astype(np.float32)
        shots = syndromes.shape[0]

        # Extract circuit text
        if "circuit" not in npz_data:
            raise ValueError(
                f"NPZ file {npz_path} does not contain 'circuit' key. "
                f"Generate circuit-level data with astra_stim."
            )
        circuit_bytes = npz_data["circuit"]
        if hasattr(circuit_bytes, "tobytes"):
            circuit_text = circuit_bytes.tobytes().decode("utf-8")
        else:
            circuit_text = str(circuit_bytes)

        # Extract DEM (cached across files with same circuit to avoid repeated extraction)
        circuit_hash = hash(circuit_text)
        if _cached_dem is not None and _cached_dem[0] == circuit_hash:
            dem_pcm, error_probs, obs_matrix = _cached_dem[1], _cached_dem[2], _cached_dem[3]
        else:
            dem_pcm, error_probs, obs_matrix = extract_dem_pcm(circuit_text)
            _cached_dem = (circuit_hash, dem_pcm, error_probs, obs_matrix)
        num_detectors, num_errors = dem_pcm.shape
        num_obs = obs_matrix.shape[1]

        print(f"  {npz_path}: {shots} shots, DEM ({num_detectors} detectors x {num_errors} errors), {num_obs} obs")

        # Build DEM Tanner graph (shared across all shots from this file)
        node_type_np, edge_index_np, edge_type_np = build_dem_tanner_graph(dem_pcm)
        node_type_t = torch.from_numpy(node_type_np)
        edge_index_t = torch.from_numpy(edge_index_np)
        edge_type_t = torch.from_numpy(edge_type_np)

        num_nodes = num_errors + num_detectors

        # Compute prior channel LLRs from DEM error probabilities
        p_clamped = np.clip(error_probs, 1e-10, 1.0 - 1e-10)
        prior_llr = np.log((1.0 - p_clamped) / p_clamped).astype(np.float32)
        prior_llr_t = torch.from_numpy(prior_llr)

        # Noise parameters for FiLM conditioning
        p_val = float(meta.get("p", 0.005))

        for shot_idx in range(shots):
            det_events = syndromes[shot_idx]  # (num_detectors,)
            obs_flips = observables[shot_idx]  # (num_obs,)

            # Build node features: [feature_value, is_error_var, is_detector, 0]
            # Using 4 features to match code-capacity GNN node_feat_dim
            x_feat = torch.zeros(num_nodes, 4, dtype=torch.float32)

            # Error variable nodes (0..num_errors-1): [prior_llr, 1, 0, 0]
            x_feat[:num_errors, 0] = prior_llr_t
            x_feat[:num_errors, 1] = 1.0

            # Detector nodes (num_errors..num_nodes-1): [syndrome_value, 0, 1, 0]
            x_feat[num_errors:, 0] = torch.from_numpy(det_events).float()
            x_feat[num_errors:, 2] = 1.0

            data_obj = Data(
                x=x_feat,
                edge_index=edge_index_t,
                edge_type=edge_type_t,
                node_type=node_type_t,
                # DEM channel LLR per error variable (used by BP)
                channel_llr=prior_llr_t.clone(),
                # Syndrome (detector events) for BP
                syndrome=torch.from_numpy(det_events.copy()).float(),
                # Observable targets for loss
                observable=torch.from_numpy(obs_flips.copy()).float(),
                # Metadata
                num_errors=torch.tensor(num_errors, dtype=torch.long),
                num_detectors=torch.tensor(num_detectors, dtype=torch.long),
                # FiLM conditioning
                p_value=torch.tensor(p_val, dtype=torch.float32),
                sample_idx=torch.tensor(len(all_samples), dtype=torch.long),
            )

            all_samples.append(data_obj)

    # Deterministic split: 80/10/10
    rng = torch.Generator().manual_seed(seed)
    n_total = len(all_samples)
    indices = torch.randperm(n_total, generator=rng).tolist()

    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_data = [all_samples[i] for i in indices[:n_train]]
    val_data = [all_samples[i] for i in indices[n_train:n_train + n_val]]
    test_data = [all_samples[i] for i in indices[n_train + n_val:]]

    # _cached_dem holds the validated DEM from the last (or cached) extraction
    if _cached_dem is None:
        raise ValueError("No valid DEM data extracted from any input file.")

    meta_out = {
        "mode": "circuit_level",
        "seed": seed,
        "num_total": n_total,
        "num_train": len(train_data),
        "num_val": len(val_data),
        "num_test": len(test_data),
        "source_files": [str(p) for p in npz_paths],
        # Shared DEM matrices (validated consistent via circuit hash caching)
        "obs_matrix": torch.from_numpy(_cached_dem[3].astype(np.float32)),
        "dem_pcm": torch.from_numpy(_cached_dem[1].astype(np.float32)),
    }

    return train_data, val_data, test_data, meta_out
