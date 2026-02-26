"""Convert NPZ syndrome data to torch_geometric Data objects with sliding windows.

Each sample is a windowed view of W consecutive detector rounds, with the
next round's syndrome as the self-supervised target.

Supports online data regeneration (Astra-style) via OnlineCodeCapDataset
which regenerates syndrome data every epoch for infinite diversity.
"""
from __future__ import annotations

import glob
import json
import math
import pathlib
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
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

    # Validate consistent code parameters across all files
    _ref_shape = None
    for _path in npz_paths:
        _d = _load_npz(_path)
        _hx = _d.get("hx", np.zeros((0, 0)))
        _hz = _d.get("hz", np.zeros((0, 0)))
        _shape = (_hx.shape, _hz.shape)
        if _ref_shape is None:
            _ref_shape = _shape
        elif _shape != _ref_shape:
            raise ValueError(
                f"Inconsistent code parameters across NPZ files: "
                f"{npz_paths[0]} has hx={_ref_shape[0]}, hz={_ref_shape[1]} "
                f"but {_path} has hx={_shape[0]}, hz={_shape[1]}"
            )

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

        # Supervised mode: load error vectors and per-shot p_values
        if mode == "supervised":
            if "z_errors" not in npz_data or "x_errors" not in npz_data:
                raise ValueError(
                    f"Supervised mode requires z_errors/x_errors in NPZ. "
                    f"Regenerate data with updated generate_codecap.py: {npz_path}"
                )
            z_errors_all = npz_data["z_errors"].astype(np.float32)  # (shots, n)
            x_errors_all = npz_data["x_errors"].astype(np.float32)  # (shots, n)
            p_values_all = npz_data.get("p_values", None)
            if p_values_all is not None:
                p_values_all = p_values_all.astype(np.float32)

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

        # Channel LLR from noise parameters (per-Pauli)
        p = float(meta.get("p", 0.01))
        eta = float(meta.get("eta", 20.0))
        p_clamped = max(min(p, 1.0 - 1e-7), 1e-7)

        # Per-Pauli channel LLRs
        pz_base = max(min(p_clamped * eta / (eta + 1), 1.0 - 1e-7), 1e-7)
        px_base = max(min(p_clamped / (eta + 1), 1.0 - 1e-7), 1e-7)
        llr_z_base = float(math.log((1.0 - pz_base) / pz_base))
        llr_x_base = float(math.log((1.0 - px_base) / px_base))
        channel_llr_z_base = torch.full((n,), llr_z_base, dtype=torch.float32)
        channel_llr_x_base = torch.full((n,), llr_x_base, dtype=torch.float32)

        # GNN input feature: average of per-Pauli LLRs (matches evaluate.py convention)
        avg_llr_base = (llr_z_base + llr_x_base) / 2.0
        channel_llr = torch.full((n,), avg_llr_base, dtype=torch.float32)

        # Reshape syndromes into (shots, num_det_rounds, total_checks)
        syn_3d = syndromes.reshape(shots, num_det_rounds, total_checks)

        # Pre-compute vectorized per-shot LLRs (avoids Python math.log in inner loop)
        if mode == "supervised" and p_values_all is not None:
            p_all = np.clip(p_values_all, 1e-7, 1.0 - 1e-7)
            pz_all = np.clip(p_all * eta / (eta + 1), 1e-7, 1.0 - 1e-7)
            px_all = np.clip(p_all / (eta + 1), 1e-7, 1.0 - 1e-7)
            llr_z_vals = np.log((1.0 - pz_all) / pz_all).astype(np.float32)  # (shots,)
            llr_x_vals = np.log((1.0 - px_all) / px_all).astype(np.float32)  # (shots,)
            llr_vals = ((llr_z_vals + llr_x_vals) / 2.0)                     # avg per-Pauli (matches eval)
            _precomputed_drift = True
        else:
            _precomputed_drift = False

        for shot_idx in range(shots):
            # Use pre-computed vectorized LLRs (drift-aware)
            if _precomputed_drift:
                shot_llr_z = torch.full((n,), llr_z_vals[shot_idx], dtype=torch.float32)
                shot_llr_x = torch.full((n,), llr_x_vals[shot_idx], dtype=torch.float32)
                shot_channel_llr = torch.full((n,), llr_vals[shot_idx], dtype=torch.float32)
            else:
                shot_llr_z = channel_llr_z_base
                shot_llr_x = channel_llr_x_base
                shot_channel_llr = channel_llr

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
                x_feat[:n, 0] = shot_channel_llr
                x_feat[:n, 1] = 1.0

                # XOR-collapse syndrome across window rounds (matches evaluate.py)
                window_t = torch.from_numpy(window)
                collapsed_syn = window_t.sum(dim=0) % 2  # (total_checks,)

                # X-checks: [collapsed_syndrome, 0, 1, 0]
                x_feat[n:n+mx, 0] = collapsed_syn[:mx]
                x_feat[n:n+mx, 2] = 1.0

                # Z-checks: [collapsed_syndrome, 0, 0, 1]
                x_feat[n+mx:, 0] = collapsed_syn[mx:]
                x_feat[n+mx:, 3] = 1.0

                data_obj = Data(
                    x=x_feat,
                    edge_index=edge_index_t,
                    edge_type=edge_type_t,
                    node_type=node_type_t,
                    channel_llr=shot_channel_llr.clone(),
                    window_syndromes=window_t,
                    target_syndrome=torch.from_numpy(target_syn.copy()).float(),
                    observable=torch.from_numpy(observables[shot_idx].copy()).float(),
                    sample_idx=torch.tensor(len(all_samples), dtype=torch.long),
                )

                # Attach per-shot p_value for FiLM conditioning
                if _precomputed_drift:
                    data_obj.p_value = torch.tensor(float(p_values_all[shot_idx]), dtype=torch.float32)
                else:
                    data_obj.p_value = torch.tensor(p_clamped, dtype=torch.float32)

                # Supervised mode: attach error vectors and per-Pauli LLRs
                if mode == "supervised":
                    data_obj.z_error = torch.from_numpy(z_errors_all[shot_idx].copy()).float()
                    data_obj.x_error = torch.from_numpy(x_errors_all[shot_idx].copy()).float()
                    data_obj.channel_llr_z = shot_llr_z.clone()
                    data_obj.channel_llr_x = shot_llr_x.clone()

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


class OnlineCodeCapDataset(Dataset):
    """Online data regeneration dataset (Astra-style).

    Instead of loading fixed data from disk, this dataset generates fresh
    syndrome/error data every epoch by sampling from the noise model.
    This provides infinite training diversity and eliminates overfitting
    to a fixed dataset.

    Args:
        hx: (mx, n) X parity check matrix
        hz: (mz, n) Z parity check matrix
        lx: (kx, n) X logical operators
        lz: (kz, n) Z logical operators
        p_base: base physical error rate
        eta: noise bias (Z/X ratio)
        samples_per_epoch: number of samples generated per epoch
        p_range: optional (p_min, p_max) for uniform p sampling per shot
        seed: random seed (incremented each epoch)
    """

    def __init__(
        self,
        hx: np.ndarray,
        hz: np.ndarray,
        lx: np.ndarray,
        lz: np.ndarray,
        p_base: float = 0.02,
        eta: float = 20.0,
        samples_per_epoch: int = 5000,
        p_range: Optional[Tuple[float, float]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.hx = np.asarray(hx, dtype=np.uint8)
        self.hz = np.asarray(hz, dtype=np.uint8)
        self.lx = np.asarray(lx, dtype=np.uint8)
        self.lz = np.asarray(lz, dtype=np.uint8)
        self.mx, self.n = self.hx.shape
        self.mz = self.hz.shape[0]
        self.p_base = p_base
        self.eta = eta
        self.samples_per_epoch = samples_per_epoch
        self.p_range = p_range
        self.base_seed = seed
        self.epoch = 0

        # Pre-build Tanner graph (shared across all samples)
        node_type_np, edge_index_np, edge_type_np = build_tanner_graph(hx, hz)
        self.node_type_t = torch.from_numpy(node_type_np)
        self.edge_index_t = torch.from_numpy(edge_index_np)
        self.edge_type_t = torch.from_numpy(edge_type_np)

        # Generate first epoch's data
        self._regenerate()

    def set_epoch(self, epoch: int):
        """Call at the start of each epoch to regenerate data."""
        if epoch != self.epoch:
            self.epoch = epoch
            self._regenerate()

    def _regenerate(self):
        """Generate fresh syndrome and error data for this epoch."""
        rng = np.random.RandomState(self.base_seed + self.epoch * 1000)
        n = self.n
        mx, mz = self.mx, self.mz
        S = self.samples_per_epoch
        eta = self.eta

        # Per-shot p values (uniform range or fixed)
        if self.p_range is not None:
            p_vals = rng.uniform(self.p_range[0], self.p_range[1], size=S).astype(np.float64)
        else:
            p_vals = np.full(S, self.p_base, dtype=np.float64)

        p_vals = np.clip(p_vals, 1e-7, 1.0 - 1e-7)
        pz_vals = p_vals * eta / (eta + 1)
        px_vals = p_vals / (eta + 1)
        pz_vals = np.clip(pz_vals, 1e-7, 1.0 - 1e-7)
        px_vals = np.clip(px_vals, 1e-7, 1.0 - 1e-7)

        # Sample errors
        z_errors = (rng.random((S, n)) < pz_vals[:, None]).astype(np.float32)
        x_errors = (rng.random((S, n)) < px_vals[:, None]).astype(np.float32)

        # Compute syndromes: hx @ z_error = x_syndrome, hz @ x_error = z_syndrome
        x_syn = (z_errors @ self.hx.T) % 2  # (S, mx)
        z_syn = (x_errors @ self.hz.T) % 2  # (S, mz)

        # Compute observables
        obs_z = (z_errors @ self.lx.T) % 2  # (S, kx)
        obs_x = (x_errors @ self.lz.T) % 2  # (S, kz)
        observables = np.concatenate([obs_z, obs_x], axis=1).astype(np.float32)

        # Compute LLRs
        llr_z_vals = np.log((1.0 - pz_vals) / pz_vals).astype(np.float32)
        llr_x_vals = np.log((1.0 - px_vals) / px_vals).astype(np.float32)
        avg_llr_vals = (llr_z_vals + llr_x_vals) / 2.0

        # Store for __getitem__
        self._z_errors = z_errors
        self._x_errors = x_errors
        self._x_syn = x_syn.astype(np.float32)
        self._z_syn = z_syn.astype(np.float32)
        self._observables = observables
        self._llr_z = llr_z_vals
        self._llr_x = llr_x_vals
        self._avg_llr = avg_llr_vals
        self._p_vals = p_vals.astype(np.float32)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Data:
        n, mx, mz = self.n, self.mx, self.mz
        num_nodes = n + mx + mz

        x_feat = torch.zeros(num_nodes, 4, dtype=torch.float32)
        x_feat[:n, 0] = float(self._avg_llr[idx])
        x_feat[:n, 1] = 1.0
        x_feat[n:n+mx, 0] = torch.from_numpy(self._x_syn[idx]).float()
        x_feat[n:n+mx, 2] = 1.0
        x_feat[n+mx:, 0] = torch.from_numpy(self._z_syn[idx]).float()
        x_feat[n+mx:, 3] = 1.0

        # Syndrome for BP: concat x_syn and z_syn
        target_syn = np.concatenate([self._x_syn[idx], self._z_syn[idx]])

        data_obj = Data(
            x=x_feat,
            edge_index=self.edge_index_t,
            edge_type=self.edge_type_t,
            node_type=self.node_type_t,
            channel_llr=torch.full((n,), float(self._avg_llr[idx]), dtype=torch.float32),
            channel_llr_z=torch.full((n,), float(self._llr_z[idx]), dtype=torch.float32),
            channel_llr_x=torch.full((n,), float(self._llr_x[idx]), dtype=torch.float32),
            target_syndrome=torch.from_numpy(target_syn.copy()).float(),
            observable=torch.from_numpy(self._observables[idx].copy()).float(),
            z_error=torch.from_numpy(self._z_errors[idx].copy()).float(),
            x_error=torch.from_numpy(self._x_errors[idx].copy()).float(),
            p_value=torch.tensor(self._p_vals[idx], dtype=torch.float32),
            sample_idx=torch.tensor(idx, dtype=torch.long),
        )
        return data_obj
