"""Build heterogeneous Tanner graph from CSS parity-check matrices Hx, Hz.

Node types:
  0 = data qubit  (n nodes)
  1 = X-check     (mx nodes)
  2 = Z-check     (mz nodes)

Edge types:
  0 = data <-> X-check  (from nonzeros of Hx)
  1 = data <-> Z-check  (from nonzeros of Hz)

All edges are stored bidirectionally (both directions in edge_index).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def build_tanner_graph(
    hx: np.ndarray,
    hz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a heterogeneous Tanner graph from CSS parity-check matrices.

    Args:
        hx: shape (mx, n) uint8 — X parity-check matrix
        hz: shape (mz, n) uint8 — Z parity-check matrix

    Returns:
        node_type:  int64 (num_nodes,) — 0=data, 1=X-check, 2=Z-check
        edge_index: int64 (2, num_edges) — COO format, bidirectional
        edge_type:  int64 (num_edges,) — 0=X-edge, 1=Z-edge
    """
    hx = np.asarray(hx, dtype=np.uint8)
    hz = np.asarray(hz, dtype=np.uint8)

    if hx.ndim != 2 or hz.ndim != 2:
        raise ValueError("Hx and Hz must be 2D matrices")
    mx, n = hx.shape
    mz, n2 = hz.shape
    if n != n2:
        raise ValueError(f"Hx has {n} columns but Hz has {n2}")

    num_nodes = n + mx + mz

    # Node types: data=0, X-check=1, Z-check=2
    node_type = np.zeros(num_nodes, dtype=np.int64)
    node_type[n : n + mx] = 1
    node_type[n + mx :] = 2

    # Build edges from Hx (X-checks connect to data qubits)
    x_checks, x_data = np.nonzero(hx)
    # X-check node indices are offset by n
    x_check_nodes = x_checks.astype(np.int64) + n
    x_data_nodes = x_data.astype(np.int64)

    # Build edges from Hz (Z-checks connect to data qubits)
    z_checks, z_data = np.nonzero(hz)
    # Z-check node indices are offset by n + mx
    z_check_nodes = z_checks.astype(np.int64) + n + mx
    z_data_nodes = z_data.astype(np.int64)

    # Bidirectional: data->check and check->data
    src_x = np.concatenate([x_data_nodes, x_check_nodes])
    dst_x = np.concatenate([x_check_nodes, x_data_nodes])
    type_x = np.zeros(len(src_x), dtype=np.int64)

    src_z = np.concatenate([z_data_nodes, z_check_nodes])
    dst_z = np.concatenate([z_check_nodes, z_data_nodes])
    type_z = np.ones(len(src_z), dtype=np.int64)

    edge_index = np.stack([
        np.concatenate([src_x, src_z]),
        np.concatenate([dst_x, dst_z]),
    ], axis=0)

    edge_type = np.concatenate([type_x, type_z])

    return node_type, edge_index, edge_type
