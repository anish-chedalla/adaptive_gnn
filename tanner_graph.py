from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


FORBIDDEN_KEYS = {"p", "eta", "p_t", "eta_t", "schedule", "drift", "drift_params"}


def reshape_detectors(
    dets_flat: np.ndarray, mx: int, mz: int, rounds: Optional[int] = None
) -> np.ndarray:
    """
    CRITICAL MAPPING FACT (deterministic):
    For each time step t = 1..rounds-1:
      first mx detectors = X-checks
      next mz detectors  = Z-checks

    Input:
      dets_flat: [shots, (rounds-1)*(mx+mz)]
    Output:
      dets_time: [shots, rounds-1, mx+mz] with check axis [X..., Z...]
    """
    if dets_flat.ndim != 2:
        raise ValueError(f"dets_flat must be 2D [shots, D]. Got shape {dets_flat.shape}")

    shots, D = dets_flat.shape
    per_t = mx + mz
    if D % per_t != 0:
        raise ValueError(
            f"Detector length D={D} not divisible by (mx+mz)={per_t}. "
            "This means your assumed ordering is broken."
        )

    rounds_minus1 = D // per_t
    if rounds is not None and rounds_minus1 != rounds - 1:
        raise ValueError(
            f"Provided rounds={rounds} inconsistent with det length: "
            f"rounds-1={rounds-1} vs inferred {rounds_minus1}"
        )
    dets_time = dets_flat.reshape(shots, rounds_minus1, per_t)
    return dets_time


def stable_edge_list_from_H(H: np.ndarray) -> List[Tuple[int, int]]:
    """
    Returns list of (row, col) where H[row, col] == 1 with stable ordering:
    - increasing row
    - within each row, increasing col
    """
    if H.dtype != np.uint8 and H.dtype != np.int64 and H.dtype != np.int32 and H.dtype != np.bool_:
        H = H.astype(np.uint8)

    rows, cols = np.nonzero(H)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    pairs.sort(key=lambda rc: (rc[0], rc[1]))
    return pairs


@dataclass(frozen=True)
class TannerGraph:
    n: int
    mx: int
    mz: int
    num_nodes: int
    edge_index: np.ndarray  # [2, E]
    edge_type: np.ndarray   # [E] 0=X-edge (Hx), 1=Z-edge (Hz)
    deg_total: np.ndarray   # [n]
    deg_x: np.ndarray       # [n]
    deg_z: np.ndarray       # [n]

    @staticmethod
    def from_hx_hz(hx: np.ndarray, hz: np.ndarray) -> "TannerGraph":
        if hx.ndim != 2 or hz.ndim != 2:
            raise ValueError("hx and hz must be 2D")
        if hx.shape[1] != hz.shape[1]:
            raise ValueError("hx and hz must have same number of columns (data qubits)")

        mx, n = hx.shape
        mz, n2 = hz.shape
        assert n == n2

        # Degrees per variable
        deg_x = hx.sum(axis=0).astype(np.int64)
        deg_z = hz.sum(axis=0).astype(np.int64)
        deg_total = deg_x + deg_z

        # Build edges with stable ordering
        hx_edges = stable_edge_list_from_H(hx)
        hz_edges = stable_edge_list_from_H(hz)

        # Node indexing (fixed):
        # variables: 0..n-1
        # checks: n..n+mx-1 are X-checks
        #         n+mx..n+mx+mz-1 are Z-checks
        edge_u = []
        edge_v = []
        edge_t = []

        # Hx edges => edge_type=0, connect X-check i to qubit j
        for i, j in hx_edges:
            c = n + i
            v = j
            # add both directions
            edge_u.extend([c, v])
            edge_v.extend([v, c])
            edge_t.extend([0, 0])

        # Hz edges => edge_type=1, connect Z-check i to qubit j
        for i, j in hz_edges:
            c = n + mx + i
            v = j
            edge_u.extend([c, v])
            edge_v.extend([v, c])
            edge_t.extend([1, 1])

        edge_index = np.array([edge_u, edge_v], dtype=np.int64)
        edge_type = np.array(edge_t, dtype=np.int64)

        num_nodes = n + mx + mz
        return TannerGraph(
            n=n, mx=mx, mz=mz, num_nodes=num_nodes,
            edge_index=edge_index, edge_type=edge_type,
            deg_total=deg_total, deg_x=deg_x, deg_z=deg_z
        )


def build_node_features(
    tg: TannerGraph, check_window: np.ndarray, W: int
) -> np.ndarray:
    """
    Build a single node feature matrix x for PyG.

    Node ordering fixed:
      var nodes: 0..n-1
      check nodes: n..n+mx+mz-1 (X checks then Z checks)

    Features (float32), shape: [N, F] where F = W + 5
      col0: node_type (0=var, 1=check)
      col1: check_type (0=X-check, 1=Z-check) else 0 for vars
      col2: deg_total (vars only)
      col3: deg_x (vars only)
      col4: deg_z (vars only)
      col5..: detector window (checks only), length W
    """
    n, mx, mz = tg.n, tg.mx, tg.mz
    m = mx + mz
    if check_window.shape != (m, W):
        raise ValueError(f"check_window shape must be {(m, W)}; got {check_window.shape}")

    N = tg.num_nodes
    F = W + 5
    x = np.zeros((N, F), dtype=np.float32)

    # Vars
    x[:n, 0] = 0.0
    x[:n, 2] = tg.deg_total.astype(np.float32)
    x[:n, 3] = tg.deg_x.astype(np.float32)
    x[:n, 4] = tg.deg_z.astype(np.float32)

    # Checks
    x[n:, 0] = 1.0
    # check_type: X first then Z
    x[n:n+mx, 1] = 0.0
    x[n+mx:n+mx+mz, 1] = 1.0
    x[n:, 5:] = check_window.astype(np.float32)
    return x


def parse_observable_support_from_circuit(circuit_text: str, n: int) -> List[List[int]]:
    """
    Parses OBSERVABLE_INCLUDE lines from Stim circuit text.

    Assumption (true for your qldpc_circuit.py):
    - final operation measures ALL data qubits in one instruction (MX or MZ)
    - the last n measurement record bits correspond to data qubits in order 0..n-1
      so:
        rec[-1]  -> qubit (n-1)
        rec[-n]  -> qubit 0
      mapping: qubit = n - k where token is rec[-k].

    Returns:
      obs_supports: list where obs_supports[obs_id] = list of qubit indices
    """
    obs_map = {}
    # Example line:
    # OBSERVABLE_INCLUDE(3) rec[-72] rec[-5] ...
    pat = re.compile(r"^OBSERVABLE_INCLUDE\((\d+)\)\s+(.*)$")

    for line in circuit_text.splitlines():
        line = line.strip()
        if not line.startswith("OBSERVABLE_INCLUDE"):
            continue
        m = pat.match(line)
        if not m:
            continue
        obs_id = int(m.group(1))
        rest = m.group(2)
        # extract rec[-k]
        recs = re.findall(r"rec\[-(\d+)\]", rest)
        qubits = []
        for k_str in recs:
            k = int(k_str)
            if k < 1 or k > n:
                # If this fires, your circuit observable is not only final-data-meas;
                # then you MUST store logical operators explicitly instead.
                raise ValueError(
                    f"Observable references rec[-{k}] but expected within last n={n} measurements."
                )
            qubit = n - k
            qubits.append(qubit)
        obs_map[obs_id] = sorted(set(qubits))

    if not obs_map:
        raise ValueError("No OBSERVABLE_INCLUDE lines found. Save circuit text in .npz for evaluation.")

    max_id = max(obs_map.keys())
    supports = []
    for i in range(max_id + 1):
        if i not in obs_map:
            raise ValueError(f"Missing observable id {i} in circuit text (non-contiguous IDs).")
        supports.append(obs_map[i])
    return supports


def infer_final_basis_from_circuit(circuit_text: str) -> str:
    """
    Heuristic: in your circuit builder, final measurements are either MZ or MX on data qubits.
    """
    # Search from bottom to top for MZ/MX.
    lines = [ln.strip() for ln in circuit_text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("MZ "):
            return "z"
        if ln.startswith("MX "):
            return "x"
    # If we can't infer, default to z (your default), but warn via exception.
    raise ValueError("Cannot infer final basis from circuit text (no MZ/MX found).")
