
"""Build Stim circuits for repeated CSS stabilizer measurements.
Handles qubit layout, CNOT scheduling, detectors, and observables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class CSSCodeSpec:
    """Minimal info needed to build a CSS stabilizer measurement circuit."""
    hx: np.ndarray  # shape (m_x, n)
    hz: np.ndarray  # shape (m_z, n)
    lx: np.ndarray  # shape (k, n) logical X operators (as binary supports)
    lz: np.ndarray  # shape (k, n) logical Z operators (as binary supports)


def _bit_support(row: np.ndarray) -> List[int]:
    """Indices where row is 1."""
    return list(np.flatnonzero(row.astype(np.int8)))


def _greedy_matching_schedule(edges: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """Pack edges into parallel layers (greedy conflict-free matching).
    Each layer has no qubit conflicts. Minimizes circuit depth and noise exposure.
    """
    remaining = edges.copy()
    remaining.sort()  # Deterministic order

    schedule: List[List[Tuple[int, int]]] = []
    while remaining:
        used_a = set()  # Ancillas used in this layer
        used_d = set()  # Data qubits used in this layer
        layer: List[Tuple[int, int]] = []
        new_remaining: List[Tuple[int, int]] = []

        for a, d in remaining:
            if a in used_a or d in used_d:  # Skip if qubit already used
                new_remaining.append((a, d))
                continue
            used_a.add(a)
            used_d.add(d)
            layer.append((a, d))

        schedule.append(layer)
        remaining = new_remaining

    return schedule


def build_qldpc_memory_circuit_text(
    code: CSSCodeSpec,
    rounds: int,
    basis: str,
    *,
    data_offset: int = 0,
) -> str:
    """
    Build a noiseless repeated-stabilizer measurement circuit.

    Qubit layout:
      data qubits: [data_offset .. data_offset+n-1]
      X ancillas:  next mx qubits
      Z ancillas:  next mz qubits

    Detectors:
      For each check i and each round r>=1:
        DETECTOR = meas(r,i) XOR meas(r-1,i)
    This avoids non-deterministic detectors from the first measurement round.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")

    basis = basis.lower().strip()
    if basis not in {"x", "z"}:
        raise ValueError("basis must be 'x' or 'z'")

    hx = np.array(code.hx, dtype=np.uint8)
    hz = np.array(code.hz, dtype=np.uint8)
    lx = np.array(code.lx, dtype=np.uint8)
    lz = np.array(code.lz, dtype=np.uint8)

    mx, n = hx.shape
    mz, n2 = hz.shape
    if n != n2:
        raise ValueError("Hx and Hz must have same number of columns")

    data = [data_offset + i for i in range(n)]
    x_anc = [data_offset + n + i for i in range(mx)]
    z_anc = [data_offset + n + mx + i for i in range(mz)]

    # Extract CNOT connections from parity-check matrices
    # Each 1 in Hx[i] means: CX from X-ancilla i to that data qubit
    x_edges: List[Tuple[int, int]] = []
    for i in range(mx):
        for q in _bit_support(hx[i]):
            x_edges.append((x_anc[i], data_offset + q))

    # Each 1 in Hz[i] means: CX from that data qubit to Z-ancilla i
    z_edges: List[Tuple[int, int]] = []
    for i in range(mz):
        for q in _bit_support(hz[i]):
            z_edges.append((z_anc[i], data_offset + q))

    # Schedule CNOTs into parallel layers
    x_schedule = _greedy_matching_schedule(x_edges)
    z_schedule = _greedy_matching_schedule(z_edges)

    # Track measurement indices for detector definitions (comparing consecutive rounds)
    meas_count = 0  # Global counter for all measurements
    x_meas_idx: List[List[int]] = []  # Per round: X ancilla measurement indices
    z_meas_idx: List[List[int]] = []  # Per round: Z ancilla measurement indices

    def rec(abs_index: int) -> str:
        """
        Convert an absolute measurement index into a Stim rec[-k] reference,
        based on current meas_count.
        """
        nonlocal meas_count
        rel = meas_count - abs_index
        if rel <= 0:
            raise RuntimeError("rec index bug (referencing future measurement)")
        return f"rec[-{rel}]"

    lines: List[str] = []

    # Start with all data qubits in |0>
    lines.append(f"R {' '.join(map(str, data))}")
    lines.append("TICK")

    # Repeated measurement cycles
    for r in range(rounds):
        # ============ X parity check measurement ============
        if mx:
            # Prepare X ancillas in |+> state
            lines.append(f"R {' '.join(map(str, x_anc))}")
            lines.append(f"H {' '.join(map(str, x_anc))}")
            lines.append("TICK")

            # CNOT layers: each ancilla touches its data qubits
            for layer in x_schedule:
                if layer:
                    pairs = " ".join(f"{a} {d}" for a, d in layer)
                    lines.append(f"CX {pairs}")
                lines.append("TICK")

            # Rotate back (so Z measurement gives X parity)
            lines.append(f"H {' '.join(map(str, x_anc))}")
            lines.append("TICK")

            # Measure X ancillas
            lines.append(f"M {' '.join(map(str, x_anc))}")
            idxs = list(range(meas_count, meas_count + mx))
            meas_count += mx
            x_meas_idx.append(idxs)

            # Detectors: syndrome = current XOR previous (0 means no error)
            if r >= 1:
                for i in range(mx):
                    prev = x_meas_idx[r - 1][i]
                    cur = x_meas_idx[r][i]
                    lines.append(f"DETECTOR {rec(prev)} {rec(cur)}")

            lines.append("TICK")

        # ============ Z parity check measurement ============
        if mz:
            # Prepare Z ancillas in |0>
            lines.append(f"R {' '.join(map(str, z_anc))}")
            lines.append("TICK")

            # CNOT layers: data controls, ancilla targets
            for layer in z_schedule:
                if layer:
                    pairs = " ".join(f"{d} {a}" for a, d in layer)
                    lines.append(f"CX {pairs}")
                lines.append("TICK")

            # Measure Z ancillas
            lines.append(f"M {' '.join(map(str, z_anc))}")
            idxs = list(range(meas_count, meas_count + mz))
            meas_count += mz
            z_meas_idx.append(idxs)

            # Detectors: syndrome differences
            if r >= 1:
                for i in range(mz):
                    prev = z_meas_idx[r - 1][i]
                    cur = z_meas_idx[r][i]
                    lines.append(f"DETECTOR {rec(prev)} {rec(cur)}")

            lines.append("TICK")

    # Final measurement of all data qubits in the target basis
    if basis == "z":
        lines.append(f"M {' '.join(map(str, data))}")
    else:
        lines.append(f"MX {' '.join(map(str, data))}")

    data_meas_idx = list(range(meas_count, meas_count + n))
    meas_count += n

    # Define observables: parity of data qubits under logical operators
    # These track encoded information protected by the code
    logicals = lz if basis == "z" else lx
    k = logicals.shape[0] if logicals.size else 0

    for li in range(k):
        supp = _bit_support(logicals[li])
        if not supp:
            continue
        # Observable = XOR of measurements at logical operator support
        terms = " ".join(rec(data_meas_idx[q]) for q in supp)
        lines.append(f"OBSERVABLE_INCLUDE({li}) {terms}")

    return "\n".join(lines) + "\n"
