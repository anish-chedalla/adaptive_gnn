"""Inject Z-biased noise into Stim circuits (Pauli channels, measurement flips).
Supports static noise and temporal drift via schedule functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import stim

# Recognized gate sets for noise injection (supersets are safe)
ANY_CLIFFORD_1_OPS: Set[str] = {
    "I", "X", "Y", "Z",
    "H",
    "S", "S_DAG",
    "SQRT_X", "SQRT_X_DAG",
    "SQRT_Y", "SQRT_Y_DAG",
    # some codegens use these aliases:
    "C_XYZ", "C_ZYX", "H_YZ",
}
ANY_CLIFFORD_2_OPS: Set[str] = {
    "CX", "CY", "CZ", "CNOT",
    "SWAP", "ISWAP",
    # sometimes used in parity-measurement constructions
    "XCX", "XCY", "XCZ",
    "YCX", "YCY", "YCZ",
    "ZCX", "ZCY", "ZCZ",
}

RESET_OPS: Set[str] = {"R", "RX", "RY", "RZ"}
MEASURE_OPS: Set[str] = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"}  # include demolition variants
ANNOTATION_OPS: Set[str] = {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "QUBIT_COORDS"}


def _pz_px_py(p: float, eta: float, bias_y: bool = False) -> Tuple[float, float, float]:
    """Convert total error rate p and bias ratio eta=pZ/pX into Pauli probabilities.
    Normalizes if sum exceeds 1.
    """
    if p <= 0:
        return 0.0, 0.0, 0.0
    if eta <= 0:
        raise ValueError(f"eta must be > 0, got {eta}")
    
    # Solve: px + pz = p, pz/px = eta → px = p/(eta+1), pz = eta*p/(eta+1)
    px = p / (eta + 1.0)
    pz = p * eta / (eta + 1.0)
    py = px if bias_y else 0.0  # Optional Y channel
    
    # Normalize if sum exceeds 1 (prevents invalid probabilities)
    s = px + py + pz
    if s > 1.0:
        px, py, pz = px / s, py / s, pz / s
    
    return float(px), float(py), float(pz)


def _collect_qubits(c: stim.Circuit) -> Set[int]:
    """Recursively find all qubit indices mentioned in circuit (including repeat blocks)."""
    qs: Set[int] = set()

    def rec(cc: stim.Circuit) -> None:
        for op in cc:
            if isinstance(op, stim.CircuitRepeatBlock):
                rec(op.body_copy())  # Recurse into repeated block
            elif isinstance(op, stim.CircuitInstruction):
                for t in op.targets_copy():
                    if t.is_qubit_target:
                        qs.add(t.value)
            else:
                raise TypeError(type(op))

    rec(c)
    return qs


def _append_pauli_channel_1(out: stim.Circuit, qubits: Sequence[int], px: float, py: float, pz: float) -> None:
    """Add single-qubit Pauli noise channel. Skips if no qubits or all probs are zero."""
    if not qubits:
        return
    if px == 0.0 and py == 0.0 and pz == 0.0:
        return
    out.append_operation("PAULI_CHANNEL_1", list(qubits), [px, py, pz])


def apply_biased_data_noise(
    circuit: stim.Circuit,
    *,
    p: float,
    eta: float = 20.0,
    data_qubits: Sequence[int],
    bias_y: bool = False,
) -> stim.Circuit:
    """Inject Z-heavy Pauli noise on data qubits after each TICK (static, no time drift)."""
    px, py, pz = _pz_px_py(p, eta, bias_y=bias_y)
    dq = sorted(set(int(q) for q in data_qubits))

    def rec(cc: stim.Circuit) -> stim.Circuit:
        out = stim.Circuit()
        for op in cc:
            if isinstance(op, stim.CircuitRepeatBlock):
                # Recursively process repeat block body
                out += rec(op.body_copy()) * op.repeat_count
            elif isinstance(op, stim.CircuitInstruction):
                out.append(op)
                # After each TICK, apply noise to data qubits
                if op.name == "TICK":
                    _append_pauli_channel_1(out, dq, px, py, pz)
            else:
                raise TypeError(type(op))
        return out

    return rec(circuit)


@dataclass(frozen=True)
class BiasedCircuitNoiseSpec:
    """Config for circuit-level biased noise (gates + idle + meas/reset flips)."""
    p: float  # Total physical error rate
    eta: float = 20.0  # Bias ratio: pZ/pX
    bias_y: bool = False  # Include Y channel
    
    meas_flip_p: Optional[float] = None  # Override for measurement errors
    reset_flip_p: Optional[float] = None  # Override for reset errors
    idle_p: Optional[float] = None  # Override for idle Pauli channels

    def probs(self) -> Tuple[float, float, float]:
        """Get (px, py, pz) from p and eta."""
        return _pz_px_py(self.p, self.eta, bias_y=self.bias_y)

    def p_idle(self) -> float:
        return self.p if self.idle_p is None else float(self.idle_p)

    def p_meas(self) -> float:
        return self.p if self.meas_flip_p is None else float(self.meas_flip_p)

    def p_reset(self) -> float:
        return self.p if self.reset_flip_p is None else float(self.reset_flip_p)


def apply_biased_circuit_noise(
    circuit: stim.Circuit,
    *,
    spec: BiasedCircuitNoiseSpec,
    qubits: Optional[Sequence[int]] = None,
) -> stim.Circuit:
    """Inject biased noise on gates, idle periods, and measurement/reset operations."""
    
    px, py, pz = spec.probs()
    p_idle = spec.p_idle()
    px_i, py_i, pz_i = _pz_px_py(p_idle, spec.eta, bias_y=spec.bias_y)

    all_qs = sorted(set(int(q) for q in (qubits if qubits is not None else _collect_qubits(circuit))))

    def meas_flip_gate(op_name: str) -> str:
        """Choose error type: MX/MRX get Z-flips, others get X-flips."""
        return "Z_ERROR" if op_name.endswith("X") else "X_ERROR"

    def rec(cc: stim.Circuit) -> stim.Circuit:
        out = stim.Circuit()
        touched: Set[int] = set()

        def flush_tick_idle() -> None:
            # Apply idle noise to qubits not touched since last TICK
            idle = [q for q in all_qs if q not in touched]
            if idle:
                _append_pauli_channel_1(out, idle, px_i, py_i, pz_i)
            touched.clear()

        for op in cc:
            if isinstance(op, stim.CircuitRepeatBlock):
                # Tick boundary before entering repeat, to keep idle logic sane
                if touched:
                    flush_tick_idle()
                out += rec(op.body_copy()) * op.repeat_count
                continue

            if not isinstance(op, stim.CircuitInstruction):
                raise TypeError(type(op))

            name = op.name

            # Annotations: passthrough, no noise added
            if name in ANNOTATION_OPS:
                if name == "TICK":
                    flush_tick_idle()
                out.append(op)
                continue

            # Measurement: classical flip model (X/Z_ERROR before measurement)
            if name in MEASURE_OPS:
                pf = spec.p_meas()
                if pf > 0:
                    qs = [t.value for t in op.targets_copy() if t.is_qubit_target]
                    out.append_operation(meas_flip_gate(name), qs, pf)
                out.append(op)
                for t in op.targets_copy():
                    if t.is_qubit_target:
                        touched.add(t.value)
                continue

            # Reset: mark touched, then apply flip errors
            if name in RESET_OPS:
                out.append(op)
                for t in op.targets_copy():
                    if t.is_qubit_target:
                        touched.add(t.value)
                pr = spec.p_reset()
                if pr > 0:
                    qs = [t.value for t in op.targets_copy() if t.is_qubit_target]
                    flip = "Z_ERROR" if name.endswith("X") else "X_ERROR"
                    out.append_operation(flip, qs, pr)
                continue

            # Measurement parity: add Pauli noise after
            if name == "MPP":
                qs = sorted({t.value for t in op.targets_copy() if (not t.is_combiner) and t.is_qubit_target})
                _append_pauli_channel_1(out, qs, px, py, pz)
                out.append(op)
                touched.update(qs)
                continue

            # Gates (1Q/2Q Clifford): add Pauli noise after gate
            out.append(op)
            qs = sorted({t.value for t in op.targets_copy() if t.is_qubit_target})
            if qs:
                touched.update(qs)

            if name in ANY_CLIFFORD_1_OPS or name in ANY_CLIFFORD_2_OPS:
                _append_pauli_channel_1(out, qs, px, py, pz)

        # idle after final tick segment
        if touched:
            flush_tick_idle()

        return out

    return rec(circuit)

# ============================================================================
# TIME-VARYING (DRIFT) NOISE: Noise parameters change with each TICK
# ============================================================================
# schedule(tick_idx) -> (p_t, eta_t) allows temporal drift in error rates.

from typing import Callable

ScheduleFn = Callable[[int], Tuple[float, float]]  # tick_idx → (p_t, eta_t)


def apply_biased_data_noise_with_schedule(
    circuit: stim.Circuit,
    *,
    schedule: ScheduleFn,
    data_qubits: Sequence[int],
    bias_y: bool = False,
) -> stim.Circuit:
    """Inject Z-biased noise on data qubits after each TICK, with drift."""
    dq = sorted(set(int(q) for q in data_qubits))

    def rec(cc: stim.Circuit, tick_state: List[int]) -> stim.Circuit:
        out = stim.Circuit()
        for op in cc:
            if isinstance(op, stim.CircuitRepeatBlock):
                out += rec(op.body_copy(), tick_state) * op.repeat_count
                continue

            if not isinstance(op, stim.CircuitInstruction):
                raise TypeError(type(op))

            out.append(op)

            if op.name == "TICK":
                t = tick_state[0]
                p_t, eta_t = schedule(t)
                px, py, pz = _pz_px_py(float(p_t), float(eta_t), bias_y=bias_y)
                _append_pauli_channel_1(out, dq, px, py, pz)
                tick_state[0] = t + 1

        return out

    return rec(circuit, [0])


def apply_biased_circuit_noise_with_schedule(
    circuit: stim.Circuit,
    *,
    schedule: ScheduleFn,
    bias_y: bool = False,
    meas_flip_p: Optional[float] = None,
    reset_flip_p: Optional[float] = None,
    idle_scale: float = 1.0,
    qubits: Optional[Sequence[int]] = None,
) -> stim.Circuit:
    """Apply biased circuit-level noise with drift keyed to TICK layers.

    - After 1Q/2Q Clifford: PAULI_CHANNEL_1 on targets using (p_t, eta_t)
    - Idles between ticks: PAULI_CHANNEL_1 on untouched qubits with p_idle(t)=idle_scale*p_t
    - Measurement/reset: simple flip model using X_ERROR/Z_ERROR; default prob is p_t unless overridden
    """
    all_qs = sorted(set(int(q) for q in (qubits if qubits is not None else _collect_qubits(circuit))))

    def meas_flip_gate(op_name: str) -> str:
        return "Z_ERROR" if op_name.endswith("X") else "X_ERROR"

    def rec(cc: stim.Circuit, tick_state: List[int]) -> stim.Circuit:
        out = stim.Circuit()
        touched: Set[int] = set()

        def current_probs():
            """Query schedule at current tick, compute gate and idle noise params."""
            t = tick_state[0]
            p_t, eta_t = schedule(t)
            px, py, pz = _pz_px_py(float(p_t), float(eta_t), bias_y=bias_y)
            px_i, py_i, pz_i = _pz_px_py(float(idle_scale) * float(p_t), float(eta_t), bias_y=bias_y)
            return float(p_t), float(eta_t), px, py, pz, px_i, py_i, pz_i

        p_t, eta_t, px, py, pz, px_i, py_i, pz_i = current_probs()

        def flush_tick_idle() -> None:
            """End of tick: apply idle noise to untouched qubits."""
            nonlocal px_i, py_i, pz_i
            idle = [q for q in all_qs if q not in touched]
            if idle:
                _append_pauli_channel_1(out, idle, px_i, py_i, pz_i)
            touched.clear()

        def advance_tick() -> None:
            """Move to next tick, update all noise parameters from schedule."""
            nonlocal p_t, eta_t, px, py, pz, px_i, py_i, pz_i
            tick_state[0] += 1
            p_t, eta_t, px, py, pz, px_i, py_i, pz_i = current_probs()

        for op in cc:
            if isinstance(op, stim.CircuitRepeatBlock):
                if touched:
                    flush_tick_idle()
                out += rec(op.body_copy(), tick_state) * op.repeat_count
                continue

            if not isinstance(op, stim.CircuitInstruction):
                raise TypeError(type(op))

            name = op.name

            if name in ANNOTATION_OPS:
                if name == "TICK":
                    flush_tick_idle()
                    out.append(op)
                    advance_tick()
                    continue
                out.append(op)
                continue

            if name in MEASURE_OPS:
                pf = p_t if meas_flip_p is None else float(meas_flip_p)
                if pf > 0:
                    qs = [t.value for t in op.targets_copy() if t.is_qubit_target]
                    out.append_operation(meas_flip_gate(name), qs, pf)
                out.append(op)
                for t in op.targets_copy():
                    if t.is_qubit_target:
                        touched.add(t.value)
                continue

            if name in RESET_OPS:
                out.append(op)
                for t in op.targets_copy():
                    if t.is_qubit_target:
                        touched.add(t.value)
                pr = p_t if reset_flip_p is None else float(reset_flip_p)
                if pr > 0:
                    qs = [t.value for t in op.targets_copy() if t.is_qubit_target]
                    flip = "Z_ERROR" if name.endswith("X") else "X_ERROR"
                    out.append_operation(flip, qs, pr)
                continue

            if name == "MPP":
                qs = sorted({t.value for t in op.targets_copy() if (not t.is_combiner) and t.is_qubit_target})
                _append_pauli_channel_1(out, qs, px, py, pz)
                out.append(op)
                touched.update(qs)
                continue

            out.append(op)
            qs = sorted({t.value for t in op.targets_copy() if t.is_qubit_target})
            if qs:
                touched.update(qs)

            if name in ANY_CLIFFORD_1_OPS or name in ANY_CLIFFORD_2_OPS:
                _append_pauli_channel_1(out, qs, px, py, pz)

        if touched:
            flush_tick_idle()

        return out

    return rec(circuit, [0])
