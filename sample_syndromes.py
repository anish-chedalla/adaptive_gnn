#!/usr/bin/env python3
"""
GNN-ready syndrome sampler for BB (bivariate bicycle) CSS QLDPC codes.
(See below for full details.)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Callable, Tuple, Optional

import numpy as np

# Stim is required for circuit construction, DEM compilation, and sampling.
try:
    import stim  # type: ignore
except ModuleNotFoundError:
    raise SystemExit(
        "Missing dependency: stim. Install it with:\n\n"
        "  pip install stim\n\n"
        "Then rerun this script."
    )

from codes_q import create_bivariate_bicycle_codes
from astra_stim.qldpc_circuit import CSSCodeSpec, build_qldpc_memory_circuit_text
from astra_stim.biased_noise import (
    apply_biased_data_noise,
    apply_biased_circuit_noise,
    apply_biased_data_noise_with_schedule,
    apply_biased_circuit_noise_with_schedule,
    BiasedCircuitNoiseSpec,
)

ScheduleFn = Callable[[int], Tuple[float, float]]  # tick_idx -> (p_t, eta_t)


# ----------------------------- BB family params ----------------------------- #
def bb_params(d: int) -> Tuple[int, int, list, list, list, list]:
    """BB family mapping (matches your existing selection logic)."""
    if d == 6:   # [[72, 12, 6]]
        return 6, 6, [3], [1, 2], [1, 2], [3]
    if d == 10:  # [[90, 8, 10]]
        return 15, 3, [9], [1, 2], [2, 7], [0]
    if d == 12:  # [[144, 12, 12]]
        return 12, 6, [3], [1, 2], [1, 2], [3]
    if d == 18:  # [[288, 12, 18]]
        return 12, 12, [3], [2, 7], [1, 2], [3]
    if d == 24:  # [[360, 12, <=24]]
        return 30, 6, [9], [1, 2], [25, 26], [3]
    if d == 34:  # [[756, 16, <=34]]
        return 21, 18, [3], [10, 17], [3, 19], [5]
    raise ValueError(f"Unsupported d={d}. Add BB parameters to bb_params().")


# ----------------------------- Drift scheduling ----------------------------- #
def make_schedule(args) -> Optional[ScheduleFn]:
    """Return tick->(p, eta) schedule or None if drift disabled."""
    if args.drift == "none":
        return None

    # Drift only makes sense if we actually use a biased noise model
    if args.noise == "none":
        raise SystemExit("Drift requested but --noise=none. That is meaningless. Use biased_data or biased_circuit.")

    if args.drift_amp <= 0:
        raise SystemExit("--drift_amp must be > 0 when --drift is enabled")
    if args.drift_period_ticks <= 0:
        raise SystemExit("--drift_period_ticks must be > 0")

    base_p = float(args.p)
    base_eta = float(args.eta)

    def schedule(tick_idx: int) -> Tuple[float, float]:
        # sine drift: scale(t) = 1 + amp*sin(2π t / period)
        phase = 2.0 * math.pi * float(tick_idx) / float(args.drift_period_ticks)
        scale = 1.0 + float(args.drift_amp) * math.sin(phase)

        p = base_p
        eta = base_eta

        if args.drift_target == "p":
            p = max(p * scale, 1e-12)     # keep positive
        else:
            eta = max(eta * scale, 1.0)   # keep eta >= 1

        return float(p), float(eta)

    return schedule


# ---------------------- GNN exports: graph + reshape ------------------------ #
def build_tanner_graph_edges(hx: np.ndarray, hz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stable node indexing convention:
      - qubit nodes:   0 .. n-1
      - X-check nodes: n .. n+mx-1
      - Z-check nodes: n+mx .. n+mx+mz-1

    We export directed edges (check -> qubit) for a bipartite graph.
    edge_type: 0 = Hx edge, 1 = Hz edge
    node_type: 0 = qubit, 1 = X-check, 2 = Z-check
    """
    mx, n = hx.shape
    mz, n2 = hz.shape
    if n != n2:
        raise ValueError(f"Hx and Hz column mismatch: {n} vs {n2}")

    src: list[int] = []
    dst: list[int] = []
    et: list[int] = []

    # X-check edges (stable ordering: row then increasing qubit index)
    for i in range(mx):
        qs = np.where(hx[i] == 1)[0]
        c = n + i
        for q in qs:
            src.append(int(c))
            dst.append(int(q))
            et.append(0)

    # Z-check edges (stable ordering)
    for i in range(mz):
        qs = np.where(hz[i] == 1)[0]
        c = n + mx + i
        for q in qs:
            src.append(int(c))
            dst.append(int(q))
            et.append(1)

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_type = np.array(et, dtype=np.int64)

    node_type = np.zeros(n + mx + mz, dtype=np.int64)
    node_type[n : n + mx] = 1
    node_type[n + mx : n + mx + mz] = 2

    return edge_index, edge_type, node_type


def reshape_detectors(dets: np.ndarray, rounds: int, mx: int, mz: int) -> np.ndarray:
    """
    CRITICAL MAPPING FACT (MUST HOLD):
      For each round r = 1..rounds-1:
        first mx detectors: X-check parity between round r and r-1
        next  mz detectors: Z-check parity between round r and r-1

    Input:
      dets: [shots, (rounds-1)*(mx+mz)]
    Output:
      dets_by_round: [shots, rounds-1, mx+mz]
    """
    if dets.ndim != 2:
        raise ValueError(f"dets must be [shots, D]. Got {dets.shape}")

    shots, D = dets.shape
    expected = (rounds - 1) * (mx + mz)
    if D != expected:
        raise ValueError(
            f"Detector count mismatch: got {D}, expected {expected} = (rounds-1)*(mx+mz). "
            "Your detector ordering/rounds assumptions are wrong."
        )
    return dets.reshape(shots, rounds - 1, mx + mz)


# ---------------------------------- main ----------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--d", type=int, required=True, help="BB family selector (6,10,12,18,24,34)")
    ap.add_argument("--rounds", type=int, default=None, help="stabilizer rounds; default=d")
    ap.add_argument("--basis", type=str, default="z", choices=["x", "z"], help="final data measurement basis")

    ap.add_argument("--noise", type=str, default="biased_data",
                    choices=["none", "biased_data", "biased_circuit"])
    ap.add_argument("--p", type=float, default=0.0, help="physical error rate p (required for biased modes)")
    ap.add_argument("--eta", type=float, default=None, help="bias eta=pZ/pX (required for biased modes)")
    ap.add_argument("--p_flip", type=float, default=None, help="meas/reset flip prob in biased_circuit; default=p")

    ap.add_argument("--shots", type=int, default=20, help="number of samples")
    ap.add_argument("--print_n", type=int, default=10, help="print first N samples")
    ap.add_argument("--out", type=str, default=None, help="optional output .npz path to save samples")

    # temporal drift
    ap.add_argument("--drift", type=str, default="none", choices=["none", "sine"], help="temporal drift model")
    ap.add_argument("--drift_target", type=str, default="p", choices=["p", "eta"], help="which parameter drifts")
    ap.add_argument("--drift_amp", type=float, default=0.0, help="relative drift amplitude (e.g. 0.3 = ±30%)")
    ap.add_argument("--drift_period_ticks", type=int, default=50, help="drift period in TICKs")

    args = ap.parse_args()

    # ---- argument validation ----
    if args.noise != "none":
        if args.p <= 0:
            raise SystemExit("For biased modes you must set --p > 0")
        if args.eta is None or args.eta <= 0:
            raise SystemExit("For biased modes you must set --eta > 0")

    schedule = make_schedule(args)

    # 1) Build BB CSS code matrices.
    l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows = bb_params(args.d)
    css, *_ = create_bivariate_bicycle_codes(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows)

    hx = np.array(css.hx, dtype=np.uint8)
    hz = np.array(css.hz, dtype=np.uint8)
    lx = np.array(css.lx, dtype=np.uint8)
    lz = np.array(css.lz, dtype=np.uint8)

    if hx.ndim != 2 or hz.ndim != 2:
        raise RuntimeError("Hx/Hz must be 2D matrices")
    if hx.shape[1] != hz.shape[1]:
        raise RuntimeError("Hx and Hz must have same number of columns")

    n = int(hx.shape[1])
    mx = int(hx.shape[0])
    mz = int(hz.shape[0])
    rounds = int(args.rounds if args.rounds is not None else args.d)
    if rounds < 2:
        raise SystemExit("--rounds must be >= 2")

    # 2) Build noiseless Stim circuit text and circuit.
    spec = CSSCodeSpec(hx=hx, hz=hz, lx=lx, lz=lz)
    base_text = build_qldpc_memory_circuit_text(spec, rounds=rounds, basis=args.basis)
    base = stim.Circuit(base_text)

    # Data qubits assumed 0..n-1 in your builder; enforce.
    if base.num_qubits < n:
        raise RuntimeError(f"Stim circuit uses num_qubits={base.num_qubits}, but n={n}. Indexing mismatch.")
    data_qubits = list(range(n))

    # Use actual circuit qubit count (your old n+mx+mz assumption can silently break).
    all_qubits = list(range(base.num_qubits))

    # 3) Inject noise (static or drifted).
    if args.noise == "none":
        circ = base

    elif args.noise == "biased_data":
        if schedule is None:
            circ = apply_biased_data_noise(
                base,
                p=float(args.p),
                eta=float(args.eta),
                data_qubits=data_qubits,
                bias_y=False,
            )
        else:
            circ = apply_biased_data_noise_with_schedule(
                base,
                schedule=schedule,
                data_qubits=data_qubits,
                bias_y=False,
            )

    elif args.noise == "biased_circuit":
        if schedule is None:
            p_flip = float(args.p) if args.p_flip is None else float(args.p_flip)
            spec_noise = BiasedCircuitNoiseSpec(
                p=float(args.p),
                eta=float(args.eta),
                bias_y=False,
                meas_flip_p=p_flip,
                reset_flip_p=p_flip,
            )
            circ = apply_biased_circuit_noise(
                base,
                spec=spec_noise,
                qubits=all_qubits,
            )
        else:
            circ = apply_biased_circuit_noise_with_schedule(
                base,
                schedule=schedule,
                bias_y=False,
                qubits=all_qubits,
                # If your schedule drifts p(t), flips should drift too. Leaving None ties them to schedule.
                meas_flip_p=None,
                reset_flip_p=None,
                idle_scale=1.0,
            )
    else:
        raise SystemExit(f"Unknown noise mode: {args.noise}")

    # 4) DEM compile gate (required for sanity + later decoder work).
    try:
        _ = circ.detector_error_model(approximate_disjoint_errors=True)
    except Exception:
        print("FAILED: circuit.detector_error_model() — stop and fix circuit/noise.", file=sys.stderr)
        raise

    # 5) Sample detectors + observables.
    sampler = circ.compile_detector_sampler()
    dets, obs = sampler.sample(int(args.shots), separate_observables=True)
    dets = dets.astype(np.uint8)  # [shots, num_detectors]
    obs = obs.astype(np.uint8)    # [shots, k]

    # ---- GNN-ready transforms ----
    dets_by_round = reshape_detectors(dets, rounds=rounds, mx=mx, mz=mz)  # [shots, rounds-1, mx+mz]
    x_dets = dets_by_round[:, :, :mx]   # [shots, rounds-1, mx]
    z_dets = dets_by_round[:, :, mx:]   # [shots, rounds-1, mz]

    edge_index, edge_type, node_type = build_tanner_graph_edges(hx, hz)

    # Hard assertions to prevent silent dataset corruption
    assert dets_by_round.shape == (args.shots, rounds - 1, mx + mz)
    assert x_dets.shape == (args.shots, rounds - 1, mx)
    assert z_dets.shape == (args.shots, rounds - 1, mz)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_type.shape[0]
    assert node_type.shape[0] == n + mx + mz

    # 6) Print summary and a few samples.
    print("=== BB QLDPC Sampling Summary ===")
    print(f"d={args.d}  (l={l}, m={m})")
    print(f"n(data)={n}  mx={mx}  mz={mz}  rounds={rounds}")
    print(f"noise={args.noise}  p={args.p}  eta={args.eta}")
    if args.drift != "none":
        print(f"drift={args.drift} target={args.drift_target} amp={args.drift_amp} period_ticks={args.drift_period_ticks}")
    print(f"num_detectors={dets.shape[1]}  num_observables={obs.shape[1]}")
    print(f"shots={args.shots}")

    pn = min(int(args.print_n), int(args.shots))
    print("\nFirst syndromes (flat detector events):")
    for i in range(pn):
        print(i, dets[i].tolist())

    print("\nFirst observables:")
    for i in range(pn):
        print(i, obs[i].tolist())

    # 7) Optional save (GNN-ready).
    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "d": int(args.d),
            "l": int(l),
            "m": int(m),
            "n": int(n),
            "mx": int(mx),
            "mz": int(mz),
            "rounds": int(rounds),
            "basis": str(args.basis),
            "noise_mode": str(args.noise),
            "drift": str(args.drift),
            "drift_target": str(args.drift_target),
            "drift_amp": float(args.drift_amp),
            "drift_period_ticks": int(args.drift_period_ticks),
            "detector_order": "per round r=1..rounds-1: [X-check detectors 0:mx] then [Z-check detectors mx:mx+mz]",
            "node_indexing": "qubits[0..n-1], X-checks[n..n+mx-1], Z-checks[n+mx..n+mx+mz-1]",
            "edge_indexing": "directed edges are (check_node -> qubit_node)",
            "no_leakage_note": "Never feed p/eta/drift params to the GNN; it must infer from detector patterns.",
            "stim_num_qubits": int(base.num_qubits),
        }
        meta_json = json.dumps(meta, sort_keys=True)

        # Save circuit texts as unicode scalars (NO pickle required on load).
        circuit_base_text = str(base)
        circuit_noisy_text = str(circ)

        np.savez_compressed(
            out_path,
            # raw sampled outputs
            syndromes=dets,                 # [shots, num_detectors]
            dets_by_round=dets_by_round,     # [shots, rounds-1, mx+mz]
            x_dets=x_dets,                   # [shots, rounds-1, mx]
            z_dets=z_dets,                   # [shots, rounds-1, mz]
            observables=obs,                 # [shots, k]

            # code definition
            hx=hx,
            hz=hz,
            lx=lx,
            lz=lz,

            # Tanner graph definition for PyG
            edge_index=edge_index,           # [2, E]
            edge_type=edge_type,             # [E] 0=Hx, 1=Hz
            node_type=node_type,             # [n+mx+mz] 0=qubit,1=X-check,2=Z-check

            # circuit text (unicode scalars; load without pickle)
            circuit_base=np.array(circuit_base_text),
            circuit_noisy=np.array(circuit_noisy_text),

            # metadata (JSON string)
            meta_json=np.array(meta_json),
        )

        print(f"\nSaved GNN-ready dataset to: {out_path}")
        print("Saved keys include: circuit_base, circuit_noisy, meta_json")

    # Done.


if __name__ == "__main__":
    main()
