"""
Sample detector syndromes from BB (bivariate bicycle) QLDPC codes with biased noise.

Supports:
- biased_data: noise on data qubits only
- biased_circuit: full circuit-level biased noise
- drifting_biased_circuit: circuit-level noise with temporal drift in p and/or eta
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from typing import Callable, Tuple

import numpy as np

try:
    import stim
except ModuleNotFoundError:
    raise SystemExit(
        "Missing dependency: stim. Install it with:\n\n"
        "  pip install stim\n\n"
        "Then rerun this script."
    )

from codes.codes_q import create_bivariate_bicycle_codes
from astra_stim.qldpc_circuit import CSSCodeSpec, build_qldpc_memory_circuit_text
from astra_stim.biased_noise import (
    apply_biased_data_noise,
    apply_biased_circuit_noise,
    apply_biased_data_noise_with_schedule,
    apply_biased_circuit_noise_with_schedule,
    BiasedCircuitNoiseSpec,
)

ScheduleFn = Callable[[int], Tuple[float, float]]  # tick_idx -> (p_t, eta_t)


def bb_params(d: int) -> Tuple[int, int, list, list, list, list]:
    """BB family mapping: distance -> (l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows)."""
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
    raise ValueError(f"Unsupported d={d}. Add your BB parameters to bb_params().")


def make_schedule(args) -> ScheduleFn | None:
    """Return a tick->(p,eta) schedule or None if drift disabled."""
    if args.drift == "none":
        return None

    if args.drift_amp <= 0:
        raise SystemExit("--drift_amp must be > 0 when --drift is enabled")
    if args.drift_period_ticks <= 0:
        raise SystemExit("--drift_period_ticks must be > 0")

    def schedule(tick_idx: int) -> Tuple[float, float]:
        phase = 2.0 * math.pi * float(tick_idx) / float(args.drift_period_ticks)
        scale = 1.0 + float(args.drift_amp) * math.sin(phase)
        p = float(args.p)
        eta = float(args.eta)
        if args.drift_target == "p":
            p = max(p * scale, 1e-9)
        else:
            eta = max(eta * scale, 1.0)
        return p, eta

    return schedule


def make_drifting_schedule(
    base_p: float, base_eta: float,
    drift_p: float, drift_eta: float,
    period_ticks: int,
) -> ScheduleFn:
    """Build a sine-wave drift schedule for drifting_biased_circuit mode."""
    def schedule(tick_idx: int) -> Tuple[float, float]:
        phase = 2.0 * math.pi * float(tick_idx) / float(period_ticks)
        s = math.sin(phase)
        p_t = max(base_p + drift_p * s, 1e-9)
        eta_t = max(base_eta + drift_eta * s, 1.0)
        return p_t, eta_t
    return schedule


def _verify_file(path: pathlib.Path) -> None:
    """Verify a file was written successfully. Exit on failure."""
    abs_path = path.resolve()
    if not abs_path.exists():
        print(f"FATAL: File does not exist: {abs_path}", file=sys.stderr)
        sys.exit(1)
    sz = abs_path.stat().st_size
    if sz == 0:
        print(f"FATAL: File is empty (0 bytes): {abs_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Verified: {abs_path} ({sz} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample detector syndromes from BB QLDPC codes with biased noise."
    )

    ap.add_argument("--d", type=int, required=True,
                    help="BB family selector (6,10,12,18,24,34)")
    ap.add_argument("--rounds", type=int, default=None,
                    help="stabilizer rounds; default=d")
    ap.add_argument("--basis", type=str, default="z", choices=["x", "z"],
                    help="final data measurement basis")

    ap.add_argument("--noise", type=str, default="biased_data",
                    choices=["none", "biased_data", "biased_circuit",
                             "drifting_biased_circuit"])
    ap.add_argument("--p", type=float, default=0.0,
                    help="physical error rate p (required for biased modes)")
    ap.add_argument("--eta", type=float, default=None,
                    help="bias eta=pZ/pX (required for biased modes)")
    ap.add_argument("--p_flip", type=float, default=None,
                    help="meas/reset flip prob in biased_circuit; default=p")

    ap.add_argument("--shots", type=int, default=20,
                    help="number of samples")
    ap.add_argument("--print_n", type=int, default=10,
                    help="print first N samples")
    ap.add_argument("--out", type=str, default=None,
                    help="optional output .npz path to save samples")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for reproducibility")

    # --- drift parameters (used by biased_data/biased_circuit with --drift) ---
    ap.add_argument("--drift", type=str, default="none", choices=["none", "sine"],
                    help="temporal drift model (for biased_data/biased_circuit)")
    ap.add_argument("--drift_target", type=str, default="p", choices=["p", "eta"],
                    help="which parameter drifts (for --drift sine)")
    ap.add_argument("--drift_amp", type=float, default=0.0,
                    help="relative drift amplitude (e.g. 0.3 = +/-30%%)")
    ap.add_argument("--drift_period_ticks", type=int, default=50,
                    help="drift period measured in TICKs")

    # --- drifting_biased_circuit specific ---
    ap.add_argument("--drift_p", type=float, default=0.0,
                    help="additive p drift amplitude for drifting_biased_circuit")
    ap.add_argument("--drift_eta", type=float, default=0.0,
                    help="additive eta drift amplitude for drifting_biased_circuit")

    args = ap.parse_args()

    # Set seed for reproducibility
    np.random.seed(args.seed)

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

    n = hx.shape[1]
    mx = hx.shape[0]
    mz = hz.shape[0]
    rounds = args.rounds if args.rounds is not None else args.d

    # 2) Build noiseless Stim circuit.
    spec = CSSCodeSpec(hx=hx, hz=hz, lx=lx, lz=lz)
    base = stim.Circuit(build_qldpc_memory_circuit_text(spec, rounds=rounds, basis=args.basis))

    data_qubits = list(range(n))
    all_qubits = list(range(n + mx + mz))

    # 3) Inject noise (static or drifted).
    if args.noise == "none":
        circ = base

    elif args.noise == "biased_data":
        if schedule is None:
            circ = apply_biased_data_noise(
                base, p=float(args.p), eta=float(args.eta),
                data_qubits=data_qubits, bias_y=False,
            )
        else:
            circ = apply_biased_data_noise_with_schedule(
                base, schedule=schedule,
                data_qubits=data_qubits, bias_y=False,
            )

    elif args.noise == "biased_circuit":
        if schedule is None:
            spec_noise = BiasedCircuitNoiseSpec(
                p=float(args.p), eta=float(args.eta), bias_y=False,
                meas_flip_p=float(args.p) if args.p_flip is None else float(args.p_flip),
                reset_flip_p=float(args.p) if args.p_flip is None else float(args.p_flip),
            )
            circ = apply_biased_circuit_noise(base, spec=spec_noise, qubits=all_qubits)
        else:
            circ = apply_biased_circuit_noise_with_schedule(
                base, schedule=schedule, bias_y=False, qubits=all_qubits,
                meas_flip_p=None, reset_flip_p=None, idle_scale=1.0,
            )

    elif args.noise == "drifting_biased_circuit":
        drift_schedule = make_drifting_schedule(
            base_p=float(args.p),
            base_eta=float(args.eta),
            drift_p=float(args.drift_p),
            drift_eta=float(args.drift_eta),
            period_ticks=args.drift_period_ticks,
        )
        circ = apply_biased_circuit_noise_with_schedule(
            base, schedule=drift_schedule, bias_y=False, qubits=all_qubits,
            meas_flip_p=None, reset_flip_p=None, idle_scale=1.0,
        )
    else:
        raise SystemExit(f"Unknown noise mode: {args.noise}")

    # 4) DEM must compile.
    try:
        circ.detector_error_model(approximate_disjoint_errors=True)
    except Exception:
        print("FAILED: circuit.detector_error_model() -- stop and fix circuit/noise.",
              file=sys.stderr)
        raise

    # 5) Sample detectors + observables.
    sampler = circ.compile_detector_sampler()
    dets, obs = sampler.sample(args.shots, separate_observables=True)
    dets = dets.astype(np.uint8)
    obs = obs.astype(np.uint8)

    # 6) Print summary and samples.
    print("=== BB QLDPC Sampling Summary ===")
    print(f"d={args.d}  (l={l}, m={m})")
    print(f"n(data)={n}  mx={mx}  mz={mz}  rounds={rounds}")
    print(f"noise={args.noise}  p={args.p}  eta={args.eta}")
    if args.noise == "drifting_biased_circuit":
        print(f"drift_p={args.drift_p}  drift_eta={args.drift_eta}  "
              f"period_ticks={args.drift_period_ticks}")
    elif args.drift != "none":
        print(f"drift={args.drift} target={args.drift_target} "
              f"amp={args.drift_amp} period_ticks={args.drift_period_ticks}")
    print(f"num_detectors={dets.shape[1]}  num_observables={obs.shape[1]}")
    print(f"shots={args.shots}  seed={args.seed}")

    pn = min(args.print_n, args.shots)
    print("\nFirst syndromes (detector events):")
    for i in range(pn):
        print(i, dets[i].tolist())

    print("\nFirst observables:")
    for i in range(pn):
        print(i, obs[i].tolist())

    # 7) Optional save with verification.
    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "d": args.d, "l": l, "m": m,
            "n": int(n), "mx": int(mx), "mz": int(mz),
            "rounds": rounds, "basis": args.basis,
            "noise": args.noise, "p": args.p, "eta": args.eta,
            "seed": args.seed, "shots": args.shots,
            "drift_p": args.drift_p, "drift_eta": args.drift_eta,
            "drift_period_ticks": args.drift_period_ticks,
        }

        np.savez_compressed(
            out_path,
            syndromes=dets,
            observables=obs,
            hx=hx,
            hz=hz,
            lx=lx,
            lz=lz,
            circuit=str(circ).encode("utf-8"),
            meta=json.dumps(meta).encode("utf-8"),
        )

        # Ensure .npz extension
        actual_path = out_path if str(out_path).endswith(".npz") else pathlib.Path(str(out_path) + ".npz")
        _verify_file(actual_path)


if __name__ == "__main__":
    main()
