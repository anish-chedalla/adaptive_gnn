"""Generate synthetic code-capacity data with optional drift.

This bypasses Stim and directly samples Z/X Pauli errors, computing
syndromes via hx/hz matrix multiplication. Produces NPZ files in the
same format as astra_stim.sample_syndromes for pipeline compatibility.

Supports multiple drift models: sine, Ornstein-Uhlenbeck, Random Telegraph Noise.

Usage:
    # Static noise
    python -m gnn_pipeline.generate_codecap --p 0.02 --eta 20 --shots 10000 --out data/static.npz

    # Sinusoidal drift in p
    python -m gnn_pipeline.generate_codecap --p 0.02 --eta 20 --shots 10000 --drift_model sine --drift_amp 0.015 --drift_period 500 --out data/drift.npz

    # OU drift
    python -m gnn_pipeline.generate_codecap --p 0.02 --eta 20 --shots 10000 --drift_model ou --drift_amp 0.015 --ou_theta 0.1 --ou_sigma 0.005 --out data/drift_ou.npz

    # RTN drift
    python -m gnn_pipeline.generate_codecap --p 0.02 --eta 20 --shots 10000 --drift_model rtn --rtn_delta 0.01 --rtn_switch 0.005 --out data/drift_rtn.npz

    # Specific code size
    python -m gnn_pipeline.generate_codecap --code 144_12_12 --p 0.02 --eta 20 --shots 5000 --out data/big.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

import numpy as np

from codes.codes_q import create_bivariate_bicycle_codes
from codes.code_registry import get_code_params, list_codes
from gnn_pipeline.drift_models import generate_drift_sequence


def generate_code_capacity_data(
    hx: np.ndarray,
    hz: np.ndarray,
    lx: np.ndarray,
    lz: np.ndarray,
    shots: int,
    p_base: float,
    eta: float,
    drift_model: str = "none",
    drift_amp: float = 0.0,
    drift_period: int = 500,
    ou_theta: float = 0.1,
    ou_sigma: float = 0.005,
    rtn_delta: float = 0.01,
    rtn_switch: float = 0.005,
    seed: int = 42,
) -> dict:
    """Generate code-capacity syndrome data with optional drift.

    Args:
        hx: (mx, n) X parity check matrix
        hz: (mz, n) Z parity check matrix
        lx: (kx, n) X logical operators
        lz: (kz, n) Z logical operators
        shots: number of samples
        p_base: base physical error rate
        eta: Z-bias ratio (pZ/pX)
        drift_model: 'none', 'sine', 'ou', or 'rtn'
        drift_amp: amplitude for sine drift
        drift_period: period for sine drift
        ou_theta: OU mean-reversion rate
        ou_sigma: OU volatility
        rtn_delta: RTN half-distance between states
        rtn_switch: RTN switching probability per shot
        seed: random seed

    Returns:
        dict with syndromes, observables, p_values, z_errors, x_errors
    """
    n = hx.shape[1]
    rng = np.random.default_rng(seed)

    # Generate the entire drift sequence upfront
    p_values = generate_drift_sequence(
        model=drift_model,
        p_base=p_base,
        shots=shots,
        rng=rng,
        amp=drift_amp,
        period=drift_period,
        theta=ou_theta,
        sigma=ou_sigma,
        p_delta=rtn_delta,
        switch_prob=rtn_switch,
    )

    all_syn = []
    all_obs = []
    all_z_err = []
    all_x_err = []

    for i in range(shots):
        p_t = p_values[i]

        # Per-Pauli rates
        pz = p_t * eta / (eta + 1)
        px = p_t / (eta + 1)

        # Sample errors
        z_err = (rng.random(n) < pz).astype(np.uint8)
        x_err = (rng.random(n) < px).astype(np.uint8)

        # Compute syndromes
        x_syndrome = (hx @ z_err) % 2  # X-checks detect Z-errors
        z_syndrome = (hz @ x_err) % 2  # Z-checks detect X-errors
        syndrome = np.concatenate([x_syndrome, z_syndrome])

        # Compute observables
        obs_from_z = (lx @ z_err) % 2
        obs_from_x = (lz @ x_err) % 2
        observable = np.concatenate([obs_from_z, obs_from_x])

        all_syn.append(syndrome)
        all_obs.append(observable)
        all_z_err.append(z_err)
        all_x_err.append(x_err)

    return {
        "syndromes": np.array(all_syn, dtype=np.float32),
        "observables": np.array(all_obs, dtype=np.float32),
        "p_values": np.array(p_values, dtype=np.float32),
        "z_errors": np.array(all_z_err, dtype=np.float32),
        "x_errors": np.array(all_x_err, dtype=np.float32),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate code-capacity data")
    parser.add_argument("--code", type=str, default="72_12_6",
                        choices=list_codes(),
                        help="Code to use (default: 72_12_6)")
    parser.add_argument("--p", type=float, default=0.02, help="Base error rate")
    parser.add_argument("--eta", type=float, default=20.0, help="Z-bias ratio")
    parser.add_argument("--shots", type=int, default=10000, help="Number of shots")
    parser.add_argument("--drift_model", type=str, default=None,
                        choices=["none", "sine", "ou", "rtn"],
                        help="Drift model (default: inferred from drift_amp)")
    parser.add_argument("--drift_amp", type=float, default=0.0,
                        help="Sinusoidal drift amplitude for p (0=static)")
    parser.add_argument("--drift_period", type=int, default=500,
                        help="Drift period in shots (sine model)")
    parser.add_argument("--ou_theta", type=float, default=0.1,
                        help="OU mean-reversion rate")
    parser.add_argument("--ou_sigma", type=float, default=0.005,
                        help="OU volatility")
    parser.add_argument("--rtn_delta", type=float, default=0.01,
                        help="RTN half-distance between states")
    parser.add_argument("--rtn_switch", type=float, default=0.005,
                        help="RTN switching probability per shot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, required=True, help="Output NPZ path")

    args = parser.parse_args(argv)

    # Infer drift model from drift_amp for backward compatibility
    if args.drift_model is None:
        drift_model = "sine" if args.drift_amp > 0 else "none"
    else:
        drift_model = args.drift_model

    # Build the code
    code_params = get_code_params(args.code)
    print(f"Building [[{args.code}]] bivariate bicycle code...")
    css, _, _ = create_bivariate_bicycle_codes(**code_params)
    hx = np.array(css.hx, dtype=np.uint8)
    hz = np.array(css.hz, dtype=np.uint8)
    lx = np.array(css.lx, dtype=np.uint8)
    lz = np.array(css.lz, dtype=np.uint8)

    n, mx, mz = hx.shape[1], hx.shape[0], hz.shape[0]
    print(f"Code: n={n}, mx={mx}, mz={mz}, k={lx.shape[0]}")

    # Drift description
    if drift_model == "none":
        drift_str = "none"
    elif drift_model == "sine":
        drift_str = f"sine(amp={args.drift_amp}, period={args.drift_period})"
    elif drift_model == "ou":
        drift_str = f"ou(theta={args.ou_theta}, sigma={args.ou_sigma})"
    elif drift_model == "rtn":
        drift_str = f"rtn(delta={args.rtn_delta}, switch={args.rtn_switch})"
    else:
        drift_str = drift_model

    print(f"Generating {args.shots} shots: p={args.p}, eta={args.eta}, drift={drift_str}")

    result = generate_code_capacity_data(
        hx, hz, lx, lz,
        shots=args.shots,
        p_base=args.p,
        eta=args.eta,
        drift_model=drift_model,
        drift_amp=args.drift_amp,
        drift_period=args.drift_period,
        ou_theta=args.ou_theta,
        ou_sigma=args.ou_sigma,
        rtn_delta=args.rtn_delta,
        rtn_switch=args.rtn_switch,
        seed=args.seed,
    )

    # Compute stats
    syn = result["syndromes"]
    obs = result["observables"]
    p_vals = result["p_values"]
    z_errs = result["z_errors"]
    x_errs = result["x_errors"]

    zero_syn = (syn.sum(axis=1) == 0).sum()
    obs_flips = (obs.sum(axis=1) > 0).sum()
    avg_z_wt = z_errs.sum(axis=1).mean()
    avg_x_wt = x_errs.sum(axis=1).mean()

    print(f"Zero-syndrome shots: {zero_syn}/{args.shots} ({100*zero_syn/args.shots:.1f}%)")
    print(f"Shots with observable flips: {obs_flips}/{args.shots} ({100*obs_flips/args.shots:.1f}%)")
    print(f"Avg error weight: Z={avg_z_wt:.2f}, X={avg_x_wt:.2f}")
    if drift_model != "none":
        print(f"p range: [{p_vals.min():.4f}, {p_vals.max():.4f}]")

    # Save
    meta = {
        "code": args.code,
        "n": int(n), "mx": int(mx), "mz": int(mz),
        "p": args.p, "eta": args.eta,
        "noise": "drifting_code_capacity" if drift_model != "none" else "code_capacity",
        "drift_model": drift_model,
        "drift_amp": args.drift_amp,
        "drift_period": args.drift_period,
        "ou_theta": args.ou_theta,
        "ou_sigma": args.ou_sigma,
        "rtn_delta": args.rtn_delta,
        "rtn_switch": args.rtn_switch,
        "shots": args.shots,
        "seed": args.seed,
    }

    np.savez_compressed(
        args.out,
        syndromes=syn,
        observables=obs,
        z_errors=z_errs,
        x_errors=x_errs,
        p_values=p_vals,
        hx=hx, hz=hz, lx=lx, lz=lz,
        meta=json.dumps(meta).encode("utf-8"),
    )
    print(f"Saved to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
