"""Adversarial stress-testing framework for the QLDPC GNN-BP pipeline.

Systematically tests the decoder under conditions where BP is known to fail:
  1. High error rates near/above pseudo-threshold
  2. Extreme bias ratios (very high or very low eta)
  3. Rapid temporal drift (fast oscillation, large amplitude)
  4. Mismatched LLR (decoder believes wrong noise level)
  5. Dense syndromes (many errors per shot)
  6. BP convergence stress (few iterations, adversarial alpha)

Each test generates data, runs all available decoders, and reports comparative
LER with confidence intervals. Results are saved to a JSON summary.

Usage:
    # Run all stress tests
    python -m gnn_pipeline.stress_test --out_dir runs/stress

    # Run a specific test suite
    python -m gnn_pipeline.stress_test --suite high_p --out_dir runs/stress

    # Run with a trained GNN model
    python -m gnn_pipeline.stress_test --gnn_model runs/sup/best_model.pt --out_dir runs/stress

    # Quick mode (fewer shots, for CI/sanity)
    python -m gnn_pipeline.stress_test --quick --out_dir runs/stress
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from codes import create_bivariate_bicycle_codes
from codes.code_registry import get_code_params
from gnn_pipeline.bp_decoder import MinSumBPDecoder
from gnn_pipeline.drift_models import generate_drift_sequence
from gnn_pipeline.tanner_graph import build_tanner_graph


# ---------------------------------------------------------------------------
# Stress-test scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class StressScenario:
    """A single adversarial test scenario."""
    name: str
    description: str
    code: str              # code name from registry
    p: float               # base physical error rate
    eta: float             # Z-bias ratio
    shots: int             # number of samples
    drift_model: str = "none"
    drift_amp: float = 0.0
    drift_period: int = 500
    ou_theta: float = 0.1
    ou_sigma: float = 0.005
    rtn_delta: float = 0.01
    rtn_switch: float = 0.005
    # Decoder config overrides
    bp_max_iter: int = 100
    bp_alpha: float = 0.8
    # Mismatch testing: if set, decoder uses this p instead of true p
    decoder_p_override: Optional[float] = None


def build_scenario_suites(quick: bool = False) -> Dict[str, List[StressScenario]]:
    """Build all adversarial test scenario suites.

    Args:
        quick: if True, use fewer shots for fast CI testing

    Returns:
        dict mapping suite name to list of scenarios
    """
    shots = 500 if quick else 3000

    suites: Dict[str, List[StressScenario]] = {}

    # ---- Suite 1: High error rates (near/above pseudo-threshold) ----
    # BB [[72,12,6]] pseudo-threshold ~4-5% for code-capacity biased noise.
    # BP degrades sharply near threshold and fails completely above it.
    suites["high_p"] = [
        StressScenario(
            name="p03_eta20",
            description="Moderate p=0.03, well below threshold",
            code="72_12_6", p=0.03, eta=20.0, shots=shots,
        ),
        StressScenario(
            name="p05_eta20",
            description="Near threshold p=0.05",
            code="72_12_6", p=0.05, eta=20.0, shots=shots,
        ),
        StressScenario(
            name="p07_eta20",
            description="Above threshold p=0.07",
            code="72_12_6", p=0.07, eta=20.0, shots=shots,
        ),
        StressScenario(
            name="p10_eta20",
            description="Far above threshold p=0.10 (BP should fail)",
            code="72_12_6", p=0.10, eta=20.0, shots=shots,
        ),
    ]

    # ---- Suite 2: Extreme bias ----
    # eta=1 (depolarizing): no Z-bias advantage, BP sees equal X/Z noise
    # eta=100: extreme bias, almost all Z-errors, very few X-errors
    # eta=0.1: inverted bias (mostly X-errors), tests if pipeline handles it
    suites["extreme_bias"] = [
        StressScenario(
            name="eta1_p03",
            description="Depolarizing (eta=1), p=0.03",
            code="72_12_6", p=0.03, eta=1.0, shots=shots,
        ),
        StressScenario(
            name="eta1_p05",
            description="Depolarizing (eta=1), p=0.05",
            code="72_12_6", p=0.05, eta=1.0, shots=shots,
        ),
        StressScenario(
            name="eta100_p03",
            description="Extreme Z-bias (eta=100), p=0.03",
            code="72_12_6", p=0.03, eta=100.0, shots=shots,
        ),
        StressScenario(
            name="eta100_p05",
            description="Extreme Z-bias (eta=100), p=0.05",
            code="72_12_6", p=0.05, eta=100.0, shots=shots,
        ),
    ]

    # ---- Suite 3: Rapid drift ----
    # Fast sine oscillation: decoder must adapt to rapidly changing noise
    # Large OU volatility: noise level is unpredictable
    # RTN with frequent switching: noise jumps between two levels rapidly
    suites["rapid_drift"] = [
        StressScenario(
            name="sine_fast_large",
            description="Fast sine drift (period=50, amp=0.03) at p=0.04",
            code="72_12_6", p=0.04, eta=20.0, shots=shots,
            drift_model="sine", drift_amp=0.03, drift_period=50,
        ),
        StressScenario(
            name="ou_volatile",
            description="High-volatility OU drift (sigma=0.02) at p=0.04",
            code="72_12_6", p=0.04, eta=20.0, shots=shots,
            drift_model="ou", ou_sigma=0.02, ou_theta=0.05,
        ),
        StressScenario(
            name="rtn_rapid",
            description="Rapid RTN switching (delta=0.025, switch=0.1) at p=0.04",
            code="72_12_6", p=0.04, eta=20.0, shots=shots,
            drift_model="rtn", rtn_delta=0.025, rtn_switch=0.1,
        ),
        StressScenario(
            name="sine_extreme",
            description="Extreme sine (amp close to p) at p=0.05",
            code="72_12_6", p=0.05, eta=20.0, shots=shots,
            drift_model="sine", drift_amp=0.045, drift_period=100,
        ),
    ]

    # ---- Suite 4: LLR mismatch (decoder uses wrong noise estimate) ----
    # This is the key real-world failure mode: calibration drift means
    # the decoder's prior doesn't match reality.
    suites["llr_mismatch"] = [
        StressScenario(
            name="underestimate_2x",
            description="True p=0.04, decoder thinks p=0.02 (underestimate 2x)",
            code="72_12_6", p=0.04, eta=20.0, shots=shots,
            decoder_p_override=0.02,
        ),
        StressScenario(
            name="overestimate_2x",
            description="True p=0.02, decoder thinks p=0.04 (overestimate 2x)",
            code="72_12_6", p=0.02, eta=20.0, shots=shots,
            decoder_p_override=0.04,
        ),
        StressScenario(
            name="underestimate_5x",
            description="True p=0.05, decoder thinks p=0.01 (underestimate 5x)",
            code="72_12_6", p=0.05, eta=20.0, shots=shots,
            decoder_p_override=0.01,
        ),
        StressScenario(
            name="wrong_bias",
            description="True eta=20, decoder thinks eta=1 (wrong bias model)",
            code="72_12_6", p=0.03, eta=20.0, shots=shots,
            # decoder uses eta=1 via p_override mechanism — handled specially
            decoder_p_override=-1.0,  # sentinel: means "use true p but wrong eta"
        ),
    ]

    # ---- Suite 5: Larger code (scaling test) ----
    # [[144,12,12]] has higher distance, so threshold is different.
    # Also tests whether GNN/BP scales to larger codes.
    suites["larger_code"] = [
        StressScenario(
            name="144_p03_eta20",
            description="[[144,12,12]] at p=0.03, eta=20",
            code="144_12_12", p=0.03, eta=20.0, shots=shots,
        ),
        StressScenario(
            name="144_p05_eta20",
            description="[[144,12,12]] at p=0.05, eta=20",
            code="144_12_12", p=0.05, eta=20.0, shots=shots,
        ),
        StressScenario(
            name="144_p05_drift",
            description="[[144,12,12]] at p=0.05 with sine drift",
            code="144_12_12", p=0.05, eta=20.0, shots=shots,
            drift_model="sine", drift_amp=0.03, drift_period=200,
        ),
    ]

    # ---- Suite 6: BP iteration stress ----
    # Very few BP iterations: tests how robust the decoder is when
    # not allowed to converge.
    suites["bp_stress"] = [
        StressScenario(
            name="bp5_p03",
            description="Only 5 BP iterations at p=0.03",
            code="72_12_6", p=0.03, eta=20.0, shots=shots,
            bp_max_iter=5,
        ),
        StressScenario(
            name="bp2_p03",
            description="Only 2 BP iterations at p=0.03 (nearly raw channel)",
            code="72_12_6", p=0.03, eta=20.0, shots=shots,
            bp_max_iter=2,
        ),
        StressScenario(
            name="alpha_low_p03",
            description="Low damping alpha=0.3 at p=0.03 (overdamped)",
            code="72_12_6", p=0.03, eta=20.0, shots=shots,
            bp_alpha=0.3,
        ),
        StressScenario(
            name="alpha_high_p03",
            description="High damping alpha=1.0 at p=0.03 (undamped, oscillation risk)",
            code="72_12_6", p=0.03, eta=20.0, shots=shots,
            bp_alpha=1.0,
        ),
    ]

    return suites


# ---------------------------------------------------------------------------
# Data generation (code-capacity, inline)
# ---------------------------------------------------------------------------

def generate_code_capacity_data(
    scenario: StressScenario,
    seed: int = 42,
) -> dict:
    """Generate code-capacity syndrome data for a stress scenario.

    Returns dict with keys matching the NPZ format expected by evaluate.py.
    """
    rng = np.random.default_rng(seed)

    # Build code
    params = get_code_params(scenario.code)
    css, _, _ = create_bivariate_bicycle_codes(**params)
    hx = css.hx.astype(np.uint8)
    hz = css.hz.astype(np.uint8)
    lx = css.lx.astype(np.uint8) if hasattr(css, 'lx') and css.lx is not None else np.eye(css.K, hx.shape[1], dtype=np.uint8)
    lz = css.lz.astype(np.uint8) if hasattr(css, 'lz') and css.lz is not None else np.eye(css.K, hz.shape[1], dtype=np.uint8)

    mx, n = hx.shape
    mz = hz.shape[0]
    k = css.K
    shots = scenario.shots

    # Generate per-shot p values (with drift)
    if scenario.drift_model != "none":
        p_values = generate_drift_sequence(
            model=scenario.drift_model,
            p_base=scenario.p,
            shots=shots,
            rng=rng,
            amp=scenario.drift_amp,
            period=scenario.drift_period,
            theta=scenario.ou_theta,
            sigma=scenario.ou_sigma,
            p_delta=scenario.rtn_delta,
            switch_prob=scenario.rtn_switch,
        )
    else:
        p_values = np.full(shots, scenario.p)

    p_values = np.clip(p_values, 1e-7, 0.5 - 1e-7).astype(np.float32)

    # Generate errors and syndromes
    syndromes_list = []
    observables_list = []
    z_errors_list = []
    x_errors_list = []

    for i in range(shots):
        p_i = float(p_values[i])
        eta = scenario.eta

        pz = p_i * eta / (eta + 1)
        px = p_i / (eta + 1)

        # Sample Z and X errors independently
        z_err = (rng.random(n) < pz).astype(np.uint8)
        x_err = (rng.random(n) < px).astype(np.uint8)

        # Compute syndromes: hx @ z_err = x_syndrome, hz @ x_err = z_syndrome
        x_syn = (hx @ z_err) % 2
        z_syn = (hz @ x_err) % 2

        syndrome = np.concatenate([x_syn, z_syn])  # (mx+mz,)

        # Observables: lx @ x_err XOR lz @ z_err
        obs_x = (lx @ x_err) % 2
        obs_z = (lz @ z_err) % 2
        obs = (obs_x + obs_z) % 2  # (k,)

        syndromes_list.append(syndrome)
        observables_list.append(obs)
        z_errors_list.append(z_err)
        x_errors_list.append(x_err)

    return {
        "syndromes": np.array(syndromes_list, dtype=np.uint8),
        "observables": np.array(observables_list, dtype=np.uint8),
        "z_errors": np.array(z_errors_list, dtype=np.uint8),
        "x_errors": np.array(x_errors_list, dtype=np.uint8),
        "p_values": p_values,
        "hx": hx,
        "hz": hz,
        "lx": lx,
        "lz": lz,
        "meta": {
            "n": n, "mx": mx, "mz": mz, "k": k,
            "p": scenario.p, "eta": scenario.eta,
            "noise": "code_capacity",
        },
    }


# ---------------------------------------------------------------------------
# Decoder runners
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if trials == 0:
        return (0.0, 1.0)
    p_hat = successes / trials
    denom = 1 + z * z / trials
    center = (p_hat + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass
class DecoderResult:
    name: str
    ler: float
    errors: int
    total: int
    convergence: float
    ci_low: float
    ci_high: float
    elapsed_s: float


def run_bp_decoder(
    data: dict,
    scenario: StressScenario,
    device: torch.device,
) -> DecoderResult:
    """Run batched BP decoder on generated data."""
    hx = data["hx"]
    hz = data["hz"]
    lx = data["lx"]
    lz = data["lz"]
    mx, n = hx.shape
    mz = hz.shape[0]
    shots = data["syndromes"].shape[0]

    # Determine decoder's view of noise
    if scenario.decoder_p_override is not None and scenario.decoder_p_override > 0:
        dec_p = scenario.decoder_p_override
        dec_eta = scenario.eta
    elif scenario.decoder_p_override is not None and scenario.decoder_p_override < 0:
        # Sentinel: wrong bias model (use true p, eta=1)
        dec_p = scenario.p
        dec_eta = 1.0
    else:
        dec_p = scenario.p
        dec_eta = scenario.eta

    dec_p = max(min(dec_p, 1.0 - 1e-7), 1e-7)
    pz = max(min(dec_p * dec_eta / (dec_eta + 1), 1.0 - 1e-7), 1e-7)
    px = max(min(dec_p / (dec_eta + 1), 1.0 - 1e-7), 1e-7)
    llr_z = float(math.log((1.0 - pz) / pz))
    llr_x = float(math.log((1.0 - px) / px))

    # Split syndrome
    syn = data["syndromes"].astype(np.float32)
    x_syn = syn[:, :mx]
    z_syn = syn[:, mx:]

    # Build decoders
    dec_z = MinSumBPDecoder(hx, max_iter=scenario.bp_max_iter,
                            alpha=scenario.bp_alpha, clamp_llr=20.0).to(device)
    dec_x = MinSumBPDecoder(hz, max_iter=scenario.bp_max_iter,
                            alpha=scenario.bp_alpha, clamp_llr=20.0).to(device)

    t0 = time.time()
    CHUNK = 512
    all_z_hard, all_x_hard, all_conv = [], [], []

    for start in range(0, shots, CHUNK):
        end = min(start + CHUNK, shots)
        B = end - start
        xs = torch.from_numpy(x_syn[start:end]).float().to(device)
        zs = torch.from_numpy(z_syn[start:end]).float().to(device)
        lz_t = torch.full((B, n), llr_z, dtype=torch.float32, device=device)
        lx_t = torch.full((B, n), llr_x, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, hz_out, cz = dec_z(xs, lz_t)
            _, hx_out, cx = dec_x(zs, lx_t)

        all_z_hard.append(hz_out.cpu().numpy())
        all_x_hard.append(hx_out.cpu().numpy())
        all_conv.append((cz & cx).cpu().numpy())

    z_err_pred = np.concatenate(all_z_hard, axis=0)
    x_err_pred = np.concatenate(all_x_hard, axis=0)
    conv = np.concatenate(all_conv, axis=0)

    # Check logical errors
    observables = data["observables"]
    logical_errors = 0
    for i in range(shots):
        pred_obs_z = (lz @ z_err_pred[i]) % 2
        pred_obs_x = (lx @ x_err_pred[i]) % 2
        pred_obs = (pred_obs_z + pred_obs_x) % 2
        if not np.array_equal(pred_obs, observables[i]):
            logical_errors += 1

    elapsed = time.time() - t0
    ler = logical_errors / shots
    ci_lo, ci_hi = wilson_ci(logical_errors, shots)

    return DecoderResult(
        name="BP",
        ler=ler,
        errors=logical_errors,
        total=shots,
        convergence=float(conv.sum()) / shots,
        ci_low=ci_lo,
        ci_high=ci_hi,
        elapsed_s=elapsed,
    )


def run_bposd_decoder(
    data: dict,
    scenario: StressScenario,
) -> Optional[DecoderResult]:
    """Run BP-OSD decoder."""
    try:
        from gnn_pipeline.bposd_decoder import run_css_bposd_decoder
    except ImportError:
        return None

    hx = data["hx"]
    hz = data["hz"]
    lx = data["lx"]
    lz = data["lz"]
    mx, n = hx.shape
    shots = data["syndromes"].shape[0]

    # Decoder noise estimate
    if scenario.decoder_p_override is not None and scenario.decoder_p_override > 0:
        dec_p = scenario.decoder_p_override
        dec_eta = scenario.eta
    elif scenario.decoder_p_override is not None and scenario.decoder_p_override < 0:
        dec_p = scenario.p
        dec_eta = 1.0
    else:
        dec_p = scenario.p
        dec_eta = scenario.eta

    dec_p = max(min(dec_p, 1.0 - 1e-7), 1e-7)
    pz = dec_p * dec_eta / (dec_eta + 1)
    px = dec_p / (dec_eta + 1)

    syn = data["syndromes"].astype(np.float32)
    x_syn = syn[:, :mx]
    z_syn = syn[:, mx:]
    observables = data["observables"]

    t0 = time.time()
    logical_errors = 0
    for i in range(shots):
        z_pred, x_pred = run_css_bposd_decoder(
            x_syn[i], z_syn[i], hx, hz, pz, px, max_iter=100, osd_order=0,
        )
        pred_obs_z = (lz @ z_pred) % 2
        pred_obs_x = (lx @ x_pred) % 2
        pred_obs = (pred_obs_z + pred_obs_x) % 2
        if not np.array_equal(pred_obs, observables[i]):
            logical_errors += 1

    elapsed = time.time() - t0
    ler = logical_errors / shots
    ci_lo, ci_hi = wilson_ci(logical_errors, shots)

    return DecoderResult(
        name="BP-OSD",
        ler=ler,
        errors=logical_errors,
        total=shots,
        convergence=1.0,  # OSD always produces an answer
        ci_low=ci_lo,
        ci_high=ci_hi,
        elapsed_s=elapsed,
    )


def run_oracle_bp(
    data: dict,
    scenario: StressScenario,
    device: torch.device,
) -> Optional[DecoderResult]:
    """Run BP with true per-shot p_values (oracle upper bound)."""
    if scenario.drift_model == "none" and scenario.decoder_p_override is None:
        return None  # No drift and no mismatch: oracle = stale

    hx = data["hx"]
    hz = data["hz"]
    lx = data["lx"]
    lz = data["lz"]
    mx, n = hx.shape
    shots = data["syndromes"].shape[0]
    p_values = data["p_values"]

    syn = data["syndromes"].astype(np.float32)
    x_syn = syn[:, :mx]
    z_syn = syn[:, mx:]
    observables = data["observables"]

    dec_z = MinSumBPDecoder(hx, max_iter=scenario.bp_max_iter,
                            alpha=scenario.bp_alpha, clamp_llr=20.0).to(device)
    dec_x = MinSumBPDecoder(hz, max_iter=scenario.bp_max_iter,
                            alpha=scenario.bp_alpha, clamp_llr=20.0).to(device)

    t0 = time.time()
    CHUNK = 512
    all_z_hard, all_x_hard, all_conv = [], [], []

    for start in range(0, shots, CHUNK):
        end = min(start + CHUNK, shots)
        B = end - start
        xs = torch.from_numpy(x_syn[start:end]).float().to(device)
        zs = torch.from_numpy(z_syn[start:end]).float().to(device)

        # Per-shot LLRs
        p_chunk = np.clip(p_values[start:end], 1e-7, 1.0 - 1e-7)
        pz_chunk = np.clip(p_chunk * scenario.eta / (scenario.eta + 1), 1e-7, 1.0 - 1e-7)
        px_chunk = np.clip(p_chunk / (scenario.eta + 1), 1e-7, 1.0 - 1e-7)
        llr_z_chunk = np.log((1.0 - pz_chunk) / pz_chunk).astype(np.float32)
        llr_x_chunk = np.log((1.0 - px_chunk) / px_chunk).astype(np.float32)

        lz_t = torch.from_numpy(
            np.broadcast_to(llr_z_chunk[:, None], (B, n)).copy()
        ).float().to(device)
        lx_t = torch.from_numpy(
            np.broadcast_to(llr_x_chunk[:, None], (B, n)).copy()
        ).float().to(device)

        with torch.no_grad():
            _, hz_out, cz = dec_z(xs, lz_t)
            _, hx_out, cx = dec_x(zs, lx_t)

        all_z_hard.append(hz_out.cpu().numpy())
        all_x_hard.append(hx_out.cpu().numpy())
        all_conv.append((cz & cx).cpu().numpy())

    z_err_pred = np.concatenate(all_z_hard, axis=0)
    x_err_pred = np.concatenate(all_x_hard, axis=0)
    conv = np.concatenate(all_conv, axis=0)

    logical_errors = 0
    for i in range(shots):
        pred_obs_z = (lz @ z_err_pred[i]) % 2
        pred_obs_x = (lx @ x_err_pred[i]) % 2
        pred_obs = (pred_obs_z + pred_obs_x) % 2
        if not np.array_equal(pred_obs, observables[i]):
            logical_errors += 1

    elapsed = time.time() - t0
    ler = logical_errors / shots
    ci_lo, ci_hi = wilson_ci(logical_errors, shots)

    return DecoderResult(
        name="Oracle BP",
        ler=ler,
        errors=logical_errors,
        total=shots,
        convergence=float(conv.sum()) / shots,
        ci_low=ci_lo,
        ci_high=ci_hi,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Run a single scenario
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: StressScenario,
    device: torch.device,
    run_bposd: bool = True,
    seed: int = 42,
) -> dict:
    """Run all decoders on a single stress scenario.

    Returns a dict with scenario info and decoder results.
    """
    print(f"\n{'='*60}")
    print(f"  {scenario.name}: {scenario.description}")
    print(f"  code={scenario.code} p={scenario.p} eta={scenario.eta} "
          f"drift={scenario.drift_model} shots={scenario.shots}")
    if scenario.decoder_p_override is not None:
        if scenario.decoder_p_override > 0:
            print(f"  LLR MISMATCH: decoder uses p={scenario.decoder_p_override}")
        elif scenario.decoder_p_override < 0:
            print(f"  BIAS MISMATCH: decoder uses eta=1 (true eta={scenario.eta})")
    print(f"{'='*60}")

    # Generate data
    t_gen = time.time()
    data = generate_code_capacity_data(scenario, seed=seed)
    print(f"  Data generated in {time.time() - t_gen:.1f}s")

    # Compute some data statistics
    syn = data["syndromes"].astype(np.float32)
    mx = data["hx"].shape[0]
    x_syn_weight = syn[:, :mx].sum(axis=1).mean()
    z_syn_weight = syn[:, mx:].sum(axis=1).mean()
    z_err_rate = data["z_errors"].mean()
    x_err_rate = data["x_errors"].mean()
    print(f"  Actual error rates: z={z_err_rate:.4f}, x={x_err_rate:.6f}")
    print(f"  Avg syndrome weight: x_checks={x_syn_weight:.1f}, z_checks={z_syn_weight:.1f}")

    if scenario.drift_model != "none":
        pv = data["p_values"]
        print(f"  p_values: min={pv.min():.4f}, max={pv.max():.4f}, "
              f"mean={pv.mean():.4f}, std={pv.std():.4f}")

    results = {"scenario": asdict(scenario), "decoders": []}

    # BP (stale LLR)
    bp_result = run_bp_decoder(data, scenario, device)
    results["decoders"].append(asdict(bp_result))
    print(f"\n  BP:       LER={bp_result.ler:.4f} [{bp_result.ci_low:.4f}, {bp_result.ci_high:.4f}] "
          f"conv={bp_result.convergence:.1%} ({bp_result.elapsed_s:.1f}s)")

    # Oracle BP (if drift or mismatch)
    oracle_result = run_oracle_bp(data, scenario, device)
    if oracle_result is not None:
        results["decoders"].append(asdict(oracle_result))
        gap = (bp_result.ler - oracle_result.ler) / max(oracle_result.ler, 1e-6) * 100
        print(f"  Oracle:   LER={oracle_result.ler:.4f} [{oracle_result.ci_low:.4f}, {oracle_result.ci_high:.4f}] "
              f"conv={oracle_result.convergence:.1%} (gap={gap:+.1f}%)")

    # BP-OSD
    if run_bposd:
        bposd_result = run_bposd_decoder(data, scenario)
        if bposd_result is not None:
            results["decoders"].append(asdict(bposd_result))
            improvement = (bp_result.ler - bposd_result.ler) / max(bp_result.ler, 1e-6) * 100
            print(f"  BP-OSD:   LER={bposd_result.ler:.4f} [{bposd_result.ci_low:.4f}, {bposd_result.ci_high:.4f}] "
                  f"(vs BP: {improvement:+.1f}%) ({bposd_result.elapsed_s:.1f}s)")

    # Data stats
    results["data_stats"] = {
        "actual_z_error_rate": float(z_err_rate),
        "actual_x_error_rate": float(x_err_rate),
        "avg_x_syndrome_weight": float(x_syn_weight),
        "avg_z_syndrome_weight": float(z_syn_weight),
    }
    if scenario.drift_model != "none":
        pv = data["p_values"]
        results["data_stats"]["p_value_min"] = float(pv.min())
        results["data_stats"]["p_value_max"] = float(pv.max())
        results["data_stats"]["p_value_mean"] = float(pv.mean())
        results["data_stats"]["p_value_std"] = float(pv.std())

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial stress-testing for QLDPC GNN-BP pipeline"
    )
    parser.add_argument("--suite", type=str, default=None,
                        help="Run only this suite (e.g. high_p, extreme_bias, rapid_drift, "
                             "llr_mismatch, larger_code, bp_stress). Default: all.")
    parser.add_argument("--gnn_model", type=str, default=None,
                        help="Path to trained GNN model checkpoint")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer shots (500 instead of 3000)")
    parser.add_argument("--no_bposd", action="store_true",
                        help="Skip BP-OSD (faster but less informative)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    suites = build_scenario_suites(quick=args.quick)

    if args.suite:
        if args.suite not in suites:
            print(f"Unknown suite '{args.suite}'. Available: {list(suites.keys())}")
            sys.exit(1)
        suites = {args.suite: suites[args.suite]}

    all_results = {}
    t_total = time.time()

    for suite_name, scenarios in suites.items():
        print(f"\n{'#'*60}")
        print(f"  SUITE: {suite_name} ({len(scenarios)} scenarios)")
        print(f"{'#'*60}")

        suite_results = []
        for scenario in scenarios:
            result = run_scenario(
                scenario, device,
                run_bposd=not args.no_bposd,
                seed=args.seed,
            )
            suite_results.append(result)

        all_results[suite_name] = suite_results

        # Print suite summary
        print(f"\n  --- Suite '{suite_name}' Summary ---")
        for r in suite_results:
            sc = r["scenario"]
            bp_res = r["decoders"][0]  # BP is always first
            bposd_res = None
            for d in r["decoders"]:
                if d["name"] == "BP-OSD":
                    bposd_res = d
                    break

            line = f"  {sc['name']:25s} BP={bp_res['ler']:.4f} (conv={bp_res['convergence']:.1%})"
            if bposd_res:
                line += f"  BP-OSD={bposd_res['ler']:.4f}"
            print(line)

    # Save all results
    results_path = out_dir / "stress_test_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print final summary table
    total_time = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  STRESS TEST COMPLETE ({total_time:.0f}s)")
    print(f"{'='*70}")

    total_scenarios = sum(len(s) for s in all_results.values())
    bp_failures = 0  # scenarios where BP conv < 50%
    bposd_wins = 0   # scenarios where BP-OSD beats BP by >5%

    for suite_name, suite_results in all_results.items():
        for r in suite_results:
            bp_res = r["decoders"][0]
            if bp_res["convergence"] < 0.5:
                bp_failures += 1
            for d in r["decoders"]:
                if d["name"] == "BP-OSD" and bp_res["ler"] > 0:
                    if (bp_res["ler"] - d["ler"]) / bp_res["ler"] > 0.05:
                        bposd_wins += 1

    print(f"  Total scenarios: {total_scenarios}")
    print(f"  BP convergence failures (<50%): {bp_failures}")
    print(f"  BP-OSD significant wins (>5% improvement): {bposd_wins}")
    print(f"  Results: {results_path}")


if __name__ == "__main__":
    main()
