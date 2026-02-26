"""BP-OSD decoder wrappers for CSS QLDPC codes.

Provides a strong baseline decoder using the `ldpc` library's BpOsdDecoder.
BP-OSD applies Ordered Statistics Decoding (OSD) as a post-processor when
standard BP fails to converge, making it significantly more powerful than
plain BP alone.

Supports both:
  - Code-capacity CSS decoding (separate hx/hz)
  - Circuit-level DEM-based decoding

Usage (via evaluate.py):
    python -m gnn_pipeline.evaluate --test_npz data/test.npz --bposd --out_dir runs/eval_bposd
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from ldpc import BpOsdDecoder
except ImportError:
    raise ImportError(
        "The 'ldpc' package is required for BP-OSD decoding. "
        "Install it with: pip install ldpc>=2.2.8"
    )


def run_css_bposd_decoder(
    x_syndrome: np.ndarray,
    z_syndrome: np.ndarray,
    hx: np.ndarray,
    hz: np.ndarray,
    error_rate_z: float,
    error_rate_x: float,
    max_iter: int = 100,
    osd_order: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run separate CSS BP-OSD decoding.

    CSS decoding splits into two independent problems:
      - hx @ z_error = x_syndrome  (X-checks detect Z-errors)
      - hz @ x_error = z_syndrome  (Z-checks detect X-errors)

    Args:
        x_syndrome: (mx,) X-check syndrome (int or float, 0/1)
        z_syndrome: (mz,) Z-check syndrome (int or float, 0/1)
        hx: (mx, n) X parity check matrix
        hz: (mz, n) Z parity check matrix
        error_rate_z: physical Z-error probability (pz)
        error_rate_x: physical X-error probability (px)
        max_iter: maximum BP iterations
        osd_order: OSD order (0 = OSD-0, fastest; higher = stronger)

    Returns:
        (z_errors, x_errors) as numpy arrays of shape (n,)
    """
    n = hx.shape[1]

    osd_method = "osd0" if osd_order == 0 else "osd_cs"

    # Decode Z-errors using hx (X-checks detect Z-errors)
    if x_syndrome.sum() > 0:
        dec_z = BpOsdDecoder(
            hx,
            error_rate=float(error_rate_z),
            bp_method="ms",
            max_iter=max_iter,
            osd_method=osd_method,
            osd_order=osd_order,
            ms_scaling_factor=0.75,
        )
        z_errors = dec_z.decode(x_syndrome.astype(np.int32))
    else:
        z_errors = np.zeros(n, dtype=np.int64)

    # Decode X-errors using hz (Z-checks detect X-errors)
    if z_syndrome.sum() > 0:
        dec_x = BpOsdDecoder(
            hz,
            error_rate=float(error_rate_x),
            bp_method="ms",
            max_iter=max_iter,
            osd_method=osd_method,
            osd_order=osd_order,
            ms_scaling_factor=0.75,
        )
        x_errors = dec_x.decode(z_syndrome.astype(np.int32))
    else:
        x_errors = np.zeros(n, dtype=np.int64)

    return z_errors.astype(np.int64), x_errors.astype(np.int64)


def run_css_bposd_with_llr(
    x_syndrome: np.ndarray,
    z_syndrome: np.ndarray,
    hx: np.ndarray,
    hz: np.ndarray,
    per_qubit_llr_z: np.ndarray,
    per_qubit_llr_x: np.ndarray,
    max_iter: int = 100,
    osd_order: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run CSS BP-OSD with per-qubit LLRs (for GNN + BP-OSD combination).

    Instead of a uniform error_rate, uses per-qubit error probabilities
    derived from GNN-corrected LLRs.

    Args:
        x_syndrome: (mx,) X-check syndrome
        z_syndrome: (mz,) Z-check syndrome
        hx: (mx, n) X parity check matrix
        hz: (mz, n) Z parity check matrix
        per_qubit_llr_z: (n,) per-qubit LLR for Z-errors (from GNN correction)
        per_qubit_llr_x: (n,) per-qubit LLR for X-errors (from GNN correction)
        max_iter: maximum BP iterations
        osd_order: OSD order (0 = fastest)

    Returns:
        (z_errors, x_errors) as numpy arrays of shape (n,)
    """
    n = hx.shape[1]

    # Convert LLRs to error probabilities: p = sigmoid(-LLR) = 1 / (1 + exp(LLR))
    def _llr_to_prob(llr):
        llr = np.clip(llr, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(llr))

    prob_z = np.clip(_llr_to_prob(per_qubit_llr_z), 1e-7, 1.0 - 1e-7).astype(np.float64)
    prob_x = np.clip(_llr_to_prob(per_qubit_llr_x), 1e-7, 1.0 - 1e-7).astype(np.float64)

    osd_method = "osd0" if osd_order == 0 else "osd_cs"

    # Decode Z-errors using hx with per-qubit error channel
    if x_syndrome.sum() > 0:
        dec_z = BpOsdDecoder(
            hx,
            error_channel=prob_z.tolist(),
            bp_method="ms",
            max_iter=max_iter,
            osd_method=osd_method,
            osd_order=osd_order,
            ms_scaling_factor=0.75,
        )
        z_errors = dec_z.decode(x_syndrome.astype(np.int32))
    else:
        z_errors = np.zeros(n, dtype=np.int64)

    # Decode X-errors using hz with per-qubit error channel
    if z_syndrome.sum() > 0:
        dec_x = BpOsdDecoder(
            hz,
            error_channel=prob_x.tolist(),
            bp_method="ms",
            max_iter=max_iter,
            osd_method=osd_method,
            osd_order=osd_order,
            ms_scaling_factor=0.75,
        )
        x_errors = dec_x.decode(z_syndrome.astype(np.int32))
    else:
        x_errors = np.zeros(n, dtype=np.int64)

    return z_errors.astype(np.int64), x_errors.astype(np.int64)


def run_dem_bposd_decoder(
    detectors: np.ndarray,
    dem_pcm: np.ndarray,
    error_probs: np.ndarray,
    obs_matrix: np.ndarray,
    max_iter: int = 100,
    osd_order: int = 0,
) -> Tuple[np.ndarray, bool]:
    """Run BP-OSD on DEM-derived PCM for circuit-level decoding.

    Args:
        detectors: (num_detectors,) raw detector events (0 or 1)
        dem_pcm: (num_detectors, num_errors) DEM-derived PCM
        error_probs: (num_errors,) prior fault probabilities
        obs_matrix: (num_errors, num_observables) observable flip map
        max_iter: maximum BP iterations
        osd_order: OSD order (0 = fastest)

    Returns:
        predicted_obs_flips: (num_observables,) predicted observable flips
        converged: whether BP-OSD found a valid correction
    """
    num_detectors, num_errors = dem_pcm.shape
    num_observables = obs_matrix.shape[1] if obs_matrix.ndim == 2 else 0

    # Handle trivial cases
    if num_errors == 0 or detectors.sum() == 0:
        return np.zeros(num_observables, dtype=np.int64), True

    osd_method = "osd0" if osd_order == 0 else "osd_cs"

    # Use per-variable error probabilities for accurate BP priors
    # DEM error probs span orders of magnitude — a single scalar is inadequate
    error_channel = np.clip(error_probs, 1e-10, 1.0 - 1e-10).astype(np.float64)

    # BP-OSD expects PCM with rows = checks (detectors), cols = variables (errors)
    decoder = BpOsdDecoder(
        dem_pcm,
        error_channel=list(error_channel),
        bp_method="ms",
        max_iter=max_iter,
        osd_method=osd_method,
        osd_order=osd_order,
        ms_scaling_factor=0.75,
    )

    decoded_faults = decoder.decode(detectors.astype(np.int32))
    decoded_faults = decoded_faults.astype(np.int64)

    # Check convergence: does the decoded error satisfy the syndrome?
    recomputed_syn = (dem_pcm @ decoded_faults) % 2
    converged = bool(np.array_equal(recomputed_syn, detectors.astype(np.int64)))

    # Map decoded faults to observable flips
    predicted_obs = (obs_matrix.T @ decoded_faults) % 2  # (num_observables,)

    return predicted_obs.astype(np.int64), converged
