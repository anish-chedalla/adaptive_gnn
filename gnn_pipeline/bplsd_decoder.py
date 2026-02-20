"""BP-LSD and BeliefFind decoder wrappers for CSS QLDPC codes.

BP-LSD applies Localised Statistics Decoding as a post-processor when BP
fails to converge, using soft information from BP to guide cluster growth.
This is generally stronger than BP-OSD for LDPC codes.

BeliefFind (BP + Union Find) is another strong post-processor that uses
the Union Find algorithm when BP fails.

Both decoders are from the `ldpc` library (>= 2.2.8).

Usage (via evaluate.py):
    python -m gnn_pipeline.evaluate --test_npz data/test.npz --bplsd --out_dir runs/eval
    python -m gnn_pipeline.evaluate --test_npz data/test.npz --belieffind --out_dir runs/eval
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from ldpc import BpLsdDecoder
except ImportError:
    raise ImportError(
        "The 'ldpc' package is required for BP-LSD decoding. "
        "Install it with: pip install ldpc>=2.2.8"
    )

try:
    from ldpc import BeliefFindDecoder
except ImportError:
    BeliefFindDecoder = None


def run_css_bplsd_decoder(
    x_syndrome: np.ndarray,
    z_syndrome: np.ndarray,
    hx: np.ndarray,
    hz: np.ndarray,
    error_rate_z: float,
    error_rate_x: float,
    max_iter: int = 100,
    lsd_order: int = 0,
    lsd_method: str = "LSD_CS",
    bits_per_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run separate CSS BP-LSD decoding.

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
        max_iter: maximum BP iterations before LSD fallback
        lsd_order: LSD order (0 = fastest, higher = stronger)
        lsd_method: LSD variant: 'LSD_0', 'LSD_E', or 'LSD_CS'
        bits_per_step: bits added to cluster per LSD step (smaller = more guided)

    Returns:
        (z_errors, x_errors) as numpy arrays of shape (n,)
    """
    n = hx.shape[1]

    # Decode Z-errors using hx (X-checks detect Z-errors)
    if x_syndrome.sum() > 0:
        dec_z = BpLsdDecoder(
            hx,
            error_rate=float(error_rate_z),
            bp_method="minimum_sum",
            max_iter=max_iter,
            ms_scaling_factor=0.75,
            lsd_order=lsd_order,
            lsd_method=lsd_method,
            bits_per_step=bits_per_step,
        )
        z_errors = dec_z.decode(x_syndrome.astype(np.int32))
    else:
        z_errors = np.zeros(n, dtype=np.int64)

    # Decode X-errors using hz (Z-checks detect X-errors)
    if z_syndrome.sum() > 0:
        dec_x = BpLsdDecoder(
            hz,
            error_rate=float(error_rate_x),
            bp_method="minimum_sum",
            max_iter=max_iter,
            ms_scaling_factor=0.75,
            lsd_order=lsd_order,
            lsd_method=lsd_method,
            bits_per_step=bits_per_step,
        )
        x_errors = dec_x.decode(z_syndrome.astype(np.int32))
    else:
        x_errors = np.zeros(n, dtype=np.int64)

    return z_errors.astype(np.int64), x_errors.astype(np.int64)


def run_css_bplsd_with_llr(
    x_syndrome: np.ndarray,
    z_syndrome: np.ndarray,
    hx: np.ndarray,
    hz: np.ndarray,
    per_qubit_llr_z: np.ndarray,
    per_qubit_llr_x: np.ndarray,
    max_iter: int = 100,
    lsd_order: int = 0,
    lsd_method: str = "LSD_CS",
    bits_per_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run CSS BP-LSD with per-qubit LLRs (for GNN + BP-LSD combination).

    Instead of a uniform error_rate, uses per-qubit error probabilities
    derived from GNN-corrected LLRs.

    Args:
        per_qubit_llr_z: (n,) per-qubit LLR for Z-errors (from GNN correction)
        per_qubit_llr_x: (n,) per-qubit LLR for X-errors (from GNN correction)
    """
    n = hx.shape[1]

    # Convert LLRs to error probabilities: p = sigmoid(-LLR) = 1 / (1 + exp(LLR))
    def _llr_to_prob(llr):
        llr = np.clip(llr, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(llr))

    prob_z = _llr_to_prob(per_qubit_llr_z).astype(np.float64)
    prob_x = _llr_to_prob(per_qubit_llr_x).astype(np.float64)

    # Clamp probabilities
    prob_z = np.clip(prob_z, 1e-7, 1.0 - 1e-7)
    prob_x = np.clip(prob_x, 1e-7, 1.0 - 1e-7)

    # Decode Z-errors using hx with per-qubit error channel
    if x_syndrome.sum() > 0:
        dec_z = BpLsdDecoder(
            hx,
            error_channel=prob_z.tolist(),
            bp_method="minimum_sum",
            max_iter=max_iter,
            ms_scaling_factor=0.75,
            lsd_order=lsd_order,
            lsd_method=lsd_method,
            bits_per_step=bits_per_step,
        )
        z_errors = dec_z.decode(x_syndrome.astype(np.int32))
    else:
        z_errors = np.zeros(n, dtype=np.int64)

    # Decode X-errors using hz with per-qubit error channel
    if z_syndrome.sum() > 0:
        dec_x = BpLsdDecoder(
            hz,
            error_channel=prob_x.tolist(),
            bp_method="minimum_sum",
            max_iter=max_iter,
            ms_scaling_factor=0.75,
            lsd_order=lsd_order,
            lsd_method=lsd_method,
            bits_per_step=bits_per_step,
        )
        x_errors = dec_x.decode(z_syndrome.astype(np.int32))
    else:
        x_errors = np.zeros(n, dtype=np.int64)

    return z_errors.astype(np.int64), x_errors.astype(np.int64)


def run_css_belief_find_decoder(
    x_syndrome: np.ndarray,
    z_syndrome: np.ndarray,
    hx: np.ndarray,
    hz: np.ndarray,
    error_rate_z: float,
    error_rate_x: float,
    max_iter: int = 100,
    uf_method: str = "inversion",
    bits_per_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run separate CSS BeliefFind (BP + Union Find) decoding.

    Args:
        x_syndrome: (mx,) X-check syndrome (int or float, 0/1)
        z_syndrome: (mz,) Z-check syndrome (int or float, 0/1)
        hx: (mx, n) X parity check matrix
        hz: (mz, n) Z parity check matrix
        error_rate_z: physical Z-error probability (pz)
        error_rate_x: physical X-error probability (px)
        max_iter: maximum BP iterations before UF fallback
        uf_method: UF local decoding method: 'inversion' or 'peeling'
        bits_per_step: bits added to cluster per UFD step

    Returns:
        (z_errors, x_errors) as numpy arrays of shape (n,)
    """
    if BeliefFindDecoder is None:
        raise ImportError(
            "BeliefFindDecoder not available in installed ldpc version. "
            "Install ldpc >= 2.2.8"
        )

    n = hx.shape[1]

    # Decode Z-errors using hx
    if x_syndrome.sum() > 0:
        dec_z = BeliefFindDecoder(
            hx,
            error_rate=float(error_rate_z),
            bp_method="minimum_sum",
            max_iter=max_iter,
            ms_scaling_factor=0.75,
            uf_method=uf_method,
            bits_per_step=bits_per_step,
        )
        z_errors = dec_z.decode(x_syndrome.astype(np.int32))
    else:
        z_errors = np.zeros(n, dtype=np.int64)

    # Decode X-errors using hz
    if z_syndrome.sum() > 0:
        dec_x = BeliefFindDecoder(
            hz,
            error_rate=float(error_rate_x),
            bp_method="minimum_sum",
            max_iter=max_iter,
            ms_scaling_factor=0.75,
            uf_method=uf_method,
            bits_per_step=bits_per_step,
        )
        x_errors = dec_x.decode(z_syndrome.astype(np.int32))
    else:
        x_errors = np.zeros(n, dtype=np.int64)

    return z_errors.astype(np.int64), x_errors.astype(np.int64)
