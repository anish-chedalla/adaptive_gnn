"""DEM-based circuit-level decoder for QLDPC codes.

Extracts a Detector Error Model (DEM) from a Stim circuit, converts it to
a binary PCM, and uses BP (or BP-OSD) to decode circuit-level syndrome data.

The DEM maps fault mechanisms to detectors and observables, enabling decoding
of circuit-level noise (gate errors, measurement errors, idle errors, drift)
which the code-capacity PCM cannot handle.

Usage (via evaluate.py):
    python -m gnn_pipeline.evaluate --test_npz data/circuit.npz --mode circuit_level --out_dir runs/eval_dem
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import stim
import torch

from gnn_pipeline.bp_decoder import MinSumBPDecoder


def extract_dem_pcm(
    circuit_text: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract DEM-derived PCM and prior error probabilities from a Stim circuit.

    Args:
        circuit_text: Stim circuit as string (stored in NPZ under "circuit" key)

    Returns:
        dem_pcm: (num_detectors, num_errors) binary matrix
                 Rows = detectors (syndrome bits), Cols = fault mechanisms (variables for BP)
        error_probs: (num_errors,) prior probability per fault mechanism
        obs_matrix: (num_errors, num_observables) which observables each fault flips
    """
    circuit = stim.Circuit(circuit_text)
    dem = circuit.detector_error_model(approximate_disjoint_errors=True)

    # Count detectors and observables from the DEM
    num_detectors = circuit.num_detectors
    num_observables = circuit.num_observables

    # Build PCM rows from DEM error instructions
    pcm_rows = []
    obs_rows = []
    probs = []

    for instruction in dem.flattened():
        if instruction.type != "error":
            continue

        prob = instruction.args_copy()[0]
        if prob < 1e-15:
            continue  # skip effectively zero-probability errors

        det_set = set()
        obs_set = set()

        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                det_set.add(target.val)
            elif target.is_logical_observable_id():
                obs_set.add(target.val)

        # Build detector row (which detectors this fault triggers)
        det_row = np.zeros(num_detectors, dtype=np.uint8)
        for d in det_set:
            if d < num_detectors:
                det_row[d] = 1

        # Build observable row (which observables this fault flips)
        obs_row = np.zeros(num_observables, dtype=np.uint8)
        for o in obs_set:
            if o < num_observables:
                obs_row[o] = 1

        pcm_rows.append(det_row)
        obs_rows.append(obs_row)
        probs.append(prob)

    if len(pcm_rows) == 0:
        # Degenerate case: no error mechanisms
        dem_pcm = np.zeros((num_detectors, 0), dtype=np.uint8)
        error_probs = np.zeros(0, dtype=np.float64)
        obs_matrix = np.zeros((0, num_observables), dtype=np.uint8)
        return dem_pcm, error_probs, obs_matrix

    # Stack: each row corresponds to one fault mechanism
    # Shape: (num_errors, num_detectors)
    pcm_errors = np.array(pcm_rows, dtype=np.uint8)
    obs_matrix = np.array(obs_rows, dtype=np.uint8)
    error_probs = np.array(probs, dtype=np.float64)

    # Transpose so rows = detectors, cols = errors (standard PCM orientation for BP)
    # BP variables are the fault mechanisms, checks are detectors
    dem_pcm = pcm_errors.T  # (num_detectors, num_errors)

    return dem_pcm, error_probs, obs_matrix


def build_dem_bp_decoder(
    dem_pcm: np.ndarray,
    error_probs: np.ndarray,
    device: torch.device,
    max_iter: int = 100,
) -> Tuple[MinSumBPDecoder, np.ndarray]:
    """Build a reusable BP decoder for DEM-based decoding.

    Call this ONCE, then pass the decoder to run_dem_bp_decoder for each shot.

    Args:
        dem_pcm: (num_detectors, num_errors) DEM-derived PCM
        error_probs: (num_errors,) prior fault probabilities
        device: torch device
        max_iter: maximum BP iterations

    Returns:
        decoder: MinSumBPDecoder instance (on device)
        channel_llr: (num_errors,) pre-computed channel LLRs
    """
    p_clamped = np.clip(error_probs, 1e-10, 1.0 - 1e-10)
    channel_llr = np.log((1.0 - p_clamped) / p_clamped).astype(np.float32)

    decoder = MinSumBPDecoder(
        dem_pcm, max_iter=max_iter, alpha=0.8, clamp_llr=20.0
    )
    decoder = decoder.to(device)

    return decoder, channel_llr


def run_dem_bp_decoder(
    detectors: np.ndarray,
    dem_pcm: np.ndarray,
    error_probs: np.ndarray,
    obs_matrix: np.ndarray,
    device: torch.device,
    max_iter: int = 100,
    decoder: MinSumBPDecoder | None = None,
    channel_llr: np.ndarray | None = None,
) -> Tuple[np.ndarray, bool]:
    """Decode circuit-level syndrome using DEM-based BP.

    Args:
        detectors: (num_detectors,) raw detector events (0 or 1)
        dem_pcm: (num_detectors, num_errors) DEM-derived PCM
        error_probs: (num_errors,) prior fault probabilities
        obs_matrix: (num_errors, num_observables) observable flip map
        device: torch device
        max_iter: maximum BP iterations
        decoder: pre-built MinSumBPDecoder (from build_dem_bp_decoder). If None, creates one.
        channel_llr: pre-computed channel LLRs (from build_dem_bp_decoder). If None, computes them.

    Returns:
        predicted_obs_flips: (num_observables,) predicted observable flips
        converged: whether BP converged (syndrome satisfied)
    """
    num_detectors, num_errors = dem_pcm.shape
    num_observables = obs_matrix.shape[1] if obs_matrix.ndim == 2 else 0

    # Handle trivial cases
    if num_errors == 0 or detectors.sum() == 0:
        return np.zeros(num_observables, dtype=np.int64), True

    # Build decoder if not provided
    if decoder is None or channel_llr is None:
        decoder, channel_llr = build_dem_bp_decoder(
            dem_pcm, error_probs, device, max_iter
        )

    syn_t = torch.from_numpy(detectors[np.newaxis, :]).float().to(device)
    llr_t = torch.from_numpy(channel_llr[np.newaxis, :]).float().to(device)

    _, hard_decision, converged = decoder(syn_t, llr_t)

    decoded_faults = hard_decision[0].cpu().numpy()  # (num_errors,)
    conv = bool(converged[0])

    # Map decoded faults to observable flips
    predicted_obs = (obs_matrix.T @ decoded_faults) % 2  # (num_observables,)

    return predicted_obs.astype(np.int64), conv
