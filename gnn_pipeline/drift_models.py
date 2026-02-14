"""Drift models for time-varying physical error rates.

Provides several stochastic and deterministic drift models that produce
per-shot error rate sequences for code-capacity simulations.

Models:
    - sine: deterministic sinusoidal drift (existing behaviour)
    - ou: Ornstein-Uhlenbeck mean-reverting Gaussian process
    - rtn: Random Telegraph Noise (two-state Markov switching)
"""
from __future__ import annotations

import numpy as np


def sine_drift(
    p_base: float,
    shots: int,
    amp: float,
    period: int = 500,
) -> np.ndarray:
    """Deterministic sinusoidal drift around p_base.

    p(t) = p_base + amp * sin(2*pi*t / period)
    """
    t = np.arange(shots, dtype=np.float64)
    p = p_base + amp * np.sin(2.0 * np.pi * t / period)
    return np.clip(p, 1e-6, 0.5).astype(np.float64)


def ou_drift(
    p_base: float,
    shots: int,
    rng: np.random.Generator,
    theta: float = 0.1,
    sigma: float = 0.005,
    dt: float = 1.0,
) -> np.ndarray:
    """Ornstein-Uhlenbeck mean-reverting Gaussian process.

    Models gradual calibration drift:
        dp = -theta * (p - p_base) * dt + sigma * dW

    Args:
        p_base: long-term mean error rate
        shots: number of time steps
        rng: numpy random generator
        theta: mean-reversion rate (higher = faster reversion)
        sigma: volatility (noise amplitude)
        dt: time step size
    """
    p = np.empty(shots, dtype=np.float64)
    p[0] = p_base
    sqrt_dt = np.sqrt(dt)

    for t in range(1, shots):
        dW = rng.standard_normal() * sqrt_dt
        p[t] = p[t - 1] + theta * (p_base - p[t - 1]) * dt + sigma * dW

    return np.clip(p, 1e-6, 0.5)


def rtn_drift(
    p_base: float,
    shots: int,
    rng: np.random.Generator,
    p_delta: float = 0.01,
    switch_prob: float = 0.005,
) -> np.ndarray:
    """Random Telegraph Noise: two-state Markov switching.

    Models TLS (two-level-system) defects in qubits. The error rate
    switches between p_base - p_delta and p_base + p_delta with
    probability switch_prob per shot.

    Args:
        p_base: midpoint error rate
        shots: number of time steps
        rng: numpy random generator
        p_delta: half-distance between the two states
        switch_prob: probability of switching state per shot
    """
    p = np.empty(shots, dtype=np.float64)
    state = 1  # +1 or -1
    p[0] = p_base + state * p_delta

    for t in range(1, shots):
        if rng.random() < switch_prob:
            state = -state
        p[t] = p_base + state * p_delta

    return np.clip(p, 1e-6, 0.5)


def generate_drift_sequence(
    model: str,
    p_base: float,
    shots: int,
    rng: np.random.Generator,
    *,
    amp: float = 0.0,
    period: int = 500,
    theta: float = 0.1,
    sigma: float = 0.005,
    p_delta: float = 0.01,
    switch_prob: float = 0.005,
) -> np.ndarray:
    """Dispatcher: generate a per-shot error rate sequence.

    Args:
        model: one of 'none', 'sine', 'ou', 'rtn'
        p_base: base physical error rate
        shots: number of samples
        rng: numpy random generator
        amp, period: sine parameters
        theta, sigma: OU parameters
        p_delta, switch_prob: RTN parameters

    Returns:
        (shots,) array of per-shot error rates
    """
    if model == "none":
        return np.full(shots, p_base, dtype=np.float64)
    elif model == "sine":
        return sine_drift(p_base, shots, amp=amp, period=period)
    elif model == "ou":
        return ou_drift(p_base, shots, rng, theta=theta, sigma=sigma)
    elif model == "rtn":
        return rtn_drift(p_base, shots, rng, p_delta=p_delta, switch_prob=switch_prob)
    else:
        raise ValueError(f"Unknown drift model: {model!r}. Choose from: none, sine, ou, rtn")
