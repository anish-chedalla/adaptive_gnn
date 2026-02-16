"""Tests for drift models (sine, OU, RTN, dispatcher)."""
from __future__ import annotations

import numpy as np
import pytest

from gnn_pipeline.drift_models import (
    sine_drift,
    ou_drift,
    rtn_drift,
    generate_drift_sequence,
)


class TestSineDrift:
    def test_output_shape(self):
        p = sine_drift(0.02, 1000, amp=0.01, period=500)
        assert p.shape == (1000,)

    def test_range_clamped(self):
        p = sine_drift(0.02, 1000, amp=0.05, period=100)
        assert p.min() >= 1e-6
        assert p.max() <= 0.5

    def test_mean_near_base(self):
        """Over a full period, mean should be close to p_base."""
        p = sine_drift(0.02, 500, amp=0.01, period=500)
        assert abs(p.mean() - 0.02) < 0.002

    def test_amplitude(self):
        """Peak-to-peak should be approximately 2*amp."""
        p = sine_drift(0.1, 2000, amp=0.05, period=500)
        assert p.max() - p.min() > 0.08  # should be close to 0.1

    def test_zero_amplitude(self):
        """With zero amplitude, all values should equal p_base."""
        p = sine_drift(0.02, 100, amp=0.0, period=500)
        np.testing.assert_allclose(p, 0.02, atol=1e-10)


class TestOUDrift:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        p = ou_drift(0.02, 1000, rng, theta=0.1, sigma=0.005)
        assert p.shape == (1000,)

    def test_range_clamped(self):
        rng = np.random.default_rng(42)
        p = ou_drift(0.02, 5000, rng, theta=0.01, sigma=0.05)
        assert p.min() >= 1e-6
        assert p.max() <= 0.5

    def test_mean_reversion(self):
        """Over many samples with strong mean-reversion, mean should be near p_base."""
        rng = np.random.default_rng(42)
        p = ou_drift(0.02, 10000, rng, theta=0.5, sigma=0.005)
        assert abs(p.mean() - 0.02) < 0.005

    def test_starts_at_p_base(self):
        rng = np.random.default_rng(42)
        p = ou_drift(0.03, 100, rng)
        assert p[0] == 0.03

    def test_reproducible(self):
        """Same seed should produce same sequence."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        p1 = ou_drift(0.02, 100, rng1)
        p2 = ou_drift(0.02, 100, rng2)
        np.testing.assert_array_equal(p1, p2)


class TestRTNDrift:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        p = rtn_drift(0.02, 1000, rng, p_delta=0.01, switch_prob=0.005)
        assert p.shape == (1000,)

    def test_range_clamped(self):
        rng = np.random.default_rng(42)
        p = rtn_drift(0.02, 1000, rng, p_delta=0.01)
        assert p.min() >= 1e-6
        assert p.max() <= 0.5

    def test_two_state_values(self):
        """RTN should only take values p_base +/- p_delta."""
        rng = np.random.default_rng(42)
        p_base, p_delta = 0.1, 0.02
        p = rtn_drift(p_base, 5000, rng, p_delta=p_delta, switch_prob=0.05)
        unique_vals = np.unique(p)
        assert len(unique_vals) == 2
        expected = sorted([p_base - p_delta, p_base + p_delta])
        np.testing.assert_allclose(sorted(unique_vals), expected, atol=1e-10)

    def test_switch_frequency(self):
        """With high switch prob, both states should appear roughly equally."""
        rng = np.random.default_rng(42)
        p = rtn_drift(0.1, 10000, rng, p_delta=0.02, switch_prob=0.5)
        frac_high = np.mean(p > 0.1)
        assert 0.3 < frac_high < 0.7

    def test_no_switching(self):
        """With switch_prob=0, should stay in initial state."""
        rng = np.random.default_rng(42)
        p = rtn_drift(0.1, 100, rng, p_delta=0.02, switch_prob=0.0)
        # All values should be the same (initial state)
        assert len(np.unique(p)) == 1


class TestGenerateDriftSequence:
    def test_none_model(self):
        rng = np.random.default_rng(42)
        p = generate_drift_sequence("none", 0.02, 100, rng)
        np.testing.assert_allclose(p, 0.02)

    def test_sine_dispatch(self):
        rng = np.random.default_rng(42)
        p = generate_drift_sequence("sine", 0.02, 100, rng, amp=0.01, period=50)
        assert p.shape == (100,)
        # Should not be constant
        assert p.std() > 0.001

    def test_ou_dispatch(self):
        rng = np.random.default_rng(42)
        p = generate_drift_sequence("ou", 0.02, 100, rng, theta=0.1, sigma=0.005)
        assert p.shape == (100,)

    def test_rtn_dispatch(self):
        rng = np.random.default_rng(42)
        p = generate_drift_sequence("rtn", 0.02, 100, rng, p_delta=0.01, switch_prob=0.1)
        assert p.shape == (100,)

    def test_unknown_raises(self):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Unknown drift model"):
            generate_drift_sequence("invalid", 0.02, 100, rng)
