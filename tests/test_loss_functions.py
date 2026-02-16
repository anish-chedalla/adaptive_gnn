"""Tests for loss functions (weighted BCE, focal, logical-aware)."""
from __future__ import annotations

import torch
import pytest

from gnn_pipeline.loss_functions import weighted_bce_loss, focal_loss, logical_aware_loss


class TestWeightedBCE:
    def test_all_zero_targets(self):
        """Loss should be finite when all targets are zero (no errors)."""
        marginals = torch.full((4, 72), 0.02)
        targets = torch.zeros(4, 72)
        loss = weighted_bce_loss(marginals, targets, pos_weight=50.0)
        assert loss.isfinite().item()
        assert loss.item() > 0.0

    def test_all_one_targets(self):
        """Loss should be finite when all targets are one."""
        marginals = torch.full((4, 72), 0.98)
        targets = torch.ones(4, 72)
        loss = weighted_bce_loss(marginals, targets, pos_weight=50.0)
        assert loss.isfinite().item()
        assert loss.item() > 0.0

    def test_perfect_predictions(self):
        """Loss should be very small when predictions match targets."""
        targets = torch.zeros(4, 72)
        targets[:, :5] = 1.0
        marginals = targets * 0.99 + (1.0 - targets) * 0.01
        loss = weighted_bce_loss(marginals, targets, pos_weight=50.0)
        assert loss.isfinite().item()
        assert loss.item() < 1.0

    def test_gradient_flows(self):
        """Gradients should flow back through the loss."""
        marginals = torch.full((2, 72), 0.05, requires_grad=True)
        targets = torch.zeros(2, 72)
        targets[0, :3] = 1.0
        loss = weighted_bce_loss(marginals, targets)
        loss.backward()
        assert marginals.grad is not None
        assert marginals.grad.isfinite().all()


class TestFocalLoss:
    def test_basic_output(self):
        """Focal loss should return a finite positive scalar."""
        marginals = torch.full((4, 72), 0.02)
        targets = torch.zeros(4, 72)
        targets[:, :3] = 1.0
        loss = focal_loss(marginals, targets, alpha=0.25, gamma=2.0)
        assert loss.isfinite().item()
        assert loss.item() > 0.0

    def test_easy_examples_downweighted(self):
        """Higher gamma should reduce loss on easy examples."""
        marginals = torch.full((4, 72), 0.01)
        targets = torch.zeros(4, 72)
        loss_g0 = focal_loss(marginals, targets, alpha=0.25, gamma=0.0)
        loss_g2 = focal_loss(marginals, targets, alpha=0.25, gamma=2.0)
        # gamma=2 down-weights easy negatives, so loss should be smaller
        assert loss_g2.item() < loss_g0.item()

    def test_gradient_flows(self):
        """Gradients should flow back through focal loss."""
        marginals = torch.full((2, 72), 0.05, requires_grad=True)
        targets = torch.zeros(2, 72)
        targets[0, :3] = 1.0
        loss = focal_loss(marginals, targets)
        loss.backward()
        assert marginals.grad is not None
        assert marginals.grad.isfinite().all()


class TestLogicalAwareLoss:
    @pytest.fixture
    def code_matrices(self):
        """Minimal logical operators for testing."""
        k, n = 12, 72
        # Simple logical operators: each row has a few non-zero entries
        lx = torch.zeros(k, n)
        lz = torch.zeros(k, n)
        for i in range(k):
            lx[i, i * 6:(i + 1) * 6] = 1.0
            lz[i, i * 6:(i + 1) * 6] = 1.0
        return lx, lz

    def test_basic_output(self, code_matrices):
        """Logical-aware loss should return finite positive scalar."""
        lx, lz = code_matrices
        marg = torch.full((2, 72), 0.02)
        tgt = torch.zeros(2, 72)
        tgt[:, :3] = 1.0
        base_fn = lambda m, t: ((m - t) ** 2).mean()
        loss = logical_aware_loss(marg, marg, tgt, tgt, lx, lz, base_fn, logical_weight=0.1)
        assert loss.isfinite().item()
        assert loss.item() > 0.0

    def test_marginals_near_zero(self, code_matrices):
        """Should handle marginals near 0 without NaN."""
        lx, lz = code_matrices
        marg = torch.full((2, 72), 1e-6)
        tgt = torch.zeros(2, 72)
        base_fn = lambda m, t: ((m - t) ** 2).mean()
        loss = logical_aware_loss(marg, marg, tgt, tgt, lx, lz, base_fn)
        assert loss.isfinite().item()

    def test_marginals_near_one(self, code_matrices):
        """Should handle marginals near 1 without NaN (tests clamping fix)."""
        lx, lz = code_matrices
        marg = torch.full((2, 72), 0.999)
        tgt = torch.zeros(2, 72)
        base_fn = lambda m, t: ((m - t) ** 2).mean()
        loss = logical_aware_loss(marg, marg, tgt, tgt, lx, lz, base_fn)
        assert loss.isfinite().item()
        assert loss.item() >= 0.0

    def test_marginals_at_half(self, code_matrices):
        """Marginals at 0.5 => maximum uncertainty, XOR prob ~0.5."""
        lx, lz = code_matrices
        marg = torch.full((2, 72), 0.5)
        tgt = torch.zeros(2, 72)
        base_fn = lambda m, t: ((m - t) ** 2).mean()
        loss = logical_aware_loss(marg, marg, tgt, tgt, lx, lz, base_fn)
        assert loss.isfinite().item()

    def test_gradient_flows(self, code_matrices):
        """Gradients should flow through logical-aware loss."""
        lx, lz = code_matrices
        marg = torch.full((2, 72), 0.05, requires_grad=True)
        tgt = torch.zeros(2, 72)
        tgt[:, :3] = 1.0
        base_fn = lambda m, t: ((m - t) ** 2).mean()
        loss = logical_aware_loss(marg, marg, tgt, tgt, lx, lz, base_fn, logical_weight=0.1)
        loss.backward()
        assert marg.grad is not None
        assert marg.grad.isfinite().all()
