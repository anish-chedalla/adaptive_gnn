"""Tests for GNN model (TannerGNN, apply_correction, forward/backward)."""
from __future__ import annotations

import numpy as np
import torch
import pytest
from torch_geometric.data import Data

from gnn_pipeline.gnn_model import TannerGNN, apply_correction
from gnn_pipeline.tanner_graph import build_tanner_graph


def _make_dummy_data(n=72, mx=36, mz=36, device="cpu"):
    """Create a minimal Data object for GNN forward pass testing."""
    num_nodes = n + mx + mz
    x = torch.randn(num_nodes, 4, device=device)
    node_type = torch.zeros(num_nodes, dtype=torch.long, device=device)
    node_type[n:n+mx] = 1
    node_type[n+mx:] = 2

    # Simple edges: each check connected to a few data qubits
    src, dst, etype = [], [], []
    for c in range(mx):
        for q in range(min(3, n)):
            s = n + c
            d = (c * 3 + q) % n
            src.extend([s, d])
            dst.extend([d, s])
            etype.extend([0, 0])
    for c in range(mz):
        for q in range(min(3, n)):
            s = n + mx + c
            d = (c * 3 + q) % n
            src.extend([s, d])
            dst.extend([d, s])
            etype.extend([1, 1])

    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    edge_type_t = torch.tensor(etype, dtype=torch.long, device=device)

    return Data(x=x, edge_index=edge_index, edge_type=edge_type_t, node_type=node_type)


class TestApplyCorrection:
    def test_additive(self):
        llr = torch.tensor([1.0, 2.0, 3.0])
        corr = torch.tensor([0.1, -0.2, 0.3])
        result = apply_correction(llr, corr, "additive")
        expected = llr + corr
        assert torch.allclose(result, expected)

    def test_multiplicative(self):
        llr = torch.tensor([1.0, 2.0, 3.0])
        corr = torch.tensor([0.1, -0.2, 0.0])
        result = apply_correction(llr, corr, "multiplicative")
        expected = llr * torch.exp(corr)
        assert torch.allclose(result, expected)

    def test_both(self):
        llr = torch.tensor([1.0, 2.0, 3.0])
        add = torch.tensor([0.1, -0.2, 0.3])
        mul = torch.tensor([0.0, 0.1, -0.1])
        result = apply_correction(llr, (add, mul), "both")
        expected = llr * torch.exp(mul) + add
        assert torch.allclose(result, expected)

    def test_unknown_mode_raises(self):
        llr = torch.tensor([1.0])
        with pytest.raises(ValueError, match="Unknown correction mode"):
            apply_correction(llr, llr, "invalid")

    def test_batched(self):
        llr = torch.randn(16, 72)
        corr = torch.randn(16, 72)
        result = apply_correction(llr, corr, "additive")
        assert result.shape == (16, 72)


class TestTannerGNNForward:
    def test_default_output_shape(self):
        """Default (additive) mode: output is (n_data,)."""
        model = TannerGNN(node_feat_dim=4, hidden_dim=32, num_mp_layers=2)
        data = _make_dummy_data(n=72, mx=36, mz=36)
        out = model(data)
        assert out.shape == (72,)

    def test_multiplicative_output_shape(self):
        """Multiplicative mode: same shape as additive."""
        model = TannerGNN(hidden_dim=32, num_mp_layers=2, correction_mode="multiplicative")
        data = _make_dummy_data()
        out = model(data)
        assert out.shape == (72,)

    def test_both_output_shape(self):
        """Both mode: returns tuple of two (n_data,) tensors."""
        model = TannerGNN(hidden_dim=32, num_mp_layers=2, correction_mode="both")
        data = _make_dummy_data()
        out = model(data)
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0].shape == (72,)
        assert out[1].shape == (72,)

    def test_residual_and_layernorm(self):
        """Model with residual + layer norm should produce valid output."""
        model = TannerGNN(
            hidden_dim=32, num_mp_layers=3,
            use_residual=True, use_layer_norm=True,
        )
        data = _make_dummy_data()
        out = model(data)
        assert out.shape == (72,)
        assert out.isfinite().all()

    def test_backward_pass(self):
        """Gradients should flow through GNN -> output -> loss."""
        model = TannerGNN(hidden_dim=32, num_mp_layers=2)
        data = _make_dummy_data()
        out = model(data)
        loss = out.sum()
        loss.backward()
        # Check at least one parameter received gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad

    def test_backward_both_mode(self):
        """Gradients should flow in 'both' correction mode."""
        model = TannerGNN(hidden_dim=32, num_mp_layers=2, correction_mode="both")
        data = _make_dummy_data()
        add_out, mul_out = model(data)
        loss = add_out.sum() + mul_out.sum()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
        assert has_grad


class TestTannerGNNWithRealGraph:
    """Test GNN with a real Tanner graph from the [[72,12,6]] code."""

    @pytest.fixture
    def real_graph(self):
        from codes.codes_q import create_bivariate_bicycle_codes
        from codes.code_registry import get_code_params
        params = get_code_params("72_12_6")
        css, _, _ = create_bivariate_bicycle_codes(**params)
        hx = np.array(css.hx, dtype=np.uint8)
        hz = np.array(css.hz, dtype=np.uint8)
        node_type_np, edge_index_np, edge_type_np = build_tanner_graph(hx, hz)
        n = hx.shape[1]
        mx, mz = hx.shape[0], hz.shape[0]
        num_nodes = n + mx + mz

        x = torch.randn(num_nodes, 4)
        return Data(
            x=x,
            edge_index=torch.from_numpy(edge_index_np).long(),
            edge_type=torch.from_numpy(edge_type_np).long(),
            node_type=torch.from_numpy(node_type_np).long(),
        )

    def test_forward_real_graph(self, real_graph):
        model = TannerGNN(hidden_dim=32, num_mp_layers=2)
        out = model(real_graph)
        # [[72,12,6]] has 72 data qubits
        assert out.shape == (72,)
        assert out.isfinite().all()
