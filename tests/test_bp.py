"""Test Min-Sum Belief Propagation decoder."""
import sys

import numpy as np
import torch


def test_bp_decoder() -> int:
    """Test Min-Sum BP decoder initialization and forward pass."""
    print("Testing BP decoder...")

    try:
        from astra_stim.sample_syndromes import bb_params
        from codes import create_bivariate_bicycle_codes
        from gnn_pipeline.bp_decoder import MinSumBPDecoder

        # Create test code
        l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows = bb_params(6)
        css, _, _ = create_bivariate_bicycle_codes(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows)
        hx = css.hx
        hz = css.hz
        mx, n = hx.shape
        mz = hz.shape[0]

        # Initialize decoder with combined PCM (CSS: stack hx and hz)
        print("  Initializing MinSumBPDecoder... ", end="")
        pcm = np.vstack([hx, hz])
        decoder = MinSumBPDecoder(pcm, max_iter=10, alpha=0.8, clamp_llr=20.0)
        print("OK")

        # Test on synthetic data (no errors)
        print("  Testing forward pass with zero syndrome... ", end="")
        zero_syndrome = np.zeros(mx + mz, dtype=np.float32)
        channel_llr = np.full(n, 10.0, dtype=np.float32)  # Strong "no error" signal

        syn_t = torch.from_numpy(zero_syndrome[np.newaxis, :]).float()
        llr_t = torch.from_numpy(channel_llr[np.newaxis, :]).float()

        marginals, hard_decision, converged = decoder(syn_t, llr_t)
        print("OK")

        # Verify output shapes
        print("  Verifying output shapes... ", end="")
        assert marginals.shape == (1, n), f"marginals shape {marginals.shape} != (1, {n})"
        assert hard_decision.shape == (1, n), f"hard_decision shape {hard_decision.shape} != (1, {n})"
        assert converged.shape == (1,), f"converged shape {converged.shape} != (1,)"
        print("OK")

        # Verify hard decision is binary
        print("  Verifying hard decisions are binary... ", end="")
        hd_val = hard_decision.numpy()
        assert np.all((hd_val == 0) | (hd_val == 1)), "Hard decision not binary"
        print("OK")

        # Test with batch
        print("  Testing batch forward pass... ", end="")
        batch_size = 4
        batch_syndrome = np.random.randint(0, 2, size=(batch_size, mx + mz)).astype(np.float32)
        batch_llr = np.random.randn(batch_size, n).astype(np.float32) * 5.0

        syn_batch = torch.from_numpy(batch_syndrome).float()
        llr_batch = torch.from_numpy(batch_llr).float()

        marginals, hard_decision, converged = decoder(syn_batch, llr_batch)
        print("OK")

        print(f"  Batch output shapes: marginals {marginals.shape}, hard_decision {hard_decision.shape}")
        assert marginals.shape == (batch_size, n)
        assert hard_decision.shape == (batch_size, n)

        # Test convergence rate
        print(f"  Convergence rate in batch: {converged.float().mean():.1%}")

        print("\nBP decoder tests passed!")
        return 0

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_bp_decoder())
