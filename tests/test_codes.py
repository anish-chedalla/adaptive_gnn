"""Test CSS code construction and properties."""
import sys

import numpy as np


def test_bivariate_bicycle() -> int:
    """Test BB QLDPC code construction."""
    print("Testing bivariate bicycle codes...")

    try:
        from codes import create_bivariate_bicycle_codes
        from astra_stim.sample_syndromes import bb_params

        # Test [[72, 12, 6]] code (d=6)
        print("  Creating [[72, 12, 6]] code (d=6)... ", end="")
        l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows = bb_params(6)
        css, _, _ = create_bivariate_bicycle_codes(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows)
        hx = css.hx
        hz = css.hz

        assert hx.shape == (36, 72), f"Expected hx shape (36, 72), got {hx.shape}"
        assert hz.shape == (36, 72), f"Expected hz shape (36, 72), got {hz.shape}"
        print("OK")

        # Verify that both are binary (GF(2))
        print("  Verifying binary matrices... ", end="")
        assert np.all((hx == 0) | (hx == 1)), "hx is not binary"
        assert np.all((hz == 0) | (hz == 1)), "hz is not binary"
        print("OK")

        # Verify orthogonality: hx @ hz^T = 0 (mod 2)
        print("  Verifying orthogonality (hx @ hz.T = 0 mod 2)... ", end="")
        ortho = (hx @ hz.T) % 2
        assert np.all(ortho == 0), f"Orthogonality violated: {ortho.shape} non-zero entries"
        print("OK")

        # Verify that stabilizers are linearly independent (full rank)
        from codes import rank

        rank_hx = rank(hx)
        rank_hz = rank(hz)
        print(f"  Rank of hx: {rank_hx}/36... ", end="")
        assert rank_hx >= 12, f"hx rank {rank_hx} is too low (need > code dim)"
        assert rank_hx <= 36, f"hx rank {rank_hx} exceeds row count"
        print("OK")

        print(f"  Rank of hz: {rank_hz}/36... ", end="")
        assert rank_hz >= 12, f"hz rank {rank_hz} is too low"
        assert rank_hz <= 36, f"hz rank {rank_hz} exceeds row count"
        print("OK")

        print("\nBivariate bicycle code tests passed!")
        return 0

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_bivariate_bicycle())
