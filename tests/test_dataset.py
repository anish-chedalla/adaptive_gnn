"""Test dataset construction from NPZ files."""
import sys
import tempfile
import pathlib

import numpy as np


def test_dataset_construction() -> int:
    """Test build_graph_dataset function."""
    print("Testing dataset construction...")

    try:
        from astra_stim.sample_syndromes import bb_params
        from codes import create_bivariate_bicycle_codes
        from gnn_pipeline.dataset import build_graph_dataset

        # Create test code
        l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows = bb_params(6)
        css, _, _ = create_bivariate_bicycle_codes(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows)
        hx = css.hx
        hz = css.hz
        mx, n = hx.shape
        mz = hz.shape[0]

        # Create synthetic NPZ data
        print("  Creating synthetic test NPZ file... ", end="")
        shots = 10
        num_det_rounds = 10
        total_checks = mx + mz

        syndromes = np.random.randint(0, 2, size=(shots, num_det_rounds, total_checks)).astype(np.float32)
        syndromes = syndromes.reshape(shots, -1)  # Flatten to (shots, num_detectors)

        observables = np.random.randint(0, 2, size=(shots, 12)).astype(np.float32)

        meta_dict = {
            "n": int(n),
            "mx": int(mx),
            "mz": int(mz),
            "p": 0.01,
            "eta": 20.0,
        }

        import json
        meta_bytes = json.dumps(meta_dict).encode("utf-8")

        # Create temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            test_npz = pathlib.Path(tmpdir) / "test.npz"
            np.savez(
                test_npz,
                syndromes=syndromes,
                observables=observables,
                hx=hx,
                hz=hz,
                lx=np.eye(n, dtype=np.uint8)[:12],
                lz=np.eye(n, dtype=np.uint8)[:12],
                meta=meta_bytes,
            )
            print("OK")

            # Test build_graph_dataset
            print("  Building graph dataset with W=4... ", end="")
            train_data, val_data, test_data, meta_out = build_graph_dataset(
                npz_paths=[str(test_npz)],
                W=4,
                mode="selfsup",
                seed=42,
            )
            print("OK")

            # Verify output
            print("  Verifying dataset properties... ", end="")
            total_samples = len(train_data) + len(val_data) + len(test_data)
            assert total_samples > 0, "No samples generated"

            # Verify split (80/10/10)
            n_train_expected = int(0.8 * total_samples)
            n_val_expected = int(0.1 * total_samples)

            assert len(train_data) >= n_train_expected - 1, "Train split too small"
            assert len(val_data) >= n_val_expected - 1, "Val split too small"
            print("OK")

            # Verify sample properties
            print("  Verifying sample structure... ", end="")
            sample = train_data[0]

            assert hasattr(sample, "x"), "Sample missing 'x' (node features)"
            assert hasattr(sample, "edge_index"), "Sample missing 'edge_index'"
            assert hasattr(sample, "edge_type"), "Sample missing 'edge_type'"
            assert hasattr(sample, "node_type"), "Sample missing 'node_type'"
            assert hasattr(sample, "channel_llr"), "Sample missing 'channel_llr'"
            assert hasattr(sample, "window_syndromes"), "Sample missing 'window_syndromes'"
            assert hasattr(sample, "target_syndrome"), "Sample missing 'target_syndrome'"
            assert hasattr(sample, "observable"), "Sample missing 'observable'"
            print("OK")

            # Verify feature dimensions
            print("  Verifying feature dimensions... ", end="")
            num_nodes = sample.x.shape[0]
            assert num_nodes == n + mx + mz, f"num_nodes {num_nodes} != {n + mx + mz}"

            assert sample.x.shape[1] == 4, f"Feature dimension {sample.x.shape[1]} != 4"
            assert sample.channel_llr.shape[0] == n, f"channel_llr shape {sample.channel_llr.shape} != ({n},)"
            print("OK")

            # Verify metadata
            print("  Verifying metadata... ", end="")
            assert "W" in meta_out
            assert "mode" in meta_out
            assert meta_out["mode"] == "selfsup"
            assert meta_out["num_train"] == len(train_data)
            assert meta_out["num_val"] == len(val_data)
            assert meta_out["num_test"] == len(test_data)
            print("OK")

        print("\nDataset construction tests passed!")
        return 0

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_dataset_construction())
