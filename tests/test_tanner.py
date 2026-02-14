"""Test Tanner graph construction."""
import sys

import numpy as np


def test_tanner_graph() -> int:
    """Test heterogeneous Tanner graph construction."""
    print("Testing Tanner graph construction...")

    try:
        from astra_stim.sample_syndromes import bb_params
        from codes import create_bivariate_bicycle_codes
        from gnn_pipeline.tanner_graph import build_tanner_graph

        # Create test code
        l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows = bb_params(6)
        css, _, _ = create_bivariate_bicycle_codes(l, m, A_x_pows, A_y_pows, B_x_pows, B_y_pows)
        hx = css.hx
        hz = css.hz
        mx, n = hx.shape
        mz = hz.shape[0]

        print(f"  Building Tanner graph for {n} data qubits, {mx} X checks, {mz} Z checks... ", end="")
        node_type, edge_index, edge_type = build_tanner_graph(hx, hz)
        print("OK")

        # Verify node types
        print("  Verifying node types... ", end="")
        expected_num_nodes = n + mx + mz
        assert len(node_type) == expected_num_nodes, f"Expected {expected_num_nodes} nodes, got {len(node_type)}"

        num_data = np.sum(node_type == 0)
        num_x_check = np.sum(node_type == 1)
        num_z_check = np.sum(node_type == 2)

        assert num_data == n, f"Expected {n} data nodes, got {num_data}"
        assert num_x_check == mx, f"Expected {mx} X-check nodes, got {num_x_check}"
        assert num_z_check == mz, f"Expected {mz} Z-check nodes, got {num_z_check}"
        print("OK")

        # Verify edge structure
        print("  Verifying edge structure... ", end="")
        assert edge_index.shape[0] == 2, f"edge_index shape[0] should be 2, got {edge_index.shape[0]}"
        num_edges = edge_index.shape[1]

        assert len(edge_type) == num_edges, f"edge_type length {len(edge_type)} != {num_edges}"
        # Count edge types
        x_edges = np.sum(edge_type == 0)
        z_edges = np.sum(edge_type == 1)
        print(f"OK ({num_edges} edges: {x_edges} X-edges, {z_edges} Z-edges)")

        # Verify X-edges connect data nodes to X-checks
        print("  Verifying X-edge connectivity... ", end="")
        for edge_idx in np.where(edge_type == 0)[0]:
            src, dst = edge_index[0, edge_idx], edge_index[1, edge_idx]
            src_type = node_type[src]
            dst_type = node_type[dst]

            # Should be bidirectional between data and X-check
            data_involved = (src_type == 0 or dst_type == 0)
            x_check_involved = (src_type == 1 or dst_type == 1)
            assert data_involved and x_check_involved, \
                f"X-edge {src}→{dst} doesn't connect data and X-check"
        print("OK")

        # Verify Z-edges connect data nodes to Z-checks
        print("  Verifying Z-edge connectivity... ", end="")
        for edge_idx in np.where(edge_type == 1)[0]:
            src, dst = edge_index[0, edge_idx], edge_index[1, edge_idx]
            src_type = node_type[src]
            dst_type = node_type[dst]

            # Should be bidirectional between data and Z-check
            data_involved = (src_type == 0 or dst_type == 0)
            z_check_involved = (src_type == 2 or dst_type == 2)
            assert data_involved and z_check_involved, \
                f"Z-edge {src}→{dst} doesn't connect data and Z-check"
        print("OK")

        print("\nTanner graph tests passed!")
        return 0

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_tanner_graph())
