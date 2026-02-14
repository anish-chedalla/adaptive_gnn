"""Test that all core modules can be imported and initialized."""
import sys


def test_imports() -> int:
    """Test all module imports."""
    print("Testing imports...")

    try:
        # Test basic packages
        print("  Importing numpy... ", end="")
        import numpy as np
        print("OK")

        print("  Importing torch... ", end="")
        import torch
        print("OK")

        print("  Importing torch_geometric... ", end="")
        from torch_geometric.data import Data
        print("OK")

        # Test codes package
        print("  Importing codes... ", end="")
        from codes import css_code, create_bivariate_bicycle_codes
        print("OK")

        # Test astra_stim package
        print("  Importing astra_stim... ", end="")
        from astra_stim import (
            CSSCodeSpec,
            build_qldpc_memory_circuit_text,
            apply_biased_data_noise,
            BiasedCircuitNoiseSpec,
        )
        print("OK")

        # Test gnn_pipeline submodules
        print("  Importing gnn_pipeline.tanner_graph... ", end="")
        from gnn_pipeline.tanner_graph import build_tanner_graph
        print("OK")

        print("  Importing gnn_pipeline.bp_decoder... ", end="")
        from gnn_pipeline.bp_decoder import MinSumBPDecoder
        print("OK")

        print("  Importing gnn_pipeline.gnn_model... ", end="")
        from gnn_pipeline.gnn_model import TannerGNN
        print("OK")

        print("  Importing gnn_pipeline.dataset... ", end="")
        from gnn_pipeline.dataset import build_graph_dataset
        print("OK")

        # Test CLI modules
        print("  Importing gnn_pipeline.build_dataset... ", end="")
        from gnn_pipeline import build_dataset
        print("OK")

        print("  Importing gnn_pipeline.train_selfsupervised... ", end="")
        from gnn_pipeline import train_selfsupervised
        print("OK")

        print("  Importing gnn_pipeline.evaluate... ", end="")
        from gnn_pipeline import evaluate
        print("OK")

        print("\nAll imports successful!")
        return 0

    except Exception as e:
        print(f"\nImport failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_imports())
