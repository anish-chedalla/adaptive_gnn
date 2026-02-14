"""CLI to build torch_geometric datasets from NPZ syndrome files.

Usage:
    python -m gnn_pipeline.build_dataset --in_glob "data/toy.npz" --mode selfsup --W 4 --out "data/toy_graph.pt"
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from typing import List

import torch

from gnn_pipeline.dataset import build_graph_dataset


def main(argv: List[str] | None = None) -> int:
    """Main entry point for dataset building CLI."""
    parser = argparse.ArgumentParser(
        description="Build train/val/test datasets from syndrome NPZ files"
    )
    parser.add_argument(
        "--in_glob",
        type=str,
        required=True,
        help="Glob pattern for input NPZ files (e.g. 'data/*.npz')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["selfsup", "supervised"],
        default="selfsup",
        help="Dataset mode: self-supervised (temporal prediction) or supervised (logical error)",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=4,
        help="Window width (number of detector rounds per sample)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val/test split",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for PyTorch graph dataset (.pt file)",
    )

    args = parser.parse_args(argv)

    # Find all matching input files
    in_glob_pattern = args.in_glob
    npz_paths = sorted(glob.glob(in_glob_pattern))

    if not npz_paths:
        print(f"ERROR: No files matching glob pattern: {in_glob_pattern}", file=sys.stderr)
        return 1

    print(f"Found {len(npz_paths)} NPZ file(s):")
    for p in npz_paths:
        print(f"  {p}")

    # Build datasets
    print(f"\nBuilding {args.mode} dataset with W={args.W}...")
    train_data, val_data, test_data, meta = build_graph_dataset(
        npz_paths=npz_paths,
        W=args.W,
        mode=args.mode,
        seed=args.seed,
    )

    print(
        f"Dataset split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test"
    )

    # Create output directory if needed
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save dataset
    dataset_dict = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }
    torch.save(dataset_dict, str(out_path))
    print(f"Saved dataset to: {out_path}")

    # Verify file was written
    if not out_path.exists():
        print(f"ERROR: Failed to write dataset file: {out_path}", file=sys.stderr)
        return 1

    file_size = out_path.stat().st_size
    if file_size == 0:
        print(f"ERROR: Dataset file is empty: {out_path}", file=sys.stderr)
        return 1

    print(f"Dataset file size: {file_size} bytes")

    # Save metadata alongside
    meta_path = out_path.parent / (out_path.stem + "_meta.json")
    meta["outfile"] = str(out_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to: {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
