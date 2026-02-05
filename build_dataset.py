from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from gnn_pipeline.tanner_graph import (
    FORBIDDEN_KEYS,
    TannerGraph,
    reshape_detectors,
    build_node_features,
)


def _load_npz(path: str) -> dict:
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


@dataclass(frozen=True)
class NPZMeta:
    path: str
    n: int
    mx: int
    mz: int
    rounds_minus1: int
    num_obs: int
    has_circuit: bool


class GraphSampleBuilder:
    """
    Builds PyG-ready tensors (but does NOT require torch to be installed here).
    Training scripts will convert numpy arrays to torch tensors.
    """

    def __init__(self, npz_path: str):
        self.raw = _load_npz(npz_path)
        self.path = npz_path

        # Leakage handling: ignore forbidden keys, never propagate them.
        # (We do NOT fail if they exist in the NPZ; we fail only if they reach tensors.)
        for k in FORBIDDEN_KEYS:
            if k in self.raw:
                # ignore silently; DO NOT store into samples
                pass

        if "hx" not in self.raw or "hz" not in self.raw:
            raise ValueError(f"{npz_path}: missing hx/hz")
        if "syndromes" not in self.raw:
            raise ValueError(f"{npz_path}: missing syndromes (detector bits)")
        if "observables" not in self.raw:
            raise ValueError(f"{npz_path}: missing observables")

        self.hx = self.raw["hx"].astype(np.uint8)
        self.hz = self.raw["hz"].astype(np.uint8)

        self.tg = TannerGraph.from_hx_hz(self.hx, self.hz)

        self.syndromes = self.raw["syndromes"].astype(np.uint8)
        self.observables = self.raw["observables"].astype(np.uint8)

        mx, mz, n = self.tg.mx, self.tg.mz, self.tg.n
        self.dets_time = reshape_detectors(self.syndromes, mx=mx, mz=mz)  # [shots, T, m]
        self.shots = self.dets_time.shape[0]
        self.T = self.dets_time.shape[1]  # rounds-1
        self.m = mx + mz

        self.has_circuit = "circuit" in self.raw

    def meta(self) -> NPZMeta:
        return NPZMeta(
            path=self.path,
            n=self.tg.n,
            mx=self.tg.mx,
            mz=self.tg.mz,
            rounds_minus1=self.T,
            num_obs=self.observables.shape[1],
            has_circuit=self.has_circuit,
        )

    def make_selfsup_item(self, shot: int, t: int, W: int) -> dict:
        """
        Self-supervised item:
          - input window: dets_time[shot, t-W:t, :]  (W slices)
          - target: next slice dets_time[shot, t, :]

        Constraints:
          W <= t <= T-1
        """
        if not (0 <= shot < self.shots):
            raise IndexError
        if not (W <= t <= self.T - 1):
            raise IndexError(f"t={t} out of range for W={W}, T={self.T}")

        window = self.dets_time[shot, t - W:t, :]          # [W, m]
        target = self.dets_time[shot, t, :]                # [m]
        # Convert to per-check window: [m, W]
        check_window = window.transpose(1, 0).astype(np.float32)

        x = build_node_features(self.tg, check_window=check_window, W=W)  # [N, F]
        N = self.tg.num_nodes
        n = self.tg.n

        var_mask = np.zeros((N,), dtype=np.bool_)
        check_mask = np.zeros((N,), dtype=np.bool_)
        var_mask[:n] = True
        check_mask[n:] = True

        # y_det_node: [N,1], only check nodes used
        y_det_node = np.zeros((N, 1), dtype=np.float32)
        y_det_node[n:, 0] = target.astype(np.float32)

        sample = {
            "x": x,
            "edge_index": self.tg.edge_index.copy(),
            "edge_type": self.tg.edge_type.copy(),
            "var_mask": var_mask,
            "check_mask": check_mask,
            "y_det": y_det_node,
            # optional extras for debugging only
            "npz_path": self.path,
            "shot": int(shot),
            "t": int(t),
        }
        return sample

    def make_finetune_item(self, shot: int, W: int) -> dict:
        """
        Fine-tune / eval item:
          - input window: last W slices dets_time[shot, T-W:T, :]
          - BP syndrome: last slice dets_time[shot, -1, :]
          - observables: raw obs bits
        """
        if not (0 <= shot < self.shots):
            raise IndexError
        if W > self.T:
            raise ValueError(f"W={W} cannot exceed T={self.T}")

        window = self.dets_time[shot, self.T - W:self.T, :]    # [W, m]
        check_window = window.transpose(1, 0).astype(np.float32)  # [m, W]
        x = build_node_features(self.tg, check_window=check_window, W=W)

        N = self.tg.num_nodes
        n = self.tg.n
        var_mask = np.zeros((N,), dtype=np.bool_)
        check_mask = np.zeros((N,), dtype=np.bool_)
        var_mask[:n] = True
        check_mask[n:] = True

        syndrome_last = self.dets_time[shot, -1, :].astype(np.float32)  # [m]
        obs = self.observables[shot].astype(np.float32)                 # [k]

        sample = {
            "x": x,
            "edge_index": self.tg.edge_index.copy(),
            "edge_type": self.tg.edge_type.copy(),
            "var_mask": var_mask,
            "check_mask": check_mask,
            "syndrome_last": syndrome_last,
            "observables": obs,
            # do NOT store forbidden keys; circuit is allowed for eval mapping
            "circuit": self.raw["circuit"].item() if self.has_circuit else None,
            "npz_path": self.path,
            "shot": int(shot),
        }
        return sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", type=str, required=True, help="Input NPZ glob, e.g. data/*.npz")
    ap.add_argument("--mode", type=str, required=True, choices=["selfsup", "finetune"])
    ap.add_argument("--W", type=int, default=5, help="Detector window length")
    ap.add_argument("--out", type=str, default=None, help="Optional output .npz (packed arrays) OR .pt (torch save)")
    ap.add_argument("--max_items", type=int, default=0, help="If >0, cap number of items")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.in_glob))
    if not paths:
        raise SystemExit(f"No files match {args.in_glob}")

    W = int(args.W)
    all_items = []
    metas = []

    for p in paths:
        b = GraphSampleBuilder(p)
        metas.append(b.meta().__dict__)

        if args.mode == "selfsup":
            # shot × t where t in [W, T-1]
            for shot in range(b.shots):
                for t in range(W, b.T):
                    all_items.append(b.make_selfsup_item(shot=shot, t=t, W=W))
                    if args.max_items and len(all_items) >= args.max_items:
                        break
                if args.max_items and len(all_items) >= args.max_items:
                    break

        else:
            for shot in range(b.shots):
                all_items.append(b.make_finetune_item(shot=shot, W=W))
                if args.max_items and len(all_items) >= args.max_items:
                    break

        if args.max_items and len(all_items) >= args.max_items:
            break

    print(f"Built {len(all_items)} items from {len(paths)} npz files. W={W} mode={args.mode}")
    print("Example meta:", metas[0])

    if args.out is None:
        return

    out = args.out
    if out.endswith(".npz"):
        # Pack as object arrays (not ideal for PyG, but portable).
        np.savez_compressed(out, items=np.array(all_items, dtype=object), metas=np.array(metas, dtype=object))
        print(f"Saved packed dataset to {out}")
    else:
        # torch save done in training scripts (preferred). Here we just write a message.
        raise SystemExit("For .pt saving, use the training scripts (they torch.save after tensor conversion).")


if __name__ == "__main__":
    main()
