"""Unified trainer: combines code-capacity and circuit-level GNN-BP training.

Merges the best features of train_supervised.py and train_circuit.py into
a single entry point with unified CLI, supporting:
  - Code-capacity (separate CSS) and circuit-level (DEM) modes
  - Online data regeneration (Astra-style, infinite diversity)
  - All loss functions: mse, bce, weighted_bce, focal, coset, constraint, observable
  - Neural BP, FiLM conditioning, interleaved GNN-BP
  - AMP, curriculum learning, gradient accumulation
  - Multi-class Pauli output mode
  - Transfer learning (freeze backbone, replace readout)
  - TensorBoard-compatible JSON logging

Usage:
    # Code-capacity supervised training
    python -m gnn_pipeline.train_unified --mode code_capacity \
        --in_glob "data/supervised_train.npz" --loss focal \
        --epochs 20 --out_dir runs/unified

    # Code-capacity with online data regeneration
    python -m gnn_pipeline.train_unified --mode code_capacity \
        --online_regen --code_npz data/code_params.npz \
        --p_range 0.01 0.04 --samples_per_epoch 10000 \
        --loss constraint --epochs 30 --out_dir runs/online

    # Circuit-level training
    python -m gnn_pipeline.train_unified --mode circuit_level \
        --in_glob "data/circuit_*.npz" --loss observable \
        --epochs 20 --out_dir runs/circuit

    # Transfer learning: freeze backbone from pretrained, train readout only
    python -m gnn_pipeline.train_unified --mode code_capacity \
        --in_glob "data/*.npz" --pretrained runs/model/best_model.pt \
        --freeze_backbone --epochs 5 --out_dir runs/transfer
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader

from gnn_pipeline.bp_decoder import MinSumBPDecoder
from gnn_pipeline.dataset import build_graph_dataset, _load_npz, OnlineCodeCapDataset
from gnn_pipeline.dem_dataset import build_circuit_dataset
from gnn_pipeline.gnn_model import TannerGNN, apply_correction
from gnn_pipeline.loss_functions import (
    weighted_bce_loss, focal_loss, coset_loss, constraint_loss,
    syndrome_consistency_loss, observable_loss, observable_loss_decomposed,
)
from gnn_pipeline.monitoring import TrainingMonitor, GradientMonitor, EpochTimer


def _compute_code_capacity_loss(
    marginals_z, marginals_x, z_error, x_error,
    loss_name, x_syndrome, z_syndrome,
    hx_t, hz_t, lx_t, lz_t,
    pos_weight=50.0, syn_weight=0.1,
):
    """Compute loss for code-capacity mode."""
    if loss_name == "mse":
        return F.mse_loss(marginals_z, z_error) + F.mse_loss(marginals_x, x_error)
    elif loss_name == "focal":
        return focal_loss(marginals_z, z_error) + focal_loss(marginals_x, x_error)
    elif loss_name == "weighted_bce":
        return (weighted_bce_loss(marginals_z, z_error, pos_weight=pos_weight)
                + weighted_bce_loss(marginals_x, x_error, pos_weight=pos_weight))
    elif loss_name == "coset":
        return (coset_loss(marginals_z, z_error, x_syndrome, hx_t, lx_t)
                + coset_loss(marginals_x, x_error, z_syndrome, hz_t, lz_t))
    elif loss_name == "constraint":
        return (constraint_loss(marginals_z, x_syndrome, hx_t, lx_t, z_error)
                + constraint_loss(marginals_x, z_syndrome, hz_t, lz_t, x_error))
    else:
        base = F.binary_cross_entropy(marginals_z, z_error) + \
               F.binary_cross_entropy(marginals_x, x_error)
        if syn_weight > 0 and hx_t is not None:
            base = base + syn_weight * (
                syndrome_consistency_loss(marginals_z, x_syndrome, hx_t)
                + syndrome_consistency_loss(marginals_x, z_syndrome, hz_t)
            )
        return base


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified GNN-BP trainer")

    # Mode
    parser.add_argument("--mode", type=str, default="code_capacity",
                        choices=["code_capacity", "circuit_level"])

    # Data
    parser.add_argument("--in_glob", type=str, default=None,
                        help="Glob pattern for NPZ files (required unless --online_regen)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--W", type=int, default=4, help="Window width (code-capacity)")

    # Online data regeneration
    parser.add_argument("--online_regen", action="store_true",
                        help="Enable online data regeneration (Astra-style)")
    parser.add_argument("--code_npz", type=str, default=None,
                        help="NPZ file with code params (hx, hz, lx, lz) for online regen")
    parser.add_argument("--p_range", type=float, nargs=2, default=None,
                        help="Uniform p range for online regen, e.g. 0.01 0.04")
    parser.add_argument("--p_base", type=float, default=0.02,
                        help="Base p for online regen (if no p_range)")
    parser.add_argument("--eta", type=float, default=20.0,
                        help="Noise bias eta (Z/X ratio)")
    parser.add_argument("--samples_per_epoch", type=int, default=5000,
                        help="Samples per epoch for online regen")

    # GNN architecture
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_mp_layers", type=int, default=3)
    parser.add_argument("--correction_mode", type=str, default="additive",
                        choices=["additive", "multiplicative", "both"])
    parser.add_argument("--output_mode", type=str, default="correction",
                        choices=["correction", "multiclass"])
    parser.add_argument("--use_residual", action="store_true")
    parser.add_argument("--use_layer_norm", action="store_true")
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--use_film", action="store_true")
    parser.add_argument("--noise_feat_dim", type=int, default=1,
                        help="Dimension of noise features for FiLM conditioning")
    parser.add_argument("--dropout", type=float, default=0.1)

    # BP
    parser.add_argument("--bp_iters", type=int, default=10)
    parser.add_argument("--neural_bp", action="store_true")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accumulate_grad", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--amp", action="store_true")

    # Loss
    parser.add_argument("--loss", type=str, default="focal",
                        choices=["mse", "bce", "weighted_bce", "focal", "coset",
                                 "constraint", "observable"])
    parser.add_argument("--pos_weight", type=float, default=50.0)
    parser.add_argument("--syn_weight", type=float, default=0.1)
    parser.add_argument("--obs_weight", type=float, default=1.0)

    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable tier-based curriculum: easy p-values first, harder later")

    # Data augmentation
    parser.add_argument("--augment", action="store_true",
                        help="Enable syndrome augmentation (random bit-flips, p=0.01)")

    # Transfer learning
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze all layers except readout (transfer learning)")

    # Input standardization
    parser.add_argument("--standardize_input", action="store_true",
                        help="Standardize LLR input to zero-mean unit-variance per batch")

    # Logging
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (default: qldpc-gnn-bp)")

    # Output
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args(argv)

    # Validate argument combinations
    if args.mode == "code_capacity" and args.loss == "observable":
        parser.error("--loss observable is only valid with --mode circuit_level")
    if args.output_mode == "multiclass":
        parser.error("--output_mode multiclass is not yet supported in the training loop")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = TrainingMonitor(
        out_dir,
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_config=vars(args),
    )

    # AMP
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("AMP enabled")
    amp_dtype = torch.float16 if use_amp else None

    # --- Load data ---
    if args.mode == "code_capacity":
        if args.online_regen:
            # Online data regeneration mode
            if args.code_npz is None:
                # Use first file from in_glob for code params
                npz_paths = sorted(glob.glob(args.in_glob)) if args.in_glob else []
                if not npz_paths:
                    print("ERROR: --code_npz or --in_glob required for online_regen")
                    return 1
                args.code_npz = npz_paths[0]

            code_data = _load_npz(args.code_npz)
            hx = code_data["hx"].astype(np.uint8)
            hz = code_data["hz"].astype(np.uint8)
            lx = code_data.get("lx", np.eye(hx.shape[1], dtype=np.uint8)[:1])
            lz = code_data.get("lz", np.eye(hz.shape[1], dtype=np.uint8)[:1])
            mx, n_qubits = hx.shape
            mz = hz.shape[0]

            p_range = tuple(args.p_range) if args.p_range else None
            train_dataset = OnlineCodeCapDataset(
                hx, hz, lx, lz,
                p_base=args.p_base, eta=args.eta,
                samples_per_epoch=args.samples_per_epoch,
                p_range=p_range, seed=args.seed,
            )
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            # Build a small fixed validation set
            val_dataset = OnlineCodeCapDataset(
                hx, hz, lx, lz,
                p_base=args.p_base, eta=args.eta,
                samples_per_epoch=min(500, args.samples_per_epoch // 5),
                p_range=p_range, seed=args.seed + 999,
            )
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            edge_types = 2
            print(f"Online regen: {args.samples_per_epoch} samples/epoch, p_range={p_range or args.p_base}")
        else:
            npz_paths = sorted(glob.glob(args.in_glob))
            if not npz_paths:
                print(f"ERROR: No files matching '{args.in_glob}'")
                return 1
            train_dataset, val_data, _, data_meta = build_graph_dataset(
                npz_paths, W=args.W, mode="supervised", seed=args.seed,
            )
            train_data = train_dataset  # alias for backward compat
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

            code_data = _load_npz(npz_paths[0])
            hx = code_data["hx"].astype(np.uint8)
            hz = code_data["hz"].astype(np.uint8)
            mx, n_qubits = hx.shape
            mz = hz.shape[0]
            edge_types = 2

        print(f"Code: n={n_qubits}, mx={mx}, mz={mz}")

        # PCM and logical operators on device
        hx_t = torch.from_numpy(hx.astype(np.float32)).to(device)
        hz_t = torch.from_numpy(hz.astype(np.float32)).to(device)
        lx_t = lz_t = None
        if "lx" in code_data and "lz" in code_data:
            lx_t = torch.from_numpy(code_data["lx"].astype(np.float32)).to(device)
            lz_t = torch.from_numpy(code_data["lz"].astype(np.float32)).to(device)

        # Build BP decoders
        dec_z = MinSumBPDecoder(
            hx, max_iter=args.bp_iters, alpha=0.8, neural_bp=args.neural_bp
        ).to(device)
        dec_x = MinSumBPDecoder(
            hz, max_iter=args.bp_iters, alpha=0.8, neural_bp=args.neural_bp
        ).to(device)

    elif args.mode == "circuit_level":
        from gnn_pipeline.dem_decoder import extract_dem_pcm

        npz_paths = sorted(glob.glob(args.in_glob))
        if not npz_paths:
            print(f"ERROR: No files matching '{args.in_glob}'")
            return 1
        train_dataset, val_data, _, data_meta = build_circuit_dataset(npz_paths, seed=args.seed)
        train_data = train_dataset
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

        sample = train_data[0]
        num_errors = int(sample.num_errors)
        num_detectors = int(sample.num_detectors)
        n_qubits = num_errors
        mx = mz = 0  # not used in circuit mode
        edge_types = 1

        obs_matrix_t = data_meta["obs_matrix"].to(device)
        dem_pcm_t = data_meta["dem_pcm"].to(device)
        dem_pcm_np = dem_pcm_t.cpu().numpy().astype(np.uint8)

        dec_z = MinSumBPDecoder(
            dem_pcm_np, max_iter=args.bp_iters, alpha=0.8, neural_bp=args.neural_bp
        ).to(device)
        dec_x = None  # Not used in circuit mode

        print(f"DEM: {num_detectors} detectors x {num_errors} errors")

    # --- Build GNN ---
    gnn = TannerGNN(
        node_feat_dim=4,
        hidden_dim=args.hidden_dim,
        num_mp_layers=args.num_mp_layers,
        edge_types=edge_types,
        dropout=args.dropout,
        correction_mode=args.correction_mode,
        use_residual=args.use_residual,
        use_layer_norm=args.use_layer_norm,
        use_attention=args.use_attention,
        use_film=args.use_film,
        noise_feat_dim=args.noise_feat_dim,
        output_mode=args.output_mode,
        standardize_input=args.standardize_input,
    ).to(device)

    # Load pretrained weights
    if args.pretrained:
        print(f"Loading pretrained model from {args.pretrained}...")
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            gnn.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            gnn.load_state_dict(ckpt, strict=False)

    if args.freeze_backbone:
        gnn.freeze_backbone()
        print("Backbone frozen (only readout trainable)")

    n_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)
    print(f"GNN: {n_params:,} trainable parameters")

    # --- Optimizer ---
    param_groups = [{"params": gnn.parameters(), "lr": args.lr}]
    if args.neural_bp:
        param_groups.append({"params": dec_z.parameters(), "lr": args.lr * 0.1})
        if dec_x is not None:
            param_groups.append({"params": dec_x.parameters(), "lr": args.lr * 0.1})
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    # --- Scheduler ---
    if args.scheduler == "cosine" and args.epochs > args.warmup_epochs:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])
    else:
        scheduler = None

    # Save config
    config = vars(args)
    config["n_params"] = n_params
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Training loop ---
    best_val_loss = float("inf")
    patience_counter = 0
    grad_mon = GradientMonitor()

    # Per-observable difficulty weights (circuit-level only, updated each epoch)
    obs_difficulty_weights = None  # None = uniform weighting initially

    print(f"\nTraining: mode={args.mode}, loss={args.loss}, epochs={args.epochs}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Online data regeneration: refresh training data each epoch
        if args.mode == "code_capacity" and args.online_regen:
            train_dataset.set_epoch(epoch)

        # Curriculum learning: filter training data by difficulty tier
        epoch_loader = train_loader
        if args.curriculum and not args.online_regen:
            # Tier 1 (epochs 1-3): easy samples only (low p)
            # Tier 2 (epochs 4-6): easy + medium
            # Tier 3 (epochs 7+): all samples
            if hasattr(train_dataset, '__len__') and len(train_dataset) > 0:
                if hasattr(train_dataset[0], 'p_value'):
                    p_vals_all = np.array([d.p_value.item() for d in train_dataset])
                    p_sorted = np.sort(p_vals_all)
                    if epoch <= 3:
                        p_thresh = p_sorted[min(len(p_sorted) - 1, len(p_sorted) // 3)]
                        indices = [i for i, d in enumerate(train_dataset) if d.p_value.item() <= p_thresh]
                    elif epoch <= 6:
                        p_thresh = p_sorted[min(len(p_sorted) - 1, 2 * len(p_sorted) // 3)]
                        indices = [i for i, d in enumerate(train_dataset) if d.p_value.item() <= p_thresh]
                    else:
                        indices = list(range(len(train_dataset)))
                    if len(indices) > 0:
                        subset = torch.utils.data.Subset(train_dataset, indices)
                        epoch_loader = DataLoader(subset, batch_size=train_loader.batch_size, shuffle=True)

        gnn.train()
        dec_z.train()
        if dec_x is not None:
            dec_x.train()
        total_loss = 0.0
        total_conv = 0.0
        num_batches = 0
        optimizer.zero_grad()

        # Per-observable loss accumulator (circuit-level)
        if args.mode == "circuit_level":
            num_obs = obs_matrix_t.shape[1]
            obs_loss_accum = torch.zeros(num_obs)
            obs_loss_count = 0

        for batch in epoch_loader:
            batch = batch.to(device)
            B = batch.num_graphs

            # Syndrome augmentation: random bit-flips for robustness
            if args.augment and args.mode == "code_capacity":
                syn_key = "target_syndrome"
                if hasattr(batch, syn_key):
                    syn_data = getattr(batch, syn_key)
                    flip_mask = (torch.rand_like(syn_data) < 0.01).float()
                    noisy_syn = torch.fmod(syn_data + flip_mask, 2.0)
                    setattr(batch, syn_key, noisy_syn)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                gnn_out = gnn(batch)

                if args.mode == "code_capacity":
                    n_checks = mx + mz
                    if args.correction_mode == "both":
                        add_c, mul_c = gnn_out
                        add_c = torch.clamp(add_c, -20.0, 20.0).view(B, n_qubits)
                        mul_c = torch.clamp(mul_c, -5.0, 5.0).view(B, n_qubits)
                        gnn_out = (add_c, mul_c)
                    else:
                        gnn_out = torch.clamp(gnn_out, -20.0, 20.0).view(B, n_qubits)

                    llr_z = batch.channel_llr_z.view(B, n_qubits)
                    llr_x = batch.channel_llr_x.view(B, n_qubits)
                    z_err = batch.z_error.view(B, n_qubits)
                    x_err = batch.x_error.view(B, n_qubits)
                    target_syn = batch.target_syndrome.view(B, n_checks)
                    x_syn = target_syn[:, :mx]
                    z_syn = target_syn[:, mx:]

                    corr_z = apply_correction(llr_z, gnn_out, args.correction_mode)
                    corr_x = apply_correction(llr_x, gnn_out, args.correction_mode)

                    marg_z, _, conv_z = dec_z(x_syn, corr_z)
                    marg_x, _, conv_x = dec_x(z_syn, corr_x)

                    loss = _compute_code_capacity_loss(
                        marg_z, marg_x, z_err, x_err,
                        args.loss, x_syn, z_syn,
                        hx_t, hz_t, lx_t, lz_t,
                        pos_weight=args.pos_weight,
                        syn_weight=args.syn_weight,
                    )
                    total_conv += ((conv_z.float() + conv_x.float()) / 2).mean().item()

                else:  # circuit_level
                    if args.correction_mode == "both":
                        add_c, mul_c = gnn_out
                        add_c = torch.clamp(add_c, -20.0, 20.0).view(B, num_errors)
                        mul_c = torch.clamp(mul_c, -5.0, 5.0).view(B, num_errors)
                        gnn_out = (add_c, mul_c)
                    else:
                        gnn_out = torch.clamp(gnn_out, -20.0, 20.0).view(B, num_errors)

                    ch_llr = batch.channel_llr.view(B, num_errors)
                    syn = batch.syndrome.view(B, num_detectors)
                    obs_tgt = batch.observable.view(B, -1)

                    corr_llr = apply_correction(ch_llr, gnn_out, args.correction_mode)
                    marg, _, conv = dec_z(syn, corr_llr)

                    loss, per_obs_losses = observable_loss_decomposed(
                        marg, obs_matrix_t, obs_tgt, syn, dem_pcm_t,
                        syn_weight=args.syn_weight, obs_weight=args.obs_weight,
                        obs_weights=obs_difficulty_weights,
                    )
                    # Track running per-observable loss for adaptive weighting
                    obs_loss_accum += per_obs_losses.cpu()
                    obs_loss_count += 1
                    total_conv += conv.float().mean().item()

            raw_loss = loss.item()
            scaled = loss / args.accumulate_grad
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            if (num_batches + 1) % args.accumulate_grad == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_mon.step(gnn)
                    nn.utils.clip_grad_norm_(gnn.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_mon.step(gnn)
                    nn.utils.clip_grad_norm_(gnn.parameters(), args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad()

            total_loss += raw_loss
            num_batches += 1

        train_loss = total_loss / max(num_batches, 1)
        train_conv = total_conv / max(num_batches, 1)

        # Update per-observable difficulty weights (circuit-level)
        if args.mode == "circuit_level" and obs_loss_count > 0:
            avg_obs_loss = obs_loss_accum / obs_loss_count  # (num_obs,)
            # Higher loss = harder observable = higher weight (softmax normalisation)
            obs_difficulty_weights = torch.softmax(avg_obs_loss, dim=0).to(device) * num_obs

        # --- Validation ---
        gnn.eval()
        dec_z.eval()
        if dec_x is not None:
            dec_x.eval()
        val_loss = 0.0
        val_conv = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                B = batch.num_graphs
                gnn_out = gnn(batch)

                if args.mode == "code_capacity":
                    n_checks = mx + mz
                    if args.correction_mode == "both":
                        add_c, mul_c = gnn_out
                        add_c = torch.clamp(add_c, -20.0, 20.0).view(B, n_qubits)
                        mul_c = torch.clamp(mul_c, -5.0, 5.0).view(B, n_qubits)
                        gnn_out = (add_c, mul_c)
                    else:
                        gnn_out = torch.clamp(gnn_out, -20.0, 20.0).view(B, n_qubits)

                    llr_z = batch.channel_llr_z.view(B, n_qubits)
                    llr_x = batch.channel_llr_x.view(B, n_qubits)
                    z_err = batch.z_error.view(B, n_qubits)
                    x_err = batch.x_error.view(B, n_qubits)
                    target_syn = batch.target_syndrome.view(B, n_checks)
                    x_syn = target_syn[:, :mx]
                    z_syn = target_syn[:, mx:]

                    corr_z = apply_correction(llr_z, gnn_out, args.correction_mode)
                    corr_x = apply_correction(llr_x, gnn_out, args.correction_mode)
                    marg_z, _, conv_z = dec_z(x_syn, corr_z)
                    marg_x, _, conv_x = dec_x(z_syn, corr_x)

                    loss = _compute_code_capacity_loss(
                        marg_z, marg_x, z_err, x_err,
                        args.loss, x_syn, z_syn,
                        hx_t, hz_t, lx_t, lz_t,
                        pos_weight=args.pos_weight,
                        syn_weight=args.syn_weight,
                    )
                    val_conv += ((conv_z.float() + conv_x.float()) / 2).mean().item()
                else:
                    if args.correction_mode == "both":
                        add_c, mul_c = gnn_out
                        add_c = torch.clamp(add_c, -20.0, 20.0).view(B, num_errors)
                        mul_c = torch.clamp(mul_c, -5.0, 5.0).view(B, num_errors)
                        gnn_out = (add_c, mul_c)
                    else:
                        gnn_out = torch.clamp(gnn_out, -20.0, 20.0).view(B, num_errors)

                    ch_llr = batch.channel_llr.view(B, num_errors)
                    syn = batch.syndrome.view(B, num_detectors)
                    obs_tgt = batch.observable.view(B, -1)
                    corr_llr = apply_correction(ch_llr, gnn_out, args.correction_mode)
                    marg, _, conv = dec_z(syn, corr_llr)

                    loss, _ = observable_loss_decomposed(
                        marg, obs_matrix_t, obs_tgt, syn, dem_pcm_t,
                        syn_weight=args.syn_weight, obs_weight=args.obs_weight,
                        obs_weights=obs_difficulty_weights,
                    )
                    val_conv += conv.float().mean().item()

                val_loss += loss.item()
                val_batches += 1

        val_loss /= max(val_batches, 1)
        val_conv /= max(val_batches, 1)

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        metrics = {
            "train_loss": train_loss, "train_conv": train_conv,
            "val_loss": val_loss, "val_conv": val_conv,
            "lr": lr_now, "time_s": elapsed,
            "grad_norm": grad_mon.last_norm,
        }
        # Per-observable loss breakdown (circuit-level only)
        if args.mode == "circuit_level" and obs_loss_count > 0:
            avg_obs = obs_loss_accum / obs_loss_count
            for k in range(len(avg_obs)):
                metrics[f"obs_{k}_loss"] = float(avg_obs[k])
        monitor.log_epoch(epoch, metrics)
        monitor.flush_csv()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} conv={train_conv:.1%} | "
            f"val_loss={val_loss:.4f} conv={val_conv:.1%} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt = {
                "model_state_dict": gnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "mode": args.mode,
                "hidden_dim": args.hidden_dim,
                "num_mp_layers": args.num_mp_layers,
                "correction_mode": args.correction_mode,
                "output_mode": args.output_mode,
                "use_residual": args.use_residual,
                "use_layer_norm": args.use_layer_norm,
                "use_attention": args.use_attention,
                "use_film": args.use_film,
                "noise_feat_dim": args.noise_feat_dim,
                "standardize_input": args.standardize_input,
                "neural_bp": args.neural_bp,
                "bp_iters": args.bp_iters,
                "edge_types": edge_types,
            }
            if args.neural_bp:
                ckpt["dec_z_state_dict"] = dec_z.state_dict()
                if dec_x is not None:
                    ckpt["dec_x_state_dict"] = dec_x.state_dict()
            if scheduler is not None:
                ckpt["scheduler_state_dict"] = scheduler.state_dict()
            if scaler is not None:
                ckpt["scaler_state_dict"] = scaler.state_dict()
            torch.save(ckpt, out_dir / "best_model.pt")
            print(f"  ** Best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    monitor.save(config)
    monitor.print_summary()
    print(f"\nTraining complete. Output: {out_dir}")
    print(f"  Metrics CSV: {monitor.csv_path}")
    print(f"  Training log: {out_dir / 'training_log.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
