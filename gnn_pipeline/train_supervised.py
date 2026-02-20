"""Supervised fine-tuning: train GNN end-to-end through differentiable BP.

The GNN outputs LLR corrections which are added to per-Pauli channel LLRs,
then fed through the differentiable MinSumBPDecoder. Loss is computed on the
soft BP marginals vs ground-truth error vectors.

Gradient path: loss -> marginals -> BP message passing -> corrected LLRs -> GNN

Usage:
    # Fine-tune from pretrained self-supervised model
    python -m gnn_pipeline.train_supervised \
        --in_glob "data/supervised_train.npz" \
        --pretrained runs/drift_train/best_model.pt \
        --epochs 10 --batch_size 16 --lr 1e-4 \
        --bp_iters 10 --out_dir runs/supervised_ft

    # Train from scratch
    python -m gnn_pipeline.train_supervised \
        --in_glob "data/supervised_train.npz" \
        --from_scratch --epochs 20 --batch_size 16 --lr 1e-3 \
        --bp_iters 10 --out_dir runs/supervised_scratch
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader

from gnn_pipeline.bp_decoder import MinSumBPDecoder
from gnn_pipeline.dataset import build_graph_dataset, _load_npz
from gnn_pipeline.gnn_model import TannerGNN, apply_correction
from gnn_pipeline.loss_functions import weighted_bce_loss, focal_loss, logical_aware_loss, syndrome_consistency_loss


def _compute_loss(
    marginals_z: torch.Tensor,
    marginals_x: torch.Tensor,
    z_error: torch.Tensor,
    x_error: torch.Tensor,
    loss_fn: str,
    pos_weight: float = 50.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    logical_weight: float = 0.1,
    lx_t: torch.Tensor | None = None,
    lz_t: torch.Tensor | None = None,
    syn_weight: float = 0.0,
    x_syndrome: torch.Tensor | None = None,
    z_syndrome: torch.Tensor | None = None,
    hx_t: torch.Tensor | None = None,
    hz_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute loss on BP marginals vs ground-truth errors (shared by train/eval)."""
    if loss_fn == "mse":
        base = F.mse_loss(marginals_z, z_error) + F.mse_loss(marginals_x, x_error)
    elif loss_fn == "bce":
        base = F.binary_cross_entropy(marginals_z, z_error) + \
               F.binary_cross_entropy(marginals_x, x_error)
    elif loss_fn == "weighted_bce":
        base = weighted_bce_loss(marginals_z, z_error, pos_weight=pos_weight) + \
               weighted_bce_loss(marginals_x, x_error, pos_weight=pos_weight)
    elif loss_fn == "focal":
        base = focal_loss(marginals_z, z_error, alpha=focal_alpha, gamma=focal_gamma) + \
               focal_loss(marginals_x, x_error, alpha=focal_alpha, gamma=focal_gamma)
    elif loss_fn == "logical_mse":
        base = logical_aware_loss(
            marginals_z, marginals_x, z_error, x_error,
            lx_t, lz_t, base_loss_fn=F.mse_loss, logical_weight=logical_weight,
        )
    elif loss_fn == "logical_bce":
        base = logical_aware_loss(
            marginals_z, marginals_x, z_error, x_error,
            lx_t, lz_t, base_loss_fn=F.binary_cross_entropy, logical_weight=logical_weight,
        )
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")

    # Syndrome consistency auxiliary loss
    if syn_weight > 0 and hx_t is not None and hz_t is not None:
        syn_loss = syndrome_consistency_loss(marginals_z, x_syndrome, hx_t) + \
                   syndrome_consistency_loss(marginals_x, z_syndrome, hz_t)
        base = base + syn_weight * syn_loss

    return base


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dec_z: MinSumBPDecoder,
    dec_x: MinSumBPDecoder,
    n_qubits: int,
    mx: int,
    mz: int,
    loss_fn: str = "mse",
    grad_clip: float = 1.0,
    pos_weight: float = 50.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    logical_weight: float = 0.1,
    lx_t: torch.Tensor | None = None,
    lz_t: torch.Tensor | None = None,
    accumulate_grad: int = 1,
    correction_mode: str = "additive",
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    syn_weight: float = 0.0,
    hx_t: torch.Tensor | None = None,
    hz_t: torch.Tensor | None = None,
    augment: bool = False,
) -> dict:
    """Train for one epoch on supervised loss through differentiable BP.

    Returns:
        dict with loss, bit_acc_z, bit_acc_x, convergence_z, convergence_x
    """
    model.train()
    total_loss = 0.0
    total_bit_acc_z = 0.0
    total_bit_acc_x = 0.0
    total_conv_z = 0.0
    total_conv_x = 0.0
    num_batches = 0
    n_checks = mx + mz
    optimizer.zero_grad()

    amp_dtype = torch.float16 if use_amp else None

    for batch in loader:
        batch = batch.to(device)
        B = batch.num_graphs

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            # GNN forward: get per-qubit LLR corrections
            gnn_out = model(batch)
            if correction_mode == "both":
                add_corr, mul_corr = gnn_out
                add_corr = torch.clamp(add_corr, -20.0, 20.0).view(B, n_qubits)
                mul_corr = torch.clamp(mul_corr, -5.0, 5.0).view(B, n_qubits)
                gnn_out = (add_corr, mul_corr)
            else:
                gnn_out = torch.clamp(gnn_out, -20.0, 20.0).view(B, n_qubits)

            # Reshape supervised data
            channel_llr_z = batch.channel_llr_z.view(B, n_qubits)
            channel_llr_x = batch.channel_llr_x.view(B, n_qubits)
            z_error = batch.z_error.view(B, n_qubits)
            x_error = batch.x_error.view(B, n_qubits)
            target_syn = batch.target_syndrome.view(B, n_checks)

            # Apply GNN corrections to per-Pauli channel LLRs
            corrected_llr_z = apply_correction(channel_llr_z, gnn_out, correction_mode)
            corrected_llr_x = apply_correction(channel_llr_x, gnn_out, correction_mode)

            # Extract CSS syndromes
            x_syndrome = target_syn[:, :mx]   # X-checks detect Z-errors
            z_syndrome = target_syn[:, mx:]   # Z-checks detect X-errors

            # Syndrome augmentation: randomly flip bits with p=0.01
            if augment:
                flip_mask_x = (torch.rand_like(x_syndrome) < 0.01).float()
                flip_mask_z = (torch.rand_like(z_syndrome) < 0.01).float()
                x_syndrome = torch.abs(x_syndrome - flip_mask_x)
                z_syndrome = torch.abs(z_syndrome - flip_mask_z)

            # Differentiable BP forward
            marginals_z, hard_z, conv_z = dec_z(x_syndrome, corrected_llr_z)
            marginals_x, hard_x, conv_x = dec_x(z_syndrome, corrected_llr_x)

            # Loss on soft marginals vs ground-truth error vectors
            loss = _compute_loss(
                marginals_z, marginals_x, z_error, x_error,
                loss_fn, pos_weight, focal_alpha, focal_gamma,
                logical_weight, lx_t, lz_t,
                syn_weight=syn_weight,
                x_syndrome=x_syndrome, z_syndrome=z_syndrome,
                hx_t=hx_t, hz_t=hz_t,
            )

        # Gradient accumulation with AMP support
        raw_loss = loss.item()
        scaled_loss = loss / accumulate_grad
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (num_batches + 1) % accumulate_grad == 0:
            if scaler is not None:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        # Metrics
        with torch.no_grad():
            bit_acc_z = (hard_z == z_error.long()).float().mean().item()
            bit_acc_x = (hard_x == x_error.long()).float().mean().item()
            conv_rate_z = conv_z.float().mean().item()
            conv_rate_x = conv_x.float().mean().item()

        total_loss += raw_loss
        total_bit_acc_z += bit_acc_z
        total_bit_acc_x += bit_acc_x
        total_conv_z += conv_rate_z
        total_conv_x += conv_rate_x
        num_batches += 1

    # Flush remaining accumulated gradients
    if num_batches % accumulate_grad != 0:
        if scaler is not None:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        optimizer.zero_grad()

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "bit_acc_z": total_bit_acc_z / n,
        "bit_acc_x": total_bit_acc_x / n,
        "convergence_z": total_conv_z / n,
        "convergence_x": total_conv_x / n,
    }


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dec_z: MinSumBPDecoder,
    dec_x: MinSumBPDecoder,
    n_qubits: int,
    mx: int,
    mz: int,
    loss_fn: str = "mse",
    pos_weight: float = 50.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    logical_weight: float = 0.1,
    lx_t: torch.Tensor | None = None,
    lz_t: torch.Tensor | None = None,
    correction_mode: str = "additive",
    syn_weight: float = 0.0,
    hx_t: torch.Tensor | None = None,
    hz_t: torch.Tensor | None = None,
) -> dict:
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_bit_acc_z = 0.0
    total_bit_acc_x = 0.0
    total_conv_z = 0.0
    total_conv_x = 0.0
    num_batches = 0
    n_checks = mx + mz

    for batch in loader:
        batch = batch.to(device)
        B = batch.num_graphs

        gnn_out = model(batch)
        if correction_mode == "both":
            add_corr, mul_corr = gnn_out
            add_corr = torch.clamp(add_corr, -20.0, 20.0).view(B, n_qubits)
            mul_corr = torch.clamp(mul_corr, -5.0, 5.0).view(B, n_qubits)
            gnn_out = (add_corr, mul_corr)
        else:
            gnn_out = torch.clamp(gnn_out, -20.0, 20.0).view(B, n_qubits)

        channel_llr_z = batch.channel_llr_z.view(B, n_qubits)
        channel_llr_x = batch.channel_llr_x.view(B, n_qubits)
        z_error = batch.z_error.view(B, n_qubits)
        x_error = batch.x_error.view(B, n_qubits)
        target_syn = batch.target_syndrome.view(B, n_checks)

        corrected_llr_z = apply_correction(channel_llr_z, gnn_out, correction_mode)
        corrected_llr_x = apply_correction(channel_llr_x, gnn_out, correction_mode)

        x_syndrome = target_syn[:, :mx]
        z_syndrome = target_syn[:, mx:]

        marginals_z, hard_z, conv_z = dec_z(x_syndrome, corrected_llr_z)
        marginals_x, hard_x, conv_x = dec_x(z_syndrome, corrected_llr_x)

        loss = _compute_loss(
            marginals_z, marginals_x, z_error, x_error,
            loss_fn, pos_weight, focal_alpha, focal_gamma,
            logical_weight, lx_t, lz_t,
            syn_weight=syn_weight,
            x_syndrome=x_syndrome, z_syndrome=z_syndrome,
            hx_t=hx_t, hz_t=hz_t,
        )

        bit_acc_z = (hard_z == z_error.long()).float().mean().item()
        bit_acc_x = (hard_x == x_error.long()).float().mean().item()
        conv_rate_z = conv_z.float().mean().item()
        conv_rate_x = conv_x.float().mean().item()

        total_loss += loss.item()
        total_bit_acc_z += bit_acc_z
        total_bit_acc_x += bit_acc_x
        total_conv_z += conv_rate_z
        total_conv_x += conv_rate_x
        num_batches += 1

    n = max(num_batches, 1)
    return {
        "loss": total_loss / n,
        "bit_acc_z": total_bit_acc_z / n,
        "bit_acc_x": total_bit_acc_x / n,
        "convergence_z": total_conv_z / n,
        "convergence_x": total_conv_x / n,
    }


def main(argv: List[str] | None = None) -> int:
    """Main entry point for supervised training CLI."""
    parser = argparse.ArgumentParser(
        description="Train GNN end-to-end through differentiable BP"
    )
    parser.add_argument(
        "--in_glob", type=str, required=True,
        help="Glob pattern for input NPZ files (must contain z_errors/x_errors)",
    )
    parser.add_argument("--W", type=int, default=4, help="Window width")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--bp_iters", type=int, default=10,
        help="BP iterations during training (fewer=faster, more=better gradients)",
    )
    parser.add_argument(
        "--loss", type=str, default="mse",
        choices=["mse", "bce", "weighted_bce", "focal", "logical_mse", "logical_bce"],
        help="Loss function on BP marginals",
    )
    parser.add_argument("--pos_weight", type=float, default=0.0,
                        help="Positive class weight for weighted_bce (0=auto from data, else fixed value)")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Focal loss alpha (default: 0.25)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma (default: 2.0)")
    parser.add_argument("--logical_weight", type=float, default=0.1,
                        help="Weight for logical-flip penalty (default: 0.1)")
    parser.add_argument("--syn_weight", type=float, default=0.1,
                        help="Weight for syndrome-consistency auxiliary loss (default: 0.1, 0 to disable)")
    parser.add_argument(
        "--grad_clip", type=float, default=1.0,
        help="Gradient clipping max norm (0 to disable)",
    )
    parser.add_argument(
        "--accumulate_grad", type=int, default=1,
        help="Gradient accumulation steps (effective batch = batch_size * this)",
    )
    parser.add_argument(
        "--scheduler", type=str, default="none", choices=["none", "cosine"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--patience", type=int, default=0,
        help="Early stopping patience in epochs (0 = disabled)",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=0,
        help="Linear LR warmup epochs (0 = disabled)",
    )
    parser.add_argument(
        "--amp", action="store_true",
        help="Enable automatic mixed-precision training (GPU only)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="AdamW weight decay (default: 1e-4, 0 to disable)",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path to pretrained model (self-supervised checkpoint)",
    )
    parser.add_argument(
        "--from_scratch", action="store_true",
        help="Train from scratch instead of fine-tuning",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64,
        help="GNN hidden dimension",
    )
    parser.add_argument(
        "--num_mp_layers", type=int, default=3,
        help="Number of GNN message-passing layers",
    )
    parser.add_argument(
        "--correction_mode", type=str, default="additive",
        choices=["additive", "multiplicative", "both"],
        help="How GNN corrections are applied to channel LLRs",
    )
    parser.add_argument(
        "--use_residual", action="store_true",
        help="Enable residual/skip connections in GNN message-passing layers",
    )
    parser.add_argument(
        "--use_layer_norm", action="store_true",
        help="Enable layer normalization in GNN message-passing layers",
    )
    parser.add_argument(
        "--use_attention", action="store_true",
        help="Enable attention-weighted message passing in GNN",
    )
    parser.add_argument(
        "--curriculum", action="store_true",
        help="Enable curriculum learning: start with easy (low-p) data, add harder gradually",
    )
    parser.add_argument(
        "--loss_phase2", type=str, default=None,
        choices=["mse", "bce", "weighted_bce", "focal", "logical_mse", "logical_bce"],
        help="Switch to this loss after --phase2_epoch (default: disabled)",
    )
    parser.add_argument(
        "--phase2_epoch", type=int, default=5,
        help="Epoch to switch to phase2 loss (default: 5)",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable syndrome bit-flip augmentation (p=0.01 per bit)",
    )
    parser.add_argument(
        "--learnable_alpha", action="store_true",
        help="Make BP damping factor (alpha) a learnable parameter, jointly optimized with GNN",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Output directory for checkpoints and logs",
    )

    args = parser.parse_args(argv)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Find input files
    npz_paths = sorted(glob.glob(args.in_glob))
    if not npz_paths:
        print(f"ERROR: No files matching {args.in_glob}", file=sys.stderr)
        return 1
    print(f"Found {len(npz_paths)} NPZ file(s)")

    # Build supervised dataset
    print(f"Building supervised dataset with W={args.W}...")
    train_data, val_data, test_data, meta = build_graph_dataset(
        npz_paths=npz_paths,
        W=args.W,
        mode="supervised",
        seed=args.seed,
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    if len(train_data) == 0:
        print("ERROR: No training data", file=sys.stderr)
        return 1

    mode_is_weighted = args.loss in ("weighted_bce",)

    # Extract PCM matrices and logical operators
    first_npz = _load_npz(npz_paths[0])
    hx = first_npz["hx"].astype(np.float32)
    hz = first_npz["hz"].astype(np.float32)
    mx, n_qubits = int(hx.shape[0]), int(hx.shape[1])
    mz = int(hz.shape[0])
    print(f"Code: n={n_qubits}, mx={mx}, mz={mz}")

    # Load logical operators (needed for logical_* losses)
    lx_t, lz_t = None, None
    if args.loss.startswith("logical_"):
        if "lx" in first_npz and "lz" in first_npz:
            lx_t = torch.tensor(first_npz["lx"].astype(np.float32))
            lz_t = torch.tensor(first_npz["lz"].astype(np.float32))
            print(f"Loaded logical operators: lx={lx_t.shape}, lz={lz_t.shape}")
        else:
            print("ERROR: logical_* loss requires lx/lz in NPZ", file=sys.stderr)
            return 1

    # Auto-compute pos_weight from data if not specified
    if args.pos_weight <= 0 and mode_is_weighted:
        z_err = first_npz.get("z_errors", None)
        x_err = first_npz.get("x_errors", None)
        if z_err is not None and x_err is not None:
            all_errors = np.concatenate([z_err.ravel(), x_err.ravel()])
            n_pos = all_errors.sum()
            n_neg = all_errors.size - n_pos
            args.pos_weight = max(1.0, n_neg / max(n_pos, 1.0))
            print(f"Auto pos_weight from data: {args.pos_weight:.1f} (error rate: {n_pos/all_errors.size:.4f})")
        else:
            args.pos_weight = 50.0
            print(f"Using default pos_weight: {args.pos_weight}")
    elif args.pos_weight <= 0:
        args.pos_weight = 50.0  # fallback

    # PCM tensors for syndrome consistency loss
    hx_t, hz_t = None, None
    if args.syn_weight > 0:
        hx_t = torch.from_numpy(hx.astype(np.float32))
        hz_t = torch.from_numpy(hz.astype(np.float32))
        print(f"Syndrome consistency loss enabled (weight={args.syn_weight})")

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move logical operators and PCMs to device if needed
    if lx_t is not None:
        lx_t = lx_t.to(device)
    if lz_t is not None:
        lz_t = lz_t.to(device)
    if hx_t is not None:
        hx_t = hx_t.to(device)
    if hz_t is not None:
        hz_t = hz_t.to(device)

    # Build BP decoders ONCE (expensive constructors)
    print(f"Building BP decoders (bp_iters={args.bp_iters})...")
    dec_z = MinSumBPDecoder(
        hx.astype(np.uint8), max_iter=args.bp_iters, alpha=0.8, clamp_llr=20.0
    ).to(device)
    dec_x = MinSumBPDecoder(
        hz.astype(np.uint8), max_iter=args.bp_iters, alpha=0.8, clamp_llr=20.0
    ).to(device)
    # Freeze BP decoder buffers (they have no learnable params by default)
    for p in dec_z.parameters():
        p.requires_grad = False
    for p in dec_x.parameters():
        p.requires_grad = False

    # Learnable alpha: make BP damping factor a trainable parameter
    alpha_params = []
    if args.learnable_alpha:
        alpha_z = dec_z.make_alpha_learnable()
        alpha_x = dec_x.make_alpha_learnable()
        alpha_params = [alpha_z, alpha_x]
        print(f"  Learnable BP alpha enabled (init={alpha_z.item():.2f})")

    print(f"  dec_z (hx): {dec_z.num_checks}x{dec_z.num_vars}, {dec_z.num_edges} edges")
    print(f"  dec_x (hz): {dec_x.num_checks}x{dec_x.num_vars}, {dec_x.num_edges} edges")

    # Build GNN model
    model = TannerGNN(
        hidden_dim=args.hidden_dim,
        num_mp_layers=args.num_mp_layers,
        correction_mode=args.correction_mode,
        use_residual=args.use_residual,
        use_layer_norm=args.use_layer_norm,
        use_attention=args.use_attention,
    )

    if args.pretrained and not args.from_scratch:
        print(f"Loading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("  Pretrained weights loaded successfully")
    else:
        print("Training from scratch")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters")

    # Optimizer and scheduler
    all_params = list(model.parameters()) + alpha_params
    if args.weight_decay > 0:
        optimizer = AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(all_params, lr=args.lr)
    scheduler = None
    if args.scheduler == "cosine":
        if args.warmup_epochs > 0:
            warmup_sched = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0,
                total_iters=args.warmup_epochs,
            )
            cosine_sched = CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - args.warmup_epochs),
                eta_min=args.lr * 0.01,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[args.warmup_epochs],
            )
            print(f"Using cosine LR scheduler with {args.warmup_epochs}-epoch warmup: "
                  f"{args.lr * 0.01:.2e} -> {args.lr} -> {args.lr * 0.01:.2e}")
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
            )
            print(f"Using cosine LR scheduler: {args.lr} -> {args.lr * 0.01}")

    # Mixed-precision setup
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Mixed-precision training enabled (AMP)")
    elif args.amp:
        print("AMP requested but CUDA not available, using fp32")

    # Create output directory
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Test loader (evaluated once after training on best model)
    test_loader = None
    if len(test_data) > 0:
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        print(f"Test set: {len(test_data)} samples (evaluated after training)")

    # Curriculum learning: extract p_value per sample for difficulty sorting
    curriculum_loaders = None
    if args.curriculum:
        # Each graph in train_data has a channel_llr attribute;
        # we can infer p from it: p = sigmoid(-llr) ~ 1/(1+exp(llr))
        # Group samples by difficulty tiers
        p_thresholds = [0.02, 0.03, 1.0]  # tier boundaries
        tier_indices = [[] for _ in range(len(p_thresholds))]
        for idx, sample in enumerate(train_data):
            # Average channel LLR -> infer approximate p
            avg_llr = sample.channel_llr.mean().item() if hasattr(sample, 'channel_llr') else 10.0
            approx_p = 1.0 / (1.0 + np.exp(avg_llr))
            for tier_idx, thresh in enumerate(p_thresholds):
                if approx_p <= thresh:
                    tier_indices[tier_idx].append(idx)
                    break

        # Create cumulative subsets: tier0, tier0+1, tier0+1+2
        from torch.utils.data import Subset
        curriculum_loaders = []
        cumulative = []
        for tier_idx in range(len(p_thresholds)):
            cumulative.extend(tier_indices[tier_idx])
            subset = Subset(train_data, list(cumulative))
            loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
            curriculum_loaders.append(loader)
        tier_sizes = [len(t) for t in tier_indices]
        print(f"Curriculum learning: {len(p_thresholds)} tiers with {tier_sizes} samples")
        print(f"  Epochs 1-3: easy only ({sum(tier_sizes[:1])} samples)")
        print(f"  Epochs 4-6: easy+medium ({sum(tier_sizes[:2])} samples)")
        print(f"  Epochs 7+: all ({sum(tier_sizes)} samples)")

    # Training loop
    patience_str = f", patience={args.patience}" if args.patience > 0 else ""
    extra_info = ""
    if args.curriculum:
        extra_info += ", curriculum"
    if args.loss_phase2:
        extra_info += f", phase2={args.loss_phase2}@epoch{args.phase2_epoch}"
    if args.augment:
        extra_info += ", augment"
    print(f"\nTraining for {args.epochs} epochs (loss={args.loss}, bp_iters={args.bp_iters}{patience_str}{extra_info})...")
    print("-" * 90)
    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    t_start = time.time()
    early_stopped = False
    epochs_completed = 0

    for epoch in range(args.epochs):
        t_epoch = time.time()

        # Curriculum: select appropriate loader
        if curriculum_loaders is not None:
            if epoch < 3:
                current_loader = curriculum_loaders[0]
            elif epoch < 6:
                current_loader = curriculum_loaders[min(1, len(curriculum_loaders) - 1)]
            else:
                current_loader = curriculum_loaders[-1]
        else:
            current_loader = train_loader

        # Loss phase switching
        current_loss = args.loss
        if args.loss_phase2 and epoch >= args.phase2_epoch:
            current_loss = args.loss_phase2

        train_metrics = train_epoch(
            model, current_loader, optimizer, device,
            dec_z, dec_x, n_qubits, mx, mz,
            loss_fn=current_loss, grad_clip=args.grad_clip,
            pos_weight=args.pos_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            logical_weight=args.logical_weight,
            lx_t=lx_t, lz_t=lz_t,
            accumulate_grad=args.accumulate_grad,
            correction_mode=args.correction_mode,
            scaler=scaler, use_amp=use_amp,
            syn_weight=args.syn_weight, hx_t=hx_t, hz_t=hz_t,
            augment=args.augment,
        )
        val_metrics = eval_epoch(
            model, val_loader, device,
            dec_z, dec_x, n_qubits, mx, mz,
            loss_fn=current_loss,
            pos_weight=args.pos_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            logical_weight=args.logical_weight,
            lx_t=lx_t, lz_t=lz_t,
            correction_mode=args.correction_mode,
            syn_weight=args.syn_weight, hx_t=hx_t, hz_t=hz_t,
        )

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t_epoch
        lr_now = optimizer.param_groups[0]["lr"]

        alpha_str = ""
        if args.learnable_alpha:
            alpha_val = dec_z.effective_alpha
            if isinstance(alpha_val, torch.Tensor):
                alpha_val = alpha_val.item()
            alpha_str = f" | alpha: {alpha_val:.4f}"

        print(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"BitAcc Z: {val_metrics['bit_acc_z']:.4f} X: {val_metrics['bit_acc_x']:.4f} | "
            f"Conv Z: {val_metrics['convergence_z']:.3f} X: {val_metrics['convergence_x']:.3f} | "
            f"LR: {lr_now:.2e}{alpha_str} | {elapsed:.1f}s"
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "lr": lr_now,
            "time_s": elapsed,
        }
        history.append(epoch_record)

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            # Save best_model.pt with architecture info for evaluate.py
            torch.save({
                "model_state_dict": model.state_dict(),
                "hidden_dim": args.hidden_dim,
                "num_mp_layers": args.num_mp_layers,
                "correction_mode": args.correction_mode,
                "use_residual": args.use_residual,
                "use_layer_norm": args.use_layer_norm,
                "use_attention": args.use_attention,
            }, str(out_dir / "best_model.pt"))

            # Rich checkpoint with architecture info + optimizer state
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hidden_dim": args.hidden_dim,
                "num_mp_layers": args.num_mp_layers,
                "node_feat_dim": 4, "edge_types": 2, "dropout": 0.1,
                "correction_mode": args.correction_mode,
                "use_residual": args.use_residual,
                "use_layer_norm": args.use_layer_norm,
                "use_attention": args.use_attention,
                "bp_iters": args.bp_iters,
                "loss_fn": args.loss,
                "epoch": epoch + 1,
                "val_loss": best_val_loss,
            }
            if scheduler is not None:
                ckpt["scheduler_state_dict"] = scheduler.state_dict()
            if scaler is not None:
                ckpt["scaler_state_dict"] = scaler.state_dict()
            torch.save(ckpt, str(out_dir / "best_checkpoint.pt"))

            print(f"  -> New best val loss: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"  Early stopping: no improvement for {args.patience} epochs")
                early_stopped = True
                epochs_completed = epoch + 1
                break

        epochs_completed = epoch + 1

    total_time = time.time() - t_start
    print("-" * 90)
    stop_reason = "early stopping" if early_stopped else "completed"
    print(f"Training {stop_reason} in {total_time:.1f}s ({total_time/60:.1f} min), "
          f"{epochs_completed}/{args.epochs} epochs")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Save final model (with architecture info)
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_dim": args.hidden_dim,
        "num_mp_layers": args.num_mp_layers,
        "correction_mode": args.correction_mode,
        "use_residual": args.use_residual,
        "use_layer_norm": args.use_layer_norm,
        "use_attention": args.use_attention,
        "epoch": epochs_completed,
    }, str(out_dir / "final_model.pt"))
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_dim": args.hidden_dim,
        "num_mp_layers": args.num_mp_layers,
        "correction_mode": args.correction_mode,
        "use_residual": args.use_residual,
        "use_layer_norm": args.use_layer_norm,
        "use_attention": args.use_attention,
        "epoch": epochs_completed,
    }, str(out_dir / "final_checkpoint.pt"))

    # Test-set evaluation on best model
    test_metrics = None
    if test_loader is not None:
        print("\nEvaluating best model on test set...")
        best_ckpt_path = out_dir / "best_checkpoint.pt"
        if best_ckpt_path.exists():
            best_ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=True)
            model.load_state_dict(best_ckpt["model_state_dict"])
        test_metrics = eval_epoch(
            model, test_loader, device,
            dec_z, dec_x, n_qubits, mx, mz,
            loss_fn=args.loss,
            pos_weight=args.pos_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            logical_weight=args.logical_weight,
            lx_t=lx_t, lz_t=lz_t,
            correction_mode=args.correction_mode,
            syn_weight=args.syn_weight, hx_t=hx_t, hz_t=hz_t,
        )
        print(
            f"Test Loss: {test_metrics['loss']:.6f} | "
            f"BitAcc Z: {test_metrics['bit_acc_z']:.4f} X: {test_metrics['bit_acc_x']:.4f} | "
            f"Conv Z: {test_metrics['convergence_z']:.3f} X: {test_metrics['convergence_x']:.3f}"
        )

    # Save training log
    log = {
        "task": "supervised",
        "model": "TannerGNN",
        "hidden_dim": args.hidden_dim,
        "num_mp_layers": args.num_mp_layers,
        "num_params": n_params,
        "num_epochs": args.epochs,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
        "patience": args.patience,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "loss_fn": args.loss,
        "bp_iters": args.bp_iters,
        "grad_clip": args.grad_clip,
        "scheduler": args.scheduler,
        "correction_mode": args.correction_mode,
        "accumulate_grad": args.accumulate_grad,
        "weight_decay": args.weight_decay,
        "pretrained": args.pretrained,
        "from_scratch": args.from_scratch,
        "seed": args.seed,
        "W": args.W,
        "num_train": len(train_data),
        "num_val": len(val_data),
        "num_test": len(test_data),
        "source_files": [str(p) for p in npz_paths],
        "best_val_loss": best_val_loss,
        "total_time_s": total_time,
        "history": history,
    }
    if test_metrics is not None:
        log["test_metrics"] = test_metrics

    log_path = out_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Saved training log to {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
