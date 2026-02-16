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
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader

from gnn_pipeline.bp_decoder import MinSumBPDecoder
from gnn_pipeline.dataset import build_graph_dataset, _load_npz
from gnn_pipeline.gnn_model import TannerGNN, apply_correction
from gnn_pipeline.loss_functions import weighted_bce_loss, focal_loss, logical_aware_loss


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

    for batch in loader:
        batch = batch.to(device)
        B = batch.num_graphs

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

        # Differentiable BP forward
        marginals_z, hard_z, conv_z = dec_z(x_syndrome, corrected_llr_z)
        marginals_x, hard_x, conv_x = dec_x(z_syndrome, corrected_llr_x)

        # Loss on soft marginals vs ground-truth error vectors
        if loss_fn == "mse":
            loss = F.mse_loss(marginals_z, z_error) + F.mse_loss(marginals_x, x_error)
        elif loss_fn == "bce":
            loss = F.binary_cross_entropy(marginals_z, z_error) + \
                   F.binary_cross_entropy(marginals_x, x_error)
        elif loss_fn == "weighted_bce":
            loss = weighted_bce_loss(marginals_z, z_error, pos_weight=pos_weight) + \
                   weighted_bce_loss(marginals_x, x_error, pos_weight=pos_weight)
        elif loss_fn == "focal":
            loss = focal_loss(marginals_z, z_error, alpha=focal_alpha, gamma=focal_gamma) + \
                   focal_loss(marginals_x, x_error, alpha=focal_alpha, gamma=focal_gamma)
        elif loss_fn == "logical_mse":
            loss = logical_aware_loss(
                marginals_z, marginals_x, z_error, x_error,
                lx_t, lz_t, base_loss_fn=F.mse_loss, logical_weight=logical_weight,
            )
        elif loss_fn == "logical_bce":
            loss = logical_aware_loss(
                marginals_z, marginals_x, z_error, x_error,
                lx_t, lz_t, base_loss_fn=F.binary_cross_entropy, logical_weight=logical_weight,
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

        # Gradient accumulation
        raw_loss = loss.item()
        scaled_loss = loss / accumulate_grad
        scaled_loss.backward()

        if (num_batches + 1) % accumulate_grad == 0:
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

        if loss_fn == "mse":
            loss = F.mse_loss(marginals_z, z_error) + F.mse_loss(marginals_x, x_error)
        elif loss_fn == "bce":
            loss = F.binary_cross_entropy(marginals_z, z_error) + \
                   F.binary_cross_entropy(marginals_x, x_error)
        elif loss_fn == "weighted_bce":
            loss = weighted_bce_loss(marginals_z, z_error, pos_weight=pos_weight) + \
                   weighted_bce_loss(marginals_x, x_error, pos_weight=pos_weight)
        elif loss_fn == "focal":
            loss = focal_loss(marginals_z, z_error, alpha=focal_alpha, gamma=focal_gamma) + \
                   focal_loss(marginals_x, x_error, alpha=focal_alpha, gamma=focal_gamma)
        elif loss_fn == "logical_mse":
            loss = logical_aware_loss(
                marginals_z, marginals_x, z_error, x_error,
                lx_t, lz_t, base_loss_fn=F.mse_loss, logical_weight=logical_weight,
            )
        elif loss_fn == "logical_bce":
            loss = logical_aware_loss(
                marginals_z, marginals_x, z_error, x_error,
                lx_t, lz_t, base_loss_fn=F.binary_cross_entropy, logical_weight=logical_weight,
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

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
    parser.add_argument("--pos_weight", type=float, default=50.0,
                        help="Positive class weight for weighted_bce (default: 50)")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Focal loss alpha (default: 0.25)")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma (default: 2.0)")
    parser.add_argument("--logical_weight", type=float, default=0.1,
                        help="Weight for logical-flip penalty (default: 0.1)")
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

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move logical operators to device if needed
    if lx_t is not None:
        lx_t = lx_t.to(device)
    if lz_t is not None:
        lz_t = lz_t.to(device)

    # Build BP decoders ONCE (expensive constructors)
    print(f"Building BP decoders (bp_iters={args.bp_iters})...")
    dec_z = MinSumBPDecoder(
        hx.astype(np.uint8), max_iter=args.bp_iters, alpha=0.8, clamp_llr=20.0
    ).to(device)
    dec_x = MinSumBPDecoder(
        hz.astype(np.uint8), max_iter=args.bp_iters, alpha=0.8, clamp_llr=20.0
    ).to(device)
    # Freeze BP decoder parameters (they have no learnable params, but be safe)
    for p in dec_z.parameters():
        p.requires_grad = False
    for p in dec_x.parameters():
        p.requires_grad = False
    print(f"  dec_z (hx): {dec_z.num_checks}x{dec_z.num_vars}, {dec_z.num_edges} edges")
    print(f"  dec_x (hz): {dec_x.num_checks}x{dec_x.num_vars}, {dec_x.num_edges} edges")

    # Build GNN model
    model = TannerGNN(
        hidden_dim=args.hidden_dim,
        num_mp_layers=args.num_mp_layers,
        correction_mode=args.correction_mode,
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
    optimizer = Adam(model.parameters(), lr=args.lr)
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

    # Training loop
    patience_str = f", patience={args.patience}" if args.patience > 0 else ""
    print(f"\nTraining for {args.epochs} epochs (loss={args.loss}, bp_iters={args.bp_iters}{patience_str})...")
    print("-" * 90)
    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    t_start = time.time()
    early_stopped = False
    epochs_completed = 0

    for epoch in range(args.epochs):
        t_epoch = time.time()

        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            dec_z, dec_x, n_qubits, mx, mz,
            loss_fn=args.loss, grad_clip=args.grad_clip,
            pos_weight=args.pos_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            logical_weight=args.logical_weight,
            lx_t=lx_t, lz_t=lz_t,
            accumulate_grad=args.accumulate_grad,
            correction_mode=args.correction_mode,
        )
        val_metrics = eval_epoch(
            model, val_loader, device,
            dec_z, dec_x, n_qubits, mx, mz,
            loss_fn=args.loss,
            pos_weight=args.pos_weight,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            logical_weight=args.logical_weight,
            lx_t=lx_t, lz_t=lz_t,
            correction_mode=args.correction_mode,
        )

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t_epoch
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:2d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"BitAcc Z: {val_metrics['bit_acc_z']:.4f} X: {val_metrics['bit_acc_x']:.4f} | "
            f"Conv Z: {val_metrics['convergence_z']:.3f} X: {val_metrics['convergence_x']:.3f} | "
            f"LR: {lr_now:.2e} | {elapsed:.1f}s"
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

            # Raw state dict (backward-compatible with evaluate.py)
            torch.save(model.state_dict(), str(out_dir / "best_model.pt"))

            # Rich checkpoint with architecture info
            torch.save({
                "model_state_dict": model.state_dict(),
                "hidden_dim": args.hidden_dim,
                "num_mp_layers": args.num_mp_layers,
                "node_feat_dim": 4, "edge_types": 2, "dropout": 0.1,
                "correction_mode": args.correction_mode,
                "bp_iters": args.bp_iters,
                "loss_fn": args.loss,
                "epoch": epoch + 1,
                "val_loss": best_val_loss,
            }, str(out_dir / "best_checkpoint.pt"))

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

    # Save final model
    torch.save(model.state_dict(), str(out_dir / "final_model.pt"))
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_dim": args.hidden_dim,
        "num_mp_layers": args.num_mp_layers,
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
