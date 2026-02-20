"""CLI to train GNN on self-supervised syndrome prediction task.

The GNN learns to correct channel LLRs such that the predicted syndrome
(from the corrected LLRs) matches the observed syndrome. No error vector
labels are needed — only syndromes.

Key improvements over naive approach:
  - Separate X/Z syndrome prediction (correct CSS decoding)
  - AdamW with weight decay for regularization
  - Cosine LR scheduler with optional warmup
  - AMP support for GPU training
  - Full checkpoint saving (architecture + optimizer state)
  - Early stopping with patience
  - Gradient accumulation for large effective batch sizes
  - Configurable GNN architecture (hidden_dim, num_mp_layers, etc.)

Usage:
    python -m gnn_pipeline.train_selfsupervised \
        --in_glob "data/*.npz" --W 4 --epochs 15 --batch_size 32 \
        --hidden_dim 128 --num_mp_layers 5 --use_residual --use_layer_norm \
        --scheduler cosine --weight_decay 1e-4 --out_dir "runs/selfsup"
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
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader

from gnn_pipeline.dataset import build_graph_dataset, _load_npz
from gnn_pipeline.gnn_model import TannerGNN, apply_correction


def _compute_syndrome_predictions(
    data_qubits_llr: torch.Tensor,
    H: torch.Tensor,
) -> torch.Tensor:
    """Convert per-qubit LLRs to syndrome predictions via exact XOR probability.

    Uses the identity: P(odd parity of X_1,...,X_k) = 0.5 * (1 - prod(1 - 2*p_i))
    where p_i = P(X_i = 1) are independent error probabilities.

    Args:
        data_qubits_llr: (B, n) per-qubit error LLRs
        H: (m, n) parity-check matrix (float tensor on same device)

    Returns:
        syndrome_pred: (B, m) predicted syndrome probabilities
    """
    # Convert LLRs to error probabilities: P(error) = sigmoid(-LLR)
    error_probs = torch.sigmoid(-data_qubits_llr)  # (B, n)

    # Compute 1 - 2*p for each qubit
    one_minus_2p = 1.0 - 2.0 * error_probs  # (B, n)

    # For each check j, compute prod_i (1 - 2*p_i)^{H[j,i]}
    # Use log-domain for numerical stability:
    log_abs = torch.log(torch.abs(one_minus_2p) + 1e-10)  # (B, n)
    prod_log = torch.matmul(log_abs, H.T)  # (B, m)
    prod_abs = torch.exp(prod_log)

    # Track sign: count how many negative (1-2p) entries per check
    is_neg = (one_minus_2p < 0).float()  # (B, n)
    neg_count = torch.matmul(is_neg, H.T)  # (B, m)
    sign = 1.0 - 2.0 * (neg_count % 2)  # +1 if even negatives, -1 if odd

    prod_val = sign * prod_abs  # (B, m)

    syndrome_pred = 0.5 * (1.0 - prod_val)  # (B, m)

    # Clamp for numerical stability before BCE
    syndrome_pred = torch.clamp(syndrome_pred, 1e-7, 1.0 - 1e-7)

    return syndrome_pred


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    Hx: torch.Tensor,
    Hz: torch.Tensor,
    n_qubits: int,
    mx: int,
    mz: int,
    correction_mode: str = "additive",
    grad_clip: float = 1.0,
    accumulate_grad: int = 1,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> float:
    """Train for one epoch on self-supervised loss (separate CSS syndrome prediction)."""
    model.train()
    total_loss = 0.0
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

            # Reshape flat tensors
            channel_llr = batch.channel_llr.view(B, n_qubits)
            target_syn = batch.target_syndrome.view(B, n_checks)

            # Apply correction
            corrected_llr = apply_correction(channel_llr, gnn_out, correction_mode)

            # Split syndrome into X-checks and Z-checks
            x_syndrome = target_syn[:, :mx]   # (B, mx) — X-checks detect Z-errors
            z_syndrome = target_syn[:, mx:]   # (B, mz) — Z-checks detect X-errors

            # Separate CSS syndrome prediction:
            # hx @ z_error -> x_syndrome, so predict x_syndrome from corrected_llr via hx
            # hz @ x_error -> z_syndrome, so predict z_syndrome from corrected_llr via hz
            pred_x_syn = _compute_syndrome_predictions(corrected_llr, Hx)  # (B, mx)
            pred_z_syn = _compute_syndrome_predictions(corrected_llr, Hz)  # (B, mz)

            # Self-supervised loss: BCE on separate syndrome predictions
            loss = nn.functional.binary_cross_entropy(pred_x_syn, x_syndrome) + \
                   nn.functional.binary_cross_entropy(pred_z_syn, z_syndrome)

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

        total_loss += raw_loss
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

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    Hx: torch.Tensor,
    Hz: torch.Tensor,
    n_qubits: int,
    mx: int,
    mz: int,
    correction_mode: str = "additive",
) -> float:
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
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

        channel_llr = batch.channel_llr.view(B, n_qubits)
        target_syn = batch.target_syndrome.view(B, n_checks)

        corrected_llr = apply_correction(channel_llr, gnn_out, correction_mode)

        x_syndrome = target_syn[:, :mx]
        z_syndrome = target_syn[:, mx:]

        pred_x_syn = _compute_syndrome_predictions(corrected_llr, Hx)
        pred_z_syn = _compute_syndrome_predictions(corrected_llr, Hz)

        loss = nn.functional.binary_cross_entropy(pred_x_syn, x_syndrome) + \
               nn.functional.binary_cross_entropy(pred_z_syn, z_syndrome)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main(argv: List[str] | None = None) -> int:
    """Main entry point for self-supervised training CLI."""
    parser = argparse.ArgumentParser(description="Train GNN on self-supervised syndrome prediction")
    parser.add_argument("--in_glob", type=str, required=True,
                        help="Glob pattern for input NPZ files")
    parser.add_argument("--W", type=int, default=4,
                        help="Window width for syndrome samples")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay (0 = plain Adam)")
    parser.add_argument("--scheduler", type=str, default="none",
                        choices=["none", "cosine"],
                        help="Learning rate scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="Linear LR warmup epochs (0 = disabled)")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--accumulate_grad", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed-precision training (GPU only)")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="GNN hidden dimension")
    parser.add_argument("--num_mp_layers", type=int, default=3,
                        help="Number of GNN message-passing layers")
    parser.add_argument("--correction_mode", type=str, default="additive",
                        choices=["additive", "multiplicative", "both"],
                        help="How GNN corrections are applied to channel LLRs")
    parser.add_argument("--use_residual", action="store_true",
                        help="Enable residual connections in GNN")
    parser.add_argument("--use_layer_norm", action="store_true",
                        help="Enable layer normalization in GNN")
    parser.add_argument("--use_attention", action="store_true",
                        help="Enable attention-weighted message passing in GNN")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for checkpoints and metadata")

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

    # Build dataset
    print(f"Building dataset with W={args.W}...")
    train_data, val_data, test_data, meta = build_graph_dataset(
        npz_paths=npz_paths,
        W=args.W,
        mode="selfsup",
        seed=args.seed,
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    if len(train_data) == 0:
        print("ERROR: No training data", file=sys.stderr)
        return 1

    # Extract PCM matrices from NPZ file
    first_npz = _load_npz(npz_paths[0])
    hx = first_npz["hx"].astype(np.float32)
    hz = first_npz["hz"].astype(np.float32)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build SEPARATE PCM tensors for CSS syndrome prediction
    n_qubits = hx.shape[1]
    mx = hx.shape[0]
    mz = hz.shape[0]
    Hx = torch.from_numpy(hx).float().to(device)  # (mx, n) for X-syndrome prediction
    Hz = torch.from_numpy(hz).float().to(device)  # (mz, n) for Z-syndrome prediction
    print(f"PCM shape: hx={hx.shape}, hz={hz.shape}")
    print(f"n_qubits={n_qubits}, mx={mx}, mz={mz}")

    # Build GNN model with configurable architecture
    model = TannerGNN(
        hidden_dim=args.hidden_dim,
        num_mp_layers=args.num_mp_layers,
        correction_mode=args.correction_mode,
        use_residual=args.use_residual,
        use_layer_norm=args.use_layer_norm,
        use_attention=args.use_attention,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} trainable parameters "
          f"(hidden_dim={args.hidden_dim}, mp_layers={args.num_mp_layers}, "
          f"residual={args.use_residual}, layer_norm={args.use_layer_norm})")

    # Optimizer
    if args.weight_decay > 0:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)
        print(f"Optimizer: Adam (lr={args.lr})")

    # LR scheduler
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
            print(f"Scheduler: cosine with {args.warmup_epochs}-epoch warmup")
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
            )
            print(f"Scheduler: cosine ({args.lr} -> {args.lr * 0.01})")

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

    # Training loop
    patience_str = f", patience={args.patience}" if args.patience > 0 else ""
    print(f"\nTraining for {args.epochs} epochs{patience_str}...")
    print("-" * 80)
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    t_start = time.time()
    early_stopped = False
    epochs_completed = 0

    for epoch in range(args.epochs):
        t_epoch = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            Hx, Hz, n_qubits, mx, mz,
            correction_mode=args.correction_mode,
            grad_clip=args.grad_clip,
            accumulate_grad=args.accumulate_grad,
            scaler=scaler, use_amp=use_amp,
        )
        val_loss = eval_epoch(
            model, val_loader, device,
            Hx, Hz, n_qubits, mx, mz,
            correction_mode=args.correction_mode,
        )

        if scheduler is not None:
            scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        elapsed = time.time() - t_epoch
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
              f"LR: {lr_now:.2e} | {elapsed:.1f}s")

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
                "node_feat_dim": 4,
                "edge_types": 2,
                "dropout": 0.1,
                "correction_mode": args.correction_mode,
                "use_residual": args.use_residual,
                "use_layer_norm": args.use_layer_norm,
                "use_attention": args.use_attention,
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
    print("-" * 80)
    stop_reason = "early stopping" if early_stopped else "completed"
    print(f"Training {stop_reason} in {total_time:.1f}s ({total_time/60:.1f} min), "
          f"{epochs_completed}/{args.epochs} epochs")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Save final model (with architecture info)
    final_ckpt = out_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_dim": args.hidden_dim,
        "num_mp_layers": args.num_mp_layers,
        "correction_mode": args.correction_mode,
        "use_residual": args.use_residual,
        "use_layer_norm": args.use_layer_norm,
        "use_attention": args.use_attention,
        "epoch": epochs_completed,
    }, str(final_ckpt))
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
    print(f"Saved final model to {final_ckpt}")

    # Verify saves
    if not final_ckpt.exists() or final_ckpt.stat().st_size == 0:
        print(f"ERROR: Failed to save final model", file=sys.stderr)
        return 1

    # Save training metadata
    meta_save = {
        "task": "selfsupervised",
        "model": "TannerGNN",
        "hidden_dim": args.hidden_dim,
        "num_mp_layers": args.num_mp_layers,
        "correction_mode": args.correction_mode,
        "use_residual": args.use_residual,
        "use_layer_norm": args.use_layer_norm,
        "use_attention": args.use_attention,
        "num_params": n_params,
        "num_epochs": args.epochs,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
        "patience": args.patience,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "grad_clip": args.grad_clip,
        "accumulate_grad": args.accumulate_grad,
        "seed": args.seed,
        "W": args.W,
        "num_train": len(train_data),
        "num_val": len(val_data),
        "source_files": [str(p) for p in npz_paths],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "total_time_s": total_time,
        "final_model": str(final_ckpt),
        "best_model": str(out_dir / "best_model.pt"),
    }

    meta_path = out_dir / "train_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_save, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
