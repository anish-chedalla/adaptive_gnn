"""CLI to train GNN on self-supervised syndrome prediction task.

Usage:
    python -m gnn_pipeline.train_selfsupervised --in_glob "data/*.npz" --W 4 --epochs 10 --batch_size 16 --out_dir "runs/selfsup_test"
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from gnn_pipeline.dataset import build_graph_dataset
from gnn_pipeline.gnn_model import TannerGNN


def _compute_syndrome_predictions(
    data_qubits_llr: torch.Tensor,
    H: torch.Tensor,
) -> torch.Tensor:
    """Convert per-qubit LLRs to syndrome predictions via exact XOR probability.

    Uses the identity: P(odd parity of X_1,...,X_k) = 0.5 * (1 - prod(1 - 2*p_i))
    where p_i = P(X_i = 1) are independent error probabilities.

    Args:
        data_qubits_llr: (B, n) per-qubit error LLRs
        H: (m, n) combined parity-check matrix (float tensor on same device)

    Returns:
        syndrome_pred: (B, m) predicted syndrome probabilities
    """
    # Convert LLRs to error probabilities: P(error) = sigmoid(-LLR)
    error_probs = torch.sigmoid(-data_qubits_llr)  # (B, n)

    # Compute 1 - 2*p for each qubit
    one_minus_2p = 1.0 - 2.0 * error_probs  # (B, n)

    # For each check j, compute prod_i (1 - 2*p_i)^{H[j,i]}
    # Use log-domain for numerical stability:
    #   log|1-2p| summed over support of H[j,:]
    log_abs = torch.log(torch.abs(one_minus_2p) + 1e-10)  # (B, n)
    # H is (m, n), log_abs is (B, n) -> matmul gives (B, m)
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
    H: torch.Tensor,
    n_qubits: int,
    n_checks: int,
) -> float:
    """Train for one epoch on self-supervised loss (syndrome prediction)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        B = batch.num_graphs

        # Forward pass: returns (B*n_qubits,) flat
        llr_corrections = model(batch)
        llr_corrections = torch.clamp(llr_corrections, -20.0, 20.0)

        # Reshape flat tensors to (B, n) and (B, m)
        llr_corrections = llr_corrections.view(B, n_qubits)
        channel_llr = batch.channel_llr.view(B, n_qubits)
        target_syn = batch.target_syndrome.view(B, n_checks)

        corrected_llr = channel_llr + llr_corrections

        # Predict syndrome from corrected LLRs
        syndrome_pred = _compute_syndrome_predictions(corrected_llr, H)

        # Self-supervised loss: BCE on syndrome prediction
        loss = nn.functional.binary_cross_entropy(syndrome_pred, target_syn)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    H: torch.Tensor,
    n_qubits: int,
    n_checks: int,
) -> float:
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        B = batch.num_graphs

        llr_corrections = model(batch)
        llr_corrections = torch.clamp(llr_corrections, -20.0, 20.0)

        llr_corrections = llr_corrections.view(B, n_qubits)
        channel_llr = batch.channel_llr.view(B, n_qubits)
        target_syn = batch.target_syndrome.view(B, n_checks)

        corrected_llr = channel_llr + llr_corrections

        syndrome_pred = _compute_syndrome_predictions(corrected_llr, H)

        loss = nn.functional.binary_cross_entropy(syndrome_pred, target_syn)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main(argv: List[str] | None = None) -> int:
    """Main entry point for self-supervised training CLI."""
    parser = argparse.ArgumentParser(description="Train GNN on self-supervised syndrome prediction")
    parser.add_argument(
        "--in_glob",
        type=str,
        required=True,
        help="Glob pattern for input NPZ files",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=4,
        help="Window width for syndrome samples",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and metadata",
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

    # Build dataset
    print(f"Building dataset with W={args.W}...")
    train_data, val_data, test_data, meta = build_graph_dataset(
        npz_paths=npz_paths,
        W=args.W,
        mode="selfsup",
        seed=args.seed,
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Extract PCM matrices from NPZ file
    if len(train_data) == 0:
        print("ERROR: No training data", file=sys.stderr)
        return 1

    from gnn_pipeline.dataset import _load_npz
    first_npz = _load_npz(npz_paths[0])
    hx = first_npz["hx"].astype(np.float32)
    hz = first_npz["hz"].astype(np.float32)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build combined PCM tensor for syndrome prediction
    n_qubits = hx.shape[1]
    n_checks = hx.shape[0] + hz.shape[0]
    H = torch.from_numpy(np.vstack([hx, hz])).float().to(device)
    print(f"PCM shape: hx={hx.shape}, hz={hz.shape}, H={H.shape}")
    print(f"n_qubits={n_qubits}, n_checks={n_checks}")

    model = TannerGNN()  # defaults: node_feat_dim=4, hidden_dim=64
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # Create output directory
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, H, n_qubits, n_checks)
        val_loss = eval_epoch(model, val_loader, device, H, n_qubits, n_checks)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = out_dir / "best_model.pt"
            torch.save(model.state_dict(), str(ckpt_path))
            print(f"  Saved best model to {ckpt_path}")

    # Save final model and metadata
    final_ckpt = out_dir / "final_model.pt"
    torch.save(model.state_dict(), str(final_ckpt))
    print(f"Saved final model to {final_ckpt}")

    # Verify saves
    if not final_ckpt.exists() or final_ckpt.stat().st_size == 0:
        print(f"ERROR: Failed to save final model", file=sys.stderr)
        return 1

    # Save training metadata
    meta_save = {
        "task": "selfsupervised",
        "model": "TannerGNN",
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "W": args.W,
        "num_train": len(train_data),
        "num_val": len(val_data),
        "source_files": [str(p) for p in npz_paths],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
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
