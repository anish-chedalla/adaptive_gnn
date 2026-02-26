"""Circuit-level supervised training: GNN-BP on DEM for QLDPC codes.

Trains a GNN to output LLR corrections for DEM fault variables, then runs
differentiable BP on the DEM PCM. Loss is computed on observable outcomes
(not per-qubit errors, which are unavailable in circuit-level sampling).

Key difference from code-capacity training:
  - PCM: DEM (detectors x fault mechanisms) instead of Hx/Hz
  - Variables: fault mechanisms (~1974) instead of data qubits (72)
  - Loss: observable flips + syndrome consistency (no ground-truth faults)
  - Single BP decoder on DEM PCM (not separate CSS decoders)

Usage:
    python -m gnn_pipeline.train_circuit \
        --in_glob "data/circuit_*.npz" \
        --epochs 20 --batch_size 8 --lr 1e-4 \
        --bp_iters 10 --out_dir runs/circuit_train
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader

from gnn_pipeline.bp_decoder import MinSumBPDecoder
from gnn_pipeline.dem_dataset import build_circuit_dataset
from gnn_pipeline.dem_decoder import extract_dem_pcm
from gnn_pipeline.gnn_model import TannerGNN, apply_correction
from gnn_pipeline.loss_functions import observable_loss
from gnn_pipeline.monitoring import TrainingMonitor


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    decoder: MinSumBPDecoder,
    num_errors: int,
    num_detectors: int,
    obs_matrix: torch.Tensor,
    dem_pcm: torch.Tensor,
    correction_mode: str = "additive",
    grad_clip: float = 1.0,
    accumulate_grad: int = 1,
    syn_weight: float = 1.0,
    obs_weight: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> dict:
    """Train for one epoch on circuit-level data."""
    model.train()
    total_loss = 0.0
    total_conv = 0.0
    total_obs_correct = 0.0
    total_obs_count = 0
    num_batches = 0
    optimizer.zero_grad()

    amp_dtype = torch.float16 if use_amp else None

    for batch in loader:
        batch = batch.to(device)
        B = batch.num_graphs

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            # GNN forward: get per-fault-variable LLR corrections
            gnn_out = model(batch)
            if correction_mode == "both":
                add_corr, mul_corr = gnn_out
                add_corr = torch.clamp(add_corr, -20.0, 20.0).view(B, num_errors)
                mul_corr = torch.clamp(mul_corr, -5.0, 5.0).view(B, num_errors)
                gnn_out = (add_corr, mul_corr)
            else:
                gnn_out = torch.clamp(gnn_out, -20.0, 20.0).view(B, num_errors)

            # Reshape batched data
            channel_llr = batch.channel_llr.view(B, num_errors)
            syndrome = batch.syndrome.view(B, num_detectors)
            obs_targets = batch.observable.view(B, -1)

            # Apply GNN corrections to DEM channel LLRs
            corrected_llr = apply_correction(channel_llr, gnn_out, correction_mode)

            # Differentiable BP on DEM PCM
            marginals, hard_decision, converged = decoder(syndrome, corrected_llr)

            # Observable loss + syndrome consistency
            loss = observable_loss(
                marginals, obs_matrix, obs_targets,
                syndrome, dem_pcm,
                syn_weight=syn_weight, obs_weight=obs_weight,
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
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if hasattr(decoder, 'w_ch'):
                    nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if hasattr(decoder, 'w_ch'):
                    nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        # Metrics
        total_loss += raw_loss
        total_conv += converged.float().mean().item()

        # Observable accuracy
        with torch.no_grad():
            pred_obs = ((hard_decision.float() @ obs_matrix.float()) % 2).float()
            n_obs = min(pred_obs.shape[1], obs_targets.shape[1])
            correct = (pred_obs[:, :n_obs] == obs_targets[:, :n_obs]).all(dim=1).float()
            total_obs_correct += correct.sum().item()
            total_obs_count += B

        num_batches += 1

    # Flush remaining accumulated gradients
    if num_batches % accumulate_grad != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if hasattr(decoder, 'w_ch'):
                nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if hasattr(decoder, 'w_ch'):
                nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad()

    return {
        "loss": total_loss / max(num_batches, 1),
        "convergence": total_conv / max(num_batches, 1),
        "obs_accuracy": total_obs_correct / max(total_obs_count, 1),
    }


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    decoder: MinSumBPDecoder,
    num_errors: int,
    num_detectors: int,
    obs_matrix: torch.Tensor,
    dem_pcm: torch.Tensor,
    correction_mode: str = "additive",
    syn_weight: float = 1.0,
    obs_weight: float = 1.0,
) -> dict:
    """Evaluate on circuit-level data."""
    model.eval()
    total_loss = 0.0
    total_conv = 0.0
    total_obs_correct = 0.0
    total_obs_count = 0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        B = batch.num_graphs

        gnn_out = model(batch)
        if correction_mode == "both":
            add_corr, mul_corr = gnn_out
            add_corr = torch.clamp(add_corr, -20.0, 20.0).view(B, num_errors)
            mul_corr = torch.clamp(mul_corr, -5.0, 5.0).view(B, num_errors)
            gnn_out = (add_corr, mul_corr)
        else:
            gnn_out = torch.clamp(gnn_out, -20.0, 20.0).view(B, num_errors)

        channel_llr = batch.channel_llr.view(B, num_errors)
        syndrome = batch.syndrome.view(B, num_detectors)
        obs_targets = batch.observable.view(B, -1)

        corrected_llr = apply_correction(channel_llr, gnn_out, correction_mode)
        marginals, hard_decision, converged = decoder(syndrome, corrected_llr)

        loss = observable_loss(
            marginals, obs_matrix, obs_targets,
            syndrome, dem_pcm,
            syn_weight=syn_weight, obs_weight=obs_weight,
        )

        total_loss += loss.item()
        total_conv += converged.float().mean().item()

        pred_obs = ((hard_decision.float() @ obs_matrix.float()) % 2).float()
        n_obs = min(pred_obs.shape[1], obs_targets.shape[1])
        correct = (pred_obs[:, :n_obs] == obs_targets[:, :n_obs]).all(dim=1).float()
        total_obs_correct += correct.sum().item()
        total_obs_count += B

        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "convergence": total_conv / max(num_batches, 1),
        "obs_accuracy": total_obs_correct / max(total_obs_count, 1),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Circuit-level GNN-BP training on DEM data"
    )

    # Data
    parser.add_argument("--in_glob", type=str, required=True,
                        help="Glob pattern for circuit-level NPZ files")
    parser.add_argument("--seed", type=int, default=42)

    # GNN architecture
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_mp_layers", type=int, default=3)
    parser.add_argument("--correction_mode", type=str, default="additive",
                        choices=["additive", "multiplicative", "both"])
    parser.add_argument("--use_residual", action="store_true")
    parser.add_argument("--use_layer_norm", action="store_true")
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--use_film", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)

    # BP
    parser.add_argument("--bp_iters", type=int, default=10,
                        help="BP iterations during training (fewer = faster)")
    parser.add_argument("--neural_bp", action="store_true",
                        help="Enable per-iteration learned BP weights")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accumulate_grad", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=2)

    # Loss
    parser.add_argument("--syn_weight", type=float, default=1.0,
                        help="Weight for syndrome consistency loss")
    parser.add_argument("--obs_weight", type=float, default=1.0,
                        help="Weight for observable prediction loss")

    # AMP
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed-precision training (GPU only)")

    # Pretrained
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model checkpoint")

    # Output
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    npz_paths = sorted(glob.glob(args.in_glob))
    if not npz_paths:
        print(f"ERROR: No files found matching '{args.in_glob}'")
        return 1

    print(f"Loading {len(npz_paths)} circuit-level NPZ file(s)...")
    train_data, val_data, test_data, data_meta = build_circuit_dataset(
        npz_paths, seed=args.seed,
    )
    print(f"Dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    if len(train_data) == 0:
        print("ERROR: No training samples loaded.")
        return 1

    # Extract DEM dimensions from first sample
    sample = train_data[0]
    num_errors = int(sample.num_errors)
    num_detectors = int(sample.num_detectors)
    num_obs = sample.observable.shape[0]
    print(f"DEM: {num_detectors} detectors x {num_errors} errors, {num_obs} observables")

    # --- Build data loaders ---
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # --- Build GNN model ---
    # DEM graph has 1 edge type (all DEM connections), vs 2 for code-capacity
    gnn = TannerGNN(
        node_feat_dim=4,
        hidden_dim=args.hidden_dim,
        num_mp_layers=args.num_mp_layers,
        edge_types=1,  # DEM: single edge type
        dropout=args.dropout,
        correction_mode=args.correction_mode,
        use_residual=args.use_residual,
        use_layer_norm=args.use_layer_norm,
        use_attention=args.use_attention,
        use_film=args.use_film,
        noise_feat_dim=1,
    ).to(device)

    # Load pretrained weights if available
    if args.pretrained:
        print(f"Loading pretrained model from {args.pretrained}...")
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            gnn.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            gnn.load_state_dict(ckpt, strict=False)
        print("  Pretrained weights loaded (strict=False for architecture changes)")

    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"GNN params: {n_params:,}")

    # --- Extract shared DEM matrices from dataset ---
    obs_matrix_t = data_meta["obs_matrix"].to(device)
    dem_pcm_t = data_meta["dem_pcm"].to(device)

    # --- Build BP decoder on DEM PCM ---
    dem_pcm_np = dem_pcm_t.cpu().numpy().astype(np.uint8)
    decoder = MinSumBPDecoder(
        dem_pcm_np, max_iter=args.bp_iters, alpha=0.8, clamp_llr=20.0,
        neural_bp=args.neural_bp,
    ).to(device)

    bp_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"BP decoder params: {bp_params} (neural_bp={args.neural_bp})")

    # --- Optimizer ---
    param_groups = [{"params": gnn.parameters(), "lr": args.lr}]
    if args.neural_bp:
        param_groups.append({"params": decoder.parameters(), "lr": args.lr * 0.1})
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    # --- AMP setup ---
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Mixed-precision training enabled (AMP)")
    elif args.amp:
        print("AMP requested but CUDA not available, using fp32")

    # --- Scheduler ---
    if args.scheduler == "cosine" and args.epochs > args.warmup_epochs:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])
    else:
        scheduler = None

    # --- Output directory ---
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["num_errors"] = num_errors
    config["num_detectors"] = num_detectors
    config["num_obs"] = num_obs
    config["gnn_params"] = n_params
    config["bp_params"] = bp_params
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # --- Training loop ---
    best_val_loss = float("inf")
    patience_counter = 0
    log_lines = []
    monitor = TrainingMonitor(out_dir)

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  correction_mode={args.correction_mode}, neural_bp={args.neural_bp}")
    print(f"  bp_iters={args.bp_iters}, batch_size={args.batch_size}")
    print(f"  syn_weight={args.syn_weight}, obs_weight={args.obs_weight}")

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        train_metrics = train_epoch(
            gnn, train_loader, optimizer, device, decoder,
            num_errors, num_detectors,
            obs_matrix=obs_matrix_t, dem_pcm=dem_pcm_t,
            correction_mode=args.correction_mode,
            grad_clip=args.grad_clip,
            accumulate_grad=args.accumulate_grad,
            syn_weight=args.syn_weight,
            obs_weight=args.obs_weight,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_metrics = eval_epoch(
            gnn, val_loader, device, decoder,
            num_errors, num_detectors,
            obs_matrix=obs_matrix_t, dem_pcm=dem_pcm_t,
            correction_mode=args.correction_mode,
            syn_weight=args.syn_weight,
            obs_weight=args.obs_weight,
        )

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t_start
        lr_now = optimizer.param_groups[0]["lr"]

        line = (
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"conv={train_metrics['convergence']:.1%} "
            f"obs_acc={train_metrics['obs_accuracy']:.1%} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"conv={val_metrics['convergence']:.1%} "
            f"obs_acc={val_metrics['obs_accuracy']:.1%} | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s"
        )
        print(line)
        log_lines.append(line)

        monitor.log_epoch(epoch, {
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_conv": train_metrics["convergence"],
            "val_conv": val_metrics["convergence"],
            "train_obs_acc": train_metrics["obs_accuracy"],
            "val_obs_acc": val_metrics["obs_accuracy"],
            "lr": lr_now,
            "time_s": elapsed,
        })
        monitor.flush_csv()

        # Checkpointing
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            checkpoint = {
                "model_state_dict": gnn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "val_obs_accuracy": val_metrics["obs_accuracy"],
                # Architecture metadata
                "mode": "circuit_level",
                "hidden_dim": args.hidden_dim,
                "num_mp_layers": args.num_mp_layers,
                "correction_mode": args.correction_mode,
                "use_residual": args.use_residual,
                "use_layer_norm": args.use_layer_norm,
                "use_attention": args.use_attention,
                "use_film": args.use_film,
                "noise_feat_dim": 1,
                "neural_bp": args.neural_bp,
                "bp_iters": args.bp_iters,
                "edge_types": 1,  # DEM
                "num_errors": num_errors,
                "num_detectors": num_detectors,
                "num_obs": num_obs,
            }
            if args.neural_bp:
                checkpoint["decoder_state_dict"] = decoder.state_dict()
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            if scaler is not None:
                checkpoint["scaler_state_dict"] = scaler.state_dict()

            torch.save(checkpoint, out_dir / "best_model.pt")
            print(f"  ** Best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # --- Test evaluation (reload best model) ---
    if test_data:
        best_ckpt_path = out_dir / "best_model.pt"
        if best_ckpt_path.exists():
            print(f"\nReloading best model from {best_ckpt_path}...")
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
            gnn.load_state_dict(best_ckpt["model_state_dict"])
            if args.neural_bp and "decoder_state_dict" in best_ckpt:
                decoder.load_state_dict(best_ckpt["decoder_state_dict"], strict=False)
        print("Evaluating on test set...")
        test_metrics = eval_epoch(
            gnn, test_loader, device, decoder,
            num_errors, num_detectors,
            obs_matrix=obs_matrix_t, dem_pcm=dem_pcm_t,
            correction_mode=args.correction_mode,
            syn_weight=args.syn_weight,
            obs_weight=args.obs_weight,
        )
        print(f"Test: loss={test_metrics['loss']:.4f} "
              f"conv={test_metrics['convergence']:.1%} "
              f"obs_acc={test_metrics['obs_accuracy']:.1%}")
        log_lines.append(f"\nTest: {test_metrics}")

    # Save training log
    with open(out_dir / "train_log.txt", "w") as f:
        f.write("\n".join(log_lines))
    monitor.save(config)

    print(f"\nTraining complete. Output: {out_dir}")
    print(f"  Metrics CSV: {monitor.csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
