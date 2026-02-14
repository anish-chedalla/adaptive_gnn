"""Ablation study: sweep over hyperparameters and compare results.

Automates training + evaluation across multiple configurations,
collects results, and outputs a CSV comparison table.

Usage:
    # Default ablation grid
    python -m gnn_pipeline.ablation \
        --train_data "data/supervised_train.npz" \
        --test_data "data/supervised_test.npz" \
        --pretrained runs/drift_train/best_model.pt \
        --epochs 10 --out_dir runs/ablation

    # Custom grid via JSON
    python -m gnn_pipeline.ablation \
        --train_data "data/supervised_train.npz" \
        --test_data "data/supervised_test.npz" \
        --grid_json ablation_grid.json \
        --epochs 10 --out_dir runs/ablation

    # Evaluate-only (skip training, read existing results)
    python -m gnn_pipeline.ablation \
        --train_data "data/supervised_train.npz" \
        --test_data "data/supervised_test.npz" \
        --out_dir runs/ablation --eval_only
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import pathlib
import subprocess
import sys
import time
from typing import List, Optional


DEFAULT_GRID = {
    "hidden_dim": [64, 128],
    "num_mp_layers": [3, 5],
    "bp_iters": [10, 20],
    "loss": ["mse", "weighted_bce", "focal"],
}


def _build_configs(grid: dict, use_pretrained: bool) -> list:
    """Generate all config combinations from the grid."""
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        if use_pretrained:
            cfg["pretrained"] = True
        else:
            cfg["pretrained"] = False
        configs.append(cfg)

    # Optionally add from-scratch variants for comparison
    if use_pretrained:
        # Add one from-scratch baseline per loss function
        for loss in grid.get("loss", ["mse"]):
            cfg = {
                "hidden_dim": grid.get("hidden_dim", [64])[0],
                "num_mp_layers": grid.get("num_mp_layers", [3])[0],
                "bp_iters": grid.get("bp_iters", [10])[0],
                "loss": loss,
                "pretrained": False,
            }
            configs.append(cfg)

    return configs


def _config_name(idx: int, cfg: dict) -> str:
    """Generate a short name for a config."""
    parts = [f"h{cfg['hidden_dim']}", f"l{cfg['num_mp_layers']}",
             f"bp{cfg['bp_iters']}", cfg["loss"]]
    if cfg.get("pretrained"):
        parts.append("pt")
    else:
        parts.append("scratch")
    return f"cfg{idx:03d}_{'_'.join(parts)}"


def _run_training(
    config_name: str,
    cfg: dict,
    train_data: str,
    pretrained_path: Optional[str],
    epochs: int,
    batch_size: int,
    lr: float,
    out_dir: pathlib.Path,
) -> bool:
    """Run training for one config as a subprocess."""
    run_dir = out_dir / config_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "gnn_pipeline.train_supervised",
        "--in_glob", train_data,
        "--hidden_dim", str(cfg["hidden_dim"]),
        "--num_mp_layers", str(cfg["num_mp_layers"]),
        "--bp_iters", str(cfg["bp_iters"]),
        "--loss", cfg["loss"],
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--scheduler", "cosine",
        "--out_dir", str(run_dir),
    ]

    if cfg.get("pretrained") and pretrained_path:
        cmd.extend(["--pretrained", pretrained_path])
    else:
        cmd.append("--from_scratch")

    # Add loss-specific args
    if cfg["loss"] == "weighted_bce":
        cmd.extend(["--pos_weight", "50"])
    elif cfg["loss"] == "focal":
        cmd.extend(["--focal_alpha", "0.25", "--focal_gamma", "2.0"])

    print(f"  CMD: {' '.join(cmd)}")

    log_path = run_dir / "train_log.txt"
    try:
        with open(log_path, "w") as log_f:
            result = subprocess.run(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                timeout=3600, cwd=str(out_dir.parent),
            )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 3600s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def _run_evaluation(
    config_name: str,
    test_data: str,
    out_dir: pathlib.Path,
    use_bposd: bool = False,
    use_mwpm: bool = False,
) -> bool:
    """Run evaluation for one config."""
    run_dir = out_dir / config_name
    model_path = run_dir / "best_model.pt"

    if not model_path.exists():
        print(f"  No model at {model_path}, skipping evaluation")
        return False

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "gnn_pipeline.evaluate",
        "--test_npz", test_data,
        "--gnn_model", str(model_path),
        "--out_dir", str(eval_dir),
    ]

    if use_bposd:
        cmd.append("--bposd")
    if use_mwpm:
        cmd.append("--mwpm")

    log_path = run_dir / "eval_log.txt"
    try:
        with open(log_path, "w") as log_f:
            result = subprocess.run(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                timeout=1800, cwd=str(out_dir.parent),
            )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 1800s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def _collect_results(configs: list, config_names: list, out_dir: pathlib.Path) -> list:
    """Collect results from all completed runs."""
    rows = []
    for cfg, name in zip(configs, config_names):
        run_dir = out_dir / name

        # Read training results
        train_log = run_dir / "train_log.txt"
        train_loss = None
        val_loss = None
        if train_log.exists():
            text = train_log.read_text(errors="replace")
            # Parse last epoch's train/val loss from log
            for line in reversed(text.splitlines()):
                if "train_loss=" in line and train_loss is None:
                    try:
                        train_loss = float(line.split("train_loss=")[1].split(",")[0].split()[0])
                    except (ValueError, IndexError):
                        pass
                if "val_loss=" in line and val_loss is None:
                    try:
                        val_loss = float(line.split("val_loss=")[1].split(",")[0].split()[0])
                    except (ValueError, IndexError):
                        pass

        # Read eval results
        eval_results_path = run_dir / "eval" / "eval_results.json"
        bp_ler = None
        gnn_bp_ler = None
        improvement = None
        bposd_ler = None
        mwpm_ler = None

        if eval_results_path.exists():
            with open(eval_results_path) as f:
                eval_data = json.load(f)
            bp_ler = eval_data.get("bp", {}).get("ler")
            gnn_bp_ler = eval_data.get("gnn_bp", {}).get("ler")
            improvement = eval_data.get("improvement_pct")
            bposd_ler = eval_data.get("bposd", {}).get("ler")
            mwpm_ler = eval_data.get("mwpm", {}).get("ler")

        row = {
            "config": name,
            "hidden_dim": cfg["hidden_dim"],
            "num_mp_layers": cfg["num_mp_layers"],
            "bp_iters": cfg["bp_iters"],
            "loss": cfg["loss"],
            "pretrained": cfg.get("pretrained", False),
            "train_loss": f"{train_loss:.6f}" if train_loss is not None else "N/A",
            "val_loss": f"{val_loss:.6f}" if val_loss is not None else "N/A",
            "bp_ler": f"{bp_ler:.6f}" if bp_ler is not None else "N/A",
            "gnn_bp_ler": f"{gnn_bp_ler:.6f}" if gnn_bp_ler is not None else "N/A",
            "improvement_pct": f"{improvement:.1f}" if improvement is not None else "N/A",
            "bposd_ler": f"{bposd_ler:.6f}" if bposd_ler is not None else "N/A",
            "mwpm_ler": f"{mwpm_ler:.6f}" if mwpm_ler is not None else "N/A",
        }
        rows.append(row)

    return rows


def _print_table(rows: list):
    """Print a formatted comparison table."""
    if not rows:
        print("No results to display.")
        return

    # Column widths
    headers = ["config", "hidden", "layers", "bp_iter", "loss", "pretrain",
               "train_loss", "val_loss", "bp_ler", "gnn_bp_ler", "improv%"]
    col_widths = [max(len(h), 10) for h in headers]

    # Header
    header_line = " | ".join(h.center(w) for h, w in zip(headers, col_widths))
    sep_line = "-+-".join("-" * w for w in col_widths)
    print(f"\n{header_line}")
    print(sep_line)

    # Sort by improvement (best first)
    sorted_rows = sorted(rows, key=lambda r: float(r["improvement_pct"])
                         if r["improvement_pct"] != "N/A" else -999, reverse=True)

    for row in sorted_rows:
        vals = [
            row["config"][:col_widths[0]],
            str(row["hidden_dim"]).center(col_widths[1]),
            str(row["num_mp_layers"]).center(col_widths[2]),
            str(row["bp_iters"]).center(col_widths[3]),
            row["loss"].center(col_widths[4]),
            str(row["pretrained"]).center(col_widths[5]),
            row["train_loss"].center(col_widths[6]),
            row["val_loss"].center(col_widths[7]),
            row["bp_ler"].center(col_widths[8]),
            row["gnn_bp_ler"].center(col_widths[9]),
            row["improvement_pct"].center(col_widths[10]),
        ]
        print(" | ".join(v.ljust(w) for v, w in zip(vals, col_widths)))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ablation study over hyperparameters")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training NPZ (glob pattern)")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test NPZ")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model for fine-tuning configs")
    parser.add_argument("--grid_json", type=str, default=None,
                        help="Custom ablation grid as JSON file")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per config")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--bposd", action="store_true",
                        help="Include BP-OSD in evaluation")
    parser.add_argument("--mwpm", action="store_true",
                        help="Include MWPM in evaluation")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, only collect and display results")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory")

    args = parser.parse_args(argv)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load or build grid
    if args.grid_json:
        with open(args.grid_json) as f:
            grid = json.load(f)
        print(f"Loaded custom grid from {args.grid_json}")
    else:
        grid = DEFAULT_GRID
        print("Using default ablation grid")

    use_pretrained = args.pretrained is not None
    configs = _build_configs(grid, use_pretrained)
    config_names = [_config_name(i, cfg) for i, cfg in enumerate(configs)]

    print(f"Total configurations: {len(configs)}")
    for i, (cfg, name) in enumerate(zip(configs, config_names)):
        pt_str = "pretrained" if cfg.get("pretrained") else "scratch"
        print(f"  [{i}] {name}: hidden={cfg['hidden_dim']}, layers={cfg['num_mp_layers']}, "
              f"bp={cfg['bp_iters']}, loss={cfg['loss']}, {pt_str}")

    # Save grid for reproducibility
    grid_path = out_dir / "ablation_grid.json"
    with open(grid_path, "w") as f:
        json.dump({"grid": grid, "configs": configs, "names": config_names}, f, indent=2)
    print(f"Saved grid to {grid_path}")

    if not args.eval_only:
        # Training phase
        t_start = time.time()
        for i, (cfg, name) in enumerate(zip(configs, config_names)):
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(configs)}] Training: {name}")
            print(f"{'='*60}")

            run_dir = out_dir / name
            if (run_dir / "best_model.pt").exists():
                print(f"  Model already exists, skipping training")
                continue

            success = _run_training(
                name, cfg,
                train_data=args.train_data,
                pretrained_path=args.pretrained,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                out_dir=out_dir,
            )
            status = "OK" if success else "FAILED"
            print(f"  Training: {status}")

        train_elapsed = time.time() - t_start
        print(f"\nTraining phase complete: {train_elapsed:.0f}s ({train_elapsed/60:.1f} min)")

        # Evaluation phase
        print(f"\n{'='*60}")
        print("Evaluation phase")
        print(f"{'='*60}")

        for i, (cfg, name) in enumerate(zip(configs, config_names)):
            eval_dir = out_dir / name / "eval"
            if (eval_dir / "eval_results.json").exists():
                print(f"  [{i+1}/{len(configs)}] {name}: already evaluated, skipping")
                continue

            print(f"  [{i+1}/{len(configs)}] Evaluating: {name}")
            success = _run_evaluation(
                name,
                test_data=args.test_data,
                out_dir=out_dir,
                use_bposd=args.bposd,
                use_mwpm=args.mwpm,
            )
            status = "OK" if success else "FAILED"
            print(f"    Eval: {status}")

    # Collect results
    print(f"\n{'='*60}")
    print("Collecting results")
    print(f"{'='*60}")

    rows = _collect_results(configs, config_names, out_dir)

    # Save CSV
    csv_path = out_dir / "ablation_results.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved results to {csv_path}")

    # Print comparison table
    _print_table(rows)

    # Find best config
    valid_rows = [r for r in rows if r["improvement_pct"] != "N/A"]
    if valid_rows:
        best = max(valid_rows, key=lambda r: float(r["improvement_pct"]))
        print(f"\nBest config: {best['config']}")
        print(f"  Improvement: {best['improvement_pct']}%")
        print(f"  GNN-BP LER: {best['gnn_bp_ler']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
