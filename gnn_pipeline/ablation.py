"""Ablation study: sweep over hyperparameters and compare results.

Automates training + evaluation across multiple configurations,
collects results, and outputs a CSV comparison table.

Supports two JSON formats:
  1. Grid format: {"hidden_dim": [64, 128], "loss": ["mse", "focal"], ...}
     Creates Cartesian product of all parameter combinations.
  2. Named configs format: {"configs": [{"name": "...", "hidden_dim": 64, ...}, ...]}
     Each config is an explicit named configuration (for component ablation).

Usage:
    # Default ablation grid
    python -m gnn_pipeline.ablation --train_data "data/*.npz" --test_data data/test.npz --epochs 10 --out_dir runs/ablation

    # Component ablation via named configs
    python -m gnn_pipeline.ablation --grid_json configs/component_ablation.json --train_data "data/*.npz" --test_data data/test.npz --epochs 15 --out_dir runs/ablation

    # Evaluate-only (skip training, read existing results)
    python -m gnn_pipeline.ablation --train_data "data/*.npz" --test_data data/test.npz --out_dir runs/ablation --eval_only
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


def _build_configs_from_grid(grid: dict, use_pretrained: bool) -> list:
    """Generate all config combinations from a grid-format dict."""
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

    if use_pretrained:
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


def _build_configs_from_named(named_configs: list) -> tuple:
    """Parse named configs format (component ablation).

    Returns (configs, config_names) where each config has all needed keys
    with sensible defaults filled in.
    """
    configs = []
    names = []
    for cfg_raw in named_configs:
        cfg = {
            "hidden_dim": cfg_raw.get("hidden_dim", 64),
            "num_mp_layers": cfg_raw.get("num_mp_layers", 3),
            "bp_iters": cfg_raw.get("bp_iters", 10),
            "loss": cfg_raw.get("loss", "focal"),
            "correction_mode": cfg_raw.get("correction_mode", "additive"),
            "neural_bp": cfg_raw.get("neural_bp", False),
            "use_film": cfg_raw.get("use_film", False),
            "use_residual": cfg_raw.get("use_residual", False),
            "use_layer_norm": cfg_raw.get("use_layer_norm", False),
            "use_attention": cfg_raw.get("use_attention", False),
            "eval_only": cfg_raw.get("eval_only", False),
            "description": cfg_raw.get("description", ""),
        }
        configs.append(cfg)
        names.append(cfg_raw.get("name", f"config_{len(configs):02d}"))
    return configs, names


def _config_name(idx: int, cfg: dict) -> str:
    """Generate a short name for a grid-format config."""
    parts = [f"h{cfg['hidden_dim']}", f"l{cfg['num_mp_layers']}",
             f"bp{cfg['bp_iters']}", cfg["loss"]]
    cm = cfg.get("correction_mode", "additive")
    if cm != "additive":
        parts.append(cm[:3])
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
    """Run training for one config as a subprocess using train_unified."""
    run_dir = out_dir / config_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "gnn_pipeline.train_unified",
        "--mode", "code_capacity",
        "--in_glob", train_data,
        "--hidden_dim", str(cfg["hidden_dim"]),
        "--num_mp_layers", str(cfg["num_mp_layers"]),
        "--bp_iters", str(cfg["bp_iters"]),
        "--loss", cfg["loss"],
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(lr),
        "--correction_mode", cfg.get("correction_mode", "additive"),
        "--scheduler", "cosine",
        "--out_dir", str(run_dir),
    ]

    # Boolean flags
    if cfg.get("neural_bp"):
        cmd.append("--neural_bp")
    if cfg.get("use_film"):
        cmd.append("--use_film")
    if cfg.get("use_residual"):
        cmd.append("--use_residual")
    if cfg.get("use_layer_norm"):
        cmd.append("--use_layer_norm")
    if cfg.get("use_attention"):
        cmd.append("--use_attention")

    # AMP for GPU
    cmd.append("--amp")

    # Pretrained model
    if cfg.get("pretrained") and pretrained_path:
        cmd.extend(["--pretrained", pretrained_path])

    print(f"  CMD: {' '.join(cmd)}")

    log_path = run_dir / "train_log.txt"
    try:
        with open(log_path, "w") as log_f:
            result = subprocess.run(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                timeout=7200, cwd=str(out_dir.parent),
            )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 7200s")
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
    use_bplsd: bool = False,
    eval_only_bp: bool = False,
) -> bool:
    """Run evaluation for one config."""
    run_dir = out_dir / config_name
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "gnn_pipeline.evaluate",
        "--test_npz", test_data,
        "--out_dir", str(eval_dir),
    ]

    # eval_only configs (plain BP baseline) have no model
    if not eval_only_bp:
        model_path = run_dir / "best_checkpoint.pt"
        if not model_path.exists():
            model_path = run_dir / "best_model.pt"
        if not model_path.exists():
            print(f"  No model at {run_dir}, skipping evaluation")
            return False
        cmd.extend(["--gnn_model", str(model_path)])

    if use_bposd:
        cmd.append("--bposd")
    if use_mwpm:
        cmd.append("--mwpm")
    if use_bplsd:
        cmd.append("--bplsd")

    log_path = run_dir / "eval_log.txt"
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


def _collect_results(configs: list, config_names: list, out_dir: pathlib.Path) -> list:
    """Collect results from all completed runs."""
    rows = []
    for cfg, name in zip(configs, config_names):
        run_dir = out_dir / name

        # Read training results
        train_loss = None
        val_loss = None
        training_json = run_dir / "training_log.json"
        if training_json.exists():
            try:
                with open(training_json) as f:
                    log_data = json.load(f)
                history = log_data.get("history", [])
                if history:
                    last_epoch = history[-1]
                    train_loss = last_epoch.get("train", {}).get("loss")
                    val_loss = last_epoch.get("val", {}).get("loss")
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # Fallback: parse train_log.txt
        if train_loss is None or val_loss is None:
            train_log = run_dir / "train_log.txt"
            if train_log.exists():
                text = train_log.read_text(errors="replace")
                for line in reversed(text.splitlines()):
                    if "Train Loss:" in line and train_loss is None:
                        try:
                            train_loss = float(line.split("Train Loss:")[1].split("|")[0].strip())
                        except (ValueError, IndexError):
                            pass
                    if "Val Loss:" in line and val_loss is None:
                        try:
                            val_loss = float(line.split("Val Loss:")[1].split("|")[0].strip())
                        except (ValueError, IndexError):
                            pass

        # Read eval results
        eval_results_path = run_dir / "eval" / "eval_results.json"
        bp_ler = None
        gnn_bp_ler = None
        improvement = None
        bposd_ler = None
        bplsd_ler = None
        mwpm_ler = None

        if eval_results_path.exists():
            with open(eval_results_path) as f:
                eval_data = json.load(f)
            bp_ler = eval_data.get("bp", {}).get("ler")
            gnn_bp_ler = eval_data.get("gnn_bp", {}).get("ler")
            improvement = eval_data.get("improvement_pct")
            bposd_ler = eval_data.get("bposd", {}).get("ler")
            bplsd_ler = eval_data.get("bplsd", {}).get("ler")
            mwpm_ler = eval_data.get("mwpm", {}).get("ler")

        row = {
            "config": name,
            "description": cfg.get("description", ""),
            "hidden_dim": cfg.get("hidden_dim", ""),
            "num_mp_layers": cfg.get("num_mp_layers", ""),
            "bp_iters": cfg.get("bp_iters", ""),
            "loss": cfg.get("loss", ""),
            "correction_mode": cfg.get("correction_mode", "additive"),
            "neural_bp": cfg.get("neural_bp", False),
            "use_film": cfg.get("use_film", False),
            "train_loss": f"{train_loss:.6f}" if train_loss is not None else "N/A",
            "val_loss": f"{val_loss:.6f}" if val_loss is not None else "N/A",
            "bp_ler": f"{bp_ler:.6f}" if bp_ler is not None else "N/A",
            "gnn_bp_ler": f"{gnn_bp_ler:.6f}" if gnn_bp_ler is not None else "N/A",
            "improvement_pct": f"{improvement:.1f}" if improvement is not None else "N/A",
            "bposd_ler": f"{bposd_ler:.6f}" if bposd_ler is not None else "N/A",
            "bplsd_ler": f"{bplsd_ler:.6f}" if bplsd_ler is not None else "N/A",
            "mwpm_ler": f"{mwpm_ler:.6f}" if mwpm_ler is not None else "N/A",
        }
        rows.append(row)

    return rows


def _print_table(rows: list):
    """Print a formatted comparison table."""
    if not rows:
        print("No results to display.")
        return

    headers = ["config", "description", "bp_ler", "gnn_bp_ler", "improv%",
               "bposd_ler", "bplsd_ler"]
    col_widths = [max(len(h), 12) for h in headers]

    # Adjust widths based on data
    for row in rows:
        col_widths[0] = max(col_widths[0], len(str(row.get("config", ""))))
        col_widths[1] = max(col_widths[1], min(len(str(row.get("description", ""))), 40))

    header_line = " | ".join(h.center(w) for h, w in zip(headers, col_widths))
    sep_line = "-+-".join("-" * w for w in col_widths)
    print(f"\n{header_line}")
    print(sep_line)

    for row in rows:
        desc = str(row.get("description", ""))[:40]
        vals = [
            str(row["config"])[:col_widths[0]],
            desc.ljust(col_widths[1]),
            str(row.get("bp_ler", "N/A")).center(col_widths[2]),
            str(row.get("gnn_bp_ler", "N/A")).center(col_widths[3]),
            str(row.get("improvement_pct", "N/A")).center(col_widths[4]),
            str(row.get("bposd_ler", "N/A")).center(col_widths[5]),
            str(row.get("bplsd_ler", "N/A")).center(col_widths[6]),
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
    parser.add_argument("--bplsd", action="store_true",
                        help="Include BP-LSD in evaluation")
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
    is_named_format = False
    if args.grid_json:
        with open(args.grid_json) as f:
            grid_data = json.load(f)
        print(f"Loaded custom grid from {args.grid_json}")

        # Detect format: named configs vs grid
        if "configs" in grid_data and isinstance(grid_data["configs"], list):
            is_named_format = True
            configs, config_names = _build_configs_from_named(grid_data["configs"])
            print(f"Using named configs format ({len(configs)} configurations)")
        else:
            grid = grid_data
            configs = _build_configs_from_grid(grid, args.pretrained is not None)
            config_names = [_config_name(i, cfg) for i, cfg in enumerate(configs)]
    else:
        grid = DEFAULT_GRID
        print("Using default ablation grid")
        configs = _build_configs_from_grid(grid, args.pretrained is not None)
        config_names = [_config_name(i, cfg) for i, cfg in enumerate(configs)]

    print(f"Total configurations: {len(configs)}")
    for i, (cfg, name) in enumerate(zip(configs, config_names)):
        desc = cfg.get("description", "")
        skip = " [eval-only]" if cfg.get("eval_only") else ""
        print(f"  [{i}] {name}: {desc}{skip}")

    # Save grid for reproducibility
    grid_path = out_dir / "ablation_grid.json"
    with open(grid_path, "w") as f:
        json.dump({"configs": configs, "names": config_names}, f, indent=2)
    print(f"Saved grid to {grid_path}")

    if not args.eval_only:
        # Training phase
        t_start = time.time()
        for i, (cfg, name) in enumerate(zip(configs, config_names)):
            # Skip eval-only configs (e.g. plain BP baseline)
            if cfg.get("eval_only"):
                print(f"\n[{i+1}/{len(configs)}] {name}: eval-only, skipping training")
                continue

            print(f"\n[{i+1}/{len(configs)}] Training: {name}")
            if cfg.get("description"):
                print(f"  {cfg['description']}")

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
        print(f"\nEvaluation phase")

        for i, (cfg, name) in enumerate(zip(configs, config_names)):
            eval_dir = out_dir / name / "eval"
            if (eval_dir / "eval_results.json").exists():
                print(f"  [{i+1}/{len(configs)}] {name}: already evaluated, skipping")
                continue

            is_eval_only_bp = cfg.get("eval_only", False)
            print(f"  [{i+1}/{len(configs)}] Evaluating: {name}")
            success = _run_evaluation(
                name,
                test_data=args.test_data,
                out_dir=out_dir,
                use_bposd=args.bposd,
                use_mwpm=args.mwpm,
                use_bplsd=args.bplsd,
                eval_only_bp=is_eval_only_bp,
            )
            status = "OK" if success else "FAILED"
            print(f"    Eval: {status}")

    # Collect results
    print(f"\nCollecting results")

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

    # Find best config (by improvement)
    valid_rows = [r for r in rows if r.get("improvement_pct", "N/A") != "N/A"]
    if valid_rows:
        best = max(valid_rows, key=lambda r: float(r["improvement_pct"]))
        print(f"\nBest config: {best['config']}")
        print(f"  Improvement: {best['improvement_pct']}%")
        print(f"  GNN-BP LER: {best['gnn_bp_ler']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
