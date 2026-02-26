"""Training monitoring and logging infrastructure.

Provides:
  - TrainingMonitor: tracks metrics, exports JSON + CSV, gradient norms,
    timing breakdown, and early stopping state.
  - GradientMonitor: lightweight gradient norm tracker for detecting
    vanishing/exploding gradient issues.

Reusable across train_supervised.py, train_circuit.py, and train_unified.py.

Usage:
    monitor = TrainingMonitor(out_dir)
    grad_mon = GradientMonitor()

    for epoch in range(epochs):
        # ... training loop ...
        grad_mon.step(model)

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_conv": train_conv,
            "val_conv": val_conv,
            "lr": current_lr,
            "grad_norm": grad_mon.last_norm,
        }
        monitor.log_epoch(epoch, metrics)
        monitor.flush_csv()

    monitor.save(config_dict)
"""
from __future__ import annotations

import csv
import json
import pathlib
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class GradientMonitor:
    """Track gradient norms to detect vanishing/exploding gradients.

    Records total gradient L2 norm and per-module norms each step.
    """

    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        self.norms: List[float] = []
        self.last_norm: float = 0.0
        self.last_per_module: Dict[str, float] = {}

    def step(self, model: nn.Module) -> float:
        """Compute and record gradient norms.

        Call this AFTER loss.backward() and BEFORE optimizer.step().

        Returns:
            Total gradient L2 norm across all parameters.
        """
        total_norm_sq = 0.0
        per_module = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                pnorm = param.grad.data.norm(2).item() ** 2
                total_norm_sq += pnorm
                # Group by top-level module name
                module_name = name.split(".")[0]
                per_module[module_name] = per_module.get(module_name, 0.0) + pnorm

        total_norm = total_norm_sq ** 0.5
        self.last_norm = total_norm
        self.last_per_module = {k: v ** 0.5 for k, v in per_module.items()}

        self.norms.append(total_norm)
        if len(self.norms) > self.max_history:
            self.norms = self.norms[-self.max_history:]

        return total_norm

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics of gradient norms."""
        if not self.norms:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "last": 0.0}
        return {
            "mean": sum(self.norms) / len(self.norms),
            "max": max(self.norms),
            "min": min(self.norms),
            "last": self.norms[-1],
            "num_steps": len(self.norms),
        }


class EpochTimer:
    """Track timing breakdown within a single epoch."""

    def __init__(self):
        self._start: float = 0.0
        self._sections: Dict[str, float] = {}
        self._current_section: Optional[str] = None
        self._section_start: float = 0.0

    def start(self):
        """Mark the beginning of an epoch."""
        self._start = time.time()
        self._sections = {}
        self._current_section = None

    def begin_section(self, name: str):
        """Start timing a named section (e.g. 'forward', 'backward', 'eval')."""
        now = time.time()
        if self._current_section is not None:
            elapsed = now - self._section_start
            self._sections[self._current_section] = (
                self._sections.get(self._current_section, 0.0) + elapsed
            )
        self._current_section = name
        self._section_start = now

    def end_section(self):
        """End the current section."""
        if self._current_section is not None:
            elapsed = time.time() - self._section_start
            self._sections[self._current_section] = (
                self._sections.get(self._current_section, 0.0) + elapsed
            )
            self._current_section = None

    def stop(self) -> Dict[str, float]:
        """Stop the epoch timer and return timing breakdown.

        Returns:
            Dict with 'total_s' and per-section times like 'train_s', 'eval_s'.
        """
        self.end_section()
        total = time.time() - self._start
        result = {"total_s": round(total, 3)}
        for name, elapsed in self._sections.items():
            result[f"{name}_s"] = round(elapsed, 3)
        return result


class TrainingMonitor:
    """Training monitor with JSON + CSV + TensorBoard + W&B logging.

    Features:
      - Per-epoch metric logging with timestamps
      - Best metric tracking (minimize loss, maximize accuracy)
      - CSV export for easy plotting
      - Optional TensorBoard logging (``use_tensorboard=True``)
      - Optional Weights & Biases logging (``use_wandb=True``)
      - Configuration saving
    """

    def __init__(
        self,
        out_dir: pathlib.Path,
        csv_filename: str = "metrics.csv",
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[dict] = None,
    ):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[dict] = []
        self.best_metrics: Dict[str, float] = {}
        self.csv_path = out_dir / csv_filename
        self._csv_initialized = False
        self._csv_columns: List[str] = []

        # --- TensorBoard ---
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=str(out_dir / "tb_logs"))
            except ImportError:
                print("Warning: tensorboard not installed, skipping TensorBoard logging. "
                      "Install with: pip install tensorboard")

        # --- Weights & Biases ---
        self._wandb_run = None
        if use_wandb:
            try:
                import wandb as _wandb
                self._wandb_run = _wandb.init(
                    project=wandb_project or "qldpc-gnn-bp",
                    config=wandb_config or {},
                    dir=str(out_dir),
                    reinit=True,
                )
            except ImportError:
                print("Warning: wandb not installed, skipping W&B logging. "
                      "Install with: pip install wandb")

    def log_epoch(self, epoch: int, metrics: dict):
        """Log metrics for one epoch.

        Args:
            epoch: epoch number (1-indexed)
            metrics: dict of metric_name -> value (float or int)
        """
        # Coerce numpy/torch scalars to plain Python types for JSON safety
        safe_metrics = {}
        for k, v in metrics.items():
            if hasattr(v, "item"):          # torch.Tensor or np scalar
                safe_metrics[k] = v.item()
            elif isinstance(v, float) and v != v:  # NaN check
                safe_metrics[k] = None
            else:
                safe_metrics[k] = v
        record = {"epoch": epoch, **safe_metrics, "timestamp": time.time()}
        self.history.append(record)

        # Update best metrics
        for key, val in metrics.items():
            if not isinstance(val, (int, float)):
                continue
            if any(key.startswith(p) for p in ("val_loss", "test_loss")):
                # Minimize loss
                if key not in self.best_metrics or val < self.best_metrics[key]:
                    self.best_metrics[key] = val
            elif any(key.endswith(s) for s in ("accuracy", "acc", "conv")):
                # Maximize accuracy/convergence
                if key not in self.best_metrics or val > self.best_metrics[key]:
                    self.best_metrics[key] = val

        # TensorBoard logging
        if self._tb_writer is not None:
            for key, val in safe_metrics.items():
                if isinstance(val, (int, float)) and val is not None:
                    self._tb_writer.add_scalar(key, val, epoch)
            self._tb_writer.flush()

        # W&B logging
        if self._wandb_run is not None:
            try:
                import wandb as _wandb
                _wandb.log({"epoch": epoch, **safe_metrics})
            except Exception:
                pass  # silently skip on logging failure

    def flush_csv(self):
        """Append the latest epoch to the CSV file.

        Call after log_epoch() each epoch for incremental writes.
        """
        if not self.history:
            return

        latest = self.history[-1]

        if not self._csv_initialized:
            # Write header on first call
            self._csv_columns = sorted(latest.keys())
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_columns)
                writer.writeheader()
                writer.writerow({k: latest.get(k, "") for k in self._csv_columns})
            self._csv_initialized = True
        else:
            # Append row — handle new columns by extending
            new_cols = [k for k in latest.keys() if k not in self._csv_columns]
            if new_cols:
                self._csv_columns.extend(new_cols)
                # Rewrite entire CSV with extended columns
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self._csv_columns)
                    writer.writeheader()
                    for row in self.history:
                        writer.writerow({k: row.get(k, "") for k in self._csv_columns})
            else:
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self._csv_columns)
                    writer.writerow({k: latest.get(k, "") for k in self._csv_columns})

    def save(self, config: dict):
        """Save full training log as JSON and close external loggers."""
        log = {
            "config": {k: v for k, v in config.items()
                       if isinstance(v, (str, int, float, bool, list, type(None)))},
            "history": self.history,
            "best_metrics": self.best_metrics,
        }
        log_path = self.out_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        self.close()

    def close(self):
        """Close TensorBoard writer and finish W&B run."""
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
        if self._wandb_run is not None:
            try:
                import wandb as _wandb
                _wandb.finish()
            except Exception:
                pass
            self._wandb_run = None

    def print_summary(self):
        """Print a summary of training metrics."""
        if not self.history:
            return
        last = self.history[-1]
        print(f"\n--- Training Summary ({len(self.history)} epochs) ---")
        for key, val in self.best_metrics.items():
            print(f"  Best {key}: {val:.6f}")
        if "train_loss" in last:
            print(f"  Final train_loss: {last['train_loss']:.6f}")
        if "val_loss" in last:
            print(f"  Final val_loss: {last['val_loss']:.6f}")
