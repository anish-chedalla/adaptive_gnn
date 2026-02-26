#!/usr/bin/env python
"""
Publication-quality figure generation for the QLDPC GNN-BP pipeline.

Generates 6 figures following Nature / Physical Review conventions:
  Figure 1: LER vs code size (the "scaling story" — headline figure)
  Figure 2: Decoder comparison bar chart with confidence intervals
  Figure 3: LER vs physical error rate (threshold-style curves)
  Figure 4: Static vs drift performance comparison
  Figure 5: Training dynamics (loss curves, convergence, alpha evolution)
  Figure 6: Relative improvement heatmap across all conditions

Usage:
  python -m gnn_pipeline.make_figures --results_dir runs --out_dir figures
  python -m gnn_pipeline.make_figures --results_dir runs --out_dir figures --format pdf
  python -m gnn_pipeline.make_figures --results_dir runs --out_dir figures --fig 1 3 6

Style reference: Nature (2024), Bravyi et al., AlphaQubit, Astra.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Matplotlib setup — must come before any pyplot import
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette (Nature standard)
# ---------------------------------------------------------------------------
COLORS = {
    "blue":        "#0072B2",
    "vermillion":  "#D55E00",
    "green":       "#009E73",
    "orange":      "#E69F00",
    "sky_blue":    "#56B4E9",
    "purple":      "#CC79A7",
    "yellow":      "#F0E442",
    "black":       "#000000",
    "grey":        "#999999",
}

# Decoder -> (color, marker, linestyle, zorder) mapping
DECODER_STYLE = {
    "Interleaved GNN-BP": (COLORS["blue"],       "o", "-",  10),
    "GNN-BP":             (COLORS["sky_blue"],    "s", "-",   9),
    "GNN + BP-LSD":       (COLORS["green"],       "D", "-",   8),
    "GNN + BP-OSD":       (COLORS["green"],       "^", "--",  7),
    "BP-OSD":             (COLORS["orange"],      "v", "--",  6),
    "BP-LSD":             (COLORS["orange"],      "p", "-.",  5),
    "BP":                 (COLORS["black"],       "x", "--",  4),
    "Oracle BP":          (COLORS["grey"],        "+", ":",   3),
}

_NEW_BLACK = "#373737"

# ---------------------------------------------------------------------------
# Global rcParams — Nature / PRA style
# ---------------------------------------------------------------------------
def _apply_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update({
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.linewidth": 0.6,
        # Ticks
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 3,
        "xtick.major.width": 0.6,
        "xtick.minor.size": 1.5,
        "xtick.minor.width": 0.4,
        "ytick.major.size": 3,
        "ytick.major.width": 0.6,
        "ytick.minor.size": 1.5,
        "ytick.minor.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        # Legend
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 5,
        # Colors
        "text.color": _NEW_BLACK,
        "axes.labelcolor": _NEW_BLACK,
        "xtick.color": _NEW_BLACK,
        "ytick.color": _NEW_BLACK,
        "axes.edgecolor": _NEW_BLACK,
        # Output
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,       # TrueType — editable in Illustrator
        "ps.fonttype": 42,
    })


# ===================================================================
#                       DATA LOADING
# ===================================================================

# Curated run directories from our experiments (default filter)
CURATED_EVAL_DIRS = {
    # [[72,12,6]] PoC — FiLM model
    "poc_eval_film_p05", "poc_eval_film_p08", "poc_eval_film_p08_sine",
    # [[72,12,6]] PoC — Baseline (no FiLM)
    "poc_eval_baseline_p08", "poc_eval_baseline_p08_sine",
    # [[144,12,12]] — FiLM model
    "big144_eval_p04", "big144_eval_p06", "big144_eval_p06_sine",
    # [[288,12,18]] — FiLM model
    "big288_eval_p03", "big288_eval_p04", "big288_eval_p04_sine",
}


def load_eval_results(results_dir: Path,
                      curated: bool = True) -> List[Dict[str, Any]]:
    """Recursively find and load eval_results.json files.

    Args:
        results_dir: root directory containing run subdirectories
        curated: if True, only load from CURATED_EVAL_DIRS (default)
    """
    results = []
    for p in sorted(results_dir.rglob("eval_results.json")):
        dirname = p.parent.name
        if curated and dirname not in CURATED_EVAL_DIRS:
            continue
        with open(p) as f:
            data = json.load(f)
        # Skip circuit-level results (different schema: 'dem' instead of 'code')
        if "code" not in data:
            continue
        data["_path"] = dirname
        results.append(data)
    return results


def load_training_log(run_dir: Path) -> Optional[Dict]:
    """Load a training_log.json from a run directory."""
    log_path = run_dir / "training_log.json"
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return None


def load_metrics_csv(run_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load metrics.csv from a run directory into column arrays."""
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        return None
    lines = csv_path.read_text().strip().split("\n")
    if len(lines) < 2:
        return None
    headers = [h.strip() for h in lines[0].split(",")]
    data = {h: [] for h in headers}
    for line in lines[1:]:
        vals = line.split(",")
        for h, v in zip(headers, vals):
            try:
                data[h].append(float(v))
            except ValueError:
                data[h].append(float("nan"))
    return {h: np.array(v) for h, v in data.items()}


def _extract_decoder_data(result: Dict, decoder_key: str) -> Optional[Dict]:
    """Extract LER + CI from a result dict for a given decoder key."""
    mapping = {
        "BP":                 "bp",
        "Oracle BP":          "oracle_bp",
        "GNN-BP":             "gnn_bp",
        "Interleaved GNN-BP": "interleaved_gnn_bp",
        "BP-OSD":             "bposd",
        "BP-LSD":             "bplsd",
        "GNN + BP-LSD":       "gnn_bplsd",
        "GNN + BP-OSD":       "gnn_bposd",
    }
    key = mapping.get(decoder_key)
    if key is None or key not in result:
        return None
    d = result[key]
    return {
        "ler": d["ler"],
        "ci_low": d.get("ler_ci_low", d["ler"]),
        "ci_high": d.get("ler_ci_high", d["ler"]),
        "errors": d.get("errors", 0),
        "shots": result.get("test_shots", 0),
        "convergence": d.get("convergence_rate"),
    }


def _get_code_label(n: int) -> str:
    """Map n -> human-readable code label."""
    labels = {72: "[[72,12,6]]", 144: "[[144,12,12]]", 288: "[[288,12,18]]"}
    return labels.get(n, f"[[{n},?,?]]")


def _panel_label(ax, label, x=-0.12, y=1.06):
    """Add bold panel label (a, b, c, ...) to axes — Nature style."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")


# ===================================================================
#          FIGURE 1 — LER vs Code Size  (The Headline)
# ===================================================================

def figure1_scaling(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """
    LER vs code size showing exponential scaling of GNN improvement.
    This is the single most important figure.
    """
    # Collect: for each code size, find the best "comparable" test
    # Use: [[72,12,6]] p=0.05 static, [[144,12,12]] p=0.04 static, [[288,12,18]] p=0.04 static
    target_tests = {
        72:  {"p": 0.05, "drift": False},
        144: {"p": 0.04, "drift": False},
        288: {"p": 0.04, "drift": False},
    }

    decoders_to_plot = [
        "BP", "GNN-BP", "Interleaved GNN-BP",
        "BP-OSD", "GNN + BP-LSD",
    ]

    # Gather data per code size
    code_data = {}  # n -> {decoder_name: {ler, ci_low, ci_high}}
    for r in results:
        n = r["code"]["n"]
        p = r["noise"]["p"]
        has_drift = r.get("_path", "").find("sine") >= 0 or \
                    r.get("_path", "").find("ou") >= 0 or \
                    r.get("_path", "").find("drift") >= 0
        if n not in target_tests:
            continue
        target = target_tests[n]
        if abs(p - target["p"]) > 0.001:
            continue
        if has_drift != target["drift"]:
            continue
        code_data[n] = {}
        for dec in decoders_to_plot:
            d = _extract_decoder_data(r, dec)
            if d is not None:
                code_data[n][dec] = d

    if len(code_data) < 2:
        print("  [Figure 1] Not enough code sizes found, skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # --- Panel (a): LER vs code size ---
    code_sizes = sorted(code_data.keys())
    x_pos = np.arange(len(code_sizes))

    for dec in decoders_to_plot:
        color, marker, ls, zorder = DECODER_STYLE.get(
            dec, (COLORS["grey"], ".", "-", 1))
        lers, ci_lo, ci_hi = [], [], []
        valid_x = []
        for i, n in enumerate(code_sizes):
            if dec in code_data.get(n, {}):
                d = code_data[n][dec]
                ler = max(d["ler"], 1e-4)  # floor for log scale
                lers.append(ler)
                ci_lo.append(max(ler - d["ci_low"], 0))
                ci_hi.append(max(d["ci_high"] - ler, 0))
                valid_x.append(i)
        if not lers:
            continue
        ax1.errorbar(valid_x, lers, yerr=[ci_lo, ci_hi],
                     fmt=marker + ls, color=color, label=dec,
                     markersize=7, capsize=3, capthick=0.8,
                     linewidth=1.4, zorder=zorder, markeredgewidth=0.6,
                     markeredgecolor="white" if marker not in ("x", "+") else color)

    ax1.set_yscale("log")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([_get_code_label(n) for n in code_sizes], fontsize=8)
    ax1.set_xlabel("Code")
    ax1.set_ylabel("Logical Error Rate (LER)")
    ax1.set_ylim(bottom=3e-4)
    ax1.legend(loc="upper left", fontsize=6.5, ncol=1)
    ax1.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    _panel_label(ax1, "a")

    # --- Panel (b): Relative improvement vs code size ---
    gnn_decoders = ["GNN-BP", "Interleaved GNN-BP", "GNN + BP-LSD"]
    bar_width = 0.22
    offsets = np.arange(len(gnn_decoders)) - (len(gnn_decoders) - 1) / 2
    offsets = offsets * bar_width

    for j, dec in enumerate(gnn_decoders):
        color = DECODER_STYLE[dec][0]
        improvements = []
        valid_x = []
        for i, n in enumerate(code_sizes):
            if dec in code_data.get(n, {}) and "BP" in code_data.get(n, {}):
                bp_ler = code_data[n]["BP"]["ler"]
                gnn_ler = code_data[n][dec]["ler"]
                if bp_ler > 0:
                    imp = (bp_ler - gnn_ler) / bp_ler * 100
                    improvements.append(imp)
                    valid_x.append(i)
        if improvements:
            bars = ax2.bar(np.array(valid_x) + offsets[j], improvements,
                          bar_width * 0.9, color=color, alpha=0.85,
                          label=dec, edgecolor="white", linewidth=0.5)
            # Value labels on bars
            for bar, imp in zip(bars, improvements):
                va = "bottom" if imp >= 0 else "top"
                y = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, y + 1,
                        f"{imp:.0f}%", ha="center", va=va,
                        fontsize=6.5, fontweight="bold", color=color)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([_get_code_label(n) for n in code_sizes], fontsize=8)
    ax2.set_xlabel("Code")
    ax2.set_ylabel("Improvement over BP (%)")
    ax2.axhline(0, color=_NEW_BLACK, linewidth=0.5, linestyle="-")
    ax2.legend(loc="upper left", fontsize=6, ncol=1)
    _panel_label(ax2, "b")

    fig.suptitle("")  # No suptitle — Nature style uses caption below
    fig.tight_layout(w_pad=2.5)
    out_path = out_dir / f"fig1_scaling.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#      FIGURE 2 — Decoder Comparison Bar Chart (per code size)
# ===================================================================

def figure2_decoder_bars(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """
    Grouped bar chart: LER for each decoder at each code size.
    Shows all 8 decoders with confidence intervals.
    """
    target_tests = {
        72:  {"p": 0.05},
        144: {"p": 0.04},
        288: {"p": 0.04},
    }

    all_decoders = [
        "BP", "Oracle BP", "GNN-BP", "Interleaved GNN-BP",
        "BP-OSD", "BP-LSD", "GNN + BP-LSD", "GNN + BP-OSD",
    ]

    code_data = {}
    for r in results:
        n = r["code"]["n"]
        p = r["noise"]["p"]
        if n not in target_tests or abs(p - target_tests[n]["p"]) > 0.001:
            continue
        path = r.get("_path", "")
        if any(k in path for k in ["sine", "ou", "rtn", "drift"]):
            continue
        code_data[n] = {}
        for dec in all_decoders:
            d = _extract_decoder_data(r, dec)
            if d is not None:
                code_data[n][dec] = d

    if not code_data:
        print("  [Figure 2] No matching results found, skipping.")
        return

    code_sizes = sorted(code_data.keys())
    n_codes = len(code_sizes)

    fig, axes = plt.subplots(1, n_codes, figsize=(3.0 * n_codes, 3.5), sharey=False)
    if n_codes == 1:
        axes = [axes]

    for idx, n in enumerate(code_sizes):
        ax = axes[idx]
        decoders_present = [d for d in all_decoders if d in code_data.get(n, {})]
        x = np.arange(len(decoders_present))
        lers = []
        ci_lo = []
        ci_hi = []
        colors = []
        for dec in decoders_present:
            d = code_data[n][dec]
            lers.append(d["ler"])
            ci_lo.append(max(d["ler"] - d["ci_low"], 0))
            ci_hi.append(max(d["ci_high"] - d["ler"], 0))
            colors.append(DECODER_STYLE.get(dec, (COLORS["grey"],))[0])

        bars = ax.bar(x, lers, color=colors, alpha=0.85,
                     edgecolor="white", linewidth=0.5, width=0.7)
        ax.errorbar(x, lers, yerr=[ci_lo, ci_hi], fmt="none",
                   ecolor=_NEW_BLACK, capsize=2.5, capthick=0.6, linewidth=0.6)

        # Highlight best
        if lers:
            best_idx = int(np.argmin(lers))
            bars[best_idx].set_edgecolor(COLORS["blue"])
            bars[best_idx].set_linewidth(1.5)

        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("Interleaved ", "Int.\n")
                           .replace("GNN + ", "GNN+\n")
                           .replace("Oracle ", "Orc.\n")
                           for d in decoders_present],
                          fontsize=5.5, rotation=45, ha="right")
        ax.set_ylabel("LER" if idx == 0 else "")
        p_val = target_tests[n]["p"]
        ax.set_title(f"{_get_code_label(n)}, p={p_val}", fontsize=9)
        ax.set_ylim(bottom=0)
        _panel_label(ax, chr(ord("a") + idx))

    fig.tight_layout(w_pad=1.5)
    out_path = out_dir / f"fig2_decoder_bars.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#   FIGURE 3 — LER vs Physical Error Rate (Threshold-style curves)
# ===================================================================

def figure3_threshold_curves(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """
    LER vs p for each code size — classic QEC threshold plot.
    Separate panel per code, all decoders overlaid.
    """
    # Group results by code size (static only)
    by_code: Dict[int, List[Dict]] = {}
    for r in results:
        n = r["code"]["n"]
        path = r.get("_path", "")
        if any(k in path for k in ["sine", "ou", "rtn", "drift"]):
            continue
        by_code.setdefault(n, []).append(r)

    if not by_code:
        print("  [Figure 3] No static results found, skipping.")
        return

    code_sizes = sorted(by_code.keys())
    n_panels = len(code_sizes)

    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.3), sharey=False)
    if n_panels == 1:
        axes = [axes]

    decoders_to_plot = [
        "BP", "GNN-BP", "Interleaved GNN-BP", "BP-OSD", "GNN + BP-LSD",
    ]

    for idx, n in enumerate(code_sizes):
        ax = axes[idx]
        rs = sorted(by_code[n], key=lambda x: x["noise"]["p"])

        for dec in decoders_to_plot:
            color, marker, ls, zorder = DECODER_STYLE.get(
                dec, (COLORS["grey"], ".", "-", 1))
            ps, lers, ci_lo, ci_hi = [], [], [], []
            for r in rs:
                d = _extract_decoder_data(r, dec)
                if d is not None:
                    ler = max(d["ler"], 2e-4)
                    ps.append(r["noise"]["p"])
                    lers.append(ler)
                    ci_lo.append(max(ler - d["ci_low"], 0))
                    ci_hi.append(max(d["ci_high"] - ler, 0))
            if not ps:
                continue
            ax.errorbar(ps, lers, yerr=[ci_lo, ci_hi],
                       fmt=marker + ls, color=color, label=dec,
                       markersize=5, capsize=2, capthick=0.5,
                       linewidth=1.2, zorder=zorder, markeredgewidth=0.4,
                       markeredgecolor="white" if marker not in ("x", "+") else color)

        ax.set_yscale("log")
        ax.set_xlabel("Physical error rate ($p$)")
        ax.set_ylabel("Logical Error Rate" if idx == 0 else "")
        ax.set_title(f"{_get_code_label(n)}", fontsize=9)
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
        ax.yaxis.set_minor_formatter(NullFormatter())
        if idx == n_panels - 1:
            ax.legend(loc="lower right", fontsize=5.5)
        _panel_label(ax, chr(ord("a") + idx))

    fig.tight_layout(w_pad=2.0)
    out_path = out_dir / f"fig3_threshold_curves.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#      FIGURE 4 — Static vs Drift Comparison
# ===================================================================

def figure4_drift_comparison(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """
    Side-by-side: static vs drift for each code size.
    Shows how decoders degrade under drift and whether GNN/FiLM helps.
    """
    # Find pairs: same code + same p, one static and one drift
    pairs: Dict[Tuple[int, float], Dict[str, Dict]] = {}
    for r in results:
        n = r["code"]["n"]
        p = r["noise"]["p"]
        path = r.get("_path", "")
        is_drift = any(k in path for k in ["sine", "ou", "rtn", "drift"])
        key = (n, p)
        if key not in pairs:
            pairs[key] = {}
        label = "drift" if is_drift else "static"
        if label not in pairs[key]:
            pairs[key][label] = r

    # Filter to pairs that have BOTH static and drift
    valid_pairs = {k: v for k, v in pairs.items() if "static" in v and "drift" in v}

    if not valid_pairs:
        print("  [Figure 4] No static/drift pairs found, skipping.")
        return

    decoders_to_plot = [
        "BP", "GNN-BP", "Interleaved GNN-BP", "BP-OSD",
    ]

    n_panels = len(valid_pairs)
    fig, axes = plt.subplots(1, min(n_panels, 4), figsize=(3.3 * min(n_panels, 4), 3.5))
    if min(n_panels, 4) == 1:
        axes = [axes]

    for idx, ((n, p), pair) in enumerate(sorted(valid_pairs.items())):
        if idx >= 4:
            break
        ax = axes[idx]
        x = np.arange(len(decoders_to_plot))
        width = 0.35

        for cond_idx, (cond, label, offset, alpha) in enumerate([
            ("static", "Static", -width / 2, 0.9),
            ("drift", "Drift", width / 2, 0.55),
        ]):
            r = pair[cond]
            lers, colors = [], []
            for dec in decoders_to_plot:
                d = _extract_decoder_data(r, dec)
                if d is not None:
                    lers.append(d["ler"])
                else:
                    lers.append(0)
                colors.append(DECODER_STYLE.get(dec, (COLORS["grey"],))[0])

            bars = ax.bar(x + offset, lers, width, alpha=alpha,
                         color=colors, edgecolor="white", linewidth=0.5,
                         label=label if idx == 0 else "")
            # Add hatching for drift bars
            if cond == "drift":
                for bar in bars:
                    bar.set_hatch("///")

        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("Interleaved ", "Int.\n")
                           for d in decoders_to_plot],
                          fontsize=6, rotation=30, ha="right")
        ax.set_ylabel("LER" if idx == 0 else "")
        ax.set_title(f"{_get_code_label(n)}, p={p}", fontsize=9)
        _panel_label(ax, chr(ord("a") + idx))

    # Shared legend
    handles = [
        mpatches.Patch(facecolor=COLORS["grey"], alpha=0.9, label="Static"),
        mpatches.Patch(facecolor=COLORS["grey"], alpha=0.55, hatch="///", label="Drift"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=7)

    fig.tight_layout(w_pad=1.5)
    out_path = out_dir / f"fig4_drift_comparison.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#    FIGURE 5 — Training Dynamics (multi-panel)
# ===================================================================

def figure5_training_dynamics(results_dir: Path, out_dir: Path, fmt: str = "pdf"):
    """
    Training curves from CSV metrics: loss, convergence, alpha, bit accuracy.
    One row per training run (e.g., poc_film, big144_film, big288_film).
    """
    run_dirs = [
        ("poc_film", "[[72,12,6]] FiLM"),
        ("big144_film", "[[144,12,12]] FiLM"),
        ("big288_film", "[[288,12,18]] FiLM"),
    ]

    valid_runs = []
    for dirname, label in run_dirs:
        d = results_dir / dirname
        csv = load_metrics_csv(d)
        if csv is not None:
            valid_runs.append((csv, label))

    if not valid_runs:
        print("  [Figure 5] No training metrics found, skipping.")
        return

    n_runs = len(valid_runs)
    fig, axes = plt.subplots(n_runs, 4, figsize=(7.2, 2.2 * n_runs))
    if n_runs == 1:
        axes = axes.reshape(1, -1)

    for row, (csv, label) in enumerate(valid_runs):
        epochs = csv.get("epoch", np.arange(len(csv.get("train_loss", []))) + 1)

        # (a) Loss curves
        ax = axes[row, 0]
        if "train_loss" in csv:
            ax.plot(epochs, csv["train_loss"], "-", color=COLORS["blue"],
                   linewidth=1.0, label="Train")
        if "val_loss" in csv:
            ax.plot(epochs, csv["val_loss"], "--", color=COLORS["vermillion"],
                   linewidth=1.0, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=6)
        if row == 0:
            ax.set_title("Loss", fontsize=8)
        ax.text(0.98, 0.95, label, transform=ax.transAxes,
                fontsize=6.5, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor=COLORS["grey"], alpha=0.8))

        # (b) Z convergence
        ax = axes[row, 1]
        if "conv_z" in csv:
            ax.plot(epochs, csv["conv_z"], "-", color=COLORS["green"], linewidth=1.0)
            ax.set_ylim(min(csv["conv_z"]) - 0.02, 1.01)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BP Conv. Rate")
        if row == 0:
            ax.set_title("Z Convergence", fontsize=8)

        # (c) Bit accuracy
        ax = axes[row, 2]
        if "bit_acc_z" in csv:
            ax.plot(epochs, csv["bit_acc_z"], "-", color=COLORS["orange"], linewidth=1.0)
            ymin = min(csv["bit_acc_z"])
            ax.set_ylim(max(ymin - 0.005, 0.95), 1.001)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Bit Accuracy (Z)")
        if row == 0:
            ax.set_title("Bit Accuracy", fontsize=8)

        # (d) Learning rate
        ax = axes[row, 3]
        if "lr" in csv:
            ax.plot(epochs, csv["lr"], "-", color=COLORS["purple"], linewidth=1.0)
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        if row == 0:
            ax.set_title("LR Schedule", fontsize=8)

    # Panel labels
    for row in range(n_runs):
        for col in range(4):
            label_char = chr(ord("a") + row * 4 + col)
            _panel_label(axes[row, col], label_char, x=-0.18, y=1.1)

    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    out_path = out_dir / f"fig5_training_dynamics.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#     FIGURE 6 — Relative Improvement Heatmap
# ===================================================================

def figure6_heatmap(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """
    Heatmap: (decoder x condition) showing % improvement over BP.
    Rows = decoders, columns = test conditions (code+p+drift).
    """
    decoders = [
        "GNN-BP", "Interleaved GNN-BP",
        "BP-OSD", "BP-LSD", "GNN + BP-LSD", "GNN + BP-OSD",
    ]

    # Short code labels for x-axis
    _short_code = {72: "72", 144: "144", 288: "288"}

    # Build conditions from all results
    conditions = []
    cond_results = []
    for r in results:
        n = r["code"]["n"]
        p = r["noise"]["p"]
        path = r.get("_path", "")
        # Skip baseline (no-film) runs to avoid duplicates
        if "baseline" in path:
            continue
        drift_label = ""
        if "sine" in path:
            drift_label = "\n+sine"
        elif "ou" in path:
            drift_label = "\n+OU"
        elif "rtn" in path:
            drift_label = "\n+RTN"
        code_short = _short_code.get(n, str(n))
        cond_label = f"n={code_short}\np={p}{drift_label}"
        conditions.append(cond_label)
        cond_results.append(r)

    if not conditions:
        print("  [Figure 6] No results for heatmap, skipping.")
        return

    # Build improvement matrix
    n_dec = len(decoders)
    n_cond = len(conditions)
    matrix = np.full((n_dec, n_cond), np.nan)

    for j, r in enumerate(cond_results):
        bp_data = _extract_decoder_data(r, "BP")
        if bp_data is None or bp_data["ler"] == 0:
            continue
        bp_ler = bp_data["ler"]
        for i, dec in enumerate(decoders):
            d = _extract_decoder_data(r, dec)
            if d is not None:
                matrix[i, j] = (bp_ler - d["ler"]) / bp_ler * 100

    # Sort columns by code size then p
    col_order = list(range(n_cond))
    col_order.sort(key=lambda j: (cond_results[j]["code"]["n"],
                                   cond_results[j]["noise"]["p"]))
    matrix = matrix[:, col_order]
    conditions = [conditions[j] for j in col_order]

    # Sizing
    cell_w = 0.65
    cell_h = 0.45
    fig_width = max(5.0, cell_w * n_cond + 3.0)
    fig_height = max(3.0, cell_h * n_dec + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Custom diverging colormap centered at 0
    from matplotlib.colors import TwoSlopeNorm
    vmin_val = np.nanmin(matrix) if not np.all(np.isnan(matrix)) else -10
    vmax_val = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 100
    abs_max = max(abs(vmin_val), abs(vmax_val), 1)

    norm = TwoSlopeNorm(vmin=-abs_max * 0.2, vcenter=0, vmax=abs_max)
    cmap = plt.cm.RdYlBu

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")

    # Annotate cells
    for i in range(n_dec):
        for j in range(n_cond):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "-", ha="center", va="center",
                       fontsize=7, color=COLORS["grey"])
                continue
            color = "white" if abs(val) > abs_max * 0.65 else _NEW_BLACK
            fontweight = "bold" if abs(val) >= 10 else "normal"
            text = f"{val:+.0f}%" if abs(val) >= 1 else f"{val:+.1f}%"
            ax.text(j, i, text, ha="center", va="center",
                   fontsize=7, color=color, fontweight=fontweight)

    # Grid lines between cells
    for i in range(n_dec + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(n_cond + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

    # Separator lines between code sizes
    prev_n = None
    for j_idx, j_orig in enumerate(col_order):
        cur_n = cond_results[j_orig]["code"]["n"]
        if prev_n is not None and cur_n != prev_n:
            ax.axvline(j_idx - 0.5, color=_NEW_BLACK, linewidth=2.0)
        prev_n = cur_n

    ax.set_xticks(np.arange(n_cond))
    ax.set_xticklabels(conditions, fontsize=6.5, ha="center")
    ax.set_yticks(np.arange(n_dec))
    ax.set_yticklabels(decoders, fontsize=8)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, pad=4)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, aspect=25)
    cbar.set_label("Improvement over BP (%)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    out_path = out_dir / f"fig6_heatmap.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#           MAIN — Generate all figures
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for QLDPC GNN-BP pipeline.")
    parser.add_argument("--results_dir", type=str, default="runs",
                        help="Directory containing evaluation runs")
    parser.add_argument("--out_dir", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "png", "svg", "eps"],
                        help="Output format (default: pdf)")
    parser.add_argument("--fig", type=int, nargs="*", default=None,
                        help="Which figures to generate (1-6). Default: all")
    parser.add_argument("--all", action="store_true",
                        help="Load ALL eval results (default: curated set only)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    _apply_style()

    # Load evaluation results
    curated = not args.all
    mode_str = "curated" if curated else "all"
    print(f"Loading {mode_str} results from {results_dir}...")
    all_results = load_eval_results(results_dir, curated=curated)
    print(f"  Found {len(all_results)} evaluation result files")

    figs_to_make = args.fig if args.fig else [1, 2, 3, 4, 5, 6]

    if 1 in figs_to_make:
        print("Generating Figure 1: LER vs Code Size (Scaling)...")
        figure1_scaling(all_results, out_dir, fmt)

    if 2 in figs_to_make:
        print("Generating Figure 2: Decoder Comparison Bars...")
        figure2_decoder_bars(all_results, out_dir, fmt)

    if 3 in figs_to_make:
        print("Generating Figure 3: Threshold Curves (LER vs p)...")
        figure3_threshold_curves(all_results, out_dir, fmt)

    if 4 in figs_to_make:
        print("Generating Figure 4: Static vs Drift...")
        figure4_drift_comparison(all_results, out_dir, fmt)

    if 5 in figs_to_make:
        print("Generating Figure 5: Training Dynamics...")
        figure5_training_dynamics(results_dir, out_dir, fmt)

    if 6 in figs_to_make:
        print("Generating Figure 6: Improvement Heatmap...")
        figure6_heatmap(all_results, out_dir, fmt)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
