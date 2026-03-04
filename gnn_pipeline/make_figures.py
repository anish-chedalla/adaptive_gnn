#!/usr/bin/env python
"""
Publication-quality figure generation for the QLDPC GNN-BP pipeline.

Auto-discovers evaluation results from any run directory structure and
generates 12 figures following Nature / Physical Review conventions:

  Figure 1: LER vs code size (the "scaling story" -- headline figure)
  Figure 2: Decoder comparison bar chart with confidence intervals
  Figure 3: LER vs physical error rate (threshold-style curves)
  Figure 4: Static vs drift performance comparison
  Figure 5: Training dynamics (loss curves, convergence, alpha evolution)
  Figure 6: Relative improvement heatmap across all conditions
  Figure 7: Multi-code threshold overlay (all codes on one plot per decoder)
  Figure 8: Component ablation bar chart (from ablation_results.csv)
  Figure 9: Transfer learning comparison (scratch vs frozen vs fine-tune)
  Figure 10: Timing / latency comparison bars
  Figure 11: BP convergence rate vs physical error rate
  Figure 12: FiLM noise estimation accuracy (syndrome p-hat vs true p)

Data sources (auto-discovered):
  - eval_results.json  (from gnn_pipeline.evaluate)
  - results.json       (from gnn_pipeline.threshold_sweep)
  - metrics.csv        (from training runs)

Usage:
  python -m gnn_pipeline.make_figures --results_dir runs --out_dir figures
  python -m gnn_pipeline.make_figures --results_dir runs --out_dir figures --format pdf
  python -m gnn_pipeline.make_figures --results_dir runs --out_dir figures --fig 1 3 7

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
# Matplotlib setup -- must come before any pyplot import
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

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
    "teal":        "#44AA99",
    "wine":        "#882255",
}

# Decoder -> (color, marker, linestyle, zorder) mapping
DECODER_STYLE = {
    "Interleaved GNN + BP-OSD": (COLORS["vermillion"], "*", "-",  12),
    "Interleaved GNN + BP-LSD": (COLORS["green"],      "d", "-",  11),
    "Interleaved GNN-BP":       (COLORS["blue"],        "o", "-",  10),
    "GNN-BP":                   (COLORS["sky_blue"],    "s", "-",   9),
    "GNN + BP-LSD":             (COLORS["teal"],        "D", "--",  8),
    "GNN + BP-OSD":             (COLORS["wine"],        "^", "--",  7),
    "BP-OSD":                   (COLORS["orange"],      "v", "--",  6),
    "BP-LSD":                   (COLORS["purple"],      "p", "-.",  5),
    "BeliefFind":               (COLORS["yellow"],      "P", "-.",  4),
    "MWPM":                     (COLORS["grey"],        "h", ":",   3),
    "BP":                       (COLORS["black"],       "x", "--",  2),
    "Oracle BP":                ("#AAAAAA",             "+", ":",   1),
}

# Canonical decoder display order (best first)
DECODER_ORDER = [
    "Interleaved GNN + BP-OSD", "Interleaved GNN + BP-LSD",
    "Interleaved GNN-BP", "GNN-BP", "GNN + BP-LSD", "GNN + BP-OSD",
    "BP-OSD", "BP-LSD", "BeliefFind", "MWPM", "BP", "Oracle BP",
]

# JSON key -> display name
DECODER_KEY_MAP = {
    "bp":                      "BP",
    "oracle_bp":               "Oracle BP",
    "gnn_bp":                  "GNN-BP",
    "interleaved_gnn_bp":      "Interleaved GNN-BP",
    "bposd":                   "BP-OSD",
    "bplsd":                   "BP-LSD",
    "gnn_bplsd":               "GNN + BP-LSD",
    "gnn_bposd":               "GNN + BP-OSD",
    "interleaved_gnn_bposd":   "Interleaved GNN + BP-OSD",
    "interleaved_gnn_bplsd":   "Interleaved GNN + BP-LSD",
    "belieffind":              "BeliefFind",
    "mwpm":                    "MWPM",
    "bp_dem":                  "BP (DEM)",
    "gnn_bp_dem":              "GNN-BP (DEM)",
}

# Reverse: display name -> JSON key
DECODER_NAME_MAP = {v: k for k, v in DECODER_KEY_MAP.items()}

_NEW_BLACK = "#373737"

# Code size -> display label
CODE_LABELS = {
    72:  "[[72,12,6]]",
    144: "[[144,12,12]]",
    288: "[[288,12,18]]",
}

# Code size -> marker for multi-code overlay plots
CODE_MARKERS = {72: "o", 144: "s", 288: "D"}
CODE_SIZES_ORDERED = [72, 144, 288]


# ---------------------------------------------------------------------------
# Global rcParams -- Nature / PRA style
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
        "pdf.fonttype": 42,       # TrueType -- editable in Illustrator
        "ps.fonttype": 42,
    })


# ===================================================================
#                     DATA LOADING (Auto-discovery)
# ===================================================================

def _detect_drift(result: Dict) -> Tuple[bool, str]:
    """Detect if a result is from drift noise. Returns (is_drift, drift_model)."""
    # Check explicit fields first
    if "drift_model" in result.get("noise", {}):
        dm = result["noise"]["drift_model"]
        if dm and dm != "none":
            return True, dm
    # Check noise_type field (from sweep results)
    nt = result.get("noise_type", "")
    if "drift" in nt:
        model = nt.replace("drift_", "")
        return True, model
    # Check path hints
    path = result.get("_path", "")
    for tag in ["sine", "ou", "rtn", "drift"]:
        if tag in path.lower():
            return True, tag
    # Check has_per_shot_p (drift indicator)
    if result.get("has_per_shot_p", False):
        return True, "unknown"
    return False, "none"


def _extract_decoder_data(result: Dict, decoder_display_name: str) -> Optional[Dict]:
    """Extract LER + CI from a result dict for a given decoder display name."""
    key = DECODER_NAME_MAP.get(decoder_display_name)
    if key is None or key not in result:
        return None
    d = result[key]
    if not isinstance(d, dict) or "ler" not in d:
        return None
    return {
        "ler": d["ler"],
        "ci_low": d.get("ler_ci_low", d.get("ci_low", d["ler"])),
        "ci_high": d.get("ler_ci_high", d.get("ci_high", d["ler"])),
        "errors": d.get("errors", 0),
        "shots": result.get("test_shots", result.get("shots", 0)),
        "convergence": d.get("convergence_rate", d.get("convergence")),
    }


def _get_code_size(result: Dict) -> Optional[int]:
    """Extract code block length n from result."""
    if "code" in result and isinstance(result["code"], dict):
        return result["code"].get("n")
    return None


def _get_p(result: Dict) -> Optional[float]:
    """Extract physical error rate from result."""
    if "noise" in result and isinstance(result["noise"], dict):
        return result["noise"].get("p")
    return result.get("p")


def _get_code_label(n: int) -> str:
    """Map n -> human-readable code label."""
    return CODE_LABELS.get(n, f"[[{n},?,?]]")


def _panel_label(ax, label, x=-0.12, y=1.06):
    """Add bold panel label (a, b, c, ...) to axes -- Nature style."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")


def _available_decoders(results: List[Dict]) -> List[str]:
    """Find all decoder display names present across results, in canonical order."""
    found = set()
    for r in results:
        for json_key, display_name in DECODER_KEY_MAP.items():
            if json_key in r and isinstance(r[json_key], dict) and "ler" in r[json_key]:
                found.add(display_name)
    return [d for d in DECODER_ORDER if d in found]


def _setup_log_y(ax, floor=2e-4):
    """Configure y-axis log scale with nice minor ticks."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.yaxis.set_minor_formatter(NullFormatter())


def load_eval_results(results_dir: Path) -> List[Dict]:
    """Recursively find and load ALL eval_results.json files."""
    results = []
    for p in sorted(results_dir.rglob("eval_results.json")):
        try:
            with open(p) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        # Skip circuit-level results (different schema)
        if "code" not in data:
            continue
        data["_path"] = p.parent.name
        data["_full_path"] = str(p)
        results.append(data)
    return results


def load_sweep_results(results_dir: Path) -> List[Dict]:
    """Recursively find and load threshold sweep results.json files.

    Returns a flat list of per-point results with code metadata attached.
    """
    flat_results = []
    for p in sorted(results_dir.rglob("results.json")):
        try:
            with open(p) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        # Must have sweep-style structure
        if "results" not in data or "configuration" not in data:
            continue
        code_meta = data.get("code", {})
        config = data.get("configuration", {})
        for point in data["results"]:
            # Attach code metadata
            if "code" not in point:
                point["code"] = code_meta
            if "noise" not in point:
                point["noise"] = {"p": point.get("p", 0), "eta": config.get("eta", 1)}
            point["_path"] = p.parent.name
            point["_source"] = "sweep"
            # Propagate noise_type
            if "noise_type" not in point:
                point["noise_type"] = "static"
            flat_results.append(point)
    return flat_results


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


def load_all_data(results_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """Load all eval and sweep results. Returns (eval_results, sweep_results)."""
    evals = load_eval_results(results_dir)
    sweeps = load_sweep_results(results_dir)
    print(f"  Found {len(evals)} eval result files, {len(sweeps)} sweep data points")
    return evals, sweeps


# ===================================================================
#          FIGURE 1 -- LER vs Code Size  (The Headline)
# ===================================================================

def figure1_scaling(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """
    LER vs code size showing how each decoder scales.
    Auto-selects the best available p-value match across code sizes.

    Panel (a): Absolute LER per code size per decoder.
    Panel (b): Relative improvement over BP per code size.
    """
    # Group by code size, pick static results, find common p-values
    by_code: Dict[int, List[Dict]] = {}
    for r in results:
        n = _get_code_size(r)
        if n is None:
            continue
        is_drift, _ = _detect_drift(r)
        if is_drift:
            continue
        by_code.setdefault(n, []).append(r)

    if len(by_code) < 2:
        print("  [Figure 1] Need >= 2 code sizes, skipping.")
        return

    # For each code size, pick the result with highest p that has both BP and
    # at least one GNN decoder (most interesting regime)
    code_data: Dict[int, Dict] = {}
    selected_p: Dict[int, float] = {}
    for n in sorted(by_code.keys()):
        # Sort by p descending to prefer harder regime
        candidates = sorted(by_code[n], key=lambda r: _get_p(r) or 0, reverse=True)
        for r in candidates:
            bp = _extract_decoder_data(r, "BP")
            if bp is None:
                continue
            # Check for at least one GNN decoder
            has_gnn = any(
                _extract_decoder_data(r, d) is not None
                for d in ["GNN-BP", "Interleaved GNN-BP", "GNN + BP-LSD"]
            )
            if has_gnn:
                code_data[n] = {}
                for dec in DECODER_ORDER:
                    d = _extract_decoder_data(r, dec)
                    if d is not None:
                        code_data[n][dec] = d
                selected_p[n] = _get_p(r)
                break
        # Fallback: use any result with BP
        if n not in code_data:
            for r in candidates:
                bp = _extract_decoder_data(r, "BP")
                if bp is not None:
                    code_data[n] = {}
                    for dec in DECODER_ORDER:
                        d = _extract_decoder_data(r, dec)
                        if d is not None:
                            code_data[n][dec] = d
                    selected_p[n] = _get_p(r)
                    break

    code_sizes = sorted(code_data.keys())
    if len(code_sizes) < 2:
        print("  [Figure 1] Not enough code sizes with data, skipping.")
        return

    decoders_present = []
    for dec in DECODER_ORDER:
        if any(dec in code_data.get(n, {}) for n in code_sizes):
            decoders_present.append(dec)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                                    gridspec_kw={"width_ratios": [3, 2]})

    # --- Panel (a): LER vs code size ---
    x_pos = np.arange(len(code_sizes))

    for dec in decoders_present:
        color, marker, ls, zorder = DECODER_STYLE.get(
            dec, (COLORS["grey"], ".", "-", 1))
        lers, ci_lo, ci_hi = [], [], []
        valid_x = []
        for i, n in enumerate(code_sizes):
            if dec in code_data.get(n, {}):
                d = code_data[n][dec]
                ler = max(d["ler"], 1e-5)  # floor for log scale
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

    _setup_log_y(ax1)
    ax1.set_xticks(x_pos)
    tick_labels = []
    for n in code_sizes:
        p_str = f"p={selected_p.get(n, 0):.3f}" if n in selected_p else ""
        tick_labels.append(f"{_get_code_label(n)}\n{p_str}")
    ax1.set_xticklabels(tick_labels, fontsize=7.5)
    ax1.set_xlabel("Code")
    ax1.set_ylabel("Logical Error Rate (LER)")
    ax1.legend(loc="upper left", fontsize=6, ncol=1)
    _panel_label(ax1, "a")

    # --- Panel (b): Relative improvement vs code size ---
    gnn_decoders = [d for d in [
        "GNN-BP", "Interleaved GNN-BP", "GNN + BP-LSD", "GNN + BP-OSD"
    ] if d in decoders_present]

    if gnn_decoders:
        bar_width = 0.8 / max(len(gnn_decoders), 1)
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
    if gnn_decoders:
        ax2.legend(loc="upper left", fontsize=6, ncol=1)
    _panel_label(ax2, "b")

    fig.tight_layout(w_pad=2.5)
    out_path = out_dir / f"fig1_scaling.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#      FIGURE 2 -- Decoder Comparison Bar Chart (per code size)
# ===================================================================

def figure2_decoder_bars(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """
    Grouped bar chart: LER for each decoder at each code size.
    Auto-discovers which decoders are available.
    """
    # Group static results by code size, pick highest-p result
    by_code: Dict[int, Dict] = {}
    for r in results:
        n = _get_code_size(r)
        if n is None:
            continue
        is_drift, _ = _detect_drift(r)
        if is_drift:
            continue
        p = _get_p(r) or 0
        if n not in by_code or p > (_get_p(by_code[n]) or 0):
            by_code[n] = r

    if not by_code:
        print("  [Figure 2] No static results found, skipping.")
        return

    code_sizes = sorted(by_code.keys())
    n_codes = len(code_sizes)

    # Find all decoders across all code sizes
    all_decoders = _available_decoders(list(by_code.values()))

    fig, axes = plt.subplots(1, n_codes, figsize=(max(3.5, 2.5 * n_codes), 3.8),
                             sharey=False, squeeze=False)

    for idx, n in enumerate(code_sizes):
        ax = axes[0, idx]
        r = by_code[n]
        p_val = _get_p(r) or 0

        decoders_here = [d for d in all_decoders if _extract_decoder_data(r, d) is not None]
        x = np.arange(len(decoders_here))
        lers, ci_lo, ci_hi, colors = [], [], [], []

        for dec in decoders_here:
            d = _extract_decoder_data(r, dec)
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
                           for d in decoders_here],
                          fontsize=5.5, rotation=45, ha="right")
        ax.set_ylabel("LER" if idx == 0 else "")
        ax.set_title(f"{_get_code_label(n)}, p={p_val:.3f}", fontsize=9)
        ax.set_ylim(bottom=0)
        _panel_label(ax, chr(ord("a") + idx))

    fig.tight_layout(w_pad=1.5)
    out_path = out_dir / f"fig2_decoder_bars.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#   FIGURE 3 -- LER vs Physical Error Rate (Threshold-style curves)
# ===================================================================

def figure3_threshold_curves(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """
    LER vs p for each code size -- classic QEC threshold plot.
    Separate panel per code, all decoders overlaid.
    Uses sweep data preferentially, falls back to individual evals.
    """
    # Group results by code size (static only), then sort by p
    by_code: Dict[int, List[Dict]] = {}
    for r in results:
        n = _get_code_size(r)
        if n is None:
            continue
        is_drift, _ = _detect_drift(r)
        if is_drift:
            continue
        by_code.setdefault(n, []).append(r)

    if not by_code:
        print("  [Figure 3] No static results found, skipping.")
        return

    code_sizes = sorted(by_code.keys())
    n_panels = len(code_sizes)

    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.3),
                             sharey=False, squeeze=False)

    all_decoders = _available_decoders(
        [r for rs in by_code.values() for r in rs])
    # Limit to interesting subset for readability
    decoders_to_plot = [d for d in [
        "BP", "GNN-BP", "Interleaved GNN-BP",
        "BP-OSD", "BP-LSD", "GNN + BP-LSD", "GNN + BP-OSD",
        "BeliefFind", "MWPM", "Oracle BP",
    ] if d in all_decoders]

    for idx, n in enumerate(code_sizes):
        ax = axes[0, idx]
        rs = sorted(by_code[n], key=lambda x: _get_p(x) or 0)

        # Deduplicate by p (prefer sweep data which has more points)
        seen_p = {}
        for r in rs:
            p = round(_get_p(r) or 0, 6)
            if p not in seen_p or r.get("_source") == "sweep":
                seen_p[p] = r
        rs = [seen_p[p] for p in sorted(seen_p.keys())]

        for dec in decoders_to_plot:
            color, marker, ls, zorder = DECODER_STYLE.get(
                dec, (COLORS["grey"], ".", "-", 1))
            ps, lers, ci_lo, ci_hi = [], [], [], []
            for r in rs:
                d = _extract_decoder_data(r, dec)
                if d is not None and d["ler"] > 0:
                    ler = max(d["ler"], 1e-5)
                    ps.append(_get_p(r))
                    lers.append(ler)
                    ci_lo.append(max(ler - d["ci_low"], 0))
                    ci_hi.append(max(d["ci_high"] - ler, 0))
            if len(ps) < 2:
                # Need at least 2 points to draw a line
                if len(ps) == 1:
                    ax.errorbar(ps, lers, yerr=[[ci_lo[0]], [ci_hi[0]]],
                               fmt=marker, color=color, label=dec,
                               markersize=6, capsize=2, zorder=zorder)
                continue
            ax.errorbar(ps, lers, yerr=[ci_lo, ci_hi],
                       fmt=marker + ls, color=color, label=dec,
                       markersize=5, capsize=2, capthick=0.5,
                       linewidth=1.2, zorder=zorder, markeredgewidth=0.4,
                       markeredgecolor="white" if marker not in ("x", "+") else color)

        _setup_log_y(ax)
        ax.set_xlabel("Physical error rate ($p$)")
        ax.set_ylabel("Logical Error Rate" if idx == 0 else "")
        ax.set_title(f"{_get_code_label(n)}", fontsize=9)
        if idx == n_panels - 1:
            ax.legend(loc="lower right", fontsize=5.5)
        _panel_label(ax, chr(ord("a") + idx))

    fig.tight_layout(w_pad=2.0)
    out_path = out_dir / f"fig3_threshold_curves.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#      FIGURE 4 -- Static vs Drift Comparison
# ===================================================================

def figure4_drift_comparison(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """
    Side-by-side: static vs drift for each code size.
    Shows how decoders degrade under drift and whether GNN/FiLM helps.
    Auto-pairs static/drift results with matching code size and p.
    """
    # Group by (code_size, p, is_drift)
    pairs: Dict[Tuple[int, float], Dict[str, Dict]] = {}
    for r in results:
        n = _get_code_size(r)
        p = _get_p(r)
        if n is None or p is None:
            continue
        is_drift, drift_model = _detect_drift(r)
        key = (n, round(p, 4))
        label = "drift" if is_drift else "static"
        if key not in pairs:
            pairs[key] = {}
        if label not in pairs[key]:
            pairs[key][label] = r

    # Filter to pairs that have BOTH static and drift
    valid_pairs = {k: v for k, v in pairs.items() if "static" in v and "drift" in v}

    if not valid_pairs:
        print("  [Figure 4] No static/drift pairs found, skipping.")
        return

    # Find available decoders across all paired results
    all_r = [v[c] for v in valid_pairs.values() for c in ["static", "drift"]]
    decoders_to_plot = [d for d in DECODER_ORDER if d in _available_decoders(all_r)]
    # Limit to ~6 decoders for readability
    if len(decoders_to_plot) > 6:
        priority = ["BP", "GNN-BP", "Interleaved GNN-BP", "BP-OSD", "GNN + BP-LSD", "Oracle BP"]
        decoders_to_plot = [d for d in priority if d in decoders_to_plot]

    n_panels = min(len(valid_pairs), 4)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.3 * n_panels, 3.5), squeeze=False)

    for idx, ((n, p), pair) in enumerate(sorted(valid_pairs.items())):
        if idx >= 4:
            break
        ax = axes[0, idx]
        x = np.arange(len(decoders_to_plot))
        width = 0.35

        for cond, label, offset, alpha in [
            ("static", "Static", -width / 2, 0.9),
            ("drift", "Drift", width / 2, 0.55),
        ]:
            r = pair[cond]
            lers, colors = [], []
            for dec in decoders_to_plot:
                d = _extract_decoder_data(r, dec)
                lers.append(d["ler"] if d is not None else 0)
                colors.append(DECODER_STYLE.get(dec, (COLORS["grey"],))[0])

            bars = ax.bar(x + offset, lers, width, alpha=alpha,
                         color=colors, edgecolor="white", linewidth=0.5,
                         label=label if idx == 0 else "")
            if cond == "drift":
                for bar in bars:
                    bar.set_hatch("///")

        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("Interleaved ", "Int.\n")
                           .replace("GNN + ", "GNN+\n")
                           for d in decoders_to_plot],
                          fontsize=6, rotation=30, ha="right")
        ax.set_ylabel("LER" if idx == 0 else "")
        ax.set_title(f"{_get_code_label(n)}, p={p}", fontsize=9)
        _panel_label(ax, chr(ord("a") + idx))

    handles = [
        mpatches.Patch(facecolor=COLORS["grey"], alpha=0.9, label="Static"),
        mpatches.Patch(facecolor=COLORS["grey"], alpha=0.55, hatch="///", label="Drift"),
    ]
    axes[0, 0].legend(handles=handles, loc="upper left", fontsize=7)

    fig.tight_layout(w_pad=1.5)
    out_path = out_dir / f"fig4_drift_comparison.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#    FIGURE 5 -- Training Dynamics (multi-panel)
# ===================================================================

def figure5_training_dynamics(results_dir: Path, out_dir: Path, fmt: str = "pdf"):
    """
    Training curves from CSV metrics: loss, convergence, bit accuracy, LR.
    Auto-discovers all training runs with metrics.csv.
    """
    # Find all directories with metrics.csv
    valid_runs = []
    for csv_path in sorted(results_dir.rglob("metrics.csv")):
        csv = load_metrics_csv(csv_path.parent)
        if csv is not None and "train_loss" in csv and len(csv["train_loss"]) >= 2:
            label = csv_path.parent.name
            valid_runs.append((csv, label))

    if not valid_runs:
        print("  [Figure 5] No training metrics found, skipping.")
        return

    # Limit to 5 runs max for readability
    if len(valid_runs) > 5:
        valid_runs = valid_runs[:5]

    n_runs = len(valid_runs)
    fig, axes = plt.subplots(n_runs, 4, figsize=(7.2, 2.2 * n_runs), squeeze=False)

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
                fontsize=6, ha="right", va="top",
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
            ax.set_ylim(max(ymin - 0.005, 0.90), 1.001)
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
#     FIGURE 6 -- Relative Improvement Heatmap
# ===================================================================

def figure6_heatmap(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """
    Heatmap: (decoder x condition) showing % improvement over BP.
    Rows = GNN/OSD decoders, columns = test conditions (code+p+noise).
    Auto-discovers all conditions from the data.
    """
    decoders = [d for d in [
        "GNN-BP", "Interleaved GNN-BP",
        "BP-OSD", "BP-LSD", "GNN + BP-LSD", "GNN + BP-OSD",
        "BeliefFind", "MWPM",
    ] if d in _available_decoders(results)]

    if not decoders:
        print("  [Figure 6] No comparison decoders found, skipping.")
        return

    # Deduplicate: keep unique (n, p, drift) combos, prefer eval over sweep
    seen = {}
    for r in results:
        n = _get_code_size(r)
        p = _get_p(r)
        if n is None or p is None:
            continue
        is_drift, drift_model = _detect_drift(r)
        key = (n, round(p, 4), is_drift, drift_model)
        # Prefer eval results over sweep points
        if key not in seen or r.get("_source") != "sweep":
            seen[key] = r

    cond_results = list(seen.values())
    if not cond_results:
        print("  [Figure 6] No results for heatmap, skipping.")
        return

    # Build condition labels
    conditions = []
    for r in cond_results:
        n = _get_code_size(r)
        p = _get_p(r)
        is_drift, drift_model = _detect_drift(r)
        drift_str = f"\n+{drift_model}" if is_drift else ""
        conditions.append(f"n={n}\np={p:.3f}{drift_str}")

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

    # Sort columns by code size then p then drift
    col_order = list(range(n_cond))
    col_order.sort(key=lambda j: (
        _get_code_size(cond_results[j]) or 0,
        _get_p(cond_results[j]) or 0,
        _detect_drift(cond_results[j])[0],
    ))
    matrix = matrix[:, col_order]
    conditions = [conditions[j] for j in col_order]
    cond_results_sorted = [cond_results[j] for j in col_order]

    # Sizing
    cell_w = 0.65
    cell_h = 0.45
    fig_width = max(5.0, cell_w * n_cond + 3.0)
    fig_height = max(3.0, cell_h * n_dec + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Custom diverging colormap centered at 0
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
    for j_idx in range(n_cond):
        cur_n = _get_code_size(cond_results_sorted[j_idx])
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
#  FIGURE 7 -- Multi-Code Threshold Overlay (the money plot)
# ===================================================================

def figure7_multicode_overlay(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """
    All code sizes on ONE plot, one panel per decoder.
    This is the classic QEC scaling plot: for a fixed decoder, show
    how LER vs p improves with increasing code size.

    If the curves cross, you can read off the threshold.
    """
    # Group static results by code size
    by_code: Dict[int, List[Dict]] = {}
    for r in results:
        n = _get_code_size(r)
        if n is None:
            continue
        is_drift, _ = _detect_drift(r)
        if is_drift:
            continue
        by_code.setdefault(n, []).append(r)

    if len(by_code) < 2:
        print("  [Figure 7] Need >= 2 code sizes with sweep data, skipping.")
        return

    code_sizes = sorted(by_code.keys())

    # Deduplicate by p per code
    code_sweep: Dict[int, List[Dict]] = {}
    for n in code_sizes:
        seen_p = {}
        for r in sorted(by_code[n], key=lambda x: _get_p(x) or 0):
            p = round(_get_p(r) or 0, 6)
            if p not in seen_p or r.get("_source") == "sweep":
                seen_p[p] = r
        if len(seen_p) >= 2:
            code_sweep[n] = [seen_p[p] for p in sorted(seen_p.keys())]

    code_sizes = sorted(code_sweep.keys())
    if len(code_sizes) < 2:
        print("  [Figure 7] Need >= 2 code sizes with multiple p-values, skipping.")
        return

    # Find decoders that appear across all code sizes
    all_decs = set(DECODER_ORDER)
    for n in code_sizes:
        n_decs = set(_available_decoders(code_sweep[n]))
        all_decs &= n_decs
    decoders_to_panel = [d for d in [
        "BP", "GNN-BP", "Interleaved GNN-BP", "BP-OSD", "GNN + BP-LSD",
    ] if d in all_decs]

    # Fallback: show decoders available in at least 2 code sizes
    if len(decoders_to_panel) < 2:
        dec_counts = {}
        for n in code_sizes:
            for d in _available_decoders(code_sweep[n]):
                dec_counts[d] = dec_counts.get(d, 0) + 1
        decoders_to_panel = [d for d in DECODER_ORDER
                             if dec_counts.get(d, 0) >= 2]

    if not decoders_to_panel:
        print("  [Figure 7] No common decoders across code sizes, skipping.")
        return

    n_panels = len(decoders_to_panel)
    ncols = min(n_panels, 3)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows),
                             squeeze=False)

    # Color per code size (distinct from decoder colors)
    code_colors = {
        72:  COLORS["blue"],
        144: COLORS["vermillion"],
        288: COLORS["green"],
    }
    # Fallback for unknown sizes
    _fallback_colors = [COLORS["orange"], COLORS["purple"], COLORS["teal"]]
    for i, n in enumerate(code_sizes):
        if n not in code_colors:
            code_colors[n] = _fallback_colors[i % len(_fallback_colors)]

    for panel_idx, dec in enumerate(decoders_to_panel):
        row, col = divmod(panel_idx, ncols)
        ax = axes[row, col]

        for n in code_sizes:
            color = code_colors[n]
            marker = CODE_MARKERS.get(n, "o")
            ps, lers, ci_lo, ci_hi = [], [], [], []
            for r in code_sweep[n]:
                d = _extract_decoder_data(r, dec)
                if d is not None and d["ler"] > 0:
                    ler = max(d["ler"], 1e-5)
                    ps.append(_get_p(r))
                    lers.append(ler)
                    ci_lo.append(max(ler - d["ci_low"], 0))
                    ci_hi.append(max(d["ci_high"] - ler, 0))
            if not ps:
                continue
            ax.errorbar(ps, lers, yerr=[ci_lo, ci_hi],
                       fmt=marker + "-", color=color,
                       label=_get_code_label(n),
                       markersize=5, capsize=2, capthick=0.5,
                       linewidth=1.2, markeredgewidth=0.4,
                       markeredgecolor="white")

        _setup_log_y(ax)
        ax.set_xlabel("Physical error rate ($p$)")
        ax.set_ylabel("LER" if col == 0 else "")
        ax.set_title(dec, fontsize=9)
        ax.legend(loc="lower right", fontsize=6)
        _panel_label(ax, chr(ord("a") + panel_idx))

    # Hide unused panels
    for panel_idx in range(n_panels, nrows * ncols):
        row, col = divmod(panel_idx, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    out_path = out_dir / f"fig7_multicode_overlay.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#  FIGURE 8 -- Component Ablation Bar Chart  (Paper Table 3)
# ===================================================================

def _load_ablation_csv(results_dir: Path) -> Optional[List[Dict]]:
    """Load ablation_results.csv from results_dir or subdirectories."""
    for csv_path in sorted(results_dir.rglob("ablation_results.csv")):
        lines = csv_path.read_text().strip().split("\n")
        if len(lines) < 2:
            continue
        headers = [h.strip() for h in lines[0].split(",")]
        rows = []
        for line in lines[1:]:
            vals = line.split(",")
            row = {}
            for h, v in zip(headers, vals):
                try:
                    row[h] = float(v)
                except ValueError:
                    row[h] = v.strip()
            rows.append(row)
        if rows:
            return rows
    return None


def figure8_ablation(results_dir: Path, out_dir: Path, fmt: str = "pdf"):
    """Component ablation bar chart: cumulative component addition."""
    rows = _load_ablation_csv(results_dir)
    if not rows:
        print("  [Figure 8] No ablation_results.csv found, skipping.")
        return

    # Extract config names and LER values
    names = [r.get("name", r.get("config", f"Config {i}")) for i, r in enumerate(rows)]
    # Clean up names for display
    display_names = []
    for n in names:
        n = str(n).replace("_", " ").replace("01 ", "").replace("02 ", "")
        n = n.replace("03 ", "").replace("04 ", "").replace("05 ", "").replace("06 ", "")
        display_names.append(n.strip().title())

    ler_key = None
    for k in ["test_ler", "ler", "eval_ler", "test_logical_error_rate"]:
        if k in rows[0]:
            ler_key = k
            break
    if ler_key is None:
        print("  [Figure 8] No LER column found in ablation CSV, skipping.")
        return

    lers = [float(r.get(ler_key, 0)) for r in rows]
    conv_key = None
    for k in ["convergence_rate", "convergence", "conv_rate"]:
        if k in rows[0]:
            conv_key = k
            break
    convs = [float(r.get(conv_key, 0)) for r in rows] if conv_key else None

    fig, ax1 = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(display_names))
    width = 0.35

    bars = ax1.bar(x, lers, width, color=COLORS["blue"], alpha=0.85,
                   edgecolor="white", linewidth=0.5, label="LER")
    ax1.set_ylabel("Logical Error Rate")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, rotation=25, ha="right", fontsize=7)

    if convs:
        ax2 = ax1.twinx()
        ax2.bar(x + width, convs, width, color=COLORS["green"], alpha=0.85,
                edgecolor="white", linewidth=0.5, label="Conv. Rate")
        ax2.set_ylabel("BP Convergence Rate")
        ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
    else:
        ax1.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    out_path = out_dir / f"fig8_ablation.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#  FIGURE 9 -- Transfer Learning Comparison  (Paper Table 4)
# ===================================================================

def figure9_transfer(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """Transfer learning comparison: scratch vs frozen vs fine-tune."""
    # Look for results with 'transfer' or 'scratch' in path
    transfer_data = {}  # {code_size: {strategy: ler}}
    for r in results:
        path = r.get("_path", "")
        n = _get_code_size(r)
        if n is None or n == 72:  # 72 is the source, not a transfer target
            continue

        # Determine strategy from path
        strategy = None
        if "frozen" in path.lower():
            strategy = "Frozen Transfer"
        elif "finetune" in path.lower() or "fine_tune" in path.lower():
            strategy = "Fine-tune"
        elif "transfer" not in path.lower():
            # Check if this is a from-scratch model for this code size
            if f"big{n}" in path.lower() and "transfer" not in path.lower():
                strategy = "From Scratch"

        if strategy is None:
            continue

        # Get best LER across decoders (use GNN-BP as reference)
        for dec_key in ["gnn_bp", "bp"]:
            if dec_key in r and isinstance(r[dec_key], dict) and "ler" in r[dec_key]:
                if n not in transfer_data:
                    transfer_data[n] = {}
                if strategy not in transfer_data[n] or r[dec_key]["ler"] < transfer_data[n][strategy]:
                    transfer_data[n][strategy] = r[dec_key]["ler"]
                break

    if not transfer_data:
        print("  [Figure 9] No transfer learning results found, skipping.")
        return

    code_sizes = sorted(transfer_data.keys())
    strategies = ["From Scratch", "Frozen Transfer", "Fine-tune"]
    strat_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    width = 0.25
    x = np.arange(len(code_sizes))

    for i, (strat, color) in enumerate(zip(strategies, strat_colors)):
        vals = [transfer_data.get(n, {}).get(strat, 0) for n in code_sizes]
        if any(v > 0 for v in vals):
            ax.bar(x + i * width, vals, width, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.5, label=strat)

    ax.set_ylabel("Logical Error Rate (GNN-BP)")
    ax.set_yscale("log")
    ax.set_xticks(x + width)
    ax.set_xticklabels([_get_code_label(n) for n in code_sizes])
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    out_path = out_dir / f"fig9_transfer.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#  FIGURE 10 -- Timing / Latency Comparison  (Paper Table 5)
# ===================================================================

def figure10_timing(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """Horizontal bar chart of per-shot decoding latency."""
    # Collect timing from eval results
    timing = {}  # {decoder_display_name: ms_per_shot}
    for r in results:
        if r.get("_source") == "sweep":
            continue  # Only use eval results for timing
        shots = r.get("test_shots", r.get("shots", 0))
        if shots <= 0:
            continue
        for json_key, display_name in DECODER_KEY_MAP.items():
            if json_key in r and isinstance(r[json_key], dict):
                sec = r[json_key].get("seconds", 0)
                if sec > 0:
                    ms = 1000 * sec / shots
                    if display_name not in timing or ms < timing[display_name]:
                        timing[display_name] = ms

    if not timing:
        print("  [Figure 10] No timing data found, skipping.")
        return

    # Sort by canonical order
    ordered = [(d, timing[d]) for d in DECODER_ORDER if d in timing]
    if not ordered:
        return

    names, values = zip(*ordered)
    colors = [DECODER_STYLE.get(n, (COLORS["grey"],))[0] for n in names]

    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(names))))
    y = np.arange(len(names))
    ax.barh(y, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Latency (ms / shot)")
    ax.set_xscale("log")
    ax.invert_yaxis()

    # Annotate values
    for i, v in enumerate(values):
        ax.text(v * 1.1, i, f"{v:.1f}", va="center", fontsize=6)

    fig.tight_layout()
    out_path = out_dir / f"fig10_timing.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#  FIGURE 11 -- BP Convergence Rate vs p  (Paper Figure 4)
# ===================================================================

def figure11_convergence(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """Convergence rate vs physical error rate for key decoders."""
    # Group by (code_size, decoder) -> list of (p, conv_rate)
    data = {}  # {(n, decoder_display): [(p, conv_rate), ...]}
    for r in results:
        is_drift, _ = _detect_drift(r)
        if is_drift:
            continue  # Only static for convergence analysis
        n = _get_code_size(r)
        p = _get_p(r)
        if n is None or p is None:
            continue
        for json_key, display_name in DECODER_KEY_MAP.items():
            if json_key in r and isinstance(r[json_key], dict):
                conv = r[json_key].get("convergence_rate")
                if conv is not None:
                    key = (n, display_name)
                    if key not in data:
                        data[key] = []
                    data[key].append((p, conv))

    if not data:
        print("  [Figure 11] No convergence data found, skipping.")
        return

    code_sizes = sorted(set(k[0] for k in data.keys()))
    # Key decoders to show
    key_decoders = ["BP", "GNN-BP", "Interleaved GNN-BP"]

    ncols = min(len(code_sizes), 3)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.5), sharey=True, squeeze=False)
    axes = axes[0]

    for col, n in enumerate(code_sizes[:ncols]):
        ax = axes[col]
        for dec in key_decoders:
            key = (n, dec)
            if key not in data:
                continue
            points = sorted(data[key], key=lambda x: x[0])
            ps, convs = zip(*points)
            style = DECODER_STYLE.get(dec, (COLORS["grey"], "o", "-", 1))
            ax.plot(ps, convs, marker=style[1], linestyle=style[2],
                    color=style[0], label=dec, markersize=4, linewidth=1.2)

        ax.set_xlabel("Physical error rate ($p$)")
        if col == 0:
            ax.set_ylabel("BP Convergence Rate")
        ax.set_title(_get_code_label(n), fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
        ax.legend(loc="lower left", fontsize=6)
        _panel_label(ax, chr(ord("a") + col))

    fig.tight_layout()
    out_path = out_dir / f"fig11_convergence.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#  FIGURE 12 -- FiLM Noise Estimation Accuracy
# ===================================================================

def figure12_film_accuracy(results: List[Dict], out_dir: Path, fmt: str = "pdf"):
    """Scatter plot: syndrome-estimated p-hat vs true p."""
    # Collect syndrome_p_estimate data from eval results
    true_ps = []
    est_ps = []
    for r in results:
        syn_est = r.get("syndrome_p_estimate", {})
        if not syn_est:
            continue
        true_p = _get_p(r)
        mean_est = syn_est.get("mean")
        if true_p is not None and mean_est is not None:
            true_ps.append(true_p)
            est_ps.append(mean_est)

    if len(true_ps) < 2:
        print("  [Figure 12] Not enough FiLM estimation data, skipping.")
        return

    true_ps = np.array(true_ps)
    est_ps = np.array(est_ps)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(true_ps, est_ps, s=20, color=COLORS["blue"], alpha=0.7,
               edgecolors="white", linewidth=0.3, zorder=3)

    # 45-degree reference
    lims = [min(true_ps.min(), est_ps.min()) * 0.8,
            max(true_ps.max(), est_ps.max()) * 1.2]
    ax.plot(lims, lims, "--", color="grey", linewidth=0.8, alpha=0.6, zorder=1)

    # R^2
    if len(true_ps) > 2:
        ss_res = np.sum((est_ps - true_ps) ** 2)
        ss_tot = np.sum((true_ps - true_ps.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.05, 0.92, f"$R^2 = {r2:.3f}$", transform=ax.transAxes,
                fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("True physical error rate ($p$)")
    ax.set_ylabel("Syndrome-estimated $\\hat{p}$")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")

    fig.tight_layout()
    out_path = out_dir / f"fig12_film_accuracy.{fmt}"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ===================================================================
#           MAIN -- Generate all figures
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for QLDPC GNN-BP pipeline.")
    parser.add_argument("--results_dir", type=str, default="runs",
                        help="Root directory containing evaluation runs (searched recursively)")
    parser.add_argument("--out_dir", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "png", "svg", "eps"],
                        help="Output format (default: pdf)")
    parser.add_argument("--fig", type=int, nargs="*", default=None,
                        help="Which figures to generate (1-7). Default: all")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = args.format

    _apply_style()

    # Load ALL data
    print(f"Loading results from {results_dir} (recursive)...")
    eval_results, sweep_results = load_all_data(results_dir)

    # Merge: eval results + sweep results into one pool
    all_results = eval_results + sweep_results

    if not all_results:
        print("No results found. Run evaluations first, then point --results_dir at the output.")
        return

    # Summary
    code_sizes_found = sorted(set(
        _get_code_size(r) for r in all_results if _get_code_size(r) is not None))
    decoders_found = _available_decoders(all_results)
    n_drift = sum(1 for r in all_results if _detect_drift(r)[0])
    print(f"  Code sizes: {[_get_code_label(n) for n in code_sizes_found]}")
    print(f"  Decoders:   {decoders_found}")
    print(f"  Drift results: {n_drift} / {len(all_results)}")

    figs_to_make = args.fig if args.fig else list(range(1, 13))

    if 1 in figs_to_make:
        print("\nFigure 1: LER vs Code Size (Scaling)...")
        figure1_scaling(all_results, out_dir, fmt)

    if 2 in figs_to_make:
        print("\nFigure 2: Decoder Comparison Bars...")
        figure2_decoder_bars(all_results, out_dir, fmt)

    if 3 in figs_to_make:
        print("\nFigure 3: Threshold Curves (LER vs p)...")
        figure3_threshold_curves(all_results, out_dir, fmt)

    if 4 in figs_to_make:
        print("\nFigure 4: Static vs Drift...")
        figure4_drift_comparison(all_results, out_dir, fmt)

    if 5 in figs_to_make:
        print("\nFigure 5: Training Dynamics...")
        figure5_training_dynamics(results_dir, out_dir, fmt)

    if 6 in figs_to_make:
        print("\nFigure 6: Improvement Heatmap...")
        figure6_heatmap(all_results, out_dir, fmt)

    if 7 in figs_to_make:
        print("\nFigure 7: Multi-Code Threshold Overlay...")
        figure7_multicode_overlay(all_results, out_dir, fmt)

    if 8 in figs_to_make:
        print("\nFigure 8: Component Ablation...")
        figure8_ablation(results_dir, out_dir, fmt)

    if 9 in figs_to_make:
        print("\nFigure 9: Transfer Learning Comparison...")
        figure9_transfer(all_results, out_dir, fmt)

    if 10 in figs_to_make:
        print("\nFigure 10: Timing / Latency...")
        figure10_timing(all_results, out_dir, fmt)

    if 11 in figs_to_make:
        print("\nFigure 11: BP Convergence Rate vs p...")
        figure11_convergence(all_results, out_dir, fmt)

    if 12 in figs_to_make:
        print("\nFigure 12: FiLM Noise Estimation Accuracy...")
        figure12_film_accuracy(all_results, out_dir, fmt)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
