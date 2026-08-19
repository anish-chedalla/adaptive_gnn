#!/usr/bin/env python3
"""Reproduce fig_F4_phase4_ler with fixes:
  - "slightly worse" instead of "significantly worse"
  - annotation moved outside the plot area (no overlap)
  - y-axis zoomed in to show the separation more clearly
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Data (read off from PDF) ---
ps = np.array([0.010, 0.020, 0.030, 0.040, 0.050])

bp_osd_ler  = np.array([6.8e-4, 7.8e-3, 2.65e-2, 7.2e-2, 1.12e-1])
bp_osd_lo   = np.array([1.4e-4, 3.0e-4, 5.0e-4,  4.0e-3, 4.0e-3])
bp_osd_hi   = np.array([1.4e-4, 3.0e-4, 5.0e-4,  4.0e-3, 4.0e-3])

gnn_osd_ler = np.array([5.2e-4, 8.2e-3, 2.65e-2, 7.4e-2, 1.14e-1])
gnn_osd_lo  = np.array([1.2e-4, 3.2e-4, 5.0e-4,  4.0e-3, 4.0e-3])
gnn_osd_hi  = np.array([1.2e-4, 3.2e-4, 5.0e-4,  4.0e-3, 4.0e-3])

# Points where GNN+OSD is slightly worse (mark with ×)
worse_mask = gnn_osd_ler > bp_osd_ler

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.0, 5.5))

# Main curves
ax.errorbar(ps, bp_osd_ler, yerr=[bp_osd_lo, bp_osd_hi],
            fmt="o-", color="#0055D4", linewidth=1.8, markersize=7,
            capsize=4, capthick=0.9, label="Serial BP-OSD", zorder=5)
ax.errorbar(ps, gnn_osd_ler, yerr=[gnn_osd_lo, gnn_osd_hi],
            fmt="s-", color="#E69F00", linewidth=1.8, markersize=7,
            capsize=4, capthick=0.9, label="GNN + OSD", zorder=5)

# Mark slightly-worse points with red ×
ax.scatter(ps[worse_mask], gnn_osd_ler[worse_mask],
           marker="x", color="#B03030", s=160, linewidths=2.5, zorder=8)

ax.set_yscale("log")

# --- Zoom y-axis: use the data range with small padding ---
y_all = np.concatenate([bp_osd_ler, gnn_osd_ler])
ax.set_ylim(3e-4, 2e-1)

ax.set_xlim(0.008, 0.055)
ax.set_xlabel(r"Physical error rate $p$")
ax.set_ylabel("Logical Error Rate")
ax.set_title(r"$[\![72, 12, 6]\!]$, $\eta = 20$, serial BP-OSD, 50k shots")

# --- Legend (inside, upper-left, clear of data) ---
ax.legend(loc="upper left", fontsize=10, frameon=True,
          framealpha=0.9, edgecolor="grey")

# --- "slightly worse" annotation: placed BELOW the plot area (no overlap) ---
# Use figure-level text in the bottom margin
ax.annotate(
    r"$\times$ = GNN slightly worse",
    xy=(0.62, 0.03),          # bottom-right of axes, below data
    xycoords="axes fraction",
    fontsize=10,
    color="#B03030",
    ha="center",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#B03030",
              alpha=0.9, linewidth=0.8),
)

out = Path(__file__).parent / "fig_F4_phase4_ler_1.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
