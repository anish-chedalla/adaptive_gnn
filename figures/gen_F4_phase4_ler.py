#!/usr/bin/env python3
"""Generate fig_F4_phase4_ler sized for a single column (~3.3 in) in a
two-column quantumarticle.  Fixes:
  - "slightly worse" instead of "significantly worse"
  - annotation boxed below data, no overlap
  - y-axis zoomed to show curve separation
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Data ---
ps = np.array([0.010, 0.020, 0.030, 0.040, 0.050])

bp_osd_ler  = np.array([6.8e-4, 7.8e-3, 2.65e-2, 6.87e-2, 1.12e-1])
bp_osd_lo   = np.array([1.4e-4, 3.0e-4, 5.0e-4,  3.0e-3,  4.0e-3])
bp_osd_hi   = np.array([1.4e-4, 3.0e-4, 5.0e-4,  3.0e-3,  4.0e-3])

gnn_osd_ler = np.array([5.2e-4, 8.2e-3, 2.65e-2, 7.10e-2, 1.14e-1])
gnn_osd_lo  = np.array([1.2e-4, 3.2e-4, 5.0e-4,  3.0e-3,  4.0e-3])
gnn_osd_hi  = np.array([1.2e-4, 3.2e-4, 5.0e-4,  3.0e-3,  4.0e-3])

worse_mask = gnn_osd_ler > bp_osd_ler

# Style: 8-9 pt fonts, column-width figure (3.3 in)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 8.5,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.3, 2.9))

ax.errorbar(ps, bp_osd_ler, yerr=[bp_osd_lo, bp_osd_hi],
            fmt="o-", color="#0055D4", capsize=2.5, capthick=0.7,
            linewidth=1.2, markersize=4.5, label="Serial BP-OSD", zorder=5)
ax.errorbar(ps, gnn_osd_ler, yerr=[gnn_osd_lo, gnn_osd_hi],
            fmt="s-", color="#E69F00", capsize=2.5, capthick=0.7,
            linewidth=1.2, markersize=4.5, label="GNN + OSD", zorder=5)

# Red x where GNN is slightly worse
ax.scatter(ps[worse_mask], gnn_osd_ler[worse_mask],
           marker="x", color="#B03030", s=60, linewidths=1.8, zorder=8)

ax.set_yscale("log")
ax.set_ylim(3e-4, 2e-1)
ax.set_xlim(0.008, 0.054)
ax.set_xlabel(r"Physical error rate $p$")
ax.set_ylabel("Logical Error Rate")
ax.set_title(
    r"$[\![72, 12, 6]\!]$, $\eta=20$, serial BP-OSD, 50k shots",
    pad=4,
)

ax.legend(loc="upper left", fontsize=7.5, frameon=True,
          framealpha=0.9, edgecolor="grey", handlelength=1.5)

# Annotation: below the data, no overlap
ax.annotate(
    r"$\times$ = GNN slightly worse",
    xy=(0.60, 0.05),
    xycoords="axes fraction",
    fontsize=7.5,
    color="#B03030",
    ha="center",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#B03030",
              alpha=0.9, linewidth=0.7),
)

# Save with name that matches \includegraphics{fig_F4_phase4_ler} in LaTeX
out = Path(__file__).parent / "fig_F4_phase4_ler.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
