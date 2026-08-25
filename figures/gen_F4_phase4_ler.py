#!/usr/bin/env python3
"""Generate fig_F4_phase4_ler — column width (3.3 in).

Fixes vs previous version:
  - All 4 p-values fully visible on log scale
  - Annotation uses mathtext italic (not LaTeX \emph)
  - Annotation placed in the UPPER area (not bottom, no clipping)
  - Proper padding so nothing is cut off
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Data from Table 5 ---
ps          = np.array([0.020, 0.030, 0.040, 0.050])
bp_osd_ler  = np.array([0.0046, 0.0377, 0.0831, 0.1515])
gnn_osd_ler = np.array([0.0050, 0.0399, 0.0876, 0.1555])

n, z = 8000, 1.96
def wilson_ci(p_hat):
    return z * np.sqrt(p_hat * (1 - p_hat) / n)

bp_ci  = wilson_ci(bp_osd_ler)
gnn_ci = wilson_ci(gnn_osd_ler)

mcn_p     = np.array([0.68, 0.63, 0.033, 3.6e-5])
worse_mask = mcn_p < 0.05   # p=0.040 and p=0.050

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         8,
    "axes.labelsize":    9,
    "axes.titlesize":    8.5,
    "axes.linewidth":    0.7,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "lines.linewidth":   1.4,
    "lines.markersize":  5.5,
    "legend.fontsize":   7.5,
    "legend.frameon":    True,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#cccccc",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.08,
    "pdf.fonttype":      42,
})

fig, ax = plt.subplots(figsize=(3.3, 3.3))

ax.errorbar(ps, bp_osd_ler, yerr=bp_ci,
            fmt="o-", color="#0055D4", capsize=3, capthick=0.8,
            linewidth=1.4, markersize=5.5,
            markeredgewidth=0.6, markeredgecolor="white",
            label="Serial BP-OSD", zorder=5)
ax.errorbar(ps, gnn_osd_ler, yerr=gnn_ci,
            fmt="s-", color="#E69F00", capsize=3, capthick=0.8,
            linewidth=1.4, markersize=5.5,
            markeredgewidth=0.6, markeredgecolor="white",
            label="GNN + OSD", zorder=5)

# Red × at points where GNN is significantly worse
ax.scatter(ps[worse_mask], gnn_osd_ler[worse_mask],
           marker="x", color="#B03030", s=80, linewidths=2.2, zorder=8)

ax.set_yscale("log")
ax.set_ylim(2e-3, 4e-1)
ax.set_xlim(0.014, 0.057)
ax.set_xlabel(r"Physical error rate $p$")
ax.set_ylabel("Logical Error Rate")
ax.set_title(r"$[\![72,12,6]\!]$, $\eta=20$, serial BP-OSD, 8k shots", pad=4)

# Legend in upper left
ax.legend(loc="upper left", handlelength=1.5, handletextpad=0.4)

# Annotation: upper-right area is empty (lines are at 0.004-0.05 for low p,
# upper right of the log plot has space above p=0.04–0.05 range).
# Place text in the middle-right where there's clear space.
ax.annotate(
    r"$\times$ = GNN $\mathit{worse}$ (McNemar $p\!<\!0.05$)",
    xy=(0.042, 0.088),
    xytext=(0.027, 0.22),
    fontsize=7,
    color="#B03030",
    ha="left",
    arrowprops=dict(arrowstyle="->", color="#B03030",
                    lw=0.8, shrinkA=4, shrinkB=4),
    bbox=dict(boxstyle="round,pad=0.25", fc="white",
              ec="#B03030", alpha=0.90, lw=0.7),
    zorder=9,
)

out = Path(__file__).parent / "fig_F4_phase4_ler.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
