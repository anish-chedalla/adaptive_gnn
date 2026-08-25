#!/usr/bin/env python3
"""Generate fig_F3_drift_adaptation — column width (3.3 in).

Legend placed BELOW the axes (ncol=2) so it never covers the lines.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

vols = np.array([0.00, 0.10, 0.35, 0.50])
oracle_ler = np.array([0.033, 0.031, 0.005, 0.004])
stale_ler  = np.array([0.033, 0.038, 0.027, 0.052])
empir_ler  = np.array([0.057, 0.057, 0.038, 0.035])
gnn_ler    = np.array([0.064, 0.067, 0.050, 0.044])

COLORS = {
    "oracle": "#009E73",
    "stale":  "#999999",
    "empir":  "#E69F00",
    "gnn":    "#0072B2",
}

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
    "lines.linewidth":   1.5,
    "lines.markersize":  6,
    "legend.fontsize":   7.5,
    "legend.frameon":    True,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#cccccc",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":      42,
})

# Extra bottom margin to hold the below-axis legend
fig, ax = plt.subplots(figsize=(3.3, 3.6))
fig.subplots_adjust(bottom=0.30)

ax.plot(vols, oracle_ler, "^-",  color=COLORS["oracle"], label="Oracle",
        markeredgewidth=0.6, markeredgecolor="white", zorder=6)
ax.plot(vols, stale_ler,  "v--", color=COLORS["stale"],  label="Stale calib.",
        markeredgewidth=0.6, markeredgecolor="white", zorder=5)
ax.plot(vols, empir_ler,  "s-",  color=COLORS["empir"],  label="Empirical",
        markeredgewidth=0.6, markeredgecolor="white", zorder=4)
ax.plot(vols, gnn_ler,    "o-",  color=COLORS["gnn"],    label="GNN",
        markeredgewidth=0.6, markeredgecolor="white", zorder=7)

# Star at the one GNN-wins point
ax.scatter([0.50], [0.044], marker="*", s=160,
           color=COLORS["gnn"], zorder=10,
           edgecolors="white", linewidths=0.6)

# Annotation: upper-right quadrant is clear (oracle drops to near 0 there,
# stale shoots up, so there is open space between stale ~0.052 and top of axes)
ax.annotate(r"GNN beats stale ($p{=}0.005$)",
            xy=(0.50, 0.044), xytext=(0.295, 0.073),
            fontsize=7, color=COLORS["gnn"],
            arrowprops=dict(arrowstyle="->", color=COLORS["gnn"],
                            lw=0.8, shrinkA=4, shrinkB=4),
            ha="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="white",
                      ec=COLORS["gnn"], alpha=0.90, lw=0.7))

ax.set_xlabel("OU drift volatility")
ax.set_ylabel("Logical Error Rate")
ax.set_ylim(-0.002, 0.082)
ax.set_xlim(-0.03, 0.58)
ax.set_xticks(vols)
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
ax.set_title(
    r"$[\![72,12,6]\!]$, $p=0.04$, $\sigma_\mathrm{OU}=1.0$, 6,000 shots",
    pad=4,
)

# Legend BELOW the axes in 2 columns — completely outside the plot area
ax.legend(loc="upper center",
          bbox_to_anchor=(0.5, -0.24),
          ncol=2,
          handlelength=1.5,
          handletextpad=0.4,
          columnspacing=0.8,
          borderpad=0.5)

out = Path(__file__).parent / "fig_F3_drift_adaptation.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
