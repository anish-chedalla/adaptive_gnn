#!/usr/bin/env python3
"""Generate fig_F3_drift_adaptation for a single column (~3.3 in) in a
two-column quantumarticle.

Data from Table 7 of the paper (tab:drift):
  LER under per-qubit OU drift by volatility.
  [[72,12,6]], sigma=1.0, W=32-round window, p=0.04, 6000 shots.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Data from Table 7 ---
vols = np.array([0.00, 0.10, 0.35, 0.50])

oracle_ler = np.array([0.033, 0.031, 0.005, 0.004])
stale_ler  = np.array([0.033, 0.038, 0.027, 0.052])
empir_ler  = np.array([0.057, 0.057, 0.038, 0.035])
gnn_ler    = np.array([0.064, 0.067, 0.050, 0.044])

# Colorblind-safe palette
COLORS = {
    "oracle": "#009E73",   # green
    "stale":  "#999999",   # grey
    "empir":  "#E69F00",   # orange
    "gnn":    "#0072B2",   # blue
}

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
    "lines.linewidth": 1.3,
    "lines.markersize": 5.5,
    "legend.fontsize": 7.5,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "grey",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.3, 2.8))

ax.plot(vols, oracle_ler, "^-", color=COLORS["oracle"], label="Oracle",
        markeredgewidth=0.6, markeredgecolor="white", zorder=6)
ax.plot(vols, stale_ler,  "v--", color=COLORS["stale"],  label="Stale calib.",
        markeredgewidth=0.6, markeredgecolor="white", zorder=5)
ax.plot(vols, empir_ler,  "s-",  color=COLORS["empir"],  label="Empirical",
        markeredgewidth=0.6, markeredgecolor="white", zorder=4)
ax.plot(vols, gnn_ler,    "o-",  color=COLORS["gnn"],    label="GNN",
        markeredgewidth=0.6, markeredgecolor="white", zorder=7)

# Highlight the one point where GNN beats stale (vol=0.5)
ax.scatter([0.50], [0.044], marker="*", s=120,
           color=COLORS["gnn"], zorder=10, edgecolors="white", linewidths=0.5)

ax.set_xlabel("OU drift volatility")
ax.set_ylabel("Logical Error Rate")
ax.set_ylim(0, 0.085)
ax.set_xlim(-0.03, 0.57)
ax.set_xticks(vols)
ax.legend(loc="upper left", handlelength=1.5, ncol=2)
ax.set_title(
    r"$[\![72,12,6]\!]$, $p=0.04$, $\sigma=1.0$, 6k shots",
    pad=4,
)

# Annotate the GNN win
ax.annotate("GNN beats\nstale ($p{=}0.005$)",
            xy=(0.50, 0.044), xytext=(0.36, 0.068),
            fontsize=6.5, color=COLORS["gnn"],
            arrowprops=dict(arrowstyle="->", color=COLORS["gnn"],
                            lw=0.8, shrinkA=2, shrinkB=2),
            ha="center")

out = Path(__file__).parent / "fig_F3_drift_adaptation.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
