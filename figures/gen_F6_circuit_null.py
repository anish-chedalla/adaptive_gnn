#!/usr/bin/env python3
"""Generate fig_F6_circuit_null for a single column (~3.3 in) in a
two-column quantumarticle.

Data from Table 8 of the paper (tab:circuit):
  Circuit-level oracle gap on [[72,12,6]], 6 rounds, eta=20, 5000 shots/point.
  Mean-prior vs per-qubit-oracle BP-OSD are statistically indistinguishable.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Data from Table 8 ---
p_means = np.array([0.001, 0.003])
mean_ler   = np.array([0.272, 0.335])
oracle_ler = np.array([0.270, 0.335])
mcn_p      = np.array([0.25, 1.00])
n_shots = 5000

z = 1.96
mean_ci   = z * np.sqrt(mean_ler   * (1 - mean_ler)   / n_shots)
oracle_ci = z * np.sqrt(oracle_ler * (1 - oracle_ler) / n_shots)

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
    "lines.markersize": 6,
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

x = np.arange(len(p_means))
width = 0.32

fig, ax = plt.subplots(figsize=(3.3, 2.8))

bars1 = ax.bar(x - width/2, mean_ler, width,
               color="#0072B2", alpha=0.85, label="Mean-prior BP-OSD",
               edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width/2, oracle_ler, width,
               color="#E69F00", alpha=0.85, label="Oracle BP-OSD",
               edgecolor="white", linewidth=0.5)

ax.errorbar(x - width/2, mean_ler, yerr=mean_ci,
            fmt="none", ecolor="black", capsize=3, capthick=0.7, linewidth=0.7)
ax.errorbar(x + width/2, oracle_ler, yerr=oracle_ci,
            fmt="none", ecolor="black", capsize=3, capthick=0.7, linewidth=0.7)

# Annotate McNemar p-values
for i, (xi, p) in enumerate(zip(x, mcn_p)):
    label = f"McNemar $p={p:.2f}$" if p > 0.01 else f"McNemar $p={p:.3f}$"
    ax.text(xi, max(mean_ler[i], oracle_ler[i]) + 0.008, label,
            ha="center", va="bottom", fontsize=6.5, color="grey")

ax.set_xticks(x)
ax.set_xticklabels([f"$p_{{\\mathrm{{mean}}}}={p}$" for p in p_means])
ax.set_ylabel("Logical Error Rate")
ax.set_ylim(0, 0.42)
ax.legend(loc="lower right", handlelength=1.2)
ax.set_title(
    r"Circuit-level oracle gap: $[\![72,12,6]\!]$, 6 rounds",
    pad=4,
)

# "No gap" annotation
ax.text(0.50, 0.88, "No statistically significant gap\nat circuit level",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=7, color="#B03030",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#B03030", alpha=0.85, linewidth=0.7))

out = Path(__file__).parent / "fig_F6_circuit_null.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
