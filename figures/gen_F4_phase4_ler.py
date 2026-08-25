#!/usr/bin/env python3
"""Generate fig_F4_phase4_ler for a single column (~3.3 in) in a
two-column quantumarticle.

Data from Table 5 of the paper (tab:gnn_null):
  GNN-BP vs serial BP-OSD on [[72,12,6]], eta=20, 8000 shots.
  McNemar paired test on identical syndromes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Data from Table 5 (paper) ---
ps = np.array([0.020, 0.030, 0.040, 0.050])

bp_osd_ler = np.array([0.0046, 0.0377, 0.0831, 0.1515])
gnn_osd_ler = np.array([0.0050, 0.0399, 0.0876, 0.1555])

# Wilson 95% CI: ±z * sqrt(p_hat*(1-p_hat)/n) with n=8000
n = 8000
z = 1.96

def wilson_ci(p_hat, n, z=1.96):
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    return z * se

bp_ci  = wilson_ci(bp_osd_ler, n)
gnn_ci = wilson_ci(gnn_osd_ler, n)

# McNemar p-values from Table 5
mcn_p = np.array([0.68, 0.63, 0.033, 3.6e-5])

# GNN worse: p < 0.05
worse_mask = mcn_p < 0.05        # p=0.040 (p=0.033) and p=0.050 (p=3.6e-5)
marginal_mask = (mcn_p >= 0.05) & (mcn_p < 1.0)  # ties at p=0.02, 0.03

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
    "lines.markersize": 5,
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

fig, ax = plt.subplots(figsize=(3.3, 2.9))

ax.errorbar(ps, bp_osd_ler, yerr=bp_ci,
            fmt="o-", color="#0055D4", capsize=2.5, capthick=0.7,
            linewidth=1.2, markersize=5, label="Serial BP-OSD", zorder=5)
ax.errorbar(ps, gnn_osd_ler, yerr=gnn_ci,
            fmt="s-", color="#E69F00", capsize=2.5, capthick=0.7,
            linewidth=1.2, markersize=5, label="GNN + OSD", zorder=5)

# Red × where GNN is statistically worse (McNemar p < 0.05)
ax.scatter(ps[worse_mask], gnn_osd_ler[worse_mask],
           marker="x", color="#B03030", s=70, linewidths=2.0, zorder=8)

ax.set_yscale("log")
ax.set_ylim(2e-3, 3e-1)
ax.set_xlim(0.015, 0.055)
ax.set_xlabel(r"Physical error rate $p$")
ax.set_ylabel("Logical Error Rate")
ax.set_title(
    r"$[\![72, 12, 6]\!]$, $\eta=20$, serial BP-OSD, 8k shots",
    pad=4,
)

ax.legend(loc="upper left", handlelength=1.5)

ax.annotate(
    r"$\times$ = GNN \emph{worse} (McNemar $p < 0.05$)",
    xy=(0.55, 0.05),
    xycoords="axes fraction",
    fontsize=7,
    color="#B03030",
    ha="center",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#B03030",
              alpha=0.9, linewidth=0.7),
)

out = Path(__file__).parent / "fig_F4_phase4_ler.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
