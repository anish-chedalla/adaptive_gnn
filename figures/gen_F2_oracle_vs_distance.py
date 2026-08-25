#!/usr/bin/env python3
"""Generate fig_F2_oracle_vs_distance for a single column (~3.3 in) in a
two-column quantumarticle.

Data from Table 6 of the paper:
  Oracle gap (% LER reduction from per-qubit over mean-prior BP-OSD).
  All McNemar p << 1e-3.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Code distances and labels
distances = np.array([6, 12, 18])
code_labels = [r"$[\![72,12,6]\!]$", r"$[\![144,12,12]\!]$", r"$[\![288,12,18]\!]$"]

# % LER reduction per (code, sigma) — from Table 6
sigma_vals  = [0.25, 0.5, 1.0]
gap_data = {
    0.25: np.array([11, 16, 18]),
    0.5:  np.array([30, 38, 48]),
    1.0:  np.array([66, 74, 87]),
}

# Colorblind-safe palette (Okabe-Ito)
colors = {
    0.25: "#56B4E9",   # sky blue
    0.5:  "#0072B2",   # blue
    1.0:  "#D55E00",   # vermillion
}
markers = {0.25: "s", 0.5: "o", 1.0: "^"}
sigma_labels = {0.25: r"$\sigma=0.25$", 0.5: r"$\sigma=0.5$", 1.0: r"$\sigma=1.0$"}

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
    "lines.linewidth": 1.4,
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

fig, ax = plt.subplots(figsize=(3.3, 2.8))

for sigma in sigma_vals:
    gaps = gap_data[sigma]
    ax.plot(distances, gaps,
            marker=markers[sigma], color=colors[sigma],
            label=sigma_labels[sigma],
            linewidth=1.4, markersize=6,
            markeredgewidth=0.6, markeredgecolor="white",
            zorder=5 - sigma_vals.index(sigma))
    # Label the top point (sigma=1.0 only) for readability
    if sigma == 1.0:
        for x, y in zip(distances, gaps):
            ax.text(x + 0.4, y - 1, f"{y}%",
                    fontsize=6.5, color=colors[sigma],
                    ha="left", va="top")

ax.set_xticks(distances)
ax.set_xticklabels(code_labels, fontsize=7.5)
ax.set_xlabel("Code")
ax.set_ylabel("LER reduction vs mean-prior (%)")
ax.set_ylim(0, 100)
ax.legend(loc="upper left", handlelength=1.5, handletextpad=0.4)
ax.set_title(r"Oracle calibration gap (per-qubit $\to$ mean-prior)", pad=4)

out = Path(__file__).parent / "fig_F2_oracle_vs_distance.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
