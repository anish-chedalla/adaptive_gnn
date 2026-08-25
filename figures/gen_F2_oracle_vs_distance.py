#!/usr/bin/env python3
"""Generate fig_F2_oracle_vs_distance — full text width (6.5 in).

Professional PRL/Nature style:
  - Endpoint labels offset to the right of the last data point
  - No overlap: labels staggered vertically if needed
  - Legend top-left in the empty corner (low-sigma values)
  - Clean tick-in style axes
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

distances   = np.array([6, 12, 18])
code_labels = [r"$[\![72,12,6]\!]$", r"$[\![144,12,12]\!]$", r"$[\![288,12,18]\!]$"]

gap_data = {
    0.25: np.array([11, 16, 18]),
    0.5:  np.array([30, 38, 48]),
    1.0:  np.array([66, 74, 87]),
}
colors  = {0.25: "#56B4E9", 0.5: "#0072B2", 1.0: "#D55E00"}
markers = {0.25: "s",       0.5: "o",        1.0: "^"}
sigma_labels = {0.25: r"$\sigma=0.25$", 0.5: r"$\sigma=0.5$", 1.0: r"$\sigma=1.0$"}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    9.5,
    "axes.linewidth":    0.7,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "lines.linewidth":   1.6,
    "lines.markersize":  7,
    "legend.fontsize":   8.5,
    "legend.frameon":    True,
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#cccccc",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":      42,
})

fig, ax = plt.subplots(figsize=(6.5, 3.2))

# --- Plot lines ---
for sigma in [0.25, 0.5, 1.0]:
    gaps = gap_data[sigma]
    ax.plot(distances, gaps,
            marker=markers[sigma], color=colors[sigma],
            label=sigma_labels[sigma],
            linewidth=1.6, markersize=7,
            markeredgewidth=0.7, markeredgecolor="white",
            zorder=5)

# --- Endpoint labels: staggered offsets to avoid overlap ---
# At d=18: sigma=0.25→18%, 0.5→48%, 1.0→87%
# Vertical gaps: 18 vs 48 = 30pp apart, 48 vs 87 = 39pp apart — no collision
endpoint_offsets = {0.25: (+0.35, -1.5), 0.5: (+0.35, +1.5), 1.0: (+0.35, +1.5)}
for sigma in [0.25, 0.5, 1.0]:
    x_end = distances[-1]
    y_end = gap_data[sigma][-1]
    dx, dy = endpoint_offsets[sigma]
    ax.text(x_end + dx, y_end + dy, f"{y_end}%",
            fontsize=8, color=colors[sigma],
            ha="left", va="center", fontweight="bold")

# --- Axes ---
ax.set_xlim(4.5, 20.5)
ax.set_ylim(-2, 100)
ax.set_xticks(distances)
ax.set_xticklabels(code_labels, fontsize=9)
ax.set_xlabel("Code")
ax.set_ylabel("LER reduction over mean-prior decoder (%)")
ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
ax.set_title(r"Oracle calibration gap (per-qubit prior $\to$ mean prior)", pad=5)

# Legend in upper-left where data is sparse (low σ values start ~11%)
ax.legend(loc="upper left", handlelength=1.8, handletextpad=0.5,
          borderpad=0.6, labelspacing=0.4)

# Horizontal reference line at 0
ax.axhline(0, color="#aaaaaa", linewidth=0.5, zorder=0)

out = Path(__file__).parent / "fig_F2_oracle_vs_distance.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
