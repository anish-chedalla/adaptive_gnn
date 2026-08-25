#!/usr/bin/env python3
"""Generate fig_F2_oracle_vs_distance — column width (3.3 in).

Endpoint labels for sigma=1.0 only, placed to the right of the last
data point with enough vertical offset to avoid the line.
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

# Extra right margin for endpoint labels
fig, ax = plt.subplots(figsize=(3.3, 3.0))
fig.subplots_adjust(right=0.82)

for sigma in [0.25, 0.5, 1.0]:
    gaps = gap_data[sigma]
    ax.plot(distances, gaps,
            marker=markers[sigma], color=colors[sigma],
            label=sigma_labels[sigma],
            linewidth=1.5, markersize=6,
            markeredgewidth=0.6, markeredgecolor="white",
            zorder=5)

# Endpoint labels placed OUTSIDE the axes on the right
# Values at d=18: sigma=0.25→18%, 0.5→48%, 1.0→87% — well separated
label_y = {0.25: 18, 0.5: 48, 1.0: 87}
for sigma in [0.25, 0.5, 1.0]:
    ax.annotate(f"{label_y[sigma]}%",
                xy=(18, label_y[sigma]),
                xytext=(18.6, label_y[sigma]),
                fontsize=7.5, color=colors[sigma],
                fontweight="bold",
                ha="left", va="center",
                annotation_clip=False)

ax.set_xlim(4.5, 18.5)
ax.set_ylim(-2, 100)
ax.set_xticks(distances)
ax.set_xticklabels(code_labels, fontsize=7.5)
ax.set_xlabel("Code")
ax.set_ylabel("LER reduction over mean-prior (%)")
ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.yaxis.set_minor_locator(plt.MultipleLocator(10))
ax.set_title(r"Oracle calibration gap", pad=4)
ax.axhline(0, color="#cccccc", linewidth=0.5, zorder=0)

ax.legend(loc="upper left", handlelength=1.6, handletextpad=0.4,
          borderpad=0.5, labelspacing=0.35)

out = Path(__file__).parent / "fig_F2_oracle_vs_distance.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
