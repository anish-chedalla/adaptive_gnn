#!/usr/bin/env python3
"""Reproduce fig_F1_baseline_decomp with non-overlapping ratio labels."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Data (read off from PDF) ---
labels = ["Flooding BP", "+ Serial\nschedule", "+ OSD-CS-10"]
lers   = [9.98e-3, 1.49e-3, 2.49e-4]
ci_lo  = [9.98e-3 - 9.40e-3, 1.49e-3 - 1.20e-3, 2.49e-4 - 1.00e-4]
ci_hi  = [1.06e-2 - 9.98e-3, 1.80e-3 - 1.49e-3, 6.00e-4 - 2.49e-4]
colors = ["#6BAED6", "#3182BD", "#2C5F7A"]

ratios  = [None, lers[0] / lers[1], lers[1] / lers[2]]   # 6.7×, 6.0×

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 11,
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

fig, ax = plt.subplots(figsize=(6.5, 5.0))

x = np.arange(len(labels))
bars = ax.bar(x, lers, color=colors, width=0.55,
              edgecolor="black", linewidth=0.8)
ax.errorbar(x, lers, yerr=[ci_lo, ci_hi], fmt="none",
            ecolor="black", capsize=5, capthick=1.0, linewidth=1.0)

ax.set_yscale("log")
ax.set_ylim(1.5e-4, 2e-2)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Logical Error Rate")
ax.set_title(r"$[\![288, 12, 18]\!]$, $p = 0.04$, $\eta = 20$, 100k shots")

# --- Ratio labels: place INSIDE each improved bar, near the top, left-aligned ---
# ratio[i] is the improvement going FROM bar i-1 TO bar i
for i, ratio in enumerate(ratios):
    if ratio is None:
        continue
    bar = bars[i]
    bx = bar.get_x() + bar.get_width() / 2   # bar centre x
    by = bar.get_height()                      # bar top (= LER value on log scale)

    # Place the label just below the top of the current bar,
    # nudged right so it sits clearly inside the bar body
    # Use a y-offset of 40% of the bar height (in log space) below the top
    y_label = by / 2.5   # geometric midpoint between top and ~floor

    ax.text(
        bx + 0.02,   # slight right-of-centre nudge
        y_label,
        f"{ratio:.1f}×",
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        color="white",
    )

out = Path(__file__).parent / "fig_F1_baseline_decomp_1.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
