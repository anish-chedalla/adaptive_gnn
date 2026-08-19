#!/usr/bin/env python3
"""Generate fig_F1_baseline_decomp sized for a single column (~3.3 in) in
a two-column quantumarticle.  Ratio labels are placed inside each bar as
centred white text so they never overlap.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- Data ---
labels = ["Flooding BP", "+ Serial\nschedule", "+ OSD-CS-10"]
lers   = [9.98e-3, 1.49e-3, 2.49e-4]
ci_lo  = [9.98e-3 - 9.40e-3, 1.49e-3 - 1.20e-3, 2.49e-4 - 1.00e-4]
ci_hi  = [1.06e-2 - 9.98e-3, 1.80e-3 - 1.49e-3, 6.00e-4 - 2.49e-4]
colors = ["#6BAED6", "#3182BD", "#2C5F7A"]
ratios = [None, lers[0] / lers[1], lers[1] / lers[2]]   # 6.7x, 6.0x

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
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(3.3, 2.7))   # matches \columnwidth in a4 twocolumn

x = np.arange(len(labels))
bars = ax.bar(x, lers, color=colors, width=0.55,
              edgecolor="black", linewidth=0.6)
ax.errorbar(x, lers, yerr=[ci_lo, ci_hi], fmt="none",
            ecolor="black", capsize=3, capthick=0.7, linewidth=0.7)

ax.set_yscale("log")
ax.set_ylim(1.5e-4, 2e-2)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Logical Error Rate")
ax.set_title(
    r"$[\![288, 12, 18]\!]$, $p=0.04$, $\eta=20$, 100k shots",
    pad=4,
)

# Ratio labels inside bars (white, centred)
for i, ratio in enumerate(ratios):
    if ratio is None:
        continue
    bx = bars[i].get_x() + bars[i].get_width() / 2
    by = bars[i].get_height()
    ax.text(bx, by / 2.5, f"{ratio:.1f}×",
            ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

# Save with name that matches \includegraphics{fig_F1_baseline_decomp} in LaTeX
out = Path(__file__).parent / "fig_F1_baseline_decomp.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
