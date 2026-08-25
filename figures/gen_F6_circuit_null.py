#!/usr/bin/env python3
"""Generate fig_F6_circuit_null — column width (3.3 in).

Professional PRL/Nature style:
  - McNemar p-values placed cleanly ABOVE each bar pair
  - No overlapping text boxes
  - "No gap" conclusion in a clean subtitle (axes title), not an in-plot box
  - Error bars sized correctly for n=5000
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

p_means    = np.array([0.001, 0.003])
mean_ler   = np.array([0.272, 0.335])
oracle_ler = np.array([0.270, 0.335])
mcn_p      = np.array([0.25, 1.00])
n_shots    = 5000

z = 1.96
mean_ci   = z * np.sqrt(mean_ler   * (1 - mean_ler)   / n_shots)
oracle_ci = z * np.sqrt(oracle_ler * (1 - oracle_ler) / n_shots)

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         8,
    "axes.labelsize":    9,
    "axes.titlesize":    8,
    "axes.linewidth":    0.7,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
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

x     = np.arange(len(p_means))
width = 0.32

fig, ax = plt.subplots(figsize=(3.3, 3.2))

bars1 = ax.bar(x - width/2, mean_ler, width,
               color="#0072B2", alpha=0.85, label="Mean-prior BP-OSD",
               edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width/2, oracle_ler, width,
               color="#E69F00", alpha=0.85, label="Oracle BP-OSD",
               edgecolor="white", linewidth=0.5)

ax.errorbar(x - width/2, mean_ler, yerr=mean_ci,
            fmt="none", ecolor="#333333", capsize=3.5, capthick=0.8, linewidth=0.8)
ax.errorbar(x + width/2, oracle_ler, yerr=oracle_ci,
            fmt="none", ecolor="#333333", capsize=3.5, capthick=0.8, linewidth=0.8)

# McNemar p-values as clean text ABOVE each bar-pair — bracketed notation
bracket_top = np.maximum(mean_ler + mean_ci, oracle_ler + oracle_ci) + 0.008
for i, (xi, p) in enumerate(zip(x, mcn_p)):
    pstr = f"$p={p:.2f}$"
    ax.text(xi, bracket_top[i], pstr,
            ha="center", va="bottom", fontsize=7.5,
            color="#444444")
    # Short bracket spanning both bars
    bx = [xi - width/2 - 0.02, xi - width/2 - 0.02,
          xi + width/2 + 0.02, xi + width/2 + 0.02]
    by = [bracket_top[i] - 0.004, bracket_top[i] - 0.001,
          bracket_top[i] - 0.001, bracket_top[i] - 0.004]
    ax.plot(bx, by, color="#888888", lw=0.7, clip_on=False)

ax.set_xticks(x)
ax.set_xticklabels([r"$p_\mathrm{mean}=0.001$",
                    r"$p_\mathrm{mean}=0.003$"], fontsize=8)
ax.set_ylabel("Logical Error Rate")
ax.set_ylim(0, 0.42)
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.02))

# Two-line title: result first, context second
ax.set_title("Circuit-level oracle gap: no significant difference\n"
             r"$[\![72,12,6]\!]$, 6 rounds, $\eta=20$, 5{,}000 shots",
             pad=4, fontsize=7.5)

ax.legend(loc="lower right", handlelength=1.3,
          handletextpad=0.4, borderpad=0.5)

out = Path(__file__).parent / "fig_F6_circuit_null.pdf"
fig.savefig(out)
plt.close(fig)
print(f"Saved {out}")
