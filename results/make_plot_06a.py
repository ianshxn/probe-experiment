"""06a: single-panel 70B explainer for plot 06 — the two source curves and the gap."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "/home/cfeng/Projects/personal/probes/plots"
R = "/home/cfeng/Projects/personal/probes/results"
SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"; S70 = "#eb6834"
plt.rcParams.update({"font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": BASE, "font.size": 10})

d70 = json.load(open(f"{R}/layer_sweep_llama33_70b.json"))
x = np.array([r["layer"] / (len(d70) - 1) * 100 for r in d70])
dec = np.array([r["decorrelated"]["auc"] for r in d70])
xf = np.array([(r["b2c"]["auc"] + r["c2b"]["auc"]) / 2 for r in d70])

fig, ax = plt.subplots(figsize=(9.6, 5.6))
ax.plot(x, dec, color=S70, lw=2, zorder=3)
ax.plot(x, xf, color=S70, lw=2, ls=(0, (4, 3)), zorder=3)
ax.axhline(0.5, color=BASE, lw=1.2, ls=":", zorder=1)
ax.text(1, 0.508, "chance", ha="left", va="bottom", color=MUTED, fontsize=9)

# direct labels on the two curves
ax.text(52, 0.965, "trained on both formats (plot 01, decorrelated)", color=S70,
        fontsize=9.5, fontweight="bold", ha="left", va="bottom")
ax.text(38, 0.655, "trained on one format, tested on the other\n(mean of plot 03's two directions)",
        color=S70, fontsize=9.5, fontweight="bold", ha="left", va="top")

# the gap arrow, at a depth where the two curves are far apart
ia = 60  # layer 60 of 0-79
ax.annotate("", xy=(x[ia], dec[ia]), xytext=(x[ia], xf[ia]),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.6), zorder=4)
ax.text(x[ia] + 1.6, (dec[ia] + xf[ia]) / 2,
        "gap between the two\ntraining conditions",
        color=INK2, fontsize=9.5, ha="left", va="center")

# the near-touch point: almost everything transfers
i19 = 19
ax.plot([x[i19]], [dec[i19]], "o", ms=8, color=S70, mec=SURF, mew=2, zorder=4)
ax.plot([x[i19]], [xf[i19]], "o", ms=8, color=S70, mec=SURF, mew=2, zorder=4)
ax.text(26.5, 0.935, f"layer 19 · gap {dec[i19]-xf[i19]:.2f}\nnearly everything transfers",
        color=S70, fontsize=9.5, fontweight="bold", ha="left", va="center")

ax.set_xlabel("relative depth through the network (%)")
ax.set_ylabel("purpose AUC (out-of-fold)")
ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(BASE)
ax.set_ylim(0.45, 1.02)

fig.legend(handles=[Line2D([], [], color=S70, lw=2, label="trained on both formats"),
                    Line2D([], [], color=S70, lw=2, ls=(0, (4, 3)), label="trained on one format, tested on the other")],
           loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=2, frameon=False,
           fontsize=9.5, labelcolor=INK2, handlelength=1.8, columnspacing=1.4)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{OUT}/06a_signal_split_explainer.png", dpi=200)
plt.close(fig)
print("wrote 06a")
