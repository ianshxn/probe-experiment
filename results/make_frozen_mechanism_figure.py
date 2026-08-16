from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "runs/tae_2026/causal/causal_20260816_8b_bf16_mechanism_frozen"
OUT = ROOT / "paper/tae_2026_workshop/figures/07_causal_transport_hero.png"


def row(rows, layer, mode):
    return next(x for x in rows if x["layer"] == layer and x["mode"] == mode)


def main() -> None:
    selectivity = json.loads((BUNDLE / "selectivity.json").read_text())
    wrong = json.loads((BUNDLE / "wrong_sign_by_format.json").read_text())
    random = json.loads((BUNDLE / "random_controls.json").read_text())
    modes = ["purpose_full", "purpose_wrong_c", "purpose_norm_matched", "purpose_shuffled_c"]
    labels = ["correct C", "wrong-sign C", "norm-matched A", "shuffled C"]
    colors = ["#2166ac", "#b2182b", "#67a9cf", "#ef8a62"]
    x = np.arange(len(modes) + 1)
    width = 0.36
    fig, (ax, ax_fmt) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.45, 1]})
    for j, layer in enumerate((5, 16)):
        vals = [row(selectivity["rows"], layer, mode)["correct_target_improvement"] - row(selectivity["rows"], layer, "purpose_main")["correct_target_improvement"] for mode in modes]
        rand_vals = [v[f"L{layer}_C_benefit"] for v in random["values"]]
        vals.append(float(np.mean(rand_vals)))
        err = [0.0] * len(modes) + [float(np.std(rand_vals, ddof=0))]
        pos = x + (j - 0.5) * width
        bars = ax.bar(pos, vals, width, label=f"L{layer}", color=[colors[k] for k in range(len(modes))] + ["#777777"], alpha=0.9, yerr=err, capsize=2, edgecolor="black", linewidth=0.25)
        if j == 1:
            for bar, value in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, value + (0.0005 if value >= 0 else -0.0011), f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=6, rotation=90)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x, labels + ["random C"], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Target-distance improvement vs A-only")
    ax.set_title("Natural target transport")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.text(0.01, 0.98, "A-only = 0", transform=ax.transAxes, va="top", fontsize=7)
    fmt_rows = [r for r in wrong["rows"] if r["layer"] == 16]
    formats = ["benchmark", "casual"]
    fmt_x = np.arange(2)
    for condition, color in [("correct_sign_C", "#2166ac"), ("wrong_sign_C", "#b2182b")]:
        vals = [next(r["mean_C_increment"] for r in fmt_rows if r["format"] == fmt and r["condition"] == condition) for fmt in formats]
        offset = -0.18 if condition == "correct_sign_C" else 0.18
        ax_fmt.bar(fmt_x + offset, vals, 0.32, label="correct C" if condition == "correct_sign_C" else "wrong C", color=color, edgecolor="black", linewidth=0.25)
    ax_fmt.axhline(0, color="black", linewidth=0.7)
    ax_fmt.set_xticks(fmt_x, ["benchmark", "casual"], fontsize=8)
    ax_fmt.set_ylabel("C increment vs A-only", fontsize=8)
    ax_fmt.set_title("L16 sign control", fontsize=9)
    ax_fmt.legend(frameon=False, fontsize=7)
    ax_fmt.tick_params(axis="y", labelsize=7)
    fig.suptitle("BF16 internal counterfactual transport", fontsize=10, y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
