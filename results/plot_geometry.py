from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    z = json.loads(args.results.read_text())["layers_results"]
    depth = np.asarray([q["layer"] for q in z])
    ratio = np.asarray([q["summary"]["norm_C_over_norm_A"] for q in z])
    b2c_auc = np.asarray([q["transfer"]["benchmark_to_casual"]["auc"] for q in z])
    b2c_ba = np.asarray([q["transfer"]["benchmark_to_casual"]["balanced_accuracy"] for q in z])
    gap = b2c_auc - b2c_ba
    delta_cos = np.asarray([q["summary"]["delta_cosine"] for q in z])
    overlap = np.asarray([np.nanmean(q["per_held_family"][0]["principal_cosines"]) for q in z])
    bproj = np.asarray([np.nanmean([f["B_projection"]["benchmark"] for f in q["per_held_family"] if f["B_projection"]["benchmark"] is not None]) for q in z])
    args.out_dir.mkdir(parents=True, exist_ok=False)

    def save(name: str, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
        ax.plot(x, y, marker="o", linewidth=1.5, markersize=3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        fig.savefig(args.out_dir / name, dpi=220)
        plt.close(fig)

    save("01_interaction_ratio_vs_depth.png", depth, ratio, "Transformer block", r"$\|C\|/\|A\|")
    save("02_cross_format_auc_vs_depth.png", depth, b2c_auc, "Transformer block", "Benchmark $\\to$ casual AUC")
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    ax.scatter(ratio, b2c_auc, c=depth, cmap="viridis", s=22)
    ax.set_xlabel(r"$\|C\|/\|A\|")
    ax.set_ylabel("Benchmark $\\to$ casual AUC")
    ax.grid(alpha=0.25)
    fig.savefig(args.out_dir / "03_interaction_ratio_vs_auc.png", dpi=220)
    plt.close(fig)
    save("04_B_projection_vs_calibration_gap.png", bproj, gap, r"$B$ projection", "AUC − balanced accuracy")
    save("05_delta_cosine_vs_depth.png", depth, delta_cos, "Transformer block", r"cosine($\delta_B,\delta_C$)")
    save("06_subspace_overlap_vs_depth.png", depth, overlap, "Transformer block", "Mean principal cosine")


if __name__ == "__main__":
    main()
