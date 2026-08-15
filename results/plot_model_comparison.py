from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    curves = {
        "Llama-3.1-8B": load(Path("results/layer_sweep_llama31_8b.json")),
        "Llama-3.3-70B": load(Path("results/layer_sweep_llama33_70b.json")),
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for key, regime, label in (("pooled", "decorrelated", "Pooled AUC"), ("b2c", "b2c", "Benchmark → casual AUC"), ("c2b", "c2b", "Casual → benchmark AUC")):
        fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
        for name, rows in curves.items():
            depth = np.asarray([row["layer"] / max(1, rows[-1]["layer"]) for row in rows])
            ax.plot(depth, [row[regime]["auc"] for row in rows], label=name)
        ax.set_xlabel("Relative depth")
        ax.set_ylabel(label)
        ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(args.out_dir / f"{key}_auc_relative_depth.png", dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    main()
