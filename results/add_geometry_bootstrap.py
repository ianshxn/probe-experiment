from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    payload = json.loads(args.input.read_text())
    rng = np.random.default_rng(20260815)
    keys = ("norm_A", "norm_B", "norm_C", "norm_C_over_norm_A", "delta_cosine", "purpose_direction_cosine")
    for layer_result in payload["layers_results"]:
        folds = layer_result["per_held_family"]
        bands: dict[str, list[float]] = {}
        for key in keys:
            values = np.asarray([fold[key] for fold in folds if fold[key] is not None], dtype=float)
            samples = np.asarray([np.nanmean(values[rng.integers(0, len(values), len(values))]) for _ in range(1000)])
            bands[key] = [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
        layer_result["family_bootstrap_conditional_on_fold_fits"] = {"seed": 20260815, "n_resamples": 1000, "unit": "purpose_family_id", "ci95": bands}
    payload["run_id"] = "geometry_20260815_8b_all_layers_v3_bootstrap"
    payload["status"] = "development_diagnostic_with_conditional_family_bootstrap"
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
