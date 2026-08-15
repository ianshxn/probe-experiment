from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

INPUT_SHA = "ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def legacy_curve(path: Path, model: str, revision: str) -> dict[str, Any]:
    rows = json.loads(path.read_text())
    return {
        "model": model,
        "revision": revision,
        "source": str(path),
        "source_sha256": sha256(path),
        "input_sha256": INPUT_SHA,
        "layers": [
            {
                "layer": int(row["layer"]),
                "confounded_auc": float(row["confounded"]["auc"]),
                "pooled_auc": float(row["decorrelated"]["auc"]),
                "benchmark_to_casual_auc": float(row["b2c"]["auc"]),
                "casual_to_benchmark_auc": float(row["c2b"]["auc"]),
                "pooled_balanced_accuracy": float(row["decorrelated"]["bal_acc"]),
                "benchmark_to_casual_balanced_accuracy": float(row["b2c"]["bal_acc"]),
                "casual_to_benchmark_balanced_accuracy": float(row["c2b"]["bal_acc"]),
            }
            for row in rows
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--llama31-70b-dir", type=Path, default=None)
    args = p.parse_args()
    models = {
        "llama31_8b": legacy_curve(Path("results/layer_sweep_llama31_8b.json"), "meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
        "llama33_70b": legacy_curve(Path("results/layer_sweep_llama33_70b.json"), "meta-llama/Llama-3.3-70B-Instruct", "6f6073b423013f6a7d4d9f39144961bfbfbc386b"),
    }
    models["llama31_8b"]["equal_n_source"] = "runs/tae_2026/equal_n/equal_n_20260815_8b_all_layers_v1/results.json"
    models["llama31_8b"]["geometry_source"] = "runs/tae_2026/geometry/geometry_20260815_8b_all_layers_v3_bootstrap/results.json"
    if args.llama31_70b_dir is not None and (args.llama31_70b_dir / "config.json").exists():
        config = json.loads((args.llama31_70b_dir / "config.json").read_text())
        models["llama31_70b"] = {
            "model": config["model"], "revision": config["model_revision"], "input_sha256": config.get("items_sha256"),
            "activation_sha256": config.get("activations_all_layers_sha256"), "status": "extraction_present; matched analyses pending",
        }
    else:
        models["llama31_70b"] = {
            "model": "meta-llama/Llama-3.1-70B-Instruct", "revision": "1605565b47bb9346c5515c34102e054115b4f98b",
            "input_sha256": INPUT_SHA, "status": "pending_slurm_51479700",
        }
    payload = {
        "schema_version": 1,
        "run_id": args.output.parent.name,
        "status": "partial_until_llama31_70b_and_matched_controls_complete",
        "git_commit": "4e64e93729026ce39c85e13800e1a40467e4f644",
        "same_input_required": INPUT_SHA,
        "models": models,
        "interpretation": "3.1-8B vs 3.1-70B is within-family scale; 3.1-70B vs 3.3-70B is same-size generation/training. Neither comparison is perfectly causal.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
