from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

INPUT_SHA = "ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022"
EXPECTED_MODELS = {
    "llama31_8b": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
    "llama31_70b": ("meta-llama/Llama-3.1-70B-Instruct", "1605565b47bb9346c5515c34102e054115b4f98b"),
    "llama33_70b": ("meta-llama/Llama-3.3-70B-Instruct", "6f6073b423013f6a7d4d9f39144961bfbfbc386b"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_geometry(path: Path, key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    model = payload.get("model", {})
    expected_model, expected_revision = EXPECTED_MODELS[key]
    if model.get("name") != expected_model or model.get("revision") != expected_revision:
        raise ValueError(f"{key}: model provenance mismatch: {model}")
    if payload.get("input_artifacts", {}).get("items_sha256") != INPUT_SHA:
        raise ValueError(f"{key}: input hash mismatch")
    if payload.get("status") != "development_diagnostic":
        raise ValueError(f"{key}: unexpected status")
    return payload


def load_equal_n(path: Path, key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    model = payload.get("model", {})
    expected_model, expected_revision = EXPECTED_MODELS[key]
    if model.get("name") != expected_model or model.get("revision") != expected_revision:
        raise ValueError(f"{key}: equal-N model provenance mismatch: {model}")
    if payload.get("input_artifacts", {}).get("items_sha256") != INPUT_SHA:
        raise ValueError(f"{key}: equal-N input hash mismatch")
    return payload


def curve(geometry: dict[str, Any], equal_n: dict[str, Any]) -> list[dict[str, Any]]:
    equal_by_layer = {int(row["layer"]): row for row in equal_n["layers_results"]}
    layers = [int(layer) for layer in geometry["layers"]]
    max_layer = max(layers) if layers else 1
    rows = []
    for item in geometry["layers_results"]:
        layer = int(item["layer"])
        summary = item["summary"]
        transfer = item["transfer"]
        equal = equal_by_layer[layer]["aggregate"]
        rows.append({
            "layer": layer,
            "relative_depth": layer / max_layer,
            "hidden_size": geometry["hidden_size"],
            "norm_C_over_norm_A": summary["norm_C_over_norm_A"],
            "delta_cosine": summary["delta_cosine"],
            "pooled_auc": transfer["pooled"]["auc"],
            "pooled_balanced_accuracy": transfer["pooled"]["balanced_accuracy"],
            "benchmark_to_casual_auc": transfer["benchmark_to_casual"]["auc"],
            "benchmark_to_casual_balanced_accuracy": transfer["benchmark_to_casual"]["balanced_accuracy"],
            "casual_to_benchmark_auc": transfer["casual_to_benchmark"]["auc"],
            "casual_to_benchmark_balanced_accuracy": transfer["casual_to_benchmark"]["balanced_accuracy"],
            "equal_n_mixed_auc": equal["equal_n_mixed"]["mean_auc"],
            "equal_n_mixed_balanced_accuracy": equal["equal_n_mixed"]["mean_balanced_accuracy"],
        })
    return rows


def plot(curves: dict[str, list[dict[str, Any]]], out_dir: Path, field: str, ylabel: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for key, rows in curves.items():
        ax.plot([r["relative_depth"] for r in rows], [r[field] for r in rows], label=key)
    ax.set_xlabel("Relative depth")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=220)
    plt.close(fig)

def write_report(curves: dict[str, list[dict[str, Any]]], geometries: dict[str, dict[str, Any]], equal_ns: dict[str, dict[str, Any]], output: Path) -> None:
    lines = [
        "# Matched three-model geometry and equal-N comparison",
        "",
        "Status: complete only when all three input payloads pass the frozen model, revision, and input-hash checks.",
        "",
        f"Input SHA256: `{INPUT_SHA}`",
        "",
        "| model | revision | hidden size | layers | max pooled AUC | max benchmark-to-casual AUC | max equal-N mixed AUC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("llama31_8b", "llama31_70b", "llama33_70b"):
        model = geometries[key]["model"]
        rows = curves[key]
        lines.append(
            f"| `{key}` | `{model['revision']}` | {geometries[key]['hidden_size']} | {len(rows)} | "
            f"{max(row['pooled_auc'] for row in rows):.4f} | "
            f"{max(row['benchmark_to_casual_auc'] for row in rows):.4f} | "
            f"{max(row['equal_n_mixed_auc'] for row in rows):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
        ]
    )
    for key in ("llama31_8b", "llama31_70b", "llama33_70b"):
        geometry_inputs = geometries[key]["input_artifacts"]
        equal_inputs = equal_ns[key].get("input_artifacts", {})
        lines.append(
            f"- `{key}` geometry activation SHA256 `{geometry_inputs.get('activations_sha256', 'missing')}`; "
            f"geometry item SHA256 `{geometry_inputs.get('items_sha256', 'missing')}`; "
            f"equal-N item SHA256 `{equal_inputs.get('items_sha256', 'missing')}`."
        )
    lines.extend(
        [
            "",
            "The three-model scale and generation comparisons are descriptive: 3.1-8B versus 3.1-70B changes scale within a model family, while 3.1-70B versus 3.3-70B changes generation/training at the same nominal size. Neither comparison is a perfectly causal estimate.",
            "",
            "The analysis fits no cross-model pooled probe and treats relative depth as a plotting coordinate, not an independent statistical unit.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", nargs=3, metavar=("8B", "31_70B", "33_70B"), type=Path, required=True)
    parser.add_argument("--equal-n", nargs=3, metavar=("8B", "31_70B", "33_70B"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    keys = ["llama31_8b", "llama31_70b", "llama33_70b"]
    geometries = {key: load_geometry(path, key) for key, path in zip(keys, args.geometry)}
    equal_ns = {key: load_equal_n(path, key) for key, path in zip(keys, args.equal_n)}
    curves = {key: curve(geometries[key], equal_ns[key]) for key in keys}
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.figures.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "status": "matched_three_model_complete",
        "analysis": "matched_full_depth_geometry_equal_n",
        "input_sha256": INPUT_SHA,
        "models": {key: geometries[key]["model"] for key in keys},
        "curves": curves,
        "scientific_boundary": "3.1B/3.1-70B is within-family scale; 3.1-70B/3.3-70B is same-size generation/training. Neither is perfectly causal.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    plot(curves, args.figures, "pooled_auc", "Pooled AUC", "pooled_auc_relative_depth.png")
    plot(curves, args.figures, "benchmark_to_casual_auc", "Benchmark to casual AUC", "benchmark_to_casual_auc_relative_depth.png")
    plot(curves, args.figures, "casual_to_benchmark_auc", "Casual to benchmark AUC", "casual_to_benchmark_auc_relative_depth.png")
    plot(curves, args.figures, "equal_n_mixed_auc", "Equal-N mixed AUC", "equal_n_auc_relative_depth.png")
    plot(curves, args.figures, "norm_C_over_norm_A", "norm(C) / norm(A)", "interaction_ratio_relative_depth.png")
    plot(curves, args.figures, "delta_cosine", "Purpose-delta cosine", "delta_cosine_relative_depth.png")
    write_report(curves, geometries, equal_ns, args.output.parent / "report.md")


if __name__ == "__main__":
    main()
