from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

INPUT_SHA = "ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022"
SEED = 2026081533


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    geometry = json.loads(args.geometry.read_text())
    if geometry.get("input_artifacts", {}).get("items_sha256") != INPUT_SHA:
        raise ValueError("geometry input hash does not match frozen bank")
    model = geometry.get("model", {})
    layers = [int(layer) for layer in geometry["layers"]]
    if len(layers) < 3:
        raise ValueError("need at least three layers for endpoint exclusion")
    excluded = {min(layers), max(layers)}
    eligible: list[dict[str, Any]] = []
    for row in geometry["layers_results"]:
        layer = int(row["layer"])
        pooled_auc = float(row["transfer"]["pooled"]["auc"])
        ratio = float(row["summary"]["norm_C_over_norm_A"])
        if layer not in excluded and pooled_auc >= 0.80:
            eligible.append({"layer": layer, "pooled_auc": pooled_auc, "norm_C_over_norm_A": ratio})
    if not eligible:
        raise RuntimeError("geometry-only eligibility rule selected no layer")
    shared = min(eligible, key=lambda row: (row["norm_C_over_norm_A"], row["layer"]))
    interaction = max(eligible, key=lambda row: (row["norm_C_over_norm_A"], -row["layer"]))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "protocol_frozen_before_endpoint_and_steering",
        "git_commit": args.git_commit,
        "geometry_sha256": sha256(args.geometry),
        "input_sha256": INPUT_SHA,
        "model": model,
        "protocol_seed": SEED,
        "layer_selection_rule": {
            "eligibility": "pooled purpose AUC >= 0.80; exclude first and last layer",
            "shared": "minimum norm(C)/norm(A) among eligible layers",
            "interaction": "maximum norm(C)/norm(A) among eligible layers",
        },
        "eligible_layers": eligible,
        "selected_layers": {"shared": int(shared["layer"]), "interaction": int(interaction["layer"])},
        "scientific_boundary": "Layer selection uses geometry only; no endpoint or intervention outcome was inspected.",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
