from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval_format_mvp.geometry import _transfer_metrics, analyze_geometry
from run_geometry import fixed_dimensional_controls


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--activations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--controls", action="store_true")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.items.open() if line.strip()]
    archive = np.load(args.activations, allow_pickle=False)
    x = np.asarray(archive["X"], dtype=np.float32)
    layers = [int(value) for value in archive["layers"].tolist()]
    if x.ndim != 3 or x.shape[1] != len(layers):
        raise ValueError(f"expected N x L x D all-layer archive, got {x.shape}")
    if [str(v) for v in archive["item_ids"].tolist()] != [str(r["item_id"]) for r in rows]:
        raise ValueError("activation item_ids do not exactly match metadata")
    if not np.isfinite(x).all():
        raise ValueError("non-finite activation")
    results = []
    for offset, layer in enumerate(layers):
        layer_x = x[:, offset, :]
        result = analyze_geometry(layer_x, rows, layer=layer)
        result["transfer"] = _transfer_metrics(layer_x, rows)
        if args.controls:
            result["fixed_dimensional_controls"] = fixed_dimensional_controls(layer_x, rows)
        results.append(result)
    payload = {
        "schema_version": 1,
        "analysis": "cross_fitted_factorial_geometry_all_layers",
        "status": "development_diagnostic",
        "run_id": args.run_id,
        "git_commit": "4e64e93729026ce39c85e13800e1a40467e4f644",
        "layers": layers,
        "hidden_size": int(x.shape[2]),
        "input_artifacts": {"items": str(args.items), "items_sha256": sha256(args.items), "activations": str(args.activations), "activations_sha256": sha256(args.activations)},
        "model": {"name": "meta-llama/Llama-3.1-8B-Instruct", "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm", "token_position": "last_prompt_token_after_chat_template_before_generation", "dtype_inference": "bfloat16", "dtype_stored": "float32"},
        "fixed_dimensional_controls": "per-layer 256-D random projections (three fixed seeds) and training-fold PCA" if args.controls else "not run",
        "layers_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
