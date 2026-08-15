from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from eval_format_mvp.geometry import _geometry_fold, _transfer_metrics, analyze_geometry, factorial_components


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fixed_dimensional_controls(x: np.ndarray, rows: list[dict[str, object]]) -> dict[str, object]:
    dimension = 256
    random_results: dict[str, object] = {}
    for seed in (1729, 2718, 31415):
        rng = np.random.default_rng(seed)
        projection = rng.normal(size=(x.shape[1], dimension)).astype(np.float32) / np.sqrt(dimension)
        projected = x @ projection
        random_results[str(seed)] = {
            "geometry": analyze_geometry(projected, rows)["summary"],
            "transfer": _transfer_metrics(projected, rows),
        }

    families = np.asarray([str(row["purpose_family_id"]) for row in rows])
    pca_folds: list[dict[str, object]] = []
    for held in sorted(set(families.tolist())):
        training = families != held
        n_components = min(dimension, int(training.sum()) - 1, x.shape[1])
        reducer = PCA(n_components=n_components, svd_solver="randomized", random_state=1729)
        reducer.fit(x[training])
        projected = reducer.transform(x)
        component = factorial_components(projected, rows)
        pca_folds.append(_geometry_fold(component, held))
    keys = ("norm_A", "norm_B", "norm_C", "norm_C_over_norm_A", "delta_cosine", "purpose_direction_cosine")
    pca_summary = {
        key: float(np.nanmean([fold[key] for fold in pca_folds if fold[key] is not None]))
        if any(fold[key] is not None for fold in pca_folds) else None
        for key in keys
    }
    return {
        "dimension": dimension,
        "random_projection_seeds": [1729, 2718, 31415],
        "random_projection": random_results,
        "pca": {"fit_within_held_family_training_rows": True, "summary": pca_summary, "per_held_family": pca_folds},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--activations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--run-id", required=True)
    p.add_argument("--controls", action="store_true")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.items.open() if line.strip()]
    archive = np.load(args.activations, allow_pickle=False)
    if [str(v) for v in archive["item_ids"].tolist()] != [str(r["item_id"]) for r in rows]:
        raise ValueError("activation item_ids do not exactly match item metadata")
    x = np.asarray(archive["X"], dtype=np.float32)
    if x.ndim == 3:
        if args.layer is None:
            raise ValueError("--layer required for all-layer archive")
        x = x[:, args.layer, :]
    elif x.ndim != 2:
        raise ValueError(f"unexpected activation shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("non-finite activation")
    result = analyze_geometry(x, rows, layer=args.layer)
    result["transfer"] = _transfer_metrics(x, rows)
    result["fixed_dimensional_controls"] = fixed_dimensional_controls(x, rows) if args.controls else {
        "required": True, "implemented_in_this_run": False, "planned_dimension": 256,
        "random_projection_seeds": [1729, 2718, 31415],
    }
    result.update({
        "schema_version": 1,
        "analysis": "cross_fitted_factorial_geometry",
        "status": "development_diagnostic",
        "run_id": args.run_id,
        "git_commit": "4e64e93729026ce39c85e13800e1a40467e4f644",
        "input_artifacts": {"items": str(args.items), "items_sha256": sha256(args.items), "activations": str(args.activations), "activations_sha256": sha256(args.activations)},
        "model": {"name": "meta-llama/Llama-3.1-8B-Instruct", "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm", "token_position": "last_prompt_token_after_chat_template_before_generation", "dtype_inference": "bfloat16", "dtype_stored": "float32"},
    })
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
