from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from eval_format_mvp.geometry import factorial_components


RUN_ID = "counterfactual_reconstruction_20260816_8b_v1"
LAYERS_OF_INTEREST = (5, 16)
RANDOM_SEEDS = (1729, 2718, 31415)
BOOTSTRAP_SEED = 2026081601
BOOTSTRAP_REPS = 10000
DIRECTIONS = (
    "purpose_db_to_eb",
    "purpose_dc_to_ec",
    "purpose_eb_to_db",
    "purpose_ec_to_dc",
    "format_eval_b_to_c",
    "format_deploy_b_to_c",
    "format_eval_c_to_b",
    "format_deploy_c_to_b",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def direction_arrays(h: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # h columns are EB, EC, DB, DC. The vectors are target minus source.
    return {
        "purpose_db_to_eb": (h[:, 2] + 2.0 * (a + c), h[:, 0]),
        "purpose_dc_to_ec": (h[:, 3] + 2.0 * (a - c), h[:, 1]),
        "purpose_eb_to_db": (h[:, 0] - 2.0 * (a + c), h[:, 2]),
        "purpose_ec_to_dc": (h[:, 1] - 2.0 * (a - c), h[:, 3]),
        "format_eval_b_to_c": (h[:, 0] - 2.0 * (b + c), h[:, 1]),
        "format_deploy_b_to_c": (h[:, 2] - 2.0 * (b - c), h[:, 3]),
        "format_eval_c_to_b": (h[:, 1] + 2.0 * (b + c), h[:, 0]),
        "format_deploy_c_to_b": (h[:, 3] + 2.0 * (b - c), h[:, 2]),
    }


def main_only_arrays(h: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "purpose_db_to_eb": (h[:, 2] + 2.0 * a, h[:, 0]),
        "purpose_dc_to_ec": (h[:, 3] + 2.0 * a, h[:, 1]),
        "purpose_eb_to_db": (h[:, 0] - 2.0 * a, h[:, 2]),
        "purpose_ec_to_dc": (h[:, 1] - 2.0 * a, h[:, 3]),
        "format_eval_b_to_c": (h[:, 0] - 2.0 * b, h[:, 1]),
        "format_deploy_b_to_c": (h[:, 2] - 2.0 * b, h[:, 3]),
        "format_eval_c_to_b": (h[:, 1] + 2.0 * b, h[:, 0]),
        "format_deploy_c_to_b": (h[:, 3] + 2.0 * b, h[:, 2]),
    }


def summarize_prediction(pred: np.ndarray, target: np.ndarray, source: np.ndarray, scale: float) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    for p, t, s in zip(pred, target, source):
        displacement = t - s
        predicted_displacement = p - s
        raw_error = float(np.linalg.norm(p - t))
        rows.append(
            {
                "raw_euclidean_error": raw_error,
                "normalized_euclidean_error": raw_error / scale if scale else None,
                "cosine_to_true_target": cosine(p, t),
                "cosine_predicted_vs_true_displacement": cosine(predicted_displacement, displacement),
            }
        )
    return rows


def mean_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in records if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in ("full", "main"):
        subset = [r for r in records if r["method"] == method]
        out[method] = {
            "n": len(subset),
            "raw_euclidean_error": mean_metric(subset, "raw_euclidean_error"),
            "normalized_euclidean_error": mean_metric(subset, "normalized_euclidean_error"),
            "cosine_to_true_target": mean_metric(subset, "cosine_to_true_target"),
            "cosine_predicted_vs_true_displacement": mean_metric(subset, "cosine_predicted_vs_true_displacement"),
        }
    full = out["full"]["raw_euclidean_error"]
    main = out["main"]["raw_euclidean_error"]
    out["C_benefit_raw_error"] = None if full is None or main is None else float(main - full)
    full_norm = out["full"]["normalized_euclidean_error"]
    main_norm = out["main"]["normalized_euclidean_error"]
    out["C_benefit_normalized_error"] = None if full_norm is None or main_norm is None else float(main_norm - full_norm)
    return out


def bootstrap_difference(raw_records: list[dict[str, Any]], layer_a: int, layer_b: int) -> dict[str, Any]:
    # Stratified complete-payload bootstrap: blocks are resampled within held-out family.
    purpose_directions = DIRECTIONS[:4]
    by_layer: dict[int, dict[tuple[str, str], dict[str, dict[str, float]]]] = {}
    for layer in (layer_a, layer_b):
        for record in raw_records:
            if record["layer"] != layer or record["direction"] not in purpose_directions:
                continue
            key = (record["held_out_family"], record["payload_block_id"])
            by_layer.setdefault(layer, {}).setdefault(key, {}).setdefault(record["direction"], {})[record["method"]] = record["raw_euclidean_error"]
    keys_by_family: dict[str, list[tuple[str, str]]] = {}
    for key in by_layer[layer_a]:
        keys_by_family.setdefault(key[0], []).append(key)

    def benefit(layer: int, key: tuple[str, str]) -> float:
        values = by_layer[layer][key]
        per_direction = [values[d]["main"] - values[d]["full"] for d in purpose_directions]
        return float(np.mean(per_direction))

    observed_values = [float(np.mean([benefit(layer, key) for key in by_layer[layer]])) for layer in (layer_a, layer_b)]
    observed = observed_values[1] - observed_values[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for i in range(BOOTSTRAP_REPS):
        selected = []
        for family, keys in keys_by_family.items():
            indices = rng.integers(0, len(keys), size=len(keys))
            selected.extend(keys[int(index)] for index in indices)
        means = [float(np.mean([benefit(layer, key) for key in selected])) for layer in (layer_a, layer_b)]
        draws[i] = means[1] - means[0]
    return {
        "contrast": f"C_benefit_purpose_L{layer_b}_minus_L{layer_a}",
        "layer_low": layer_a,
        "layer_high": layer_b,
        "observed": float(observed),
        "bootstrap_unit": "complete held-out payload block, stratified within held-out purpose family",
        "benefit_aggregation": "mean of four purpose directions per payload block",
        "seed": BOOTSTRAP_SEED,
        "repetitions": BOOTSTRAP_REPS,
        "percentile_ci_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "fraction_positive": float(np.mean(draws > 0.0)),
    }


def analyze_space(x: np.ndarray, rows: list[dict[str, Any]], layers: list[int], *, projection_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Full per-payload records are retained for the raw space; controls receive summaries.
    raw_records: list[dict[str, Any]] = []
    layer_summaries: list[dict[str, Any]] = []
    for layer_offset, layer in enumerate(layers):
        comp = factorial_components(x[:, layer_offset, :], rows)
        h = np.asarray(comp["h"], dtype=np.float64)
        families = np.asarray(comp["families"])
        blocks = list(comp["block_ids"])
        fold_records: list[dict[str, Any]] = []
        ratios: list[float] = []
        for held in sorted(set(families.tolist())):
            train = families != held
            test = families == held
            a_hat = np.mean(comp["A"][train], axis=0)
            b_hat = np.mean(comp["B"][train], axis=0)
            c_hat = np.mean(comp["C"][train], axis=0)
            a_norm = float(np.linalg.norm(a_hat))
            ratios.append(float(np.linalg.norm(c_hat) / a_norm) if a_norm else float("nan"))
            train_scale = float(np.sqrt(np.mean(np.sum(h[train] ** 2, axis=1))))
            full = direction_arrays(h[test], np.broadcast_to(a_hat, (int(test.sum()), x.shape[2])), np.broadcast_to(b_hat, (int(test.sum()), x.shape[2])), np.broadcast_to(c_hat, (int(test.sum()), x.shape[2])))
            main = main_only_arrays(h[test], np.broadcast_to(a_hat, (int(test.sum()), x.shape[2])), np.broadcast_to(b_hat, (int(test.sum()), x.shape[2])))
            held_blocks = [blocks[i] for i in np.flatnonzero(test)]
            for direction in DIRECTIONS:
                full_pred, target = full[direction]
                main_pred, _ = main[direction]
                # source is recovered from the direction-specific target/source algebra.
                if direction == "purpose_db_to_eb": source = h[test, 2]
                elif direction == "purpose_dc_to_ec": source = h[test, 3]
                elif direction == "purpose_eb_to_db": source = h[test, 0]
                elif direction == "purpose_ec_to_dc": source = h[test, 1]
                elif direction == "format_eval_b_to_c": source = h[test, 0]
                elif direction == "format_deploy_b_to_c": source = h[test, 2]
                elif direction == "format_eval_c_to_b": source = h[test, 1]
                else: source = h[test, 3]
                full_metrics = summarize_prediction(full_pred, target, source, train_scale)
                main_metrics = summarize_prediction(main_pred, target, source, train_scale)
                for block, fm, mm in zip(held_blocks, full_metrics, main_metrics):
                    for method, metric in (("full", fm), ("main", mm)):
                        fold_records.append({
                            "projection": projection_name,
                            "layer": int(layer),
                            "held_out_family": str(held),
                            "payload_block_id": str(block),
                            "direction": direction,
                            "method": method,
                            **metric,
                        })
        layer_summaries.append({
            "layer": int(layer),
            "projection": projection_name,
            "training_fold_norm_C_over_norm_A_mean": float(np.nanmean(ratios)),
            "by_direction": {direction: summarize_records([r for r in fold_records if r["direction"] == direction]) for direction in DIRECTIONS},
        })
        raw_records.extend(fold_records)
    return raw_records, layer_summaries


def pca_controls(x: np.ndarray, rows: list[dict[str, Any]], layers: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    family_array = np.asarray([str(row["purpose_family_id"]) for row in rows])
    for layer_offset, layer in enumerate(layers):
        layer_x = x[:, layer_offset, :]
        for held in sorted(set(family_array.tolist())):
            train = family_array != held
            reducer = PCA(n_components=min(256, int(train.sum()) - 1, layer_x.shape[1]), svd_solver="randomized", random_state=1729)
            projected = reducer.fit_transform(layer_x[train])
            test_projected = reducer.transform(layer_x[~train])
            combined = np.empty((len(rows), projected.shape[1]), dtype=np.float64)
            combined[train] = projected
            combined[~train] = test_projected
            projected_records, _ = analyze_space(combined[:, None, :], rows, [layer], projection_name="pca256")
            purpose = [r for r in projected_records if r["direction"].startswith("purpose_")]
            out.append({"layer": int(layer), "held_out_family": held, "summary": {direction: summarize_records([r for r in purpose if r["direction"] == direction]) for direction in DIRECTIONS[:4]}})
    return out


def random_controls(x: np.ndarray, rows: list[dict[str, Any]], layers: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seed in RANDOM_SEEDS:
        rng = np.random.default_rng(seed)
        for layer_offset, layer in enumerate(layers):
            projection = rng.normal(size=(x.shape[2], 256)).astype(np.float32) / np.sqrt(256.0)
            projected = np.asarray(x[:, layer_offset, :] @ projection, dtype=np.float64)
            records, _ = analyze_space(projected[:, None, :], rows, [layer], projection_name=f"random256_seed{seed}")
            purpose = [r for r in records if r["direction"].startswith("purpose_")]
            out.append({"seed": seed, "layer": int(layer), "summary": {direction: summarize_records([r for r in purpose if r["direction"] == direction]) for direction in DIRECTIONS[:4]}})
    return out


def write_plots(output: Path, summaries: list[dict[str, Any]], raw_records: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    layers = np.asarray([s["layer"] for s in summaries])
    ratio = np.asarray([s["training_fold_norm_C_over_norm_A_mean"] for s in summaries])
    benefits = []
    for s in summaries:
        values = [s["by_direction"][d]["C_benefit_raw_error"] for d in DIRECTIONS[:4]]
        benefits.append(float(np.mean([v for v in values if v is not None])))
    benefits_array = np.asarray(benefits)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(layers, ratio, marker="o", label=r"$\|C\|/\|A\|$")
    ax.set_xlabel("Transformer block")
    ax.set_ylabel(r"training-fold $\|C\|/\|A\|$")
    ax2 = ax.twinx()
    ax2.plot(layers, benefits_array, color="tab:red", marker="s", label="C benefit")
    ax2.set_ylabel("purpose C benefit (raw error)", color="tab:red")
    ax.set_title("Interaction ratio and counterfactual reconstruction benefit")
    fig.tight_layout()
    fig.savefig(output / "interaction_ratio_vs_reconstruction_benefit.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(ratio, benefits_array)
    for l, xx, yy in zip(layers, ratio, benefits_array):
        ax.annotate(str(l), (xx, yy), fontsize=7)
    ax.set_xlabel(r"$\|C\|/\|A\|$")
    ax.set_ylabel("purpose C benefit (raw error)")
    ax.set_title("All-layer relationship")
    fig.tight_layout()
    fig.savefig(output / "interaction_ratio_vs_benefit_scatter.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-pca", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.items.open(encoding="utf-8") if line.strip()]
    archive = np.load(args.activations, allow_pickle=False)
    item_ids = [str(v) for v in archive["item_ids"].tolist()]
    if item_ids != [str(row["item_id"]) for row in rows]:
        raise ValueError("activation item_ids do not exactly match canonical rows")
    x = np.asarray(archive["X"], dtype=np.float32)
    layers = [int(v) for v in archive["layers"].tolist()]
    if x.ndim != 3 or x.shape[1] != len(layers) or not np.isfinite(x).all():
        raise ValueError(f"expected finite N x L x D archive, got {x.shape}")
    if set(layers) != set(range(32)):
        raise ValueError(f"expected exact 32-layer 8B archive, got {layers}")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    raw_records, summaries = analyze_space(x, rows, layers, projection_name="raw")
    primary = {
        "L5": next(s for s in summaries if s["layer"] == 5),
        "L16": next(s for s in summaries if s["layer"] == 16),
    }
    bootstrap = bootstrap_difference(raw_records, 5, 16)
    all_layer_ratio = np.asarray([s["training_fold_norm_C_over_norm_A_mean"] for s in summaries])
    all_layer_benefit = np.asarray([np.mean([s["by_direction"][d]["C_benefit_raw_error"] for d in DIRECTIONS[:4]]) for s in summaries])
    pearson = float(np.corrcoef(all_layer_ratio, all_layer_benefit)[0, 1])
    spearman = float(np.corrcoef(np.argsort(np.argsort(all_layer_ratio)), np.argsort(np.argsort(all_layer_benefit)))[0, 1])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "DEVELOPMENT_OFFLINE_MECHANISM",
        "analysis": "cross_fitted_representational_counterfactual_reconstruction",
        "run_id": RUN_ID,
        "scientific_boundary": "Observational/representational; no modified model forward pass and no behavioral causal claim.",
        "model": {"name": "meta-llama/Llama-3.1-8B-Instruct", "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "layers": layers, "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm", "token_position": "last_prompt_token_after_chat_template_before_generation", "dtype_stored": "float32"},
        "fit_rule": "For each held-out purpose family, A/B/C are means of complete training payload-block factorial components; held-out payload blocks are never used for fitting.",
        "normalization": "Normalized Euclidean error divides by the RMS activation norm over training rows in the held-out-family fold.",
        "purpose_transport": {"full": "DB->EB: h_DB+2(A+C); DC->EC: h_DC+2(A-C); reverse signs", "main_only": "same transports with C omitted"},
        "format_transport": {"full": "benchmark->casual: -2(B+pC), reverse signs", "main_only": "same transports with C omitted"},
        "primary_layers": {"shared": 5, "interaction_heavy": 16},
        "bootstrap": bootstrap,
        "all_layer_relationship": {"metric": "mean purpose C benefit across four purpose directions", "pearson_ratio_vs_benefit": pearson, "spearman_ratio_vs_benefit": spearman},
        "input_artifacts": {"items": str(args.items), "items_sha256": sha256(args.items), "activations": str(args.activations), "activations_sha256": sha256(args.activations), "n_items": len(rows), "n_payload_blocks": len(factorial_components(x[:, 0, :], rows)["block_ids"])},
        "layer_summaries": summaries,
        "primary_layer_summaries": primary,
        "raw_records": raw_records,
        "pca256_controls": [] if args.skip_pca else pca_controls(x, rows, layers),
        "random256_controls": [] if args.skip_random else random_controls(x, rows, layers),
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_plots(output, summaries, raw_records)
    report = [
        "# Cross-fitted representational counterfactual reconstruction",
        "",
        "Status: `DEVELOPMENT_OFFLINE_MECHANISM`.",
        "",
        "This is an observational activation-space reconstruction. It does not continue the model forward and is not behavioral causality or causal steering.",
        "",
        f"- L5 purpose C benefit: {primary['L5']['by_direction']['purpose_db_to_eb']['C_benefit_raw_error']:.6f} (DB->EB); mean four-purpose-direction benefit {np.mean([primary['L5']['by_direction'][d]['C_benefit_raw_error'] for d in DIRECTIONS[:4]]):.6f}.",
        f"- L16 purpose C benefit: {primary['L16']['by_direction']['purpose_db_to_eb']['C_benefit_raw_error']:.6f} (DB->EB); mean four-purpose-direction benefit {np.mean([primary['L16']['by_direction'][d]['C_benefit_raw_error'] for d in DIRECTIONS[:4]]):.6f}.",
        f"- Paired L16-minus-L5 bootstrap contrast: {bootstrap['observed']:.6f}; 95% CI [{bootstrap['percentile_ci_95'][0]:.6f}, {bootstrap['percentile_ci_95'][1]:.6f}].",
        f"- All-layer ratio/benefit association: Pearson {pearson:.4f}; Spearman {spearman:.4f}.",
        "",
        "Raw per-payload metrics, PCA controls, random-projection controls, provenance, and fixed seeds are in `results.json`.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
