#!/usr/bin/env python3
"""Reciprocal spent-bank development of cue-balanced linear probes.

Bank 1 and bank 2 are both spent.  Each recipe is trained with one bank and
evaluated on the other, in both directions and on both 8B stand-ins.  A recipe
must pass every reciprocal control gate before core leave-one-frame-family-out
diagnostics are computed.  Naturalistic transfer is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROBE_CHECK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROBE_CHECK))
sys.path.insert(0, str(PROBE_CHECK / "development" / "layer_selection"))
from analyze import auc, bacc, load, load_acts, pick_C  # noqa: E402
from cue_balanced import RECIPES, Recipe, fit_estimator, score  # noqa: E402
from cue_invariant import FAMILIES, LANGUAGES, metadata_arrays, read_metadata  # noqa: E402
from explicit_contract import stable_seed  # noqa: E402


ROOT = PROBE_CHECK.parent
EXPECTED_MODELS = {
    "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
}
LAYER = 6
POSITION = "mean"
BANKS = {
    "bank1": "development_controls",
    "bank2": "fresh_confirmation",
}
THRESHOLDS = {
    "pooled_auc": 0.75,
    "stratified_auc": 0.65,
    "bootstrap_lower": 0.50,
    "cue_effect": 0.25,
    "core_point": 0.65,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fast_cluster_auc_interval(
    y: np.ndarray,
    value: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> list[float]:
    """Exact whole-block AUC bootstrap using observation multiplicities.

    Resampling a block duplicates every observation in that block.  Weighted
    pairwise positive-negative comparisons therefore give exactly the same AUC
    as materializing the duplicated rows, while evaluating all replicates in
    one vectorized operation.
    """

    unique = np.unique(groups)
    group_index = {group: i for i, group in enumerate(unique)}
    positive = [np.flatnonzero((groups == group) & (y == 1)) for group in unique]
    negative = [np.flatnonzero((groups == group) & (y == 0)) for group in unique]
    kernel = np.zeros((len(unique), len(unique)), dtype=np.float64)
    for i, pos in enumerate(positive):
        for j, neg in enumerate(negative):
            comparisons = value[pos, None] - value[neg][None, :]
            kernel[i, j] = float(
                np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)
            )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(unique, size=(samples, len(unique)), replace=True)
    weights = np.zeros((samples, len(unique)), dtype=np.float64)
    for column in range(chosen.shape[1]):
        indices = np.fromiter(
            (group_index[group] for group in chosen[:, column]),
            dtype=np.int64,
            count=samples,
        )
        weights[np.arange(samples), indices] += 1.0
    positive_counts = np.array([len(indices) for indices in positive], dtype=float)
    negative_counts = np.array([len(indices) for indices in negative], dtype=float)
    numerator = np.einsum("bi,ij,bj->b", weights, kernel, weights, optimize=True)
    denominator = (weights @ positive_counts) * (weights @ negative_counts)
    draws = numerator / denominator
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def auc_record(
    y: np.ndarray,
    value: np.ndarray,
    groups: np.ndarray,
    *,
    threshold: float,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    point = float(auc(y, value))
    interval = fast_cluster_auc_interval(
        y, value, groups, seed=seed, samples=samples
    )
    return {
        "auc": point,
        "ci95": interval,
        "threshold": threshold,
        "pass": bool(point >= threshold and interval[0] > THRESHOLDS["bootstrap_lower"]),
        "n": int(len(y)),
        "blocks": int(len(np.unique(groups))),
    }


def control_record(
    estimator: dict[str, Any],
    x_target: np.ndarray,
    target_rows: list[dict[str, Any]],
    fit_data: dict[str, np.ndarray],
    *,
    seed: int,
    samples: int,
    seed_parts: tuple[object, ...],
    score_function=score,
) -> dict[str, Any]:
    arrays = metadata_arrays(target_rows)
    purpose = (arrays["intended_purpose"] == "evaluation").astype(int)
    target_score = score_function(estimator, x_target, arrays["language"])
    pooled: dict[str, Any] = {}
    surfaces: dict[str, Any] = {}
    cue_strata: dict[str, Any] = {}
    cue_effects: dict[str, Any] = {}
    passes: list[bool] = []
    fit_language = fit_data["lang"][estimator["fit_mask"]]
    for family in FAMILIES:
        for language in LANGUAGES:
            base = (arrays["cue_family"] == family) & (arrays["language"] == language)
            key = f"{family}|{language}"
            pooled[key] = auc_record(
                purpose[base],
                target_score[base],
                arrays["control_block_id"][base],
                threshold=THRESHOLDS["pooled_auc"],
                seed=stable_seed(seed, *seed_parts, "pooled", key),
                samples=samples,
            )
            passes.append(pooled[key]["pass"])
            fitting = estimator["fit_score"][fit_language == language]
            fit_sd = float(fitting.std())
            effect = float(
                (
                    target_score[base & (arrays["lexical_cue"] == "evaluation")].mean()
                    - target_score[base & (arrays["lexical_cue"] == "deployment")].mean()
                )
                / fit_sd
            )
            cue_effects[key] = {
                "effect_fitting_sd": effect,
                "max_absolute": THRESHOLDS["cue_effect"],
                "pass": bool(abs(effect) <= THRESHOLDS["cue_effect"]),
            }
            passes.append(cue_effects[key]["pass"])
            for surface in ("external", "institutional"):
                mask = base & (arrays["surface"] == surface)
                cell = f"{key}|{surface}"
                surfaces[cell] = auc_record(
                    purpose[mask],
                    target_score[mask],
                    arrays["control_block_id"][mask],
                    threshold=THRESHOLDS["stratified_auc"],
                    seed=stable_seed(seed, *seed_parts, "surface", cell),
                    samples=samples,
                )
                passes.append(surfaces[cell]["pass"])
            for cue in ("evaluation", "deployment"):
                mask = base & (arrays["lexical_cue"] == cue)
                cell = f"{key}|cue={cue}"
                cue_strata[cell] = auc_record(
                    purpose[mask],
                    target_score[mask],
                    arrays["control_block_id"][mask],
                    threshold=THRESHOLDS["stratified_auc"],
                    seed=stable_seed(seed, *seed_parts, "cue", cell),
                    samples=samples,
                )
                passes.append(cue_strata[cell]["pass"])
    return {
        "pooled_purpose_auc": pooled,
        "surface_purpose_auc": surfaces,
        "within_lexical_cue_purpose_auc": cue_strata,
        "lexical_cue_main_effect": cue_effects,
        "all_control_gates_pass": bool(all(passes)),
        "purpose_auc_floor": float(
            min(v["auc"] for section in (pooled, surfaces, cue_strata) for v in section.values())
        ),
        "purpose_ci_lower_floor": float(
            min(v["ci95"][0] for section in (pooled, surfaces, cue_strata) for v in section.values())
        ),
        "max_abs_cue_effect_fitting_sd": float(
            max(abs(v["effect_fitting_sd"]) for v in cue_effects.values())
        ),
    }


def reciprocal_core(
    recipe: Recipe,
    x_fit: np.ndarray,
    data: dict[str, np.ndarray],
    x_source: np.ndarray,
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    prediction = np.full(len(data["y"]), -1, dtype=int)
    decision = np.zeros(len(data["y"]), dtype=float)
    chosen: dict[str, float] = {}
    for held in sorted(set(data["fam"].tolist())):
        train = (data["fam"] != held) & (data["expl"] == "explicit")
        test = data["fam"] == held
        selected_c, _ = pick_C(x_fit, data, train)
        estimator = fit_estimator(
            x_fit,
            data,
            x_source,
            source_rows,
            recipe,
            fit_mask=train,
            selected_c=selected_c,
        )
        fold_score = score(estimator, x_fit[test], data["lang"][test])
        decision[test] = fold_score
        prediction[test] = (fold_score > 0).astype(int)
        chosen[held] = float(selected_c)
    if np.any(prediction < 0):
        raise RuntimeError("not every core item received an outer-fold prediction")

    fold_language = {
        f"{family}|{language}": float(
            bacc(
                data["y"][(data["fam"] == family) & (data["lang"] == language)],
                prediction[(data["fam"] == family) & (data["lang"] == language)],
            )
        )
        for family in sorted(set(data["fam"].tolist()))
        for language in LANGUAGES
    }
    language_auc = {
        language: float(
            auc(
                data["y"][data["lang"] == language],
                decision[data["lang"] == language],
            )
        )
        for language in LANGUAGES
    }
    implicit = {
        language: float(
            bacc(
                data["y"][(data["expl"] == "implicit") & (data["lang"] == language)],
                prediction[(data["expl"] == "implicit") & (data["lang"] == language)],
            )
        )
        for language in LANGUAGES
    }
    surfaces = {
        f"{surface}|{language}": float(
            bacc(
                data["y"][(data["surf"] == surface) & (data["lang"] == language)],
                prediction[(data["surf"] == surface) & (data["lang"] == language)],
            )
        )
        for surface in ("external", "institutional")
        for language in LANGUAGES
    }
    floor = float(min([*fold_language.values(), *language_auc.values(), *implicit.values(), *surfaces.values()]))
    return {
        "role": "spent_data_reciprocal_core_development",
        "outer_selected_C": chosen,
        "held_family_by_language_bacc": fold_language,
        "language_auc": language_auc,
        "implicit_bacc": implicit,
        "surface_by_language_bacc": surfaces,
        "core_floor": floor,
        "all_core_point_gates_pass": bool(floor >= THRESHOLDS["core_point"]),
    }


def load_banks(acts: Path) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray], dict[str, tuple[np.ndarray, list[dict[str, Any]]]]]:
    config, keep, data = load(acts, "all")
    if config.get("analysis") != "cue_invariant_fresh_confirmation_v1":
        raise RuntimeError(f"{acts}: not the frozen same-run activation package")
    x_fit = load_acts(acts, LAYER, POSITION, keep)
    banks = {}
    for name, relative in BANKS.items():
        directory = acts / relative
        rows = read_metadata(directory / "meta.jsonl")
        x = load_acts(directory, LAYER, POSITION)
        if len(rows) != 256 or len(x) != 256:
            raise RuntimeError(f"{acts}/{relative}: expected 256 controls")
        banks[name] = (x, rows)
    return config, x_fit, data, banks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "probe_check" / "metrics" / "cue_balanced_development.json",
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 2000:
        raise SystemExit("formal development requires at least 2000 bootstraps")
    result: dict[str, Any] = {
        "analysis": "cue_balanced_reciprocal_development_v1",
        "status": "both_confirmation_banks_spent_development_only",
        "cell": {"position": POSITION, "layer": LAYER},
        "recipes": [recipe.__dict__ for recipe in RECIPES],
        "thresholds": THRESHOLDS,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "selection_rule": (
            "Eligible only if every reciprocal-bank control gate and every "
            "reciprocal core point gate passes on both models. Among eligible "
            "recipes maximize joint minimum core floor, then target-bank purpose "
            "AUC floor, then minimize maximum absolute cue effect."
        ),
        "naturalistic_transfer": "not_loaded_or_scored",
        "models": {},
        "inputs": {},
    }
    loaded: dict[str, tuple[np.ndarray, dict[str, np.ndarray], dict[str, tuple[np.ndarray, list[dict[str, Any]]]]]] = {}
    seen: set[str] = set()
    for acts in args.acts:
        config, x_fit, data, banks = load_banks(acts)
        model = str(config.get("model"))
        if model not in EXPECTED_MODELS or model in seen:
            raise SystemExit(f"unexpected or duplicate model: {model}")
        seen.add(model)
        loaded[model] = (x_fit, data, banks)
        result["inputs"][model] = {
            "config_sha256": sha256(acts / "config.json"),
            "fitting_meta_sha256": sha256(acts / "meta.jsonl"),
            "bank1_meta_sha256": sha256(acts / BANKS["bank1"] / "meta.jsonl"),
            "bank2_meta_sha256": sha256(acts / BANKS["bank2"] / "meta.jsonl"),
        }
        result["models"][model] = {}
        for recipe in RECIPES:
            recipe_entry: dict[str, Any] = {"reciprocal_controls": {}}
            for source_name, target_name in (("bank1", "bank2"), ("bank2", "bank1")):
                print(f"{model}: {recipe.name} {source_name}->{target_name}", flush=True)
                x_source, source_rows = banks[source_name]
                x_target, target_rows = banks[target_name]
                estimator = fit_estimator(x_fit, data, x_source, source_rows, recipe)
                record = control_record(
                    estimator,
                    x_target,
                    target_rows,
                    data,
                    seed=args.seed,
                    samples=args.bootstrap_samples,
                    seed_parts=(model, recipe.name, source_name, target_name),
                )
                record["fit"] = {
                    "selected_C": estimator["selected_C"],
                    "inner_cv_bacc": estimator["inner_cv_bacc"],
                    "l2_strength": estimator["l2_strength"],
                    "optimization": estimator["optimization"],
                }
                recipe_entry["reciprocal_controls"][f"{source_name}_to_{target_name}"] = record
            recipe_entry["all_reciprocal_control_gates_pass"] = bool(
                all(v["all_control_gates_pass"] for v in recipe_entry["reciprocal_controls"].values())
            )
            result["models"][model][recipe.name] = recipe_entry
    if seen != EXPECTED_MODELS:
        raise SystemExit(f"both frozen models are required; found {sorted(seen)}")

    control_eligible = [
        recipe
        for recipe in RECIPES
        if all(result["models"][model][recipe.name]["all_reciprocal_control_gates_pass"] for model in seen)
    ]
    result["control_eligible_recipes"] = [recipe.name for recipe in control_eligible]
    for recipe in control_eligible:
        for model in sorted(seen):
            x_fit, data, banks = loaded[model]
            core: dict[str, Any] = {}
            for source_name in BANKS:
                print(f"{model}: {recipe.name} core via {source_name}", flush=True)
                x_source, source_rows = banks[source_name]
                core[source_name] = reciprocal_core(recipe, x_fit, data, x_source, source_rows)
            result["models"][model][recipe.name]["reciprocal_core"] = core
            result["models"][model][recipe.name]["all_reciprocal_core_gates_pass"] = bool(
                all(v["all_core_point_gates_pass"] for v in core.values())
            )

    eligible = []
    for recipe in control_eligible:
        entries = [result["models"][model][recipe.name] for model in sorted(seen)]
        if not all(entry["all_reciprocal_core_gates_pass"] for entry in entries):
            continue
        control_records = [
            record
            for entry in entries
            for record in entry["reciprocal_controls"].values()
        ]
        core_records = [record for entry in entries for record in entry["reciprocal_core"].values()]
        eligible.append(
            {
                "recipe": recipe.name,
                "core_floor": min(record["core_floor"] for record in core_records),
                "purpose_auc_floor": min(record["purpose_auc_floor"] for record in control_records),
                "max_abs_cue_effect": max(record["max_abs_cue_effect_fitting_sd"] for record in control_records),
            }
        )
    eligible.sort(
        key=lambda row: (row["core_floor"], row["purpose_auc_floor"], -row["max_abs_cue_effect"], row["recipe"]),
        reverse=True,
    )
    result["eligible_recipes"] = eligible
    result["selected_recipe"] = eligible[0]["recipe"] if eligible else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"selected: {result['selected_recipe']}")


if __name__ == "__main__":
    main()
