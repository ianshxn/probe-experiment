#!/usr/bin/env python3
"""Held-cue-context development for the cue-balanced probe recipes.

The two original banks and all three augmentation templates are spent
development contexts.  For each recipe and model, leave one whole cue context
out, fit with the other four, and evaluate the untouched context with the full
control gate.  Only recipes passing every held-context gate receive core-data
diagnostics.  Naturalistic transfer is never loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROBE_CHECK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROBE_CHECK))
from analyze import load, load_acts, pick_C  # noqa: E402
from cue_balanced import RECIPES, Recipe, fit_estimator  # noqa: E402
from cue_invariant import read_metadata  # noqa: E402
from develop_cue_balanced import (  # noqa: E402
    EXPECTED_MODELS,
    LAYER,
    POSITION,
    THRESHOLDS,
    control_record,
    reciprocal_core,
    sha256,
)


ROOT = PROBE_CHECK.parent
EXPECTED_ANALYSIS = "cue_balanced_augmented_development_v1"
ORIGINAL_CONTEXTS = {
    "bank1_original": ("bank1", 256),
    "bank2_original": ("bank2", 256),
}
AUGMENTED_CONTEXTS = {
    "administrative_middle",
    "category_bracket_middle",
    "index_footer",
}


def load_contexts(
    acts: Path,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, tuple[np.ndarray, list[dict[str, Any]]]],
]:
    config, keep, data = load(acts, "all")
    if config.get("analysis") != EXPECTED_ANALYSIS or config.get("development_only") is not True:
        raise RuntimeError(f"{acts}: not the expanded development activation package")
    if config.get("cell") != {"position": POSITION, "layer": LAYER}:
        raise RuntimeError(f"{acts}: wrong layer/position cell")
    if config.get("naturalistic_transfer") != "not_loaded":
        raise RuntimeError(f"{acts}: naturalistic-transfer seal is not recorded")
    x_fit = load_acts(acts, LAYER, POSITION, keep)
    contexts: dict[str, tuple[np.ndarray, list[dict[str, Any]]]] = {}
    for context, (relative, expected_n) in ORIGINAL_CONTEXTS.items():
        directory = acts / relative
        rows = read_metadata(directory / "meta.jsonl")
        x = load_acts(directory, LAYER, POSITION)
        if len(rows) != expected_n or len(x) != expected_n:
            raise RuntimeError(f"{directory}: expected {expected_n} rows")
        contexts[context] = (x, [{**row, "cue_template_id": context} for row in rows])

    directory = acts / "augmentation"
    augmented_rows = read_metadata(directory / "meta.jsonl")
    x_augmented = load_acts(directory, LAYER, POSITION)
    if len(augmented_rows) != 1536 or len(x_augmented) != 1536:
        raise RuntimeError(f"{directory}: expected 1,536 rows")
    found = {str(row.get("cue_template_id")) for row in augmented_rows}
    if found != AUGMENTED_CONTEXTS:
        raise RuntimeError(f"{directory}: unexpected cue templates {sorted(found)}")
    template_array = np.array([str(row["cue_template_id"]) for row in augmented_rows])
    for context in sorted(AUGMENTED_CONTEXTS):
        mask = template_array == context
        rows = [row for row, include in zip(augmented_rows, mask, strict=True) if include]
        if int(mask.sum()) != 512:
            raise RuntimeError(f"{directory}: {context} does not have 512 rows")
        contexts[context] = (x_augmented[mask], rows)
    if len(contexts) != 5:
        raise RuntimeError("expanded development requires exactly five cue contexts")
    return config, x_fit, data, contexts


def concatenate_contexts(
    contexts: dict[str, tuple[np.ndarray, list[dict[str, Any]]]],
    names: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    return (
        np.concatenate([contexts[name][0] for name in names], axis=0),
        [row for name in names for row in contexts[name][1]],
    )


def input_record(acts: Path) -> dict[str, str]:
    files = {
        "config": acts / "config.json",
        "fitting_meta": acts / "meta.jsonl",
        "fitting_acts": acts / f"acts_layer{LAYER}_{POSITION}.pt",
        "bank1_meta": acts / "bank1" / "meta.jsonl",
        "bank1_acts": acts / "bank1" / f"acts_layer{LAYER}_{POSITION}.pt",
        "bank2_meta": acts / "bank2" / "meta.jsonl",
        "bank2_acts": acts / "bank2" / f"acts_layer{LAYER}_{POSITION}.pt",
        "augmentation_meta": acts / "augmentation" / "meta.jsonl",
        "augmentation_acts": acts / "augmentation" / f"acts_layer{LAYER}_{POSITION}.pt",
    }
    return {f"{name}_sha256": sha256(path) for name, path in files.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "probe_check" / "metrics" / "cue_balanced_augmented_development.json",
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 2000:
        raise SystemExit("formal development requires at least 2,000 bootstraps")
    result: dict[str, Any] = {
        "analysis": "cue_balanced_held_context_development_v1",
        "status": "development_only_all_inputs_spent",
        "cell": {"position": POSITION, "layer": LAYER},
        "recipes": [recipe.__dict__ for recipe in RECIPES],
        "thresholds": THRESHOLDS,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "selection_rule": (
            "Eligible only if every one-of-five held-cue-context control gate and "
            "the leave-one-frame-family core point gate pass on both models. Among "
            "eligible recipes maximize the joint minimum core floor, then the held-"
            "context purpose-AUC floor, then minimize maximum absolute cue effect."
        ),
        "naturalistic_transfer": "not_loaded_or_scored",
        "models": {},
        "inputs": {},
    }
    loaded: dict[
        str,
        tuple[np.ndarray, dict[str, np.ndarray], dict[str, tuple[np.ndarray, list[dict[str, Any]]]]],
    ] = {}
    seen: set[str] = set()
    for acts in args.acts:
        config, x_fit, data, contexts = load_contexts(acts)
        model = str(config.get("model"))
        if model not in EXPECTED_MODELS or model in seen:
            raise SystemExit(f"unexpected or duplicate model: {model}")
        seen.add(model)
        loaded[model] = (x_fit, data, contexts)
        result["inputs"][model] = input_record(acts)
        result["models"][model] = {}
        explicit = data["expl"] == "explicit"
        selected_c, inner_bacc = pick_C(x_fit, data, explicit)
        for recipe in RECIPES:
            entry: dict[str, Any] = {
                "held_context_controls": {},
                "shared_selected_C": float(selected_c),
                "shared_inner_cv_bacc": float(inner_bacc),
            }
            for held in sorted(contexts):
                train_names = [name for name in sorted(contexts) if name != held]
                x_train, rows_train = concatenate_contexts(contexts, train_names)
                x_target, rows_target = contexts[held]
                print(f"{model}: {recipe.name}; held context {held}", flush=True)
                estimator = fit_estimator(
                    x_fit,
                    data,
                    x_train,
                    rows_train,
                    recipe,
                    selected_c=selected_c,
                )
                record = control_record(
                    estimator,
                    x_target,
                    rows_target,
                    data,
                    seed=args.seed,
                    samples=args.bootstrap_samples,
                    seed_parts=(model, recipe.name, "held_context", held),
                )
                record["training_contexts"] = train_names
                record["fit"] = {
                    "selected_C": estimator["selected_C"],
                    "l2_strength": estimator["l2_strength"],
                    "optimization": estimator["optimization"],
                }
                entry["held_context_controls"][held] = record
            entry["all_held_context_control_gates_pass"] = bool(
                all(
                    record["all_control_gates_pass"]
                    for record in entry["held_context_controls"].values()
                )
            )
            result["models"][model][recipe.name] = entry
    if seen != EXPECTED_MODELS:
        raise SystemExit(f"both frozen models are required; found {sorted(seen)}")

    control_eligible = [
        recipe
        for recipe in RECIPES
        if all(
            result["models"][model][recipe.name]["all_held_context_control_gates_pass"]
            for model in seen
        )
    ]
    result["control_eligible_recipes"] = [recipe.name for recipe in control_eligible]
    for recipe in control_eligible:
        for model in sorted(seen):
            x_fit, data, contexts = loaded[model]
            x_all, rows_all = concatenate_contexts(contexts, sorted(contexts))
            print(f"{model}: {recipe.name}; core diagnostics", flush=True)
            core = reciprocal_core(recipe, x_fit, data, x_all, rows_all)
            result["models"][model][recipe.name]["core"] = core

    eligible: list[dict[str, Any]] = []
    for recipe in control_eligible:
        entries = [result["models"][model][recipe.name] for model in sorted(seen)]
        if not all(entry["core"]["all_core_point_gates_pass"] for entry in entries):
            continue
        controls = [
            record
            for entry in entries
            for record in entry["held_context_controls"].values()
        ]
        eligible.append(
            {
                "recipe": recipe.name,
                "core_floor": min(entry["core"]["core_floor"] for entry in entries),
                "purpose_auc_floor": min(record["purpose_auc_floor"] for record in controls),
                "max_abs_cue_effect": max(
                    record["max_abs_cue_effect_fitting_sd"] for record in controls
                ),
            }
        )
    eligible.sort(
        key=lambda row: (
            row["core_floor"],
            row["purpose_auc_floor"],
            -row["max_abs_cue_effect"],
            row["recipe"],
        ),
        reverse=True,
    )
    result["eligible_recipes"] = eligible
    result["selected_recipe"] = eligible[0]["recipe"] if eligible else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"selected: {result['selected_recipe']}")


if __name__ == "__main__":
    main()
