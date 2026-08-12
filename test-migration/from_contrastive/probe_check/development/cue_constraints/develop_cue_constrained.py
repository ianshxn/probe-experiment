#!/usr/bin/env python3
"""Post-result development of stronger cue constraints on spent contexts."""

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
from analyze import auc, bacc, pick_C  # noqa: E402
from cue_constrained import (  # noqa: E402
    CONSTRAINED_RECIPES,
    CONSTRAINED_RECIPES_ROUND2,
    ConstrainedRecipe,
    fit_constrained,
    prepare_basis,
    score_constrained,
)
from cue_invariant import LANGUAGES  # noqa: E402
from develop_augmented_cue_balanced import (  # noqa: E402
    EXPECTED_MODELS,
    concatenate_contexts,
    input_record,
    load_contexts,
)
from develop_cue_balanced import (  # noqa: E402
    LAYER,
    POSITION,
    THRESHOLDS,
    control_record,
)


ROOT = PROBE_CHECK.parent
PRIOR_RESULT = ROOT / "probe_check" / "metrics" / "cue_balanced_augmented_development.json"
EXPECTED_PRIOR_RESULT_SHA256 = "3b8b6ec3368fee0b2755d71ec9e40053d15e81b5ba15da27ed7ad4cfddb59c8a"
ROUND1_RESULT = ROOT / "probe_check" / "metrics" / "cue_constrained_development.json"
EXPECTED_ROUND1_RESULT_SHA256 = "8d76b7f746f148d878e346cfbb58a48e3dbd2f1444249b5bdd7ac9745559af73"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def constrained_core(
    recipe: ConstrainedRecipe,
    x_fit: np.ndarray,
    data: dict[str, np.ndarray],
    x_controls: np.ndarray,
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    prediction = np.full(len(data["y"]), -1, dtype=int)
    decision = np.zeros(len(data["y"]), dtype=float)
    chosen: dict[str, float] = {}
    constraints: dict[str, Any] = {}
    for held in sorted(set(data["fam"].tolist())):
        train = (data["fam"] != held) & (data["expl"] == "explicit")
        test = data["fam"] == held
        selected_c, _ = pick_C(x_fit, data, train)
        prepared = None
        if recipe.hard_families:
            prepared = prepare_basis(
                x_fit,
                data,
                x_controls,
                control_rows,
                recipe.hard_families,
                fit_mask=train,
            )
        estimator = fit_constrained(
            x_fit,
            data,
            x_controls,
            control_rows,
            recipe,
            fit_mask=train,
            selected_c=selected_c,
            prepared_basis=prepared,
        )
        fold_score = score_constrained(
            estimator, x_fit[test], data["lang"][test]
        )
        decision[test] = fold_score
        prediction[test] = (fold_score > 0).astype(int)
        chosen[held] = float(selected_c)
        if estimator["constraint"] is not None:
            constraints[held] = estimator["constraint"]
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
    floor = float(
        min(
            [
                *fold_language.values(),
                *language_auc.values(),
                *implicit.values(),
                *surfaces.values(),
            ]
        )
    )
    return {
        "role": "spent_five_context_core_development",
        "outer_selected_C": chosen,
        "constraints": constraints,
        "held_family_by_language_bacc": fold_language,
        "language_auc": language_auc,
        "implicit_bacc": implicit,
        "surface_by_language_bacc": surfaces,
        "core_floor": floor,
        "all_core_point_gates_pass": bool(floor >= THRESHOLDS["core_point"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", type=Path, nargs="+", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--round", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 2000:
        raise SystemExit("formal development requires at least 2,000 bootstraps")
    trigger_path = PRIOR_RESULT if args.round == 1 else ROUND1_RESULT
    trigger_hash = (
        EXPECTED_PRIOR_RESULT_SHA256
        if args.round == 1
        else EXPECTED_ROUND1_RESULT_SHA256
    )
    recipes = (
        CONSTRAINED_RECIPES
        if args.round == 1
        else CONSTRAINED_RECIPES_ROUND2
    )
    if args.output is None:
        args.output = ROOT / "probe_check" / "metrics" / (
            "cue_constrained_development.json"
            if args.round == 1
            else "cue_constrained_tradeoff_development.json"
        )
    checkpoint = args.output.with_suffix(".partial.json")
    if sha256(trigger_path) != trigger_hash:
        raise SystemExit("the triggering development result hash is stale")
    prior = json.loads(trigger_path.read_text(encoding="utf-8"))
    if prior.get("selected_recipe") is not None:
        raise SystemExit("the triggering result no longer records a null selection")

    result: dict[str, Any] = {
        "analysis": (
            "cue_constrained_held_context_development_v1"
            if args.round == 1
            else "cue_constrained_tradeoff_development_v1"
        ),
        "status": "post_result_development_all_inputs_spent",
        "development_round": args.round,
        "trigger_result_sha256": trigger_hash,
        "cell": {"position": POSITION, "layer": LAYER},
        "recipes": [recipe.__dict__ for recipe in recipes],
        "thresholds": THRESHOLDS,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "selection_rule": (
            "Eligible only if every one-of-five held-cue-context control gate and "
            "the leave-one-frame-family core point gate pass on both models. Among "
            "eligible recipes maximize joint minimum core floor, then held-context "
            "purpose-AUC floor, then minimize maximum absolute cue effect."
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
        basis_cache: dict[tuple[str, tuple[str, ...]], tuple[np.ndarray, dict[str, Any]]] = {}
        for recipe in recipes:
            entry: dict[str, Any] = {
                "held_context_controls": {},
                "shared_selected_C": float(selected_c),
                "shared_inner_cv_bacc": float(inner_bacc),
            }
            for held in sorted(contexts):
                train_names = [name for name in sorted(contexts) if name != held]
                x_train, rows_train = concatenate_contexts(contexts, train_names)
                x_target, rows_target = contexts[held]
                prepared = None
                if recipe.hard_families:
                    cache_key = (held, recipe.hard_families)
                    if cache_key not in basis_cache:
                        print(
                            f"{model}: prepare {list(recipe.hard_families)}; held {held}",
                            flush=True,
                        )
                        basis_cache[cache_key] = prepare_basis(
                            x_fit,
                            data,
                            x_train,
                            rows_train,
                            recipe.hard_families,
                        )
                    prepared = basis_cache[cache_key]
                print(f"{model}: {recipe.name}; held context {held}", flush=True)
                try:
                    estimator = fit_constrained(
                        x_fit,
                        data,
                        x_train,
                        rows_train,
                        recipe,
                        selected_c=selected_c,
                        prepared_basis=prepared,
                    )
                except RuntimeError as error:
                    entry["optimization_failure"] = {
                        "held_context": held,
                        "error": str(error),
                    }
                    print(
                        f"{model}: {recipe.name}; INELIGIBLE NUMERICAL FAILURE: {error}",
                        flush=True,
                    )
                    break
                record = control_record(
                    estimator,
                    x_target,
                    rows_target,
                    data,
                    seed=args.seed,
                    samples=args.bootstrap_samples,
                    seed_parts=(model, recipe.name, "held_context", held),
                    score_function=score_constrained,
                )
                record["training_contexts"] = train_names
                record["fit"] = {
                    "selected_C": estimator["selected_C"],
                    "l2_strength": estimator["l2_strength"],
                    "optimization": estimator["optimization"],
                    "constraint": estimator["constraint"],
                }
                entry["held_context_controls"][held] = record
            entry["all_held_context_control_gates_pass"] = bool(
                "optimization_failure" not in entry
                and len(entry["held_context_controls"]) == len(contexts)
                and all(
                    record["all_control_gates_pass"]
                    for record in entry["held_context_controls"].values()
                )
            )
            result["models"][model][recipe.name] = entry
            result["checkpoint"] = {
                "complete": False,
                "last_completed_model": model,
                "last_completed_recipe": recipe.name,
            }
            write_json(checkpoint, result)
    if seen != EXPECTED_MODELS:
        raise SystemExit(f"both frozen models are required; found {sorted(seen)}")

    control_eligible = [
        recipe
        for recipe in recipes
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
            print(f"{model}: {recipe.name}; gated core diagnostics", flush=True)
            core = constrained_core(recipe, x_fit, data, x_all, rows_all)
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
    result.pop("checkpoint", None)
    write_json(args.output, result)
    write_json(checkpoint, {"complete": True, "result_sha256": sha256(args.output)})
    print(f"wrote {args.output}")
    print(f"selected: {result['selected_recipe']}")


if __name__ == "__main__":
    main()
