#!/usr/bin/env python3
"""Develop a cue-invariant purpose probe on the spent confirmation bank.

This is development analysis, never fresh confirmation.  Nuisance bases are
cross-fitted by payload block: every reported control score comes from a basis
that did not see that block.  Naturalistic transfer is never loaded.
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
from analyze import auc, bacc, fold_scores, load, load_acts  # noqa: E402
from cue_invariant import (  # noqa: E402
    FAMILIES,
    LANGUAGES,
    fit_probe,
    metadata_arrays,
    read_metadata,
    score_controls,
)
from explicit_contract import cluster_interval, stable_seed  # noqa: E402


ROOT = PROBE_CHECK.parent
EXPECTED_MODELS = {
    "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
}
LAYER = 6
POSITION = "mean"
METHODS = ("baseline", "axes4", "svd90")
THRESHOLDS = {
    "pooled_auc": 0.75,
    "stratified_auc": 0.65,
    "bootstrap_lower": 0.50,
    "cue_effect": 0.25,
    "core_point": 0.65,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc_record(
    y: np.ndarray,
    score: np.ndarray,
    groups: np.ndarray,
    *,
    threshold: float,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    point = float(auc(y, score))
    interval = cluster_interval(
        y, score, groups, auc, seed=seed, samples=samples
    )
    return {
        "auc": point,
        "ci95": interval,
        "threshold": threshold,
        "pass": bool(point >= threshold and interval[0] > THRESHOLDS["bootstrap_lower"]),
        "n": int(len(y)),
        "blocks": int(len(np.unique(groups))),
    }


def crossfit_controls(
    acts: Path,
    method: str,
    x_fit: np.ndarray,
    data: dict[str, np.ndarray],
    x_control: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    arrays = metadata_arrays(rows)
    blocks = arrays["control_block_id"]
    purpose_y = (arrays["intended_purpose"] == "evaluation").astype(int)
    score = np.zeros(len(rows), dtype=float)
    fold_records: dict[str, Any] = {}
    for block in sorted(set(blocks.tolist())):
        train = blocks != block
        test = ~train
        fitted = fit_probe(
            x_fit,
            data,
            x_control,
            rows,
            control_keep=train,
            method=method,
        )
        raw = score_controls(fitted, x_control[test], [rows[i] for i in np.flatnonzero(test)])
        fit_language = data["lang"][fitted["fit_mask"]]
        test_language = arrays["language"][test]
        normalized = np.zeros(test.sum(), dtype=float)
        for language in LANGUAGES:
            fitting = fitted["fit_score"][fit_language == language]
            selected = test_language == language
            normalized[selected] = (raw[selected] - fitting.mean()) / fitting.std()
        score[test] = normalized
        fold_records[block] = {
            "basis": fitted["basis_record"],
            "selected_C": fitted["selected_C"],
            "inner_cv_bacc": fitted["inner_cv_bacc"],
        }

    pooled: dict[str, Any] = {}
    surfaces: dict[str, Any] = {}
    cue_strata: dict[str, Any] = {}
    cue_effects: dict[str, Any] = {}
    passes: list[bool] = []
    for family in FAMILIES:
        for language in LANGUAGES:
            base = (arrays["cue_family"] == family) & (
                arrays["language"] == language
            )
            key = f"{family}|{language}"
            pooled[key] = auc_record(
                purpose_y[base],
                score[base],
                blocks[base],
                threshold=THRESHOLDS["pooled_auc"],
                seed=stable_seed(seed, method, "pooled", key),
                samples=samples,
            )
            passes.append(pooled[key]["pass"])
            effect = float(
                score[base & (arrays["lexical_cue"] == "evaluation")].mean()
                - score[base & (arrays["lexical_cue"] == "deployment")].mean()
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
                    purpose_y[mask],
                    score[mask],
                    blocks[mask],
                    threshold=THRESHOLDS["stratified_auc"],
                    seed=stable_seed(seed, method, "surface", cell),
                    samples=samples,
                )
                passes.append(surfaces[cell]["pass"])
            for cue in ("evaluation", "deployment"):
                mask = base & (arrays["lexical_cue"] == cue)
                cell = f"{key}|cue={cue}"
                cue_strata[cell] = auc_record(
                    purpose_y[mask],
                    score[mask],
                    blocks[mask],
                    threshold=THRESHOLDS["stratified_auc"],
                    seed=stable_seed(seed, method, "cue", cell),
                    samples=samples,
                )
                passes.append(cue_strata[cell]["pass"])
    return {
        "status": "spent_bank_block_crossfit_development_only",
        "folds": fold_records,
        "pooled_purpose_auc": pooled,
        "surface_purpose_auc": surfaces,
        "within_lexical_cue_purpose_auc": cue_strata,
        "lexical_cue_main_effect": cue_effects,
        "all_control_gates_pass": bool(all(passes)),
        "purpose_auc_floor": float(
            min(
                value["auc"]
                for section in (pooled, surfaces, cue_strata)
                for value in section.values()
            )
        ),
        "purpose_ci_lower_floor": float(
            min(
                value["ci95"][0]
                for section in (pooled, surfaces, cue_strata)
                for value in section.values()
            )
        ),
        "max_abs_cue_effect_fitting_sd": float(
            max(abs(value["effect_fitting_sd"]) for value in cue_effects.values())
        ),
    }


def core_point_diagnostic(
    method: str,
    x_fit: np.ndarray,
    data: dict[str, np.ndarray],
    x_control: np.ndarray,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    fitted = fit_probe(x_fit, data, x_control, rows, method=method)
    pred, score, chosen = fold_scores(
        fitted["x_fit_projected"],
        data,
        fitted["selected_C"],
        fit_mask=fitted["fit_mask"],
        inner_cv=True,
    )
    fold_language: dict[str, float] = {}
    for family in sorted(set(data["fam"].tolist())):
        for language in LANGUAGES:
            mask = (data["fam"] == family) & (data["lang"] == language)
            fold_language[f"{family}|{language}"] = bacc(data["y"][mask], pred[mask])
    language_auc = {
        language: auc(data["y"][data["lang"] == language], score[data["lang"] == language])
        for language in LANGUAGES
    }
    implicit = {
        language: bacc(
            data["y"][(data["lang"] == language) & (data["expl"] == "implicit")],
            pred[(data["lang"] == language) & (data["expl"] == "implicit")],
        )
        for language in LANGUAGES
    }
    surfaces = {
        f"{surface}|{language}": bacc(
            data["y"][(data["surf"] == surface) & (data["lang"] == language)],
            pred[(data["surf"] == surface) & (data["lang"] == language)],
        )
        for surface in ("external", "institutional")
        for language in LANGUAGES
    }
    floor = float(min([*fold_language.values(), *language_auc.values(), *implicit.values(), *surfaces.values()]))
    return {
        "role": "development_point_diagnostic_not_fresh_evidence",
        "basis": fitted["basis_record"],
        "selected_C": fitted["selected_C"],
        "inner_cv_bacc": fitted["inner_cv_bacc"],
        "outer_selected_C": chosen,
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
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "probe_check" / "metrics" / "cue_invariant_development.json",
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 2000:
        raise SystemExit("development comparison requires at least 2000 bootstrap samples")

    result: dict[str, Any] = {
        "analysis": "cue_invariant_probe_development_v1",
        "status": "spent_confirmatory_bank_reclassified_as_development",
        "cell": {"position": POSITION, "layer": LAYER},
        "methods": list(METHODS),
        "thresholds": THRESHOLDS,
        "selection_rule": (
            "Among methods passing every held-block control gate and every core "
            "point gate on both models, maximize the joint minimum core floor; "
            "then purpose-AUC floor; then minimize maximum absolute cue effect."
        ),
        "naturalistic_transfer": "not_loaded_or_scored",
        "models": {},
    }
    seen: set[str] = set()
    for acts in args.acts:
        config, keep, data = load(acts, "all")
        model = str(config["model"])
        if model not in EXPECTED_MODELS or model in seen:
            raise SystemExit(f"unexpected or duplicate model: {model}")
        seen.add(model)
        x_fit = load_acts(acts, LAYER, POSITION, keep)
        control_dir = acts / "confirmatory_controls"
        rows = read_metadata(control_dir / "meta.jsonl")
        x_control = load_acts(control_dir, LAYER, POSITION)
        entry: dict[str, Any] = {}
        for method in METHODS:
            print(f"{model}: {method}", flush=True)
            entry[method] = {
                "crossfit_controls": crossfit_controls(
                    acts,
                    method,
                    x_fit,
                    data,
                    x_control,
                    rows,
                    seed=args.seed,
                    samples=args.bootstrap_samples,
                ),
                "core": core_point_diagnostic(method, x_fit, data, x_control, rows),
            }
        result["models"][model] = entry
    if seen != EXPECTED_MODELS:
        raise SystemExit(f"both frozen 8B models are required; found {sorted(seen)}")

    eligible = []
    for method in METHODS:
        entries = [result["models"][model][method] for model in sorted(seen)]
        if all(
            entry["crossfit_controls"]["all_control_gates_pass"]
            and entry["core"]["all_core_point_gates_pass"]
            for entry in entries
        ):
            eligible.append(
                {
                    "method": method,
                    "core_floor": min(entry["core"]["core_floor"] for entry in entries),
                    "purpose_auc_floor": min(entry["crossfit_controls"]["purpose_auc_floor"] for entry in entries),
                    "max_abs_cue_effect": max(entry["crossfit_controls"]["max_abs_cue_effect_fitting_sd"] for entry in entries),
                }
            )
    eligible.sort(
        key=lambda row: (
            row["core_floor"],
            row["purpose_auc_floor"],
            -row["max_abs_cue_effect"],
            row["method"],
        ),
        reverse=True,
    )
    result["eligible_methods"] = eligible
    result["selected_method"] = eligible[0]["method"] if eligible else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"selected: {result['selected_method']}")


if __name__ == "__main__":
    main()
