#!/usr/bin/env python3
"""No-search evaluator for the untouched cue-constrained confirmation bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROBE_CHECK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROBE_CHECK))
sys.path.insert(0, str(PROBE_CHECK / "development" / "cue_constraints"))
sys.path.insert(0, str(PROBE_CHECK / "confirmations" / "01_format_matched"))
from analyze import load, load_acts  # noqa: E402
from cue_constrained import ConstrainedRecipe, fit_constrained, score_constrained  # noqa: E402
from cue_invariant import read_metadata  # noqa: E402
from develop_cue_balanced import THRESHOLDS, control_record  # noqa: E402
from extract_confirmatory_8b import sha256_file  # noqa: E402


LAYER = 6
POSITION = "mean"
BOOTSTRAP_SAMPLES = 2000
SEED = 20260805
EXPECTED_MODELS = {
    "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
}
EXPECTED_ITEMS_SHA256 = "97796dcf3f54918cb7b55c009f6458832fcf75d880df50252ffdf8415c6bc84e"
EXPECTED_BANK1_SHA256 = "fbd320168c38e14321e50cab1872144a3d32e171d4a56afca63dcb6f782131b9"
EXPECTED_BANK2_SHA256 = "0f93681d626c2e4d6d7c91acf9c8301f42f1203f45136d32441a2ef68de654fb"
EXPECTED_AUGMENTATION_SHA256 = "ce7a41bb44b6ede12dee5455b0e8a586d06f15feb491ce4c91e16fdf74bb2b0c"
EXPECTED_DEVELOPMENT_RENDERED_SHA256 = "31b3092fa2da6c4bfc56372cb8d1fbf59fc0960b9846ce8ba94988f673932493"
EXPECTED_FRESH_SHA256 = "676a7e688d1c88af3fd0fab9e12b09ea91ab6cbff3a64be03c3dbae745277c86"
EXPECTED_RESULT_SHA256 = "def7b012c957547300cb2a7a06f1ae6de8b40a197bb609c0debfa0522a7b23d6"
DEVELOPMENT_RESULT = PROBE_CHECK / "metrics" / "cue_constrained_tradeoff_development.json"
FROZEN_RECIPE = ConstrainedRecipe(
    name="balanced_penalty_256",
    control_mass=1.0,
    cue_penalty=256.0,
    hard_families=(),
)


def evaluate_model(acts: Path) -> dict[str, Any]:
    config, keep, data = load(acts, "all")
    model = str(config.get("model"))
    if model not in EXPECTED_MODELS:
        raise RuntimeError(f"unexpected model: {model}")
    expected_config = {
        "analysis": "cue_constrained_fresh_confirmation_v1",
        "n_layers": 32,
        "layers_stored": [LAYER],
        "positions": [POSITION],
        "gating_cell": {"position": POSITION, "layer": LAYER},
        "fitting_recipe": {
            "name": "balanced_penalty_256",
            "base": "explicit_only",
            "control_mass": 1.0,
            "cue_penalty": 256.0,
            "hard_families": [],
        },
        "n_items": 1200,
        "n_development_controls": 2048,
        "n_fresh_confirmation_controls": 256,
        "items_sha256": EXPECTED_ITEMS_SHA256,
        "bank1_sha256": EXPECTED_BANK1_SHA256,
        "bank2_sha256": EXPECTED_BANK2_SHA256,
        "augmentation_sha256": EXPECTED_AUGMENTATION_SHA256,
        "development_controls_rendered_sha256": EXPECTED_DEVELOPMENT_RENDERED_SHA256,
        "fresh_controls_sha256": EXPECTED_FRESH_SHA256,
        "development_result_sha256": EXPECTED_RESULT_SHA256,
        "model_load_dtype": "bfloat16",
        "load_in_4bit": False,
        "naturalistic_transfer": "not_loaded",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(f"{acts}: config {key} is not frozen: {config.get(key)!r}")
    if "A100" not in str(config.get("accelerator", "")).upper():
        raise RuntimeError(f"{acts}: activations were not recorded on an A100")

    x_fit = load_acts(acts, LAYER, POSITION, keep)
    development_dir = acts / "development_controls"
    development_rows = read_metadata(development_dir / "meta.jsonl")
    x_development = load_acts(development_dir, LAYER, POSITION)
    fresh_dir = acts / "fresh_confirmation"
    fresh_rows = read_metadata(fresh_dir / "meta.jsonl")
    x_fresh = load_acts(fresh_dir, LAYER, POSITION)
    if len(development_rows) != 2048 or len(fresh_rows) != 256:
        raise RuntimeError(f"{acts}: control metadata count changed")

    fitted = fit_constrained(
        x_fit,
        data,
        x_development,
        development_rows,
        FROZEN_RECIPE,
    )
    controls = control_record(
        fitted,
        x_fresh,
        fresh_rows,
        data,
        seed=SEED,
        samples=BOOTSTRAP_SAMPLES,
        seed_parts=(model, "untouched_confirmation"),
        score_function=score_constrained,
    )
    return {
        "model": model,
        "cell": {"position": POSITION, "layer": LAYER},
        "fitting_recipe": fitted["recipe"],
        "selected_C": fitted["selected_C"],
        "inner_cv_bacc": fitted["inner_cv_bacc"],
        "l2_strength": fitted["l2_strength"],
        "optimization": fitted["optimization"],
        "fresh_controls": controls,
        "confirmatory_gate_pass": controls["all_control_gates_pass"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("probe_check/metrics/cue_constrained_fresh_confirmation.json"),
    )
    args = parser.parse_args()
    if sha256_file(DEVELOPMENT_RESULT) != EXPECTED_RESULT_SHA256:
        raise SystemExit("the frozen cue-constrained development result is missing or changed")
    result: dict[str, Any] = {
        "analysis": "cue_constrained_fresh_confirmation_v1",
        "status": "scored_once_after_frozen_activation_extraction",
        "development_result_sha256": EXPECTED_RESULT_SHA256,
        "recipe": {
            "name": "balanced_penalty_256",
            "base": "explicit_only",
            "control_mass": 1.0,
            "cue_penalty": 256.0,
            "hard_families": [],
        },
        "cell": {"position": POSITION, "layer": LAYER},
        "thresholds": THRESHOLDS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": SEED,
        "models": {},
        "naturalistic_transfer": "not_loaded_or_scored",
    }
    seen: set[str] = set()
    for acts in args.acts:
        model_result = evaluate_model(acts)
        model = str(model_result["model"])
        if model in seen:
            raise SystemExit(f"duplicate model: {model}")
        seen.add(model)
        result["models"][model] = model_result
    if seen != EXPECTED_MODELS:
        raise SystemExit(f"both frozen models are required; found {sorted(seen)}")
    result["joint_confirmatory_gate_pass"] = bool(
        all(entry["confirmatory_gate_pass"] for entry in result["models"].values())
    )
    result["decision"] = (
        "cue_constrained_probe_eligible_for_naturalistic_gate"
        if result["joint_confirmatory_gate_pass"]
        else "cue_constrained_probe_rejected_by_fresh_confirmation"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(result["decision"])


if __name__ == "__main__":
    main()
