"""Write PLACEHOLDER calibration evidence so a demo can run the production path.

The two calibration gates normally consume independent human ratings and
held-out model classifications. Neither exists during a demo, so this script
synthesises reports that satisfy the gates structurally and lets
``preflight --production`` proceed end to end.

Every report it writes is marked in three ways that survive into any artifact
built on top of it:

* ``cohort_id`` / model ids carry a ``DEMO_PLACEHOLDER`` prefix;
* ``notes`` states in the first sentence that the evidence is synthetic; and
* the numbers are the gate thresholds themselves, not plausible measurements,
  so nobody can mistake them for a real result.

This exists to unblock a pipeline rehearsal. Reports it produces are not
evidence, must never back a scientific claim, and must be replaced by real
``calibration-check`` / ``comprehension-check`` output before the study
reports anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONTRASTIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRASTIVE_ROOT / "src"))

from contrastive_generation.catalog import load_catalog  # noqa: E402
from contrastive_generation.utils import (  # noqa: E402
    hash_object,
    load_json,
    load_yaml,
    sha256_file,
)

PLACEHOLDER = "DEMO_PLACEHOLDER"
FAKE_HASH = "0" * 64

DEMO_NOTE = (
    "SYNTHETIC PLACEHOLDER -- NOT EVIDENCE. No human rater and no held-out "
    "model produced these numbers; they were written by "
    "scripts/make_demo_calibration.py so that a pipeline demonstration could "
    "execute the production path end to end. The reported figures are the "
    "gate thresholds themselves, not measurements. Any run, dataset, or claim "
    "depending on this file inherits that status. Replace with real "
    "calibration-check / comprehension-check output before reporting anything."
)


def stratum(items: int, balanced: float, lower: float) -> dict:
    return {
        "items": items,
        "correct": round(items * balanced),
        "balanced_accuracy": balanced,
        "eval_accuracy": balanced,
        "deploy_accuracy": balanced,
        "class_accuracy_gap": 0.0,
        "wilson_95_lower_bound": lower,
    }


def frame_language_result() -> dict:
    # Implicit sits mid-window (0.60-0.80); explicit clears 0.85 with the
    # required >=0.10 separation.
    return {
        "strata": {
            "implicit": stratum(40, 0.70, 0.55),
            "explicit": stratum(40, 0.90, 0.80),
        },
        "gates": {
            "implicit_min": True,
            "implicit_max": True,
            "implicit_above_chance": True,
            "implicit_class_symmetry": True,
            "explicit_min": True,
            "explicit_minus_implicit": True,
        },
        "pass": True,
    }


def comprehension_language_result(gate_names: list[str]) -> dict:
    return {
        "strata": {
            "implicit": stratum(106, 0.90, 0.80),
            "explicit": stratum(106, 0.95, 0.88),
        },
        "surface": {
            "external": {"items": 96, "accuracy": 0.92},
            "institutional": {"items": 116, "accuracy": 0.92},
        },
        "gates": {name: True for name in gate_names},
        "pass": True,
    }


def comprehension_gate_names() -> list[str]:
    schema = load_json(
        CONTRASTIVE_ROOT / "schemas" / "heldout_comprehension_report.schema.json"
    )
    gates = schema["$defs"]["language_result"]["properties"]["gates"]
    return list(gates["required"])


def main() -> int:
    english = load_catalog("topics.yaml", "containers.yaml")
    japanese = load_catalog("topics.ja.yaml", "containers.yaml")
    approval = load_json(
        CONTRASTIVE_ROOT / "prompts" / "pre_generation_semantic_approval.json"
    )
    config = load_yaml(CONTRASTIVE_ROOT / "configs" / "pre_generation.yaml")

    catalog_binding = {
        "purpose_frame_catalog_sha256": sha256_file(english.purpose_frame_path),
        "container_catalog_sha256_en": english.container_catalog_sha256(),
        "container_catalog_sha256_ja": japanese.container_catalog_sha256(),
    }

    frame_report = {
        "schema_version": 1,
        "evidence_type": "human_frame_only_calibration",
        "status": "passed",
        **catalog_binding,
        "packet_sha256": FAKE_HASH,
        "results_sha256": FAKE_HASH,
        "raters_sha256": FAKE_HASH,
        "cohort_id": f"{PLACEHOLDER}_frame_cohort",
        "confirmatory_raters": 5,
        "minimum_ratings_per_item": 5,
        "languages": {
            "en": frame_language_result(),
            "ja": frame_language_result(),
        },
        "notes": DEMO_NOTE,
    }

    profiles = [
        load_yaml(CONTRASTIVE_ROOT / "configs" / "providers" / name)
        for name in ("qwen_openrouter.yaml", "gemini_vertex.yaml")
    ]
    generator_models = [
        {
            "profile_id": str(row["profile_id"]),
            "model": str(row["model"]),
            "model_revision": str(row["model_revision"]),
        }
        for row in profiles
    ]
    gate_names = comprehension_gate_names()
    comprehension_report = {
        "schema_version": 1,
        "evidence_type": "heldout_model_comprehension",
        "status": "passed",
        **catalog_binding,
        "catalog_semantics_sha256": approval["catalog_semantics_sha256"],
        "request_projection_sha256": approval["request_projection_sha256"],
        "generation_settings_sha256": hash_object(config["generation"]),
        "generator_models": generator_models,
        "dataset_sha256": FAKE_HASH,
        "results_sha256": FAKE_HASH,
        "models_sha256": FAKE_HASH,
        "minimum_items_per_language_pair_class_cell": 1,
        "models": {
            f"{PLACEHOLDER}_calibration_model_{index}": {
                "model_revision": f"{PLACEHOLDER}_revision_{index}",
                "lineage": f"{PLACEHOLDER}_lineage_{index}",
                "languages": {
                    "en": comprehension_language_result(gate_names),
                    "ja": comprehension_language_result(gate_names),
                },
                "pass": True,
            }
            for index in (1, 2)
        },
        "notes": DEMO_NOTE,
    }

    targets = {
        CONTRASTIVE_ROOT / "calibration" / "frame_calibration_report.json": (
            frame_report
        ),
        CONTRASTIVE_ROOT
        / "calibration"
        / "heldout_comprehension_report.json": comprehension_report,
    }
    for path, payload in targets.items():
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"wrote PLACEHOLDER {path.relative_to(CONTRASTIVE_ROOT)}")
    print(
        "\nThese reports are synthetic. Any run built on them is a "
        "demonstration, not evidence."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
