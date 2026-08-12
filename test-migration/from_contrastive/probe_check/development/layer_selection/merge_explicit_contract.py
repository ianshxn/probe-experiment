#!/usr/bin/env python3
"""Merge completed per-model explicit-contract forecasts without recomputation."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    documents = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.inputs
    ]
    result = deepcopy(documents[0])
    result["models"] = {}
    for path, document in zip(args.inputs, documents):
        if document["analysis"] != result["analysis"]:
            raise SystemExit(f"analysis mismatch in {path}")
        if document["thresholds"] != result["thresholds"]:
            raise SystemExit(f"threshold mismatch in {path}")
        overlap = set(result["models"]) & set(document["models"])
        if overlap:
            raise SystemExit(f"duplicate models in {path}: {sorted(overlap)}")
        result["models"].update(document["models"])

    common: set[str] | None = None
    for model in result["models"].values():
        keys = set(model["cells"])
        common = keys if common is None else common & keys
    common = common or set()

    def joint(flag: str) -> list[str]:
        return sorted(
            key
            for key in common
            if all(
                model["cells"][key][flag]
                for model in result["models"].values()
            )
        )

    result["joint_cells"] = {
        "evaluated": len(common),
        "core_controlled_gates_pass_all_models": joint(
            "core_controlled_gates_pass"
        ),
        "all_available_machine_gates_pass_all_models": joint(
            "all_available_machine_gates_pass"
        ),
        "all_available_gates_with_threshold_free_controls_pass_all_models": joint(
            "all_available_gates_with_threshold_free_controls_pass"
        ),
    }
    result["joint_status"] = (
        "not_adoptable_under_current_machine_contract; "
        "naturalistic_transfer_not_scored"
    )
    result["source_files"] = [str(path) for path in args.inputs]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
