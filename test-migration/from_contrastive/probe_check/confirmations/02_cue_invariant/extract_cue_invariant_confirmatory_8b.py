#!/usr/bin/env python3
"""Extract the frozen cue-invariant 8B confirmation inputs at mean layer 6."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROBE_CHECK = Path(__file__).resolve().parents[2]
ROOT = PROBE_CHECK.parent
sys.path.insert(
    0, str(PROBE_CHECK / "confirmations" / "01_format_matched")
)

from extract_confirmatory_8b import (
    CONTROL_METADATA_FIELDS,
    EXPECTED_ITEMS_SHA256,
    FIT_METADATA_FIELDS,
    FROZEN_LAYER,
    FROZEN_MODELS,
    extract_rows,
    read_jsonl,
    sha256_file,
    validate_inputs,
)


DEFAULT_ITEMS = ROOT / "data" / "processed" / "demo16_v2" / "probe_items.jsonl"
DEFAULT_DEVELOPMENT = ROOT / "probe_check" / "format_matched_controls.jsonl"
DEFAULT_FRESH = ROOT / "probe_check" / "cue_invariant_confirmatory_controls.jsonl"
DEFAULT_DEVELOPMENT_RESULT = ROOT / "probe_check" / "metrics" / "cue_invariant_development.json"
EXPECTED_DEVELOPMENT_SHA256 = "fbd320168c38e14321e50cab1872144a3d32e171d4a56afca63dcb6f782131b9"
EXPECTED_FRESH_SHA256 = "0f93681d626c2e4d6d7c91acf9c8301f42f1203f45136d32441a2ef68de654fb"
DEVELOPMENT_RESULT_SHA256 = "f8ae340fdf900748dd0ea254d7bdcb4d0b4922ba6a76cb004f9c42828919677d"
EXPECTED_HELPER_SHA256 = "8f7710adfbe63c9572d2a2e770a801bd6f0ae0704cc75f9d8a98827449ab294c"
POSITIONS = ["mean"]


def validate_helper() -> None:
    helper = (
        PROBE_CHECK
        / "confirmations"
        / "01_format_matched"
        / "extract_confirmatory_8b.py"
    )
    actual = sha256_file(helper)
    if actual != EXPECTED_HELPER_SHA256:
        raise ValueError(
            "extract_confirmatory_8b.py is missing or incompatible: "
            f"SHA-256 {actual}; expected {EXPECTED_HELPER_SHA256}"
        )


def validate_development_result(path: Path) -> None:
    actual = sha256_file(path)
    if actual != DEVELOPMENT_RESULT_SHA256:
        raise ValueError(f"development-result hash mismatch: {actual}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("selected_method") != "svd90":
        raise ValueError("development result does not select the frozen svd90 method")


def validate_fresh(path: Path, development: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actual = sha256_file(path)
    if actual != EXPECTED_FRESH_SHA256:
        raise ValueError(f"fresh-control hash mismatch: {actual}")
    rows = sorted(read_jsonl(path), key=lambda row: str(row["control_id"]))
    if len(rows) != 256 or len({row["control_id"] for row in rows}) != 256:
        raise ValueError("fresh confirmation must contain 256 unique controls")
    cells = Counter(
        (
            row["language"],
            row["surface"],
            row["cue_family"],
            row["intended_purpose"],
            row["lexical_cue"],
        )
        for row in rows
    )
    if len(cells) != 32 or set(cells.values()) != {8}:
        raise ValueError("fresh confirmation factorial is not balanced")
    if {row["text"] for row in rows} & {row["text"] for row in development}:
        raise ValueError("fresh and development control text overlaps")
    return rows


def extract_model(
    model_id: str,
    items: list[dict[str, Any]],
    development: list[dict[str, Any]],
    fresh: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("the frozen confirmation requires a CUDA A100")
    accelerator = torch.cuda.get_device_name(0)
    if "A100" not in accelerator.upper():
        raise RuntimeError(f"the frozen confirmation requires an A100; found {accelerator}")

    slug = model_id.split("/")[-1]
    out = args.out / slug
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "dtype": torch.bfloat16,
    }
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    if model.config.num_hidden_layers != 32:
        raise RuntimeError(f"{model_id} is not a frozen 32-layer 8B model")

    for label, rows, directory, fields in (
        ("fitting items", items, out, FIT_METADATA_FIELDS),
        ("spent development controls", development, out / "development_controls", CONTROL_METADATA_FIELDS),
        ("fresh confirmation controls", fresh, out / "fresh_confirmation", CONTROL_METADATA_FIELDS),
    ):
        print(f"{model_id}: {label}", flush=True)
        extract_rows(
            model,
            tokenizer,
            rows,
            directory,
            batch_size=args.batch_size,
            metadata_fields=fields,
            positions=POSITIONS,
        )
    config = {
        "analysis": "cue_invariant_fresh_confirmation_v1",
        "model": model_id,
        "n_layers": 32,
        "layers_stored": [FROZEN_LAYER],
        "positions": POSITIONS,
        "gating_cell": {"position": "mean", "layer": FROZEN_LAYER},
        "fitting_recipe": "explicit_only",
        "cue_invariance_method": {
            "method": "matched_cue_difference_svd",
            "variance_fraction": 0.90,
            "development_result_sha256": DEVELOPMENT_RESULT_SHA256,
        },
        "n_items": len(items),
        "n_development_controls": len(development),
        "n_fresh_confirmation_controls": len(fresh),
        "items_sha256": EXPECTED_ITEMS_SHA256,
        "development_controls_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "fresh_controls_sha256": EXPECTED_FRESH_SHA256,
        "activation_storage_dtype": "float16",
        "model_load_dtype": "bfloat16",
        "load_in_4bit": False,
        "accelerator": accelerator,
        "naturalistic_transfer": "not_loaded",
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--development-controls", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--fresh-controls", type=Path, default=DEFAULT_FRESH)
    parser.add_argument(
        "--development-result", type=Path, default=DEFAULT_DEVELOPMENT_RESULT
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "probe_check" / "out" / "cue_invariant_confirmatory_8b",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--models", nargs="+", choices=FROZEN_MODELS, default=FROZEN_MODELS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    validate_helper()
    validate_development_result(args.development_result)
    items, development = validate_inputs(args.items, args.development_controls)
    fresh = validate_fresh(args.fresh_controls, development)
    print(f"fitting items:       {len(items)} ({EXPECTED_ITEMS_SHA256})")
    print(f"development controls:{len(development)} ({EXPECTED_DEVELOPMENT_SHA256})")
    print(f"fresh controls:      {len(fresh)} ({EXPECTED_FRESH_SHA256})")
    print(f"models:              {args.models}")
    print("method:              SVD90 cue projection; explicit-only; mean layer 6")
    print("naturalistic:        sealed and not loaded")
    if args.check_only:
        print("READY FOR CUE-INVARIANT 8B ACTIVATION EXTRACTION")
        return
    for model_id in args.models:
        extract_model(model_id, items, development, fresh, args)


if __name__ == "__main__":
    main()
