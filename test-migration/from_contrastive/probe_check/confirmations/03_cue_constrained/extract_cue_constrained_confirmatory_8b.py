#!/usr/bin/env python3
"""Extract the frozen cue-constrained confirmation package at mean layer 6."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
DEFAULT_BANK1 = ROOT / "probe_check" / "format_matched_controls.jsonl"
DEFAULT_BANK2 = ROOT / "probe_check" / "cue_invariant_confirmatory_controls.jsonl"
DEFAULT_AUGMENTATION = ROOT / "probe_check" / "cue_balanced_development_augmentation.jsonl"
DEFAULT_FRESH = ROOT / "probe_check" / "cue_constrained_confirmatory_controls.jsonl"
DEFAULT_RESULT = ROOT / "probe_check" / "metrics" / "cue_constrained_tradeoff_development.json"
EXPECTED_HELPER_SHA256 = "8f7710adfbe63c9572d2a2e770a801bd6f0ae0704cc75f9d8a98827449ab294c"
EXPECTED_BANK1_SHA256 = "fbd320168c38e14321e50cab1872144a3d32e171d4a56afca63dcb6f782131b9"
EXPECTED_BANK2_SHA256 = "0f93681d626c2e4d6d7c91acf9c8301f42f1203f45136d32441a2ef68de654fb"
EXPECTED_AUGMENTATION_SHA256 = "ce7a41bb44b6ede12dee5455b0e8a586d06f15feb491ce4c91e16fdf74bb2b0c"
EXPECTED_FRESH_SHA256 = "676a7e688d1c88af3fd0fab9e12b09ea91ab6cbff3a64be03c3dbae745277c86"
EXPECTED_RESULT_SHA256 = "def7b012c957547300cb2a7a06f1ae6de8b40a197bb609c0debfa0522a7b23d6"
POSITIONS = ["mean"]


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual}; expected {expected}")


def validate_control_rows(
    path: Path, expected_hash: str, expected_count: int, label: str
) -> list[dict[str, Any]]:
    require_hash(path, expected_hash, label)
    rows = sorted(read_jsonl(path), key=lambda row: str(row["control_id"]))
    if len(rows) != expected_count or len({row["control_id"] for row in rows}) != expected_count:
        raise ValueError(f"{label} must contain {expected_count} unique controls")
    return rows


def validate_package(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    helper = (
        PROBE_CHECK
        / "confirmations"
        / "01_format_matched"
        / "extract_confirmatory_8b.py"
    )
    require_hash(helper, EXPECTED_HELPER_SHA256, "extraction helper")
    require_hash(args.development_result, EXPECTED_RESULT_SHA256, "development result")
    result = json.loads(args.development_result.read_text(encoding="utf-8"))
    if result.get("selected_recipe") != "balanced_penalty_256":
        raise ValueError("development result no longer selects balanced_penalty_256")

    items, bank1 = validate_inputs(args.items, args.bank1)
    require_hash(args.bank1, EXPECTED_BANK1_SHA256, "spent bank 1")
    bank2 = validate_control_rows(args.bank2, EXPECTED_BANK2_SHA256, 256, "spent bank 2")
    augmentation = validate_control_rows(
        args.augmentation, EXPECTED_AUGMENTATION_SHA256, 1536, "spent augmentation"
    )
    fresh = validate_control_rows(args.fresh, EXPECTED_FRESH_SHA256, 256, "fresh bank")
    development = sorted([*bank1, *bank2, *augmentation], key=lambda row: str(row["control_id"]))
    if len(development) != 2048 or len({row["control_id"] for row in development}) != 2048:
        raise ValueError("the five spent contexts must contain 2,048 unique controls")
    fresh_cells = Counter(
        (
            row["language"], row["surface"], row["cue_family"],
            row["intended_purpose"], row["lexical_cue"],
        )
        for row in fresh
    )
    if len(fresh_cells) != 32 or set(fresh_cells.values()) != {8}:
        raise ValueError("fresh bank is not a balanced 2 x 2 x 2 factorial")
    all_texts = [
        *(row["text"] for row in items),
        *(row["text"] for row in development),
        *(row["text"] for row in fresh),
    ]
    if len(all_texts) != len(set(all_texts)):
        raise ValueError("confirmation package contains duplicate rendered text")
    return items, development, fresh


def rows_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        raise RuntimeError("the frozen cue-constrained confirmation requires a CUDA A100")
    accelerator = torch.cuda.get_device_name(0)
    if "A100" not in accelerator.upper():
        raise RuntimeError(
            f"the frozen cue-constrained confirmation requires an A100; found {accelerator}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the assigned A100 does not report bfloat16 support")

    slug = model_id.split("/")[-1]
    out = args.out / slug
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        low_cpu_mem_usage=True,
        dtype=torch.bfloat16,
    ).eval()
    if model.config.num_hidden_layers != 32:
        raise RuntimeError(f"{model_id} is not a frozen 32-layer 8B stand-in")

    for label, rows, directory, fields in (
        ("fitting items", items, out, FIT_METADATA_FIELDS),
        ("five spent cue contexts", development, out / "development_controls", CONTROL_METADATA_FIELDS),
        ("untouched confirmation bank", fresh, out / "fresh_confirmation", CONTROL_METADATA_FIELDS),
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
        "analysis": "cue_constrained_fresh_confirmation_v1",
        "model": model_id,
        "n_layers": 32,
        "layers_stored": [FROZEN_LAYER],
        "positions": POSITIONS,
        "gating_cell": {"position": "mean", "layer": FROZEN_LAYER},
        "fitting_recipe": {
            "name": "balanced_penalty_256",
            "base": "explicit_only",
            "control_mass": 1.0,
            "cue_penalty": 256.0,
            "hard_families": [],
        },
        "n_items": len(items),
        "n_development_controls": len(development),
        "n_fresh_confirmation_controls": len(fresh),
        "items_sha256": EXPECTED_ITEMS_SHA256,
        "bank1_sha256": EXPECTED_BANK1_SHA256,
        "bank2_sha256": EXPECTED_BANK2_SHA256,
        "augmentation_sha256": EXPECTED_AUGMENTATION_SHA256,
        "development_controls_rendered_sha256": rows_sha256(development),
        "fresh_controls_sha256": EXPECTED_FRESH_SHA256,
        "development_result_sha256": EXPECTED_RESULT_SHA256,
        "base_extraction_helper_sha256": EXPECTED_HELPER_SHA256,
        "activation_storage_dtype": "float16",
        "model_load_dtype": "bfloat16",
        "load_in_4bit": False,
        "accelerator": accelerator,
        "naturalistic_transfer": "not_loaded",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=DEFAULT_ITEMS)
    parser.add_argument("--bank1", type=Path, default=DEFAULT_BANK1)
    parser.add_argument("--bank2", type=Path, default=DEFAULT_BANK2)
    parser.add_argument("--augmentation", type=Path, default=DEFAULT_AUGMENTATION)
    parser.add_argument("--fresh", type=Path, default=DEFAULT_FRESH)
    parser.add_argument("--development-result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "probe_check" / "out" / "cue_constrained_confirmatory_8b",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--models", nargs="+", choices=FROZEN_MODELS, default=FROZEN_MODELS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    items, development, fresh = validate_package(args)
    print(f"fitting items:              {len(items)} ({EXPECTED_ITEMS_SHA256})")
    print(f"spent development controls:{len(development)}")
    print(f"fresh controls:             {len(fresh)} ({EXPECTED_FRESH_SHA256})")
    print(f"models:                     {args.models}")
    print("method:                     balanced penalty 256; mean layer 6")
    print("naturalistic:               sealed and not loaded")
    if args.check_only:
        print("READY FOR CUE-CONSTRAINED 8B CONFIRMATION EXTRACTION")
        return
    for model_id in args.models:
        extract_model(model_id, items, development, fresh, args)


if __name__ == "__main__":
    main()
