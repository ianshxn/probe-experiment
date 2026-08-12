#!/usr/bin/env python3
"""Extract the expanded cue-balanced development package at mean layer 6.

All inputs are development data.  The original fitting set, both spent control
banks, and the cue-context augmentation are re-extracted in one model session
so activation differences cannot be attributed to a changed runtime.
"""

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
DEFAULT_BANK1 = ROOT / "probe_check" / "format_matched_controls.jsonl"
DEFAULT_BANK2 = ROOT / "probe_check" / "cue_invariant_confirmatory_controls.jsonl"
DEFAULT_AUGMENTATION = ROOT / "probe_check" / "cue_balanced_development_augmentation.jsonl"
DEFAULT_TRIGGER = ROOT / "probe_check" / "metrics" / "cue_balanced_development.json"
EXPECTED_HELPER_SHA256 = "8f7710adfbe63c9572d2a2e770a801bd6f0ae0704cc75f9d8a98827449ab294c"
EXPECTED_BANK1_SHA256 = "fbd320168c38e14321e50cab1872144a3d32e171d4a56afca63dcb6f782131b9"
EXPECTED_BANK2_SHA256 = "0f93681d626c2e4d6d7c91acf9c8301f42f1203f45136d32441a2ef68de654fb"
EXPECTED_AUGMENTATION_SHA256 = "ce7a41bb44b6ede12dee5455b0e8a586d06f15feb491ce4c91e16fdf74bb2b0c"
EXPECTED_TRIGGER_SHA256 = "a7fd030bd0d3220d61d4c95a2e67cb91d2b6c033ff6c5a3e8c8dc5219c90781f"
POSITIONS = ["mean"]
AUGMENTATION_METADATA_FIELDS = CONTROL_METADATA_FIELDS + [
    "cue_template_id",
    "source_bank",
    "source_control_id",
]


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual}; expected {expected}")


def validate_package(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    helper = (
        PROBE_CHECK
        / "confirmations"
        / "01_format_matched"
        / "extract_confirmatory_8b.py"
    )
    require_hash(helper, EXPECTED_HELPER_SHA256, "extraction helper")
    require_hash(args.trigger_result, EXPECTED_TRIGGER_SHA256, "trigger result")
    trigger = json.loads(args.trigger_result.read_text(encoding="utf-8"))
    if "selected_recipe" not in trigger or trigger["selected_recipe"] is not None:
        raise ValueError("trigger result must record no selected reciprocal recipe")

    items, bank1 = validate_inputs(args.items, args.bank1)
    require_hash(args.bank1, EXPECTED_BANK1_SHA256, "bank 1")
    require_hash(args.bank2, EXPECTED_BANK2_SHA256, "bank 2")
    require_hash(args.augmentation, EXPECTED_AUGMENTATION_SHA256, "augmentation")
    bank2 = sorted(read_jsonl(args.bank2), key=lambda row: str(row["control_id"]))
    augmentation = sorted(
        read_jsonl(args.augmentation), key=lambda row: str(row["control_id"])
    )
    if len(bank2) != 256 or len({row["control_id"] for row in bank2}) != 256:
        raise ValueError("bank 2 must contain 256 unique rows")
    if len(augmentation) != 1536 or len({row["control_id"] for row in augmentation}) != 1536:
        raise ValueError("augmentation must contain 1,536 unique rows")
    augmentation_cells = Counter(
        (
            row["cue_template_id"],
            row["language"],
            row["surface"],
            row["cue_family"],
            row["intended_purpose"],
            row["lexical_cue"],
        )
        for row in augmentation
    )
    if len(augmentation_cells) != 96 or set(augmentation_cells.values()) != {16}:
        raise ValueError("augmentation is not balanced within template")
    all_texts = [
        *(row["text"] for row in items),
        *(row["text"] for row in bank1),
        *(row["text"] for row in bank2),
        *(row["text"] for row in augmentation),
    ]
    if len(all_texts) != len(set(all_texts)):
        raise ValueError("development package contains duplicate rendered text")
    return items, bank1, bank2, augmentation


def extract_model(
    model_id: str,
    items: list[dict[str, Any]],
    bank1: list[dict[str, Any]],
    bank2: list[dict[str, Any]],
    augmentation: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("cue-balanced development extraction requires a CUDA A100")
    accelerator = torch.cuda.get_device_name(0)
    if "A100" not in accelerator.upper():
        raise RuntimeError(f"cue-balanced development requires an A100; found {accelerator}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the assigned A100 runtime does not report bfloat16 support")

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
        raise RuntimeError(f"{model_id} is not a 32-layer 8B stand-in")

    packages = (
        ("fitting items", items, out, FIT_METADATA_FIELDS),
        ("spent bank 1", bank1, out / "bank1", CONTROL_METADATA_FIELDS),
        ("spent bank 2", bank2, out / "bank2", CONTROL_METADATA_FIELDS),
        (
            "cue-context augmentation",
            augmentation,
            out / "augmentation",
            AUGMENTATION_METADATA_FIELDS,
        ),
    )
    for label, rows, directory, fields in packages:
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
        "analysis": "cue_balanced_augmented_development_v1",
        "development_only": True,
        "model": model_id,
        "n_layers": 32,
        "layers_stored": [FROZEN_LAYER],
        "positions": POSITIONS,
        "cell": {"position": "mean", "layer": FROZEN_LAYER},
        "n_items": len(items),
        "n_bank1_controls": len(bank1),
        "n_bank2_controls": len(bank2),
        "n_augmentation_controls": len(augmentation),
        "items_sha256": EXPECTED_ITEMS_SHA256,
        "bank1_sha256": EXPECTED_BANK1_SHA256,
        "bank2_sha256": EXPECTED_BANK2_SHA256,
        "augmentation_sha256": EXPECTED_AUGMENTATION_SHA256,
        "trigger_result_sha256": EXPECTED_TRIGGER_SHA256,
        "helper_sha256": EXPECTED_HELPER_SHA256,
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
    parser.add_argument("--trigger-result", type=Path, default=DEFAULT_TRIGGER)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "probe_check" / "out" / "cue_balanced_development_8b",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--models", nargs="+", choices=FROZEN_MODELS, default=FROZEN_MODELS)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    items, bank1, bank2, augmentation = validate_package(args)
    print(f"fitting items: {len(items)} ({EXPECTED_ITEMS_SHA256})")
    print(f"spent bank 1: {len(bank1)} ({EXPECTED_BANK1_SHA256})")
    print(f"spent bank 2: {len(bank2)} ({EXPECTED_BANK2_SHA256})")
    print(f"augmentation: {len(augmentation)} ({EXPECTED_AUGMENTATION_SHA256})")
    print(f"models: {args.models}")
    print("cell: mean layer 6; exact bfloat16; A100 required")
    print("naturalistic: sealed and not loaded")
    if args.check_only:
        print("READY FOR CUE-BALANCED DEVELOPMENT EXTRACTION")
        return
    for model_id in args.models:
        extract_model(model_id, items, bank1, bank2, augmentation, args)


if __name__ == "__main__":
    main()
