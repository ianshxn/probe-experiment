#!/usr/bin/env python3
"""Extract the frozen 8B confirmatory cell and its declared diagnostic.

This script is intentionally narrow. It re-extracts the unchanged demo fitting
items and the previously unscored format-matched controls in the same model run,
at mean layer 6 (the frozen gate) and last layer 6 (diagnostic only). It never
loads the sealed naturalistic-transfer set.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROBE_CHECK = Path(__file__).resolve().parents[2]
ROOT = PROBE_CHECK.parent
DEFAULT_ITEMS = ROOT / "data" / "processed" / "demo16_v2" / "probe_items.jsonl"
DEFAULT_CONTROLS = ROOT / "probe_check" / "format_matched_controls.jsonl"
EXPECTED_ITEMS_SHA256 = "97796dcf3f54918cb7b55c009f6458832fcf75d880df50252ffdf8415c6bc84e"
EXPECTED_CONTROLS_SHA256 = "fbd320168c38e14321e50cab1872144a3d32e171d4a56afca63dcb6f782131b9"
FROZEN_MODELS = [
    "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
]
FROZEN_LAYER = 6
POSITIONS = ["mean", "last"]
GATING_POSITION = "mean"

# The processed fitting JSONL intentionally carries only rendered-item fields.
# The original activation extraction joined cue_locus from catalog 2.3.2 before
# writing meta.jsonl, and analyze.load requires that field even when fitting on
# explicit items only. Keep the same frozen join here without adding a PyYAML
# dependency to the GPU handoff environment.
CUE_LOCUS_BY_PAIR = {
    "p01": "both",
    "p02": "verb_only",
    "p03": "both",
    "p04": "verb_only",
    "p05": "both",
    "p06": "noun_only",
    "p07": "noun_only",
    "p08": "both",
    "p09": "noun_only",
    "p10": "verb_only",
    "p11": "noun_only",
    "p12": "noun_only",
    "p13": "noun_only",
    "p14": "noun_only",
    "p15": "noun_only",
    "p16": "noun_only",
    "p17": "noun_only",
    "p18": "noun_only",
    "p19": "noun_only",
    "p20": "noun_only",
}
FIT_METADATA_FIELDS = [
    "item_id",
    "pair_id",
    "language",
    "class",
    "label",
    "explicitness",
    "frame_family_id",
    "surface",
    "job_id",
    "content_sha256",
    "content_spec_id",
    "topic_id",
    "cue_locus",
]
CONTROL_METADATA_FIELDS = [
    "control_id",
    "control_block_id",
    "purpose_variant_id",
    "cue_variant_id",
    "language",
    "surface",
    "cue_family",
    "intended_purpose",
    "lexical_cue",
    "cue_term",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain one JSON object per line")
    return rows


def validate_inputs(items_path: Path, controls_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    item_hash = sha256_file(items_path)
    control_hash = sha256_file(controls_path)
    if item_hash != EXPECTED_ITEMS_SHA256:
        raise ValueError(
            f"fitting item hash mismatch: {item_hash}; expected {EXPECTED_ITEMS_SHA256}"
        )
    if control_hash != EXPECTED_CONTROLS_SHA256:
        raise ValueError(
            f"confirmatory control hash mismatch: {control_hash}; expected "
            f"{EXPECTED_CONTROLS_SHA256}"
        )
    items = [
        row
        for row in read_jsonl(items_path)
        if row.get("source") == "matched_generated"
    ]
    items.sort(key=lambda row: str(row["item_id"]))
    pair_ids = {str(row["pair_id"]) for row in items}
    if pair_ids != set(CUE_LOCUS_BY_PAIR):
        raise ValueError(
            "fitting item pair identifiers do not match the frozen catalog: "
            f"{sorted(pair_ids)}"
        )
    items = [
        {**row, "cue_locus": CUE_LOCUS_BY_PAIR[str(row["pair_id"])]}
        for row in items
    ]
    controls = read_jsonl(controls_path)
    controls.sort(key=lambda row: str(row["control_id"]))
    if len(items) != 1200:
        raise ValueError(f"expected 1200 fitting items, found {len(items)}")
    if len(controls) != 256:
        raise ValueError(f"expected 256 confirmatory controls, found {len(controls)}")
    item_cells = Counter(
        (row["language"], row["explicitness"], row["class"]) for row in items
    )
    if set(item_cells.values()) != {150} or len(item_cells) != 8:
        raise ValueError(f"fitting grid is not balanced: {dict(item_cells)}")
    control_cells = Counter(
        (
            row["language"],
            row["surface"],
            row["cue_family"],
            row["intended_purpose"],
            row["lexical_cue"],
        )
        for row in controls
    )
    if set(control_cells.values()) != {8} or len(control_cells) != 32:
        raise ValueError(f"confirmatory grid is not balanced: {dict(control_cells)}")
    if {row["text"] for row in items} & {row["text"] for row in controls}:
        raise ValueError("confirmatory controls overlap fitting text")
    return items, controls


def extract_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    out: Path,
    *,
    batch_size: int,
    metadata_fields: list[str],
    positions: list[str] = POSITIONS,
) -> None:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    if not positions or not set(positions) <= set(POSITIONS):
        raise ValueError(f"unsupported extraction positions: {positions}")
    features: dict[str, list[Any]] = {position: [] for position in positions}
    metadata = out / "meta.jsonl"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        with torch.inference_mode():
            for start in range(0, len(rows), batch_size):
                chunk = rows[start : start + batch_size]
                texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": str(row["text"])}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for row in chunk
                ]
                encoded = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                if encoded["attention_mask"].shape[1] > 1024:
                    raise RuntimeError(
                        "a confirmatory prompt exceeds 1024 tokens; no truncation is allowed"
                    )
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                hidden = model(**encoded, output_hidden_states=True).hidden_states[
                    FROZEN_LAYER
                ]
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                if "last" in positions:
                    features["last"].append(hidden[:, -1].float().cpu())
                if "mean" in positions:
                    features["mean"].append(
                        ((hidden * mask).sum(1) / mask.sum(1)).float().cpu()
                    )
                for row in chunk:
                    handle.write(
                        json.dumps(
                            {
                                key: row[key]
                                for key in metadata_fields
                                if key in row
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                done = min(start + batch_size, len(rows))
                print(f"    {done}/{len(rows)}", flush=True)
    for position, parts in features.items():
        torch.save(
            torch.cat(parts).half(),
            out / f"acts_layer{FROZEN_LAYER}_{position}.pt",
        )


def extract_model(model_id: str, items: list[dict[str, Any]], controls: list[dict[str, Any]], args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    slug = model_id.split("/")[-1]
    out = args.out / slug
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {"device_map": "auto", "low_cpu_mem_usage": True}
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    if model.config.num_hidden_layers != 32:
        raise RuntimeError(
            f"{model_id} has {model.config.num_hidden_layers} layers; this frozen "
            "8B confirmation requires a 32-layer model"
        )
    print(f"{model_id}: fitting items", flush=True)
    extract_rows(
        model,
        tokenizer,
        items,
        out,
        batch_size=args.batch_size,
        metadata_fields=FIT_METADATA_FIELDS,
    )
    print(f"{model_id}: frozen confirmatory controls", flush=True)
    extract_rows(
        model,
        tokenizer,
        controls,
        out / "confirmatory_controls",
        batch_size=args.batch_size,
        metadata_fields=CONTROL_METADATA_FIELDS,
    )
    config = {
        "analysis": "format_matched_confirmatory_controls_v1",
        "model": model_id,
        "n_layers": 32,
        "layers_stored": [FROZEN_LAYER],
        "positions": POSITIONS,
        "gating_cell": {"position": GATING_POSITION, "layer": FROZEN_LAYER},
        "diagnostic_only_cells": [{"position": "last", "layer": FROZEN_LAYER}],
        "fitting_recipe": "explicit_only",
        "n_items": len(items),
        "n_confirmatory_controls": len(controls),
        "items_sha256": EXPECTED_ITEMS_SHA256,
        "controls_sha256": EXPECTED_CONTROLS_SHA256,
        "activation_storage_dtype": "float16",
        "model_load_dtype": args.dtype if not args.load_in_4bit else "4bit_nf4",
        "load_in_4bit": args.load_in_4bit,
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
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "probe_check" / "out" / "confirmatory_8b"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=FROZEN_MODELS,
        default=FROZEN_MODELS,
        help=(
            "one or both frozen 8B model identifiers; selecting models only "
            "splits the extraction across resumable Colab cells"
        ),
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    items, controls = validate_inputs(args.items, args.controls)
    print(f"fitting items: {len(items)} ({EXPECTED_ITEMS_SHA256})")
    print(f"controls:      {len(controls)} ({EXPECTED_CONTROLS_SHA256})")
    print(f"models:        {args.models}")
    print(f"gate:          mean layer 6; last layer 6 is diagnostic only")
    print("naturalistic:  sealed and not loaded")
    if args.check_only:
        print("READY FOR 8B ACTIVATION EXTRACTION")
        return
    for model_id in args.models:
        extract_model(model_id, items, controls, args)


if __name__ == "__main__":
    main()
