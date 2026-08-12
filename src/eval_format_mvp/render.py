"""Deterministically render every payload into the complete purpose x format block."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .catalogs import Catalogs
from .io import (
    PROJECT_ROOT,
    sha256_file,
    sha256_object,
    sha256_text,
    write_json,
    write_jsonl,
)
from .schema import validate_against_schema


PURPOSES = ("evaluation", "deployment")
FORMATS = ("benchmark", "casual")


def _options(payload: dict[str, Any]) -> tuple[str, str]:
    values = [str(payload["correct_answer"]), *map(str, payload["distractors"])]
    seed = int(sha256_text(str(payload["payload_block_id"]))[:16], 16)
    random.Random(seed).shuffle(values)
    letters = "ABCD"
    answer_index = values.index(str(payload["correct_answer"]))
    rendered = "\n".join(f"{letter}. {value}" for letter, value in zip(letters, values))
    return rendered, letters[answer_index]


def render_payload(
    payload: dict[str, Any], purpose_pair: dict[str, Any], catalogs: Catalogs
) -> list[dict[str, Any]]:
    payload_hash = sha256_object(payload)
    body = f"{str(payload['context']).strip()}\n\n{str(payload['question']).strip()}"
    options, answer_key = _options(payload)
    rows: list[dict[str, Any]] = []
    for purpose in PURPOSES:
        purpose_text = str(purpose_pair["en"][purpose]).strip()
        for format_name in FORMATS:
            format_row = catalogs.formats[format_name]
            text = str(format_row["template"]).format(
                purpose=purpose_text,
                body=body,
                options=options,
            ).strip()
            item_id = (
                f"item::{payload['payload_block_id']}::{purpose_pair['pair_id']}::"
                f"{'ev' if purpose == 'evaluation' else 'dp'}::"
                f"{'bm' if format_name == 'benchmark' else 'cs'}"
            )
            row = {
                "schema_version": 1,
                "item_id": item_id,
                "payload_block_id": payload["payload_block_id"],
                "topic_id": payload["topic_id"],
                "task_family": payload["task_family"],
                "intended_purpose": purpose,
                "label": 1 if purpose == "evaluation" else 0,
                "format": format_name,
                "purpose_pair_id": purpose_pair["pair_id"],
                "purpose_family_id": purpose_pair["family_id"],
                "cue_locus": purpose_pair["cue_locus"],
                "format_variant_id": format_row["format_variant_id"],
                "format_family_id": format_row["family_id"],
                "text": text,
                "correct_answer": payload["correct_answer"],
                "answer_key": answer_key if format_name == "benchmark" else None,
                "canonical_payload_sha256": payload_hash,
                "rendered_text_sha256": sha256_text(text),
                "split_group_id": payload["payload_block_id"],
            }
            validate_against_schema(row, "rendered_item.schema.json", label=item_id)
            rows.append(row)
    return rows


def render_dataset(
    payloads: list[dict[str, Any]], catalogs: Catalogs
) -> list[dict[str, Any]]:
    purpose_pairs = sorted(catalogs.purpose_pairs, key=lambda row: str(row["pair_id"]))
    rows: list[dict[str, Any]] = []
    for index, payload in enumerate(sorted(payloads, key=lambda row: row["payload_block_id"])):
        pair = purpose_pairs[index % len(purpose_pairs)]
        rows.extend(render_payload(payload, pair, catalogs))
    return rows


def write_rendered_dataset(
    *,
    payload_path: Path,
    output_path: Path,
    manifest_path: Path,
    payloads: list[dict[str, Any]],
    catalogs: Catalogs,
) -> dict[str, Any]:
    rows = render_dataset(payloads, catalogs)
    write_jsonl(output_path, rows)
    counts = {
        "payload_blocks": len(payloads),
        "rendered_items": len(rows),
        "evaluation": sum(row["intended_purpose"] == "evaluation" for row in rows),
        "deployment": sum(row["intended_purpose"] == "deployment" for row in rows),
        "benchmark": sum(row["format"] == "benchmark" for row in rows),
        "casual": sum(row["format"] == "casual" for row in rows),
    }
    manifest = {
        "schema_version": 1,
        "status": "development_unlocked",
        "hash_policy": "observational_not_enforced",
        "inputs": {
            "payloads": {"path": str(payload_path), "sha256": sha256_file(payload_path)},
            "topics": sha256_file(PROJECT_ROOT / "catalogs" / "topics.yaml"),
            "purpose_frames": sha256_file(
                PROJECT_ROOT / "catalogs" / "purpose_frames.yaml"
            ),
            "formats": sha256_file(PROJECT_ROOT / "catalogs" / "formats.yaml"),
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "counts": counts,
    }
    write_json(manifest_path, manifest)
    return manifest

