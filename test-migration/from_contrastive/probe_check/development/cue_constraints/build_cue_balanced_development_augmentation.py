#!/usr/bin/env python3
"""Build extra cue contexts from the two already-spent development banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROBE_CHECK = Path(__file__).resolve().parents[2]
ROOT = PROBE_CHECK.parent
sys.path.insert(0, str(ROOT / "src"))
from contrastive_generation.post_generation import _validate_asset  # noqa: E402
from contrastive_generation.utils import (  # noqa: E402
    ContrastiveError,
    load_json,
    sha256_file,
)


DEFAULT_SPEC = ROOT / "validation" / "cue_balanced_development_augmentation.yaml"
DEFAULT_SCHEMA = ROOT / "schemas" / "cue_balanced_development_augmentation.schema.json"
DEFAULT_OUTPUT = ROOT / "probe_check" / "cue_balanced_development_augmentation.jsonl"
REQUIRED_FIELDS = {
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
    "text",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rendered_bytes(items: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in items
    ).encode("utf-8")


def validate_and_render(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    value = _validate_asset(path, DEFAULT_SCHEMA.name)
    trigger = ROOT / value["trigger_result"]["path"]
    if sha256_file(trigger) != value["trigger_result"]["sha256"]:
        raise ContrastiveError("cue-balanced trigger-result hash is stale")
    result = load_json(trigger)
    if "selected_recipe" not in result:
        raise ContrastiveError("trigger result does not declare selected_recipe")
    if result["selected_recipe"] is not value["trigger_result"]["required_selected_recipe"]:
        raise ContrastiveError("trigger result no longer records the expected null selection")

    separator = value["rendering"]["separator"]
    source_rows: list[tuple[str, dict[str, Any], list[str]]] = []
    source_texts: set[str] = set()
    source_names: set[str] = set()
    for source in value["source_banks"]:
        source_name = source["source_bank"]
        if source_name in source_names:
            raise ContrastiveError(f"duplicate source-bank name: {source_name}")
        source_names.add(source_name)
        source_path = ROOT / source["path"]
        if sha256_file(source_path) != source["sha256"]:
            raise ContrastiveError(f"source-bank hash is stale: {source_name}")
        rows = read_jsonl(source_path)
        if len(rows) != 256 or len({row.get("control_id") for row in rows}) != 256:
            raise ContrastiveError(f"{source_name} is not a 256-item unique bank")
        for row in rows:
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ContrastiveError(f"{source_name} row lacks {sorted(missing)}")
            fields = str(row["text"]).split(separator)
            if len(fields) != 3:
                raise ContrastiveError(f"{source_name}/{row['control_id']} is not three-field text")
            purpose, cue_line, payload = fields
            term = str(row["cue_term"])
            if cue_line.casefold().count(term.casefold()) != 1:
                raise ContrastiveError(f"{source_name}/{row['control_id']} cue is not isolated")
            if term.casefold() in (purpose + " " + payload).casefold():
                raise ContrastiveError(f"{source_name}/{row['control_id']} cue leaks outside cue line")
            source_texts.add(str(row["text"]))
            source_rows.append((source_name, row, [purpose, cue_line, payload]))
    if len(source_rows) != 512 or len(source_texts) != 512:
        raise ContrastiveError("the two spent sources must provide 512 unique texts")

    templates = value["rendering"]["templates"]
    template_ids = [row["cue_template_id"] for row in templates]
    if len(set(template_ids)) != len(template_ids):
        raise ContrastiveError("cue-template identifiers must be unique")
    items: list[dict[str, Any]] = []
    for template in templates:
        template_id = template["cue_template_id"]
        order = template["field_order"]
        if set(order) != {"purpose", "cue_line", "payload"}:
            raise ContrastiveError(f"{template_id} must render all three fields once")
        for source_name, row, fields in source_rows:
            purpose, _, payload = fields
            term = str(row["cue_term"])
            cue_line = template["cue_line_templates"][row["language"]].format(
                cue_term=term
            )
            rendered = separator.join(
                {"purpose": purpose, "cue_line": cue_line, "payload": payload}[field]
                for field in order
            )
            if rendered.casefold().count(term.casefold()) != 1:
                raise ContrastiveError(f"{template_id}/{row['control_id']} cue isolation failed")
            items.append(
                {
                    **{key: row[key] for key in REQUIRED_FIELDS - {"control_id", "control_block_id", "text"}},
                    "control_id": f"cba_{template_id}_{source_name}_{row['control_id']}",
                    "control_block_id": f"cba_{template_id}_{source_name}_{row['control_block_id']}",
                    "cue_template_id": template_id,
                    "source_bank": source_name,
                    "source_control_id": row["control_id"],
                    "text": rendered,
                }
            )
    ids = [row["control_id"] for row in items]
    texts = [row["text"] for row in items]
    if len(items) != 1536 or len(set(ids)) != 1536 or len(set(texts)) != 1536:
        raise ContrastiveError("augmentation must contain 1,536 unique items")
    if set(texts) & source_texts:
        raise ContrastiveError("augmentation unexpectedly reproduces a source text")
    cells = Counter(
        (
            row["cue_template_id"],
            row["language"],
            row["surface"],
            row["cue_family"],
            row["intended_purpose"],
            row["lexical_cue"],
        )
        for row in items
    )
    if len(cells) != 96 or set(cells.values()) != {16}:
        raise ContrastiveError("augmentation is not balanced within every cue template")
    return {**value, "items": items}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    value = validate_and_render(args.spec)
    payload = rendered_bytes(value["items"])
    digest = hashlib.sha256(payload).hexdigest()
    print(f"spec:     {args.spec} ({sha256_file(args.spec)})")
    print(f"items:    {len(value['items'])}")
    print(f"rendered: {digest}")
    if args.check_only:
        if not args.output.exists() or args.output.read_bytes() != payload:
            raise SystemExit(f"{args.output} is missing or stale")
        print("status: development-only, balanced, provenance-bound, and current")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"wrote:    {args.output}")


if __name__ == "__main__":
    main()
