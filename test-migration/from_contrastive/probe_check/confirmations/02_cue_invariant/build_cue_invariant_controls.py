#!/usr/bin/env python3
"""Validate and materialize the second fresh cue-invariant confirmation bank."""

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
from contrastive_generation.post_generation import (  # noqa: E402
    _validate_asset,
    validate_format_matched_controls,
)
from contrastive_generation.utils import (  # noqa: E402
    ContrastiveError,
    load_json,
    load_yaml,
    sha256_file,
)


DEFAULT_SPEC = ROOT / "validation" / "cue_invariant_confirmatory_controls.yaml"
DEFAULT_SCHEMA = ROOT / "schemas" / "cue_invariant_confirmatory_controls.schema.json"
DEFAULT_OUTPUT = ROOT / "probe_check" / "cue_invariant_confirmatory_controls.jsonl"
DEVELOPMENT_RESULT = ROOT / "probe_check" / "metrics" / "cue_invariant_development.json"
SPENT_ITEMS = ROOT / "probe_check" / "format_matched_controls.jsonl"


def render_bytes(items: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in items
    ).encode("utf-8")


def validate_bank(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    value = _validate_asset(path, DEFAULT_SCHEMA.name)
    if value["development"]["result_sha256"] != sha256_file(DEVELOPMENT_RESULT):
        raise ContrastiveError("cue-invariant development result hash is stale")
    if value["development"]["spent_controls_sha256"] != sha256_file(SPENT_ITEMS):
        raise ContrastiveError("spent-control hash is stale")
    development = load_json(DEVELOPMENT_RESULT)
    if development.get("selected_method") != "svd90":
        raise ContrastiveError("development artifact does not select svd90")

    purposes = {row["purpose_variant_id"]: row for row in value["purpose_variants"]}
    cues = {row["cue_variant_id"]: row for row in value["cue_variants"]}
    blocks = value["blocks"]
    if len(purposes) != 16 or len(cues) != 8 or len({row["control_block_id"] for row in blocks}) != 16:
        raise ContrastiveError("second confirmation identifiers must be unique")
    if Counter(row["surface"] for row in blocks) != {"external": 8, "institutional": 8}:
        raise ContrastiveError("second confirmation needs eight blocks per surface")
    for surface in ("external", "institutional"):
        used = {row["cue_variant_id"] for row in blocks if row["surface"] == surface}
        if used != set(cues):
            raise ContrastiveError(f"{surface} blocks must cross all cue variants")

    first = validate_format_matched_controls()
    old_texts = {row["text"] for row in first["items"]}
    old_terms = {
        (row["language"], row["cue_term"].casefold()) for row in first["items"]
    }
    legacy = load_yaml(ROOT / "validation" / "lexical_controls.yaml")
    old_terms.update(
        (row["language"], row["cue_term"].casefold()) for row in legacy["items"]
    )
    canonical = {
        "en": {"evaluation": "evaluation", "deployment": "deployment"},
        "ja": {"evaluation": "評価", "deployment": "運用"},
    }
    new_implicit: set[tuple[str, str]] = set()
    for cue_id, row in cues.items():
        for language in ("en", "ja"):
            if row["explicit_lexeme"][language] != canonical[language]:
                raise ContrastiveError(f"{cue_id}/{language} changes canonical explicit cues")
            for term in row["implicit_cue"][language].values():
                key = (language, str(term).casefold())
                if key in old_terms or key in new_implicit:
                    raise ContrastiveError(f"implicit cue is not fresh: {key}")
                new_implicit.add(key)

    sep = value["rendering"]["separator"]
    templates = value["rendering"]["cue_line_templates"]
    items: list[dict[str, Any]] = []
    for block in blocks:
        block_id = block["control_block_id"]
        purpose = purposes.get(block["purpose_variant_id"])
        cue = cues.get(block["cue_variant_id"])
        if purpose is None or cue is None or purpose["surface"] != block["surface"]:
            raise ContrastiveError(f"invalid variant reference in {block_id}")
        for language in ("en", "ja"):
            payload = str(block["payload"][language]).strip()
            minimum, maximum = ((200, 1400) if language == "en" else (80, 700))
            if not minimum <= len(payload) <= maximum:
                raise ContrastiveError(f"{block_id}/{language} payload length {len(payload)} is invalid")
            for family in ("explicit_lexeme", "implicit_cue"):
                terms = cue[family][language]
                invariant = (
                    purpose[language]["evaluation"]
                    + " "
                    + purpose[language]["deployment"]
                    + " "
                    + payload
                ).casefold()
                if any(str(term).casefold() in invariant for term in terms.values()):
                    raise ContrastiveError(f"{block_id}/{family}/{language} leaks a cue outside its cue line")
                for intended in ("evaluation", "deployment"):
                    for lexical in ("evaluation", "deployment"):
                        term = str(terms[lexical])
                        text = sep.join(
                            [
                                purpose[language][intended],
                                templates[language].format(cue_term=term),
                                payload,
                            ]
                        )
                        counterpart = str(terms["deployment" if lexical == "evaluation" else "evaluation"])
                        folded = text.casefold()
                        if folded.count(term.casefold()) != 1 or counterpart.casefold() in folded:
                            raise ContrastiveError(f"{block_id} cue isolation failed")
                        items.append(
                            {
                                "control_id": f"ci2_{language}_{block_id[4:]}_{'xl' if family == 'explicit_lexeme' else 'ic'}_{'ev' if intended == 'evaluation' else 'dp'}_{'ev' if lexical == 'evaluation' else 'dp'}",
                                "control_block_id": block_id,
                                "purpose_variant_id": block["purpose_variant_id"],
                                "cue_variant_id": block["cue_variant_id"],
                                "language": language,
                                "surface": block["surface"],
                                "cue_family": family,
                                "intended_purpose": intended,
                                "lexical_cue": lexical,
                                "cue_term": term,
                                "text": text,
                            }
                        )
    ids = [row["control_id"] for row in items]
    texts = [row["text"] for row in items]
    if len(items) != 256 or len(set(ids)) != 256 or len(set(texts)) != 256:
        raise ContrastiveError("second confirmation must contain 256 unique items")
    if set(texts) & old_texts:
        raise ContrastiveError("second confirmation reuses first-bank text")
    cells = Counter(
        (row["language"], row["surface"], row["cue_family"], row["intended_purpose"], row["lexical_cue"])
        for row in items
    )
    if len(cells) != 32 or set(cells.values()) != {8}:
        raise ContrastiveError("second confirmation factorial is unbalanced")
    return {**value, "items": items}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    value = validate_bank(args.spec)
    payload = render_bytes(value["items"])
    print(f"spec:     {args.spec} ({sha256_file(args.spec)})")
    print(f"items:    {len(value['items'])}")
    print(f"rendered: {hashlib.sha256(payload).hexdigest()}")
    if args.check_only:
        if args.output.exists() and args.output.read_bytes() != payload:
            raise SystemExit(f"{args.output} is stale")
        print("status: frozen, fresh, balanced, and unscored")
        return
    args.output.write_bytes(payload)
    print(f"wrote:    {args.output}")


if __name__ == "__main__":
    main()
