#!/usr/bin/env python3
"""Validate and materialize the untouched cue-constrained confirmation bank."""

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
from contrastive_generation.utils import ContrastiveError, sha256_file  # noqa: E402


DEFAULT_SPEC = ROOT / "validation" / "cue_constrained_confirmatory_controls.yaml"
DEFAULT_SCHEMA = ROOT / "schemas" / "cue_constrained_confirmatory_controls.schema.json"
DEFAULT_OUTPUT = ROOT / "probe_check" / "cue_constrained_confirmatory_controls.jsonl"
DEVELOPMENT_RESULT = ROOT / "probe_check" / "metrics" / "cue_constrained_tradeoff_development.json"
SPENT_BANKS = (
    ROOT / "probe_check" / "format_matched_controls.jsonl",
    ROOT / "probe_check" / "cue_invariant_confirmatory_controls.jsonl",
    ROOT / "probe_check" / "cue_balanced_development_augmentation.jsonl",
)
EXPECTED_SELECTED_RECIPE = "balanced_penalty_256"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render_bytes(items: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in items
    ).encode("utf-8")


def normalized_cue_line(row: dict[str, Any], separator: str) -> str:
    term = str(row["cue_term"])
    candidates = [
        field
        for field in str(row["text"]).split(separator)
        if term.casefold() in field.casefold()
    ]
    if len(candidates) != 1:
        raise ContrastiveError(f"cannot isolate cue line in {row['control_id']}")
    return candidates[0].casefold().replace(term.casefold(), "{cue_term}")


def validate_bank(path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    value = _validate_asset(path, DEFAULT_SCHEMA.name)
    if value["development"]["result_sha256"] != sha256_file(DEVELOPMENT_RESULT):
        raise ContrastiveError("cue-constrained development result hash is stale")
    result = json.loads(DEVELOPMENT_RESULT.read_text(encoding="utf-8"))
    if result.get("selected_recipe") != EXPECTED_SELECTED_RECIPE:
        raise ContrastiveError("development no longer selects balanced_penalty_256")

    separator = value["rendering"]["separator"]
    old_rows = [row for source in SPENT_BANKS for row in read_jsonl(source)]
    old_texts = {str(row["text"]) for row in old_rows}
    old_terms = {
        (str(row["language"]), str(row["cue_term"]).casefold()) for row in old_rows
    }
    old_lines = {
        (str(row["language"]), normalized_cue_line(row, separator))
        for row in old_rows
    }
    new_templates = value["rendering"]["cue_line_templates"]
    for language, template in new_templates.items():
        if (language, template.casefold()) in old_lines:
            raise ContrastiveError(f"{language} cue-line context is not fresh")

    purposes = {row["purpose_variant_id"]: row for row in value["purpose_variants"]}
    cues = {row["cue_variant_id"]: row for row in value["cue_variants"]}
    blocks = value["blocks"]
    if len(purposes) != 16 or len(cues) != 8:
        raise ContrastiveError("confirmation identifiers are not unique")
    if len({row["control_block_id"] for row in blocks}) != 16:
        raise ContrastiveError("confirmation block identifiers are not unique")
    if Counter(row["surface"] for row in blocks) != {"external": 8, "institutional": 8}:
        raise ContrastiveError("confirmation requires eight blocks per surface")
    for surface in ("external", "institutional"):
        if {row["cue_variant_id"] for row in blocks if row["surface"] == surface} != set(cues):
            raise ContrastiveError(f"{surface} blocks must cross every cue variant")

    canonical = {
        "en": {"evaluation": "evaluation", "deployment": "deployment"},
        "ja": {"evaluation": "評価", "deployment": "運用"},
    }
    new_implicit: set[tuple[str, str]] = set()
    all_new_terms: dict[str, set[str]] = {"en": set(), "ja": set()}
    for cue_id, cue in cues.items():
        for language in ("en", "ja"):
            if cue["explicit_lexeme"][language] != canonical[language]:
                raise ContrastiveError(f"{cue_id}/{language} changes canonical explicit cues")
            for term in cue["implicit_cue"][language].values():
                key = (language, str(term).casefold())
                if key in old_terms or key in new_implicit:
                    raise ContrastiveError(f"implicit cue is not fresh: {key}")
                new_implicit.add(key)
            for family in ("explicit_lexeme", "implicit_cue"):
                all_new_terms[language].update(
                    str(term).casefold() for term in cue[family][language].values()
                )

    items: list[dict[str, Any]] = []
    for block in blocks:
        block_id = block["control_block_id"]
        purpose = purposes.get(block["purpose_variant_id"])
        cue = cues.get(block["cue_variant_id"])
        if purpose is None or cue is None or purpose["surface"] != block["surface"]:
            raise ContrastiveError(f"invalid variant reference in {block_id}")
        for language in ("en", "ja"):
            payload = str(block["payload"][language]).strip()
            invariant = (
                purpose[language]["evaluation"]
                + " "
                + purpose[language]["deployment"]
                + " "
                + payload
            ).casefold()
            leaked = sorted(term for term in all_new_terms[language] if term in invariant)
            if leaked:
                raise ContrastiveError(
                    f"{block_id}/{language} contains bank cue terms outside the cue line: {leaked}"
                )
            for family in ("explicit_lexeme", "implicit_cue"):
                terms = cue[family][language]
                if any(str(term).casefold() in invariant for term in terms.values()):
                    raise ContrastiveError(
                        f"{block_id}/{family}/{language} leaks a cue outside its cue line"
                    )
                for intended in ("evaluation", "deployment"):
                    for lexical in ("evaluation", "deployment"):
                        term = str(terms[lexical])
                        cue_line = new_templates[language].format(cue_term=term)
                        text = separator.join(
                            [purpose[language][intended], cue_line, payload]
                        )
                        counterpart = str(
                            terms[
                                "deployment" if lexical == "evaluation" else "evaluation"
                            ]
                        )
                        folded = text.casefold()
                        if folded.count(term.casefold()) != 1 or counterpart.casefold() in folded:
                            raise ContrastiveError(f"{block_id} cue isolation failed")
                        items.append(
                            {
                                "control_id": (
                                    f"cc3_{language}_{block_id[4:]}_"
                                    f"{'xl' if family == 'explicit_lexeme' else 'ic'}_"
                                    f"{'ev' if intended == 'evaluation' else 'dp'}_"
                                    f"{'ev' if lexical == 'evaluation' else 'dp'}"
                                ),
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
        raise ContrastiveError("confirmation must contain 256 unique items")
    if set(texts) & old_texts:
        raise ContrastiveError("confirmation reuses a spent rendered prompt")
    cells = Counter(
        (
            row["language"],
            row["surface"],
            row["cue_family"],
            row["intended_purpose"],
            row["lexical_cue"],
        )
        for row in items
    )
    if len(cells) != 32 or set(cells.values()) != {8}:
        raise ContrastiveError("confirmation factorial is unbalanced")
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
        if not args.output.exists() or args.output.read_bytes() != payload:
            raise SystemExit(f"{args.output} is missing or stale")
        print("status: frozen, fresh, balanced, and unscored")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"wrote:    {args.output}")


if __name__ == "__main__":
    main()
