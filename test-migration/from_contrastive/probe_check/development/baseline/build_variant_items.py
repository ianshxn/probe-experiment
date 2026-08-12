#!/usr/bin/env python
"""Build a variant probe item set by substituting purpose-frame lines offline.

Why this exists
---------------
`RESULT_newframes_8b.md` established that a frame rewrite can be measured
without regenerating the corpus: the purpose frame occupies the first line of
every rendered item, so substituting that line yields a set that differs from
the base in the frame and nothing else. Payload, folds, container, generator and
every hash-bearing field stay byte-identical, which is what makes the before/
after comparison clean.

That loop was previously run by hand. This script makes it reproducible and
guards the two things a hand-run substitution gets wrong: silently editing more
than the frame line, and producing a frame pair that would not survive the
catalog's `cue_locus` rules once it is committed to `purpose_frames.yaml`.

Standard library only, and deliberately outside the package: adding a dependency
to `contrastive/pyproject.toml` would move `request_projection_sha256` and
invalidate the semantic approval record, for the same reason `comprehension_check/`
is standard-library only.

Usage
-----
    python probe_check/development/baseline/build_variant_items.py --variant v1_judge
    python probe_check/development/baseline/build_variant_items.py --all
    python probe_check/development/baseline/build_variant_items.py --all --check-only

Run from `contrastive/`. Writes
`data/processed/demo16_v2/probe_items.<variant>.jsonl`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import sys
from collections import Counter

PROBE_CHECK = pathlib.Path(__file__).resolve().parents[2]
REPO = PROBE_CHECK.parent
DATA = REPO / "data" / "processed" / "demo16_v2"
SPEC = pathlib.Path(__file__).resolve().with_name("frame_variants.json")
FRAMES = REPO / "purpose_frames.yaml"

# Mirrors contrastive_generation.catalog.IMPLICIT_CUE_SPAN_CAP. Duplicated
# rather than imported because this script must not depend on the package.
IMPLICIT_CUE_SPAN_CAP = 2


class BuildError(Exception):
    pass


def cue_spans(
    eval_text: str, deploy_text: str, language: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Contiguous spans in which the two class members differ.

    Mirrors ``contrastive_generation.catalog.cue_spans`` exactly. English is
    compared on whitespace tokens and Japanese on characters. If that function
    changes, change this one.
    """
    if language == "en":
        left, right = eval_text.split(" "), deploy_text.split(" ")
        join = " ".join
    else:
        left, right = list(eval_text), list(deploy_text)
        join = "".join
    eval_spans: list[str] = []
    deploy_spans: list[str] = []
    matcher = difflib.SequenceMatcher(None, left, right)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i2 > i1:
            eval_spans.append(join(left[i1:i2]))
        if j2 > j1:
            deploy_spans.append(join(right[j1:j2]))
    return tuple(eval_spans), tuple(deploy_spans)


def read_catalog_frames() -> dict[tuple[str, str, str], str]:
    """Parse the frame text out of purpose_frames.yaml without a YAML dependency.

    The file is machine-written and flat, so a targeted line scan is safe here.
    Anything unexpected raises rather than guessing.
    """
    frames: dict[tuple[str, str, str], str] = {}
    pair_id = language = None
    explicitness: dict[str, str] = {}
    locus: dict[str, str] = {}
    current_pair = None
    for raw in FRAMES.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("- pair_id:"):
            current_pair = stripped.split(":", 1)[1].strip()
            pair_id, language = current_pair, None
            continue
        if current_pair is None:
            continue
        if stripped.startswith("explicitness:"):
            explicitness[current_pair] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("cue_locus:"):
            locus[current_pair] = stripped.split(":", 1)[1].strip()
        elif stripped in ("en:", "ja:"):
            language = stripped[:-1]
        elif language and stripped.startswith(("eval:", "deploy:")):
            key, value = stripped.split(":", 1)
            frames[(pair_id, language, key)] = value.strip()
    if not frames:
        raise BuildError(f"No frames parsed from {FRAMES}")
    return {
        "frames": frames,
        "explicitness": explicitness,
        "cue_locus": locus,
    }  # type: ignore[return-value]


def validate_catalog_rules(
    catalog: dict, overrides: dict[tuple[str, str, str], str], variant: str
) -> list[str]:
    """Check the variant's frame set against the rules preflight will enforce.

    This is a pre-check, not a substitute: the authoritative validation runs in
    `_validate_cue_locus` when a winning frame is committed to the catalog. The
    point of running it here is to avoid spending GPU time measuring a candidate
    that could never be adopted.
    """
    frames = dict(catalog["frames"])
    frames.update(overrides)
    implicit = [
        pair
        for pair, kind in catalog["explicitness"].items()
        if kind == "implicit"
    ]
    notes: list[str] = []
    for language in ("en", "ja"):
        span_counts: dict[str, Counter[str]] = {
            "eval": Counter(),
            "deploy": Counter(),
        }
        for pair in sorted(implicit):
            locus = catalog["cue_locus"][pair]
            eval_spans, deploy_spans = cue_spans(
                frames[(pair, language, "eval")],
                frames[(pair, language, "deploy")],
                language,
            )
            observed = max(len(eval_spans), len(deploy_spans))
            if locus == "both":
                if observed < 2:
                    raise BuildError(
                        f"{variant}: {language} pair {pair} declares cue_locus "
                        f"'both' but differs in {observed} span(s)"
                    )
            elif observed != 1:
                raise BuildError(
                    f"{variant}: {language} pair {pair} declares cue_locus "
                    f"{locus!r} but differs in {observed} spans "
                    f"{eval_spans} / {deploy_spans}; a single-locus pair must "
                    "vary in exactly one place"
                )
            span_counts["eval"].update(eval_spans)
            span_counts["deploy"].update(deploy_spans)
        for class_name, counter in span_counts.items():
            for span, count in sorted(counter.items()):
                if count > IMPLICIT_CUE_SPAN_CAP:
                    raise BuildError(
                        f"{variant}: {language} implicit {class_name} cue span "
                        f"{span!r} occurs in {count} pairs "
                        f"(cap {IMPLICIT_CUE_SPAN_CAP}); it would separate the "
                        "classes on its own and survive the holdout"
                    )
                if count == IMPLICIT_CUE_SPAN_CAP:
                    notes.append(
                        f"{language} {class_name} span {span!r} is at the cap "
                        f"({count}/{IMPLICIT_CUE_SPAN_CAP})"
                    )
    return notes


def build(variant: str, spec: dict, catalog: dict, check_only: bool) -> dict:
    overrides: dict[tuple[str, str, str], str] = {}
    for pair, by_language in spec["variants"][variant]["frames"].items():
        for language, by_class in by_language.items():
            for class_name, text in by_class.items():
                overrides[(pair, language, class_name)] = text

    notes = validate_catalog_rules(catalog, overrides, variant)

    base_path = DATA / spec["base_item_set"]
    rows = [json.loads(line) for line in base_path.read_text(encoding="utf-8").splitlines()]

    changed = 0
    touched_pairs: Counter[str] = Counter()
    out_lines: list[str] = []
    for row in rows:
        key = (row["pair_id"], row["language"], row["class"])
        text = row["text"]
        if key in overrides:
            head, sep, tail = text.partition("\n")
            expected = catalog["frames"][key]
            if head != expected:
                raise BuildError(
                    f"{variant}: item {row['item_id']} first line does not match "
                    f"the catalog frame for {key}.\n  first line: {head!r}\n"
                    f"  catalog   : {expected!r}\n"
                    "The base item set was generated from a different catalog "
                    "revision; substitution would silently compare two changes."
                )
            if not sep:
                raise BuildError(
                    f"{variant}: item {row['item_id']} has no line break after "
                    "the purpose frame"
                )
            new_text = overrides[key] + sep + tail
            # Guard the invariant the whole method rests on: only line 1 moves.
            if new_text.partition("\n")[2] != tail:
                raise BuildError(f"{variant}: substitution altered the payload")
            row = {**row, "text": new_text}
            changed += 1
            touched_pairs[f"{key[0]}/{key[1]}/{key[2]}"] += 1
        out_lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))

    if changed == 0:
        raise BuildError(f"{variant}: no items matched the override keys")

    out_path = DATA / f"probe_items.{variant}.jsonl"
    if not check_only:
        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    return {
        "variant": variant,
        "items_changed": changed,
        "items_total": len(rows),
        "by_slot": dict(sorted(touched_pairs.items())),
        "output": str(out_path.relative_to(REPO)),
        "written": not check_only,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate and report without writing the item set",
    )
    args = parser.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    catalog = read_catalog_frames()

    names = list(spec["variants"]) if args.all else args.variant
    if not names:
        parser.error("pass --variant NAME (repeatable) or --all")
    unknown = [n for n in names if n not in spec["variants"]]
    if unknown:
        parser.error(f"unknown variant(s): {unknown}")

    for name in names:
        report = build(name, spec, catalog, args.check_only)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
