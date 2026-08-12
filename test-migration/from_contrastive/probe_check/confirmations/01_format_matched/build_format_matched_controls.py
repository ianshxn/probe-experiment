#!/usr/bin/env python3
"""Materialize the frozen format-matched control specification as JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROBE_CHECK = Path(__file__).resolve().parents[2]
ROOT = PROBE_CHECK.parent
sys.path.insert(0, str(ROOT / "src"))

from contrastive_generation.post_generation import (  # noqa: E402
    FORMAT_MATCHED_CONTROLS_PATH,
    validate_format_matched_controls,
)
from contrastive_generation.utils import sha256_file  # noqa: E402


def rendered_bytes(items: list[dict[str, object]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in items
        )
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=FORMAT_MATCHED_CONTROLS_PATH
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "probe_check" / "format_matched_controls.jsonl",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    value = validate_format_matched_controls(args.spec)
    payload = rendered_bytes(value["items"])
    rendered_sha256 = hashlib.sha256(payload).hexdigest()
    print(f"spec:     {args.spec} ({sha256_file(args.spec)})")
    print(f"items:    {len(value['items'])}")
    print(f"rendered: {rendered_sha256}")

    if args.check_only:
        if args.output.exists() and args.output.read_bytes() != payload:
            raise SystemExit(
                f"{args.output} is stale; rebuild it without --check-only"
            )
        print(
            "status: frozen and structurally valid; no activations were loaded "
            "or scored"
        )
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"wrote:    {args.output}")


if __name__ == "__main__":
    main()
