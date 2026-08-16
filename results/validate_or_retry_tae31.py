from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval_format_mvp.extract import extract_activations
from validate_activation_archive import validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True, help="frozen canonical UTF-8/LF bank")
    parser.add_argument("--legacy-archive", type=Path, required=True, help="user-owned archive; never modified")
    parser.add_argument("--retry-items", type=Path, required=True)
    parser.add_argument("--retry-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    source = "user_owned_archive"
    selected_archive = args.legacy_archive
    try:
        validation = validate(items=args.items, activations=args.legacy_archive, model=args.model, revision=args.revision)
    except (OSError, ValueError) as original_error:
        source = "parent_owned_canonical_retry"
        if args.retry_output.exists():
            raise FileExistsError(f"retry output already exists: {args.retry_output}") from original_error
        config = extract_activations(
            items_path=args.retry_items,
            output_dir=args.retry_output,
            model_name=args.model,
            model_revision=args.revision,
            layer=0,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device_map="auto",
            all_layers=True,
        )
        del config
        selected_archive = args.retry_output / "activations.npz"
        validation = validate(items=args.items, activations=selected_archive, model=args.model, revision=args.revision)
        validation["recovery_reason"] = str(original_error)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "validated_archive_selected",
        "source": source,
        "selected_archive": str(selected_archive),
        "selected_all_layers": str(selected_archive.parent / "activations_all_layers.npz"),
        "validation": validation,
        "scientific_boundary": "archive provenance and selection only; no model, probe, or causal claim",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
