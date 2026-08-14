"""Command-line interface for the English purpose-by-format MVP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import analyze_activations
from .catalogs import load_catalogs
from .extract import extract_activations
from .generation import generate_validated_payloads
from .io import MVPError, PROJECT_ROOT, load_jsonl, write_json
from .payloads import load_and_validate_payloads, write_generation_requests
from .render import write_rendered_dataset
from .validate import validate_rendered_dataset


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-format-mvp",
        description="English matched-payload purpose x format probing MVP",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    requests = commands.add_parser("requests", help="write offline generation requests")
    requests.add_argument("--out", type=_path, required=True)
    requests.add_argument("--per-topic", type=int, default=5)

    generate = commands.add_parser(
        "generate",
        help="generate with one OpenRouter model and validate with another",
    )
    generate.add_argument("--generator-profile", type=_path, required=True)
    generate.add_argument("--validator-profile", type=_path, required=True)
    generate.add_argument("--out", type=_path, required=True)
    generate.add_argument("--run-dir", type=_path, required=True)
    generate.add_argument("--per-topic", type=int, default=5)
    generate.add_argument("--max-generation-attempts", type=int, default=6)
    generate.add_argument("--max-semantic-attempts", type=int, default=3)
    generate.add_argument("--max-validator-attempts", type=int, default=2)
    generate.add_argument(
        "--max-attempts",
        type=int,
        help="legacy alias for --max-semantic-attempts; generator budget remains separate",
    )
    generate.add_argument("--limit", type=int)

    payloads = commands.add_parser("validate-payloads", help="validate payload JSONL")
    payloads.add_argument("--payloads", type=_path, required=True)

    render = commands.add_parser("render", help="render the complete English 2x2")
    render.add_argument("--payloads", type=_path, required=True)
    render.add_argument("--out", type=_path, required=True)
    render.add_argument("--manifest", type=_path, required=True)

    rendered = commands.add_parser(
        "validate-rendered", help="reconstruct and validate rendered JSONL"
    )
    rendered.add_argument("--payloads", type=_path, required=True)
    rendered.add_argument("--items", type=_path, required=True)
    rendered.add_argument("--report", type=_path)

    extract = commands.add_parser("extract", help="extract fixed model activations")
    extract.add_argument("--items", type=_path, required=True)
    extract.add_argument("--out-dir", type=_path, required=True)
    extract.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct"
    )
    extract.add_argument("--revision")
    extract.add_argument("--layer", type=int, default=24)
    extract.add_argument("--batch-size", type=int, default=8)
    extract.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="maximum allowed prompt tokens; longer prompts fail rather than truncate",
    )

    analyze = commands.add_parser("analyze", help="run grouped probe comparisons")
    analyze.add_argument("--items", type=_path, required=True)
    analyze.add_argument("--activations", type=_path, required=True)
    analyze.add_argument("--out", type=_path, required=True)
    analyze.add_argument("--C", type=float, default=0.1)
    return parser


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        catalogs = load_catalogs(PROJECT_ROOT)
        if args.command == "requests":
            count = write_generation_requests(args.out, catalogs, args.per_topic)
            _print({"status": "written", "requests": count, "path": str(args.out)})
        elif args.command == "generate":
            semantic_attempts = (
                args.max_attempts
                if args.max_attempts is not None
                else args.max_semantic_attempts
            )
            _print(
                generate_validated_payloads(
                    catalogs=catalogs,
                    generator_profile_path=args.generator_profile,
                    validator_profile_path=args.validator_profile,
                    output_path=args.out,
                    run_dir=args.run_dir,
                    per_topic=args.per_topic,
                    max_generation_attempts=args.max_generation_attempts,
                    max_semantic_attempts=semantic_attempts,
                    max_validator_attempts=args.max_validator_attempts,
                    limit=args.limit,
                )
            )
        elif args.command == "validate-payloads":
            rows = load_and_validate_payloads(args.payloads, catalogs)
            _print(
                {
                    "status": "passed",
                    "payloads": len(rows),
                    "topics": len({row["topic_id"] for row in rows}),
                }
            )
        elif args.command == "render":
            rows = load_and_validate_payloads(args.payloads, catalogs)
            manifest = write_rendered_dataset(
                payload_path=args.payloads,
                output_path=args.out,
                manifest_path=args.manifest,
                payloads=rows,
                catalogs=catalogs,
            )
            rendered_rows = load_jsonl(args.out)
            manifest["validation"] = validate_rendered_dataset(
                rendered_rows, rows, catalogs
            )
            write_json(args.manifest, manifest)
            _print(manifest)
        elif args.command == "validate-rendered":
            payload_rows = load_and_validate_payloads(args.payloads, catalogs)
            rendered_rows = load_jsonl(args.items)
            report = validate_rendered_dataset(rendered_rows, payload_rows, catalogs)
            if args.report:
                write_json(args.report, report)
            _print(report)
        elif args.command == "extract":
            _print(
                extract_activations(
                    items_path=args.items,
                    output_dir=args.out_dir,
                    model_name=args.model,
                    model_revision=args.revision,
                    layer=args.layer,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )
            )
        elif args.command == "analyze":
            _print(
                analyze_activations(
                    items_path=args.items,
                    activations_path=args.activations,
                    output_path=args.out,
                    c=args.C,
                )
            )
        else:  # pragma: no cover
            parser.error(f"Unknown command {args.command}")
    except MVPError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
