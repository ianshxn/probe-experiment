"""Command-line interface for contrastive preflight and generation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .planning import load_provider_profile
from .calibration import (
    validate_frame_calibration_results,
    validate_heldout_comprehension_results,
    write_frame_calibration_packet,
)
from .pre_generation import (
    apply_pre_generation_overrides,
    load_pre_generation_config,
    verify_pre_generation_run,
    with_pre_generation_provider,
    write_pre_generation_plan,
)
from .lease import RunLease
from .post_generation import assemble_post_generation
from .publication import atomic_create_json, resolve_unaliased_output_path
from .runner import GenerationRunner, load_generation_run_config
from .utils import CONTRASTIVE_ROOT, ContrastiveError


DEFAULT_PRE_GENERATION_CONFIG = (
    CONTRASTIVE_ROOT / "configs" / "pre_generation.yaml"
)
DEFAULT_GENERATION_CONFIG = (
    CONTRASTIVE_ROOT / "configs" / "generation_run.yaml"
)


def _csv(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _preflight(args: argparse.Namespace) -> int:
    config = apply_pre_generation_overrides(
        with_pre_generation_provider(
            load_pre_generation_config(args.config),
            load_provider_profile(args.provider_config),
        ),
        topics=_csv(args.topic),
        content_specs=_csv(args.content_spec),
        languages=_csv(args.language),
        mode="production" if args.production else None,
    )
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "contrastive_preflight_%Y%m%d_%H%M%S"
    )
    run_dir, plan, _ = write_pre_generation_plan(config, run_id)
    _print({
        "status": "planned",
        "run_dir": str(run_dir),
        "run_id": run_id,
        "counts": plan["counts"],
        "preflight": plan["preflight"],
        "cost_estimate": plan["cost_estimate"],
        "plan_sha256": plan["plan_sha256"],
    })
    return 0


def _verify(args: argparse.Namespace) -> int:
    _print(verify_pre_generation_run(args.run_dir))
    return 0


def _generate(args: argparse.Namespace) -> int:
    config = load_generation_run_config(args.config)
    if args.run_dir_a is not None:
        config["run_dir_a"] = str(args.run_dir_a)
    if args.run_dir_b is not None:
        config["run_dir_b"] = str(args.run_dir_b)
    if args.production:
        config["production"] = True
    _print(GenerationRunner(config).run())
    return 0


def _lease_release(args: argparse.Namespace) -> int:
    holder = RunLease.current(args.run_dir / "lease.json")
    _print({"current_holder": holder, "action": "lease-release"})
    sys.stdout.flush()
    released = RunLease.force_release(args.run_dir)
    _print({"status": "released" if released else "not_held"})
    return 0


def _postprocess(args: argparse.Namespace) -> int:
    _print(
        assemble_post_generation(
            run_dir_a=args.run_dir_a,
            run_dir_b=args.run_dir_b,
            output_dir=args.output_dir,
            production=bool(args.production),
        )
    )
    return 0


def _calibration_packet(args: argparse.Namespace) -> int:
    _print(
        write_frame_calibration_packet(
            args.output_dir,
            seed=int(args.seed),
        )
    )
    return 0


def _calibration_check(args: argparse.Namespace) -> int:
    report = validate_frame_calibration_results(
        args.packet_dir,
        args.results,
        args.raters,
    )
    if args.output is not None:
        atomic_create_json(
            resolve_unaliased_output_path(args.output),
            report,
        )
    _print(report)
    return 0 if report["status"] == "passed" else 1


def _comprehension_check(args: argparse.Namespace) -> int:
    report = validate_heldout_comprehension_results(
        args.dataset_dir,
        args.results,
        args.models,
    )
    if args.output is not None:
        atomic_create_json(
            resolve_unaliased_output_path(args.output),
            report,
        )
    _print(report)
    return 0 if report["status"] == "passed" else 1


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="contrastive-data")
    commands = value.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser(
        "preflight",
        help=(
            "compile content specs and write an immutable template-blind "
            "generation plan without making model calls"
        ),
    )
    preflight.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PRE_GENERATION_CONFIG,
    )
    preflight.add_argument("--run-id")
    preflight.add_argument(
        "--provider-config",
        type=Path,
        required=True,
        help="model/provider profile inside contrastive/",
    )
    preflight.add_argument("--topic", action="append")
    preflight.add_argument("--content-spec", action="append")
    preflight.add_argument("--language", action="append")
    preflight.add_argument(
        "--production",
        action="store_true",
        help="require the full bilingual grid and approved semantic review",
    )
    preflight.set_defaults(function=_preflight)
    verify = commands.add_parser(
        "verify",
        help="rehash and re-audit an immutable pre-generation run",
    )
    verify.add_argument("run_dir", type=Path)
    verify.set_defaults(function=_verify)
    generate = commands.add_parser(
        "generate",
        help="generate, cross-verify, render, and publish a verified run pair",
    )
    generate.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_GENERATION_CONFIG,
    )
    generate.add_argument("--run-dir-a", type=Path)
    generate.add_argument("--run-dir-b", type=Path)
    generate.add_argument(
        "--production",
        action="store_true",
        help="enforce distinct production-mode non-mock provider runs",
    )
    generate.set_defaults(function=_generate)
    postprocess = commands.add_parser(
        "postprocess",
        help=(
            "verify generated artifacts and publish grouped, structurally complete "
            "fitting and frozen validation datasets"
        ),
    )
    postprocess.add_argument("--run-dir-a", type=Path, required=True)
    postprocess.add_argument("--run-dir-b", type=Path, required=True)
    postprocess.add_argument("--output-dir", type=Path, required=True)
    postprocess.add_argument(
        "--production",
        action="store_true",
        help="require complete production preflight and generation grids",
    )
    postprocess.set_defaults(function=_postprocess)
    calibration_packet = commands.add_parser(
        "calibration-packet",
        help="write an immutable blinded frame-only human-rating packet",
    )
    calibration_packet.add_argument("--output-dir", type=Path, required=True)
    calibration_packet.add_argument("--seed", type=int, default=48271)
    calibration_packet.set_defaults(function=_calibration_packet)
    calibration_check = commands.add_parser(
        "calibration-check",
        help="validate blinded frame-only rating results against frozen gates",
    )
    calibration_check.add_argument("packet_dir", type=Path)
    calibration_check.add_argument("results", type=Path)
    calibration_check.add_argument(
        "--raters",
        type=Path,
        required=True,
        help="confirmatory rater cohort metadata and blinding attestations",
    )
    calibration_check.add_argument("--output", type=Path)
    calibration_check.set_defaults(function=_calibration_check)
    comprehension_check = commands.add_parser(
        "comprehension-check",
        help=(
            "validate held-out calibration-model classifications of complete "
            "rendered prompts"
        ),
    )
    comprehension_check.add_argument("dataset_dir", type=Path)
    comprehension_check.add_argument("results", type=Path)
    comprehension_check.add_argument("--models", type=Path, required=True)
    comprehension_check.add_argument("--output", type=Path)
    comprehension_check.set_defaults(function=_comprehension_check)
    lease_release = commands.add_parser(
        "lease-release",
        help="break-glass removal of a generation run lease",
    )
    lease_release.add_argument("run_dir", type=Path)
    lease_release.set_defaults(function=_lease_release)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.function(args))
    except ContrastiveError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
