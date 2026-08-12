"""Replay Japanese surface QC over archived generation candidates without APIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CONTRASTIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRASTIVE_ROOT / "src"))

from contrastive_generation.runner import load_generation_run_config
from contrastive_generation.utils import (
    ContrastiveError,
    load_json,
    load_yaml,
    resolve_local_path,
)
from contrastive_generation.verification import deterministic_qc


EXPECTED_SLICE_BASELINE = {
    "candidate_artifacts": 117,
    "parseable_candidates": 115,
    "malformed_candidates": 2,
    "japanese_candidates": 78,
    "english_candidates": 37,
    "japanese_surface_pass": 78,
    "former_japanese_surface_failures": 61,
    "former_failures_now_surface_pass": 61,
    "former_failures_with_cue_hits": 13,
    "japanese_candidates_with_fullwidth_digits": 0,
    "english_proxy_surface_failures": 37,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ContrastiveError(f"Could not read JSONL {path}: {exc}") from exc


def _attempt_index(
    run_dirs: tuple[Path, Path],
) -> dict[str, dict[str, Any]]:
    attempts: dict[str, dict[str, Any]] = {}
    for run_dir in dict.fromkeys(run_dirs):
        for job in _read_jsonl(run_dir / "manifest.jsonl"):
            for attempt in job["generation_attempts"]:
                attempt_id = str(attempt["attempt_id"])
                if attempt_id in attempts:
                    raise ContrastiveError(
                        f"Duplicate archived attempt ID: {attempt_id}"
                    )
                attempts[attempt_id] = {
                    "language": str(job["language"]),
                    "request_path": str(attempt["request_path"]),
                    "candidate_path": str(attempt["candidate_output_path"]),
                }
    return attempts


def _old_qc_index(run_dir_a: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    state_dir = run_dir_a / "state"
    for path in sorted(state_dir.glob("*.json")):
        state = load_json(path)
        for attempt in state["attempts"]:
            candidate_path = attempt.get("candidate_path")
            qc = attempt.get("deterministic_qc")
            if candidate_path and isinstance(qc, dict):
                result[Path(str(candidate_path)).stem] = qc
    return result


def replay(
    *,
    run_dir_a: Path,
    run_dir_b: Path,
    config_path: Path,
) -> dict[str, int]:
    config = load_generation_run_config(config_path)
    policy = config["verification"]["japanese_surface"]
    resolved_config = load_json(run_dir_a / "resolved_config.json")
    cue_lexicon = load_yaml(
        resolve_local_path(resolved_config["paths"]["cue_lexicon"])
    )
    attempts = _attempt_index((run_dir_a, run_dir_b))
    old_qc = _old_qc_index(run_dir_a)
    summary = {key: 0 for key in EXPECTED_SLICE_BASELINE}

    for attempt_id, metadata in sorted(attempts.items()):
        candidate_path = resolve_local_path(metadata["candidate_path"])
        if not candidate_path.is_file():
            continue
        summary["candidate_artifacts"] += 1
        candidate = load_json(candidate_path)
        if candidate.get("parsed_output") is None:
            summary["malformed_candidates"] += 1
            continue
        summary["parseable_candidates"] += 1
        request = load_json(resolve_local_path(metadata["request_path"]))
        language = metadata["language"]
        qc, parsed = deterministic_qc(
            str(candidate["raw_text"]),
            request["response_schema"],
            language,
            cue_lexicon,
            japanese_surface=policy,
        )
        if parsed is None:
            raise ContrastiveError(
                f"Archived parseable candidate no longer parses: {attempt_id}"
            )

        if language == "ja":
            summary["japanese_candidates"] += 1
            if qc["checks"]["japanese_surface"]:
                summary["japanese_surface_pass"] += 1
            if qc["checks"]["fullwidth_digit_hits"]:
                summary["japanese_candidates_with_fullwidth_digits"] += 1
            previous = old_qc.get(attempt_id, {})
            if previous.get("checks", {}).get("japanese_surface") is False:
                summary["former_japanese_surface_failures"] += 1
                if qc["checks"]["japanese_surface"]:
                    summary["former_failures_now_surface_pass"] += 1
                if not qc["checks"]["cue_free"]:
                    summary["former_failures_with_cue_hits"] += 1
        elif language == "en":
            summary["english_candidates"] += 1
            proxy_qc, proxy_parsed = deterministic_qc(
                str(candidate["raw_text"]),
                request["response_schema"],
                "ja",
                cue_lexicon,
                japanese_surface=policy,
            )
            if proxy_parsed is None:
                raise ContrastiveError(
                    f"English proxy no longer parses: {attempt_id}"
                )
            if not proxy_qc["checks"]["japanese_surface"]:
                summary["english_proxy_surface_failures"] += 1
        else:
            raise ContrastiveError(
                f"Unexpected archived language for {attempt_id}: {language}"
            )

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir-a",
        type=Path,
        default=CONTRASTIVE_ROOT / "runs" / "slice_qwen",
    )
    parser.add_argument(
        "--run-dir-b",
        type=Path,
        default=CONTRASTIVE_ROOT / "runs" / "slice_gemini",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONTRASTIVE_ROOT / "configs" / "generation_run.yaml",
    )
    parser.add_argument("--assert-slice-baseline", action="store_true")
    args = parser.parse_args(argv)

    summary = replay(
        run_dir_a=args.run_dir_a.resolve(),
        run_dir_b=args.run_dir_b.resolve(),
        config_path=args.config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.assert_slice_baseline and summary != EXPECTED_SLICE_BASELINE:
        print(
            "Archived slice replay differs from the expected baseline.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
