"""Validated two-model payload generation for the English MVP."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .catalogs import Catalogs
from .io import (
    MVPError,
    PROJECT_ROOT,
    load_json,
    load_jsonl,
    sha256_file,
    sha256_object,
    write_json,
    write_jsonl,
)
from .openrouter import (
    OpenRouterResponseError,
    assert_independent_models,
    call_structured,
    load_model_profile,
    public_profile,
)
from .payloads import build_generation_requests, validate_payloads
from .schema import validate_against_schema


StructuredCaller = Callable[..., dict[str, Any]]
VALIDATION_PROMPT = PROJECT_ROOT / "prompts" / "payload_validation.txt"
GENERATION_SCHEMA = PROJECT_ROOT / "schemas" / "generation_response.schema.json"
VALIDATION_SCHEMA = PROJECT_ROOT / "schemas" / "validation_response.schema.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_context(exc: Exception) -> dict[str, Any]:
    context: dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }
    if exc.__cause__ is not None:
        context["cause_type"] = type(exc.__cause__).__name__
        context["cause_message"] = str(exc.__cause__)
    if isinstance(exc, OpenRouterResponseError):
        context["openrouter_provenance"] = exc.provenance
        context["openrouter_raw_response"] = exc.raw_response
    return context


def _persist_attempt_error(
    *,
    run_dir: Path,
    attempts_path: Path,
    attempts: list[dict[str, Any]],
    config: dict[str, Any],
    attempt_record: dict[str, Any],
    stage: str,
    exc: Exception,
    fatal: bool,
) -> Path:
    """Persist the attempt plus an independently readable error artifact."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    payload_id = str(attempt_record.get("payload_block_id", "unknown"))
    error_path = run_dir / "errors" / f"error_{stamp}_{payload_id}_{stage}.json"
    attempt_record["error_record"] = str(error_path.relative_to(run_dir))
    attempts.append(attempt_record)
    write_jsonl(attempts_path, attempts)
    record = {
        "schema_version": 1,
        "record_type": "attempt_error",
        "status": "failed" if fatal else "rejected_and_continuing",
        "fatal": fatal,
        "recorded_at": _now(),
        "stage": stage,
        "run_directory": str(run_dir.resolve()),
        "payload_block_id": attempt_record.get("payload_block_id"),
        "request_id": attempt_record.get("request_id"),
        "generation_attempt": attempt_record.get("generation_attempt"),
        "semantic_attempt": attempt_record.get("semantic_attempt"),
        "validator_api_attempt": attempt_record.get("validator_api_attempt"),
        "error": _error_context(exc),
        "attempt_record": attempt_record,
        "run_config_sha256": sha256_object(config),
        "attempt_log_sha256": sha256_file(attempts_path),
    }
    write_json(error_path, record)
    return error_path


def _candidate(
    content: dict[str, Any], request_row: dict[str, Any]
) -> dict[str, Any]:
    validate_against_schema(
        content,
        "generation_response.schema.json",
        label=f"generator output for {request_row['payload_block_id']}",
    )
    return {
        "schema_version": 1,
        "payload_block_id": request_row["payload_block_id"],
        "topic_id": request_row["topic_id"],
        "task_family": request_row["task_family"],
        "context": content["context"],
        "question": content["question"],
        "correct_answer": content["correct_answer"],
        "distractors": content["distractors"],
    }


def _validation_user(
    payload: dict[str, Any], consistency_feedback: list[str] | None = None
) -> str:
    prompt = VALIDATION_PROMPT.read_text(encoding="utf-8")
    visible = {
        key: payload[key]
        for key in (
            "topic_id",
            "task_family",
            "context",
            "question",
            "correct_answer",
            "distractors",
        )
    }
    rendered = prompt.format(
        payload_json=json.dumps(visible, ensure_ascii=False, indent=2)
    ).strip()
    if consistency_feedback:
        rendered += (
            "\n\nYour previous judgment was internally inconsistent. Re-evaluate the "
            "same payload independently and correct these structural contradictions:\n- "
            + "\n- ".join(consistency_feedback)
        )
    return rendered


ISSUE_CHECKS = {
    "not_self_contained": "self_contained",
    "incorrect_answer": "correct_answer",
    "ambiguous_answer": "unique_answer",
    "invalid_distractor": "all_distractors_wrong",
    "implausible_distractor": "distractors_plausible",
    "short_answer_unsuitable": "short_answer_suitable",
    "cross_format_mismatch": "cross_format_equivalent",
    "purpose_leakage": "purpose_neutral",
    "format_leakage": "format_neutral",
    "temporally_unstable": "temporally_stable",
}


def validation_consistency_errors(value: dict[str, Any]) -> list[str]:
    """Find contradictions that make a semantic judgment unusable."""
    validate_against_schema(
        value, "validation_response.schema.json", label="semantic validation"
    )
    errors: list[str] = []
    checks = value["checks"]
    all_true = all(bool(result) for result in checks.values())
    clean = not value["invalid_distractor_indices"] and not value["issues"]
    if value["verdict"] == "pass":
        if not all_true:
            errors.append("verdict is pass but at least one check is false")
        if not clean:
            errors.append("verdict is pass but issues or invalid distractors are present")
        if value["confidence"] != "high":
            errors.append("verdict is pass but confidence is not high")
    elif all_true and clean:
        errors.append("verdict is not pass despite all checks passing with no issues")

    if value["invalid_distractor_indices"] and checks["all_distractors_wrong"]:
        errors.append("invalid distractors are listed but all_distractors_wrong is true")
    for issue in value["issues"]:
        check = ISSUE_CHECKS.get(issue["code"])
        if check and checks[check]:
            errors.append(f"issue {issue['code']} is present but check {check} is true")
    if any(issue["code"] == "other" for issue in value["issues"]) and all_true:
        errors.append("an 'other' issue is present but every check is true")
    return errors


def validation_passes(value: dict[str, Any]) -> bool:
    consistency_errors = validation_consistency_errors(value)
    if consistency_errors:
        raise MVPError("Inconsistent semantic validation: " + "; ".join(consistency_errors))
    return (
        value["verdict"] == "pass"
        and value["confidence"] == "high"
        and all(bool(result) for result in value["checks"].values())
        and not value["invalid_distractor_indices"]
        and not value["issues"]
    )


def _feedback(attempts: list[dict[str, Any]], payload_id: str) -> str | None:
    previous = [
        row
        for row in attempts
        if row["payload_block_id"] == payload_id
        and row.get("status")
        in {
            "generator_output_rejection",
            "validator_output_rejection",
            "semantic_rejection",
            # Compatibility with runs created before API and output failures were
            # given separate statuses.
            "generator_or_deterministic_rejection",
        }
        and not str(row.get("error", "")).startswith("OpenRouter ")
    ]
    if not previous:
        return None
    latest = previous[-1]
    if latest.get("validation"):
        value = latest["validation"]
        issues = value.get("issues", [])
        if issues:
            return "; ".join(f"{row['code']}: {row['message']}" for row in issues)
        failed = [key for key, passed in value.get("checks", {}).items() if not passed]
        if failed:
            return "Failed checks: " + ", ".join(failed)
        return f"Validator verdict={value.get('verdict')} confidence={value.get('confidence')}"
    if latest.get("error"):
        return str(latest["error"])
    return None


def _generator_user(request_row: dict[str, Any], feedback: str | None) -> str:
    user = str(request_row["user"])
    if feedback:
        user += (
            "\n\nA previous attempt was rejected for the following reason. Generate a new "
            "payload that avoids the problem; do not discuss the rejection in your output.\n"
            f"Rejection feedback: {feedback}"
        )
    return user


def _run_configuration(
    *,
    generator: dict[str, Any],
    validator: dict[str, Any],
    per_topic: int,
    max_generation_attempts: int,
    max_semantic_attempts: int,
    max_validator_attempts: int,
    output_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "development_unlocked",
        "hash_policy": "observational_with_internal_consistency",
        "request_order_policy": "sample_round_robin_by_topic_v1",
        "validator_consistency_policy": "issue_check_alignment_v1",
        "generator": public_profile(generator),
        "validator": public_profile(validator),
        "per_topic": per_topic,
        "max_generation_attempts_per_payload": max_generation_attempts,
        "max_semantic_attempts_per_payload": max_semantic_attempts,
        "max_validator_attempts_per_candidate": max_validator_attempts,
        "accepted_output_path": str(output_path.resolve()),
        "source_hashes": {
            "generation_prompt": sha256_file(
                PROJECT_ROOT / "prompts" / "payload_generation.txt"
            ),
            "validation_prompt": sha256_file(VALIDATION_PROMPT),
            "generation_schema": sha256_file(GENERATION_SCHEMA),
            "validation_schema": sha256_file(VALIDATION_SCHEMA),
            "topics": sha256_file(PROJECT_ROOT / "catalogs" / "topics.yaml"),
        },
    }


def _prepare_run_config(path: Path, expected: dict[str, Any]) -> None:
    if path.exists():
        actual = load_json(path)
        if actual != expected:
            raise MVPError(
                f"Run configuration changed since {path} was created; use a new run directory"
            )
    else:
        write_json(path, expected)


def _validated_existing(
    output_path: Path, catalogs: Catalogs
) -> list[dict[str, Any]]:
    rows = load_jsonl(output_path) if output_path.exists() else []
    if rows:
        validate_payloads(rows, catalogs)
        for row in rows:
            validation = row.get("generation", {}).get("semantic_validation", {})
            try:
                passed = validation_passes(validation)
            except MVPError as exc:
                raise MVPError(
                    f"Existing output {output_path} contains malformed semantic "
                    f"validation for {row['payload_block_id']}: {exc}"
                ) from exc
            if not passed:
                raise MVPError(
                    f"Existing output {output_path} contains payload "
                    f"{row['payload_block_id']} without a complete high-confidence "
                    "validator pass"
                )
    return rows


def _legacy_infrastructure_failure(row: dict[str, Any]) -> bool:
    return row.get("status") == "generator_or_deterministic_rejection" and str(
        row.get("error", "")
    ).startswith("OpenRouter ")


def _consumes_generation_attempt(row: dict[str, Any]) -> bool:
    status = row.get("status")
    if status in {"accepted", "semantic_rejection", "generator_output_rejection"}:
        return True
    return status == "generator_or_deterministic_rejection" and not _legacy_infrastructure_failure(
        row
    )


def _consumes_semantic_attempt(row: dict[str, Any]) -> bool:
    status = row.get("status")
    return status in {"accepted", "semantic_rejection"}


def _interleave_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order sample 1 across every topic before sample 2, and so on."""
    original_order = {id(row): index for index, row in enumerate(requests)}
    return sorted(
        requests,
        key=lambda row: (
            int(str(row["payload_block_id"]).rsplit("_", 1)[1]),
            original_order[id(row)],
        ),
    )


def generate_validated_payloads(
    *,
    catalogs: Catalogs,
    generator_profile_path: Path,
    validator_profile_path: Path,
    output_path: Path,
    run_dir: Path,
    per_topic: int,
    max_generation_attempts: int = 6,
    max_semantic_attempts: int = 3,
    max_validator_attempts: int = 2,
    limit: int | None = None,
    caller: StructuredCaller = call_structured,
) -> dict[str, Any]:
    """Generate, independently validate, and persist only accepted payloads."""
    for name, value in {
        "max_generation_attempts": max_generation_attempts,
        "max_semantic_attempts": max_semantic_attempts,
        "max_validator_attempts": max_validator_attempts,
    }.items():
        if value < 1:
            raise MVPError(f"{name} must be positive")
    if max_generation_attempts < max_semantic_attempts:
        raise MVPError(
            "max_generation_attempts must be at least max_semantic_attempts"
        )
    if limit is not None and limit < 1:
        raise MVPError("limit must be positive")

    generator = load_model_profile(generator_profile_path, expected_role="generator")
    validator = load_model_profile(validator_profile_path, expected_role="validator")
    assert_independent_models(generator, validator)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = _run_configuration(
        generator=generator,
        validator=validator,
        per_topic=per_topic,
        max_generation_attempts=max_generation_attempts,
        max_semantic_attempts=max_semantic_attempts,
        max_validator_attempts=max_validator_attempts,
        output_path=output_path,
    )
    _prepare_run_config(run_dir / "run_config.json", config)

    requests = _interleave_requests(
        build_generation_requests(catalogs, per_topic=per_topic)
    )
    if limit is not None:
        requests = requests[:limit]
    accepted_rows = _validated_existing(output_path, catalogs)
    accepted = {str(row["payload_block_id"]) for row in accepted_rows}
    attempts_path = run_dir / "attempts.jsonl"
    attempts = load_jsonl(attempts_path) if attempts_path.exists() else []
    generation_counts: dict[str, int] = {}
    semantic_counts: dict[str, int] = {}
    for row in attempts:
        payload_id = str(row["payload_block_id"])
        if _consumes_generation_attempt(row):
            generation_counts[payload_id] = generation_counts.get(payload_id, 0) + 1
        if _consumes_semantic_attempt(row):
            semantic_counts[payload_id] = semantic_counts.get(payload_id, 0) + 1

    generation_schema = load_json(GENERATION_SCHEMA)
    validation_schema = load_json(VALIDATION_SCHEMA)
    exhausted: dict[str, str] = {}
    for request_row in requests:
        payload_id = str(request_row["payload_block_id"])
        if payload_id in accepted:
            continue
        generation_count = generation_counts.get(payload_id, 0)
        semantic_count = semantic_counts.get(payload_id, 0)
        while (
            generation_count < max_generation_attempts
            and semantic_count < max_semantic_attempts
        ):
            attempt_number = generation_count + 1
            attempt_record: dict[str, Any] = {
                "schema_version": 1,
                "payload_block_id": payload_id,
                "request_id": request_row["request_id"],
                "attempt": attempt_number,
                "generation_attempt": attempt_number,
                "started_at": _now(),
                "status": "started",
            }
            feedback = _feedback(attempts, payload_id)
            try:
                author = caller(
                    generator,
                    system=request_row["system"],
                    user=_generator_user(request_row, feedback),
                    response_schema=generation_schema,
                    schema_name="canonical_payload",
                )
            except (MVPError, KeyError, TypeError) as exc:
                attempt_record.update(
                    status="generator_api_error", error=str(exc), completed_at=_now()
                )
                error_path = _persist_attempt_error(
                    run_dir=run_dir,
                    attempts_path=attempts_path,
                    attempts=attempts,
                    config=config,
                    attempt_record=attempt_record,
                    stage="generator_api",
                    exc=exc,
                    fatal=True,
                )
                raise MVPError(
                    f"Generator API failed for {payload_id}; the run stopped without "
                    f"consuming a generation or semantic attempt. Fix the API/configuration "
                    f"issue and rerun the same command to resume. Error record: "
                    f"{error_path}. Cause: {exc}"
                ) from exc

            generation_count += 1
            generation_counts[payload_id] = generation_count
            try:
                attempt_record["generator"] = {
                    "provenance": author["provenance"],
                    "parsed": author["parsed"],
                    "raw_response": author["raw_response"],
                }
                candidate = _candidate(author["parsed"], request_row)
                validate_payloads([candidate], catalogs)
            except (MVPError, KeyError, TypeError) as exc:
                attempt_record.update(
                    status="generator_output_rejection",
                    error=str(exc),
                    completed_at=_now(),
                )
                _persist_attempt_error(
                    run_dir=run_dir,
                    attempts_path=attempts_path,
                    attempts=attempts,
                    config=config,
                    attempt_record=attempt_record,
                    stage="generator_output_validation",
                    exc=exc,
                    fatal=False,
                )
                continue

            validator_trials: list[dict[str, Any]] = []
            consistency_feedback: list[str] | None = None
            judgment: dict[str, Any] | None = None
            validation: dict[str, Any] | None = None
            for validator_attempt in range(1, max_validator_attempts + 1):
                try:
                    current_judgment = caller(
                        validator,
                        system=(
                            "You are an independent scientific stimulus validator. "
                            "Return only schema-valid, internally consistent JSON and fail closed."
                        ),
                        user=_validation_user(candidate, consistency_feedback),
                        response_schema=validation_schema,
                        schema_name="canonical_payload_validation",
                    )
                except (MVPError, KeyError, TypeError) as exc:
                    attempt_record["validator_attempts"] = validator_trials
                    attempt_record.update(
                        status="validator_api_error",
                        validator_api_attempt=validator_attempt,
                        error=str(exc),
                        completed_at=_now(),
                    )
                    error_path = _persist_attempt_error(
                        run_dir=run_dir,
                        attempts_path=attempts_path,
                        attempts=attempts,
                        config=config,
                        attempt_record=attempt_record,
                        stage="validator_api_or_response_parsing",
                        exc=exc,
                        fatal=True,
                    )
                    raise MVPError(
                        f"Validator API failed for {payload_id}; the run stopped without "
                        f"consuming a semantic attempt. Fix the API/configuration issue and "
                        f"rerun the same command to resume. Error record: {error_path}. "
                        f"Cause: {exc}"
                    ) from exc

                current_validation = current_judgment.get("parsed")
                try:
                    if not isinstance(current_validation, dict):
                        raise MVPError("Validator parsed output is not an object")
                    consistency_errors = validation_consistency_errors(current_validation)
                except (MVPError, KeyError, TypeError) as exc:
                    consistency_errors = [str(exc)]
                validator_trials.append(
                    {
                        "validator_attempt": validator_attempt,
                        "provenance": current_judgment.get("provenance"),
                        "parsed": current_validation,
                        "raw_response": current_judgment.get("raw_response"),
                        "consistency_errors": consistency_errors,
                    }
                )
                if consistency_errors:
                    consistency_feedback = consistency_errors
                    continue
                judgment = current_judgment
                validation = current_validation
                break

            attempt_record["validator_attempts"] = validator_trials
            if judgment is None or validation is None:
                inconsistency_error = MVPError(
                    "Validator remained internally inconsistent after "
                    f"{max_validator_attempts} attempts"
                )
                attempt_record.update(
                    status="validator_inconsistency_error",
                    error=str(inconsistency_error),
                    completed_at=_now(),
                )
                error_path = _persist_attempt_error(
                    run_dir=run_dir,
                    attempts_path=attempts_path,
                    attempts=attempts,
                    config=config,
                    attempt_record=attempt_record,
                    stage="validator_consistency",
                    exc=inconsistency_error,
                    fatal=True,
                )
                raise MVPError(
                    f"Validator remained internally inconsistent for {payload_id}; "
                    "the run stopped without consuming a semantic attempt. Inspect the "
                    f"recorded validator attempts and {error_path}, then rerun the same "
                    "command to resume."
                )

            passed = validation_passes(validation)
            semantic_count += 1
            semantic_counts[payload_id] = semantic_count
            attempt_record["semantic_attempt"] = semantic_count
            attempt_record["validator"] = {
                "provenance": judgment["provenance"],
                "raw_response": judgment["raw_response"],
            }
            attempt_record["validation"] = validation
            attempt_record["status"] = "accepted" if passed else "semantic_rejection"
            attempt_record["completed_at"] = _now()
            attempts.append(attempt_record)
            write_jsonl(attempts_path, attempts)
            if not passed:
                continue

            candidate["generation"] = {
                "request_id": request_row["request_id"],
                "attempt": attempt_number,
                "generator_profile": generator["profile_id"],
                "generator_model": generator["model"],
                "generator_canonical_slug": generator["canonical_slug"],
                "generator_response_id": author["provenance"].get("response_id"),
                "validator_profile": validator["profile_id"],
                "validator_model": validator["model"],
                "validator_canonical_slug": validator["canonical_slug"],
                "validator_response_id": judgment["provenance"].get("response_id"),
                "semantic_validation": validation,
                "attempt_record_sha256": sha256_object(attempt_record),
            }
            accepted_rows.append(candidate)
            accepted.add(payload_id)
            write_jsonl(output_path, accepted_rows)
            break
        if payload_id not in accepted:
            if semantic_count >= max_semantic_attempts and generation_count >= max_generation_attempts:
                exhausted[payload_id] = "semantic_and_generation_budgets"
            elif semantic_count >= max_semantic_attempts:
                exhausted[payload_id] = "semantic_budget"
            else:
                exhausted[payload_id] = "generation_budget"

    targeted = [str(row["payload_block_id"]) for row in requests]
    target_accepted = [payload_id for payload_id in targeted if payload_id in accepted]
    report = {
        "schema_version": 1,
        "status": "completed" if len(target_accepted) == len(targeted) else "completed_with_exhausted",
        "generator": generator["model"],
        "validator": validator["model"],
        "targeted_payloads": len(targeted),
        "accepted_payloads": len(target_accepted),
        "exhausted_payloads": sorted(exhausted),
        "exhaustion_reasons": exhausted,
        "attempt_records": len(attempts),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path) if output_path.exists() else None,
        "run_config_sha256": sha256_object(config),
        "completed_at": _now(),
    }
    write_json(run_dir / "run_report.json", report)
    return report
