from __future__ import annotations

from pathlib import Path

import pytest

from eval_format_mvp.catalogs import load_catalogs
from eval_format_mvp.generation import (
    _interleave_requests,
    _prior_semantic_concerns,
    _validation_user,
    generate_validated_payloads,
    validation_consistency_errors,
)
from eval_format_mvp.io import MVPError, load_json, load_jsonl
from eval_format_mvp.openrouter import (
    OpenRouterResponseError,
    build_request_body,
    load_model_profile,
    provider_response_schema,
)
from eval_format_mvp.payloads import build_generation_requests


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "configs" / "grok43_generator.openrouter.yaml"
VALIDATOR = ROOT / "configs" / "gpt54_validator.openrouter.yaml"


def _result(parsed: dict, profile: dict, call_number: int) -> dict:
    raw = {
        "id": f"response-{call_number}",
        "model": profile["model"],
        "provider": "test-provider",
        "choices": [{"message": {"content": "{}"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    return {
        "parsed": parsed,
        "raw_response": raw,
        "provenance": {
            "profile_id": profile["profile_id"],
            "requested_model": profile["model"],
            "canonical_slug": profile["canonical_slug"],
            "returned_model": profile["model"],
            "response_id": raw["id"],
            "provider": raw["provider"],
            "usage": raw["usage"],
            "request_sha256": "0" * 64,
            "transport_attempt": 1,
        },
    }


def _payload() -> dict:
    return {
        "context": "A list contains three red tokens and two blue tokens.",
        "question": "How many tokens are in the list?",
        "correct_answer": "5",
        "distractors": ["2", "3", "6"],
    }


def _judgment(*, passed: bool) -> dict:
    checks = {
        "self_contained": True,
        "correct_answer": True,
        "unique_answer": True,
        "all_distractors_wrong": passed,
        "distractors_plausible": True,
        "distractors_mutually_distinct": True,
        "option_independent": True,
        "short_answer_suitable": True,
        "cross_format_equivalent": True,
        "purpose_neutral": True,
        "format_neutral": True,
        "temporally_stable": True,
    }
    return {
        "verdict": "pass" if passed else "fail",
        "checks": checks,
        "normalized_answer": "5",
        "option_independent_answer": "5",
        "unstated_assumptions": [],
        "reasonable_alternative_answers": [],
        "normalized_distractor_answers": ["2", "3", "6"],
        "invalid_distractor_indices": [] if passed else [2],
        "equivalent_distractor_pairs": [],
        "issues": []
        if passed
        else [{"code": "invalid_distractor", "message": "Replace distractor 2."}],
        "confidence": "high",
    }


def test_grok_gpt_rejection_retry_acceptance_and_resume(tmp_path: Path) -> None:
    catalogs = load_catalogs(ROOT)
    calls: list[dict] = []
    validation_number = 0

    def fake_caller(profile, **kwargs):
        nonlocal validation_number
        calls.append({"role": profile["role"], "user": kwargs["user"]})
        if profile["role"] == "generator":
            return _result(_payload(), profile, len(calls))
        validation_number += 1
        return _result(
            _judgment(passed=validation_number == 2), profile, len(calls)
        )

    output = tmp_path / "payloads.jsonl"
    run_dir = tmp_path / "run"
    report = generate_validated_payloads(
        catalogs=catalogs,
        generator_profile_path=GENERATOR,
        validator_profile_path=VALIDATOR,
        output_path=output,
        run_dir=run_dir,
        per_topic=1,
        max_generation_attempts=3,
        max_semantic_attempts=3,
        limit=1,
        caller=fake_caller,
    )
    assert report["status"] == "completed"
    assert report["accepted_payloads"] == 1
    assert [row["role"] for row in calls] == [
        "generator",
        "validator",
        "generator",
        "validator",
    ]
    assert "invalid_distractor" in calls[2]["user"]
    assert "Earlier candidates for this same payload slot" not in calls[1]["user"]
    assert "invalid_distractor: Replace distractor 2." in calls[3]["user"]
    assert "Treat them as audit leads, not as presumed facts" in calls[3]["user"]

    accepted = load_jsonl(output)
    assert len(accepted) == 1
    provenance = accepted[0]["generation"]
    assert provenance["attempt"] == 2
    assert provenance["generator_model"] == "x-ai/grok-4.3"
    assert provenance["validator_model"] == "openai/gpt-5.4"
    assert provenance["semantic_validation"]["verdict"] == "pass"
    attempts = load_jsonl(run_dir / "attempts.jsonl")
    assert [row["status"] for row in attempts] == ["semantic_rejection", "accepted"]
    assert attempts[0]["prior_semantic_concerns"] == []
    assert attempts[1]["prior_semantic_concerns"] == [
        "invalid_distractor: Replace distractor 2."
    ]
    assert load_json(run_dir / "run_report.json")["accepted_payloads"] == 1

    def no_more_calls(*args, **kwargs):  # pragma: no cover - called only on failure
        raise AssertionError("A completed payload should be resumed without API calls")

    resumed = generate_validated_payloads(
        catalogs=catalogs,
        generator_profile_path=GENERATOR,
        validator_profile_path=VALIDATOR,
        output_path=output,
        run_dir=run_dir,
        per_topic=1,
        max_generation_attempts=3,
        max_semantic_attempts=3,
        limit=1,
        caller=no_more_calls,
    )
    assert resumed["status"] == "completed"
    assert resumed["attempt_records"] == 2


def test_validator_uncertainty_exhausts_fail_closed(tmp_path: Path) -> None:
    catalogs = load_catalogs(ROOT)

    def uncertain_caller(profile, **kwargs):
        if profile["role"] == "generator":
            return _result(_payload(), profile, 1)
        value = _judgment(passed=True)
        value["verdict"] = "uncertain"
        value["confidence"] = "medium"
        value["checks"]["unique_answer"] = False
        value["issues"] = [{"code": "other", "message": "Could not establish uniqueness."}]
        return _result(value, profile, 2)

    output = tmp_path / "payloads.jsonl"
    report = generate_validated_payloads(
        catalogs=catalogs,
        generator_profile_path=GENERATOR,
        validator_profile_path=VALIDATOR,
        output_path=output,
        run_dir=tmp_path / "run",
        per_topic=1,
        max_generation_attempts=2,
        max_semantic_attempts=2,
        limit=1,
        caller=uncertain_caller,
    )
    assert report["status"] == "completed_with_exhausted"
    assert report["accepted_payloads"] == 0
    assert report["exhausted_payloads"] == ["pb_t001_01"]
    assert not output.exists()


def test_openrouter_profiles_enforce_structured_no_fallback_requests() -> None:
    generator = load_model_profile(GENERATOR, expected_role="generator")
    validator = load_model_profile(VALIDATOR, expected_role="validator")
    body = build_request_body(
        generator,
        system="system",
        user="user",
        response_schema={"type": "object"},
        schema_name="test_schema",
    )
    assert body["model"] == "x-ai/grok-4.3"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["provider"] == {
        "require_parameters": True,
        "allow_fallbacks": False,
        "data_collection": "deny",
    }
    assert body["reasoning"] == {"effort": "medium"}
    assert body["temperature"] == 0.6
    assert body["top_p"] == 0.9
    assert body["max_tokens"] == 1536
    validator_body = build_request_body(
        validator,
        system="system",
        user="user",
        response_schema={"type": "object"},
        schema_name="test_schema",
    )
    assert validator_body["model"] == "openai/gpt-5.4"
    assert validator_body["reasoning"] == {"effort": "medium"}
    assert validator_body["max_tokens"] == 4096
    assert "temperature" not in validator_body
    assert "top_p" not in validator_body
    assert generator["model"].split("/", 1)[0] != validator["model"].split("/", 1)[0]


def test_validation_prompt_includes_exact_output_shape() -> None:
    payload = {
        "topic_id": "t001",
        "task_family": "classification",
        **_payload(),
    }
    rendered = _validation_user(payload)
    for key in (
        "verdict",
        "checks",
        "self_contained",
        "correct_answer",
        "unique_answer",
        "all_distractors_wrong",
        "distractors_plausible",
        "distractors_mutually_distinct",
        "option_independent",
        "short_answer_suitable",
        "cross_format_equivalent",
        "purpose_neutral",
        "format_neutral",
        "temporally_stable",
        "normalized_answer",
        "option_independent_answer",
        "unstated_assumptions",
        "reasonable_alternative_answers",
        "normalized_distractor_answers",
        "invalid_distractor_indices",
        "equivalent_distractor_pairs",
        "issues",
        "confidence",
    ):
        assert f'"{key}"' in rendered
    assert '"context": "A list contains three red tokens and two blue tokens."' in rendered
    assert "0 versus 1, 0 versus 2, and 1 versus 2" in rendered
    assert "reasonable interpretation or counterexample" in rendered
    assert "Judge plausibility as an error model" in rendered
    assert "Do not mark a distractor implausible merely" in rendered
    assert "semantic granularity requested by the" in rendered
    assert '"No, because X" and "No, because Y"' in rendered
    assert "symbolic or textual construction" in rendered
    assert "Apply the option-removal test" in rendered
    assert "Independently verify every embedded query" in rendered
    assert 'Do not treat claims such as "the parser accepts"' in rendered
    assert "silently reinterpret the artifact" in rendered
    assert "reasonable semantics allowed by the supplied context" in rendered
    assert "a correct answer does not excuse an inconsistent context" in rendered


def test_prior_semantic_concerns_are_cumulative_deduplicated_and_scoped() -> None:
    issue = {
        "code": "ambiguous_answer",
        "message": "The SQL dialect is unspecified.",
    }
    attempts = [
        {
            "payload_block_id": "pb_t002_01",
            "status": "semantic_rejection",
            "validation": {"issues": [issue]},
        },
        {
            "payload_block_id": "pb_t002_01",
            "status": "semantic_rejection",
            "validation": {
                "issues": [
                    issue,
                    {
                        "code": "not_self_contained",
                        "message": "Required execution assumptions are missing.",
                    },
                ]
            },
        },
        {
            "payload_block_id": "pb_t002_01",
            "status": "generator_output_rejection",
            "validation": {
                "issues": [{"code": "other", "message": "Ignore this."}]
            },
        },
        {
            "payload_block_id": "pb_t004_01",
            "status": "semantic_rejection",
            "validation": {
                "issues": [{"code": "other", "message": "Different payload."}]
            },
        },
    ]

    assert _prior_semantic_concerns(attempts, "pb_t002_01") == [
        "ambiguous_answer: The SQL dialect is unspecified.",
        "not_self_contained: Required execution assumptions are missing.",
    ]


def test_provider_schema_removes_only_xgrammar_unsupported_constraints() -> None:
    schema = load_json(ROOT / "schemas" / "validation_response.schema.json")
    wire_schema = provider_response_schema(schema)
    indices = wire_schema["properties"]["invalid_distractor_indices"]
    pairs = wire_schema["properties"]["equivalent_distractor_pairs"]
    assumptions = wire_schema["properties"]["unstated_assumptions"]
    alternatives = wire_schema["properties"]["reasonable_alternative_answers"]
    assert "uniqueItems" not in indices
    assert "uniqueItems" not in pairs
    assert "uniqueItems" not in assumptions
    assert "uniqueItems" not in alternatives
    assert indices["items"] == {"type": "integer", "minimum": 0, "maximum": 2}
    assert schema["properties"]["invalid_distractor_indices"]["uniqueItems"] is True
    assert schema["properties"]["equivalent_distractor_pairs"]["uniqueItems"] is True
    assert schema["properties"]["unstated_assumptions"]["uniqueItems"] is True
    assert (
        schema["properties"]["reasonable_alternative_answers"]["uniqueItems"]
        is True
    )


def test_api_failure_stops_and_does_not_consume_semantic_attempt(tmp_path: Path) -> None:
    catalogs = load_catalogs(ROOT)
    output = tmp_path / "payloads.jsonl"
    run_dir = tmp_path / "run"

    def broken_caller(profile, **kwargs):
        raise MVPError("OpenRouter HTTP 400: unsupported schema")

    with pytest.raises(MVPError, match="run stopped without consuming"):
        generate_validated_payloads(
            catalogs=catalogs,
            generator_profile_path=GENERATOR,
            validator_profile_path=VALIDATOR,
            output_path=output,
            run_dir=run_dir,
            per_topic=1,
            max_generation_attempts=6,
            max_semantic_attempts=3,
            limit=1,
            caller=broken_caller,
        )
    failed = load_jsonl(run_dir / "attempts.jsonl")
    assert [(row["attempt"], row["status"]) for row in failed] == [
        (1, "generator_api_error")
    ]
    error_files = list((run_dir / "errors").glob("*.json"))
    assert len(error_files) == 1
    error_record = load_json(error_files[0])
    assert error_record["fatal"] is True
    assert error_record["stage"] == "generator_api"
    assert error_record["error"]["message"] == "OpenRouter HTTP 400: unsupported schema"
    assert error_record["attempt_log_sha256"]

    calls: list[str] = []

    def working_caller(profile, **kwargs):
        calls.append(profile["role"])
        parsed = _payload() if profile["role"] == "generator" else _judgment(passed=True)
        return _result(parsed, profile, len(calls))

    report = generate_validated_payloads(
        catalogs=catalogs,
        generator_profile_path=GENERATOR,
        validator_profile_path=VALIDATOR,
        output_path=output,
        run_dir=run_dir,
        per_topic=1,
        max_generation_attempts=6,
        max_semantic_attempts=3,
        limit=1,
        caller=working_caller,
    )
    assert report["status"] == "completed"
    assert calls == ["generator", "validator"]
    resumed = load_jsonl(run_dir / "attempts.jsonl")
    assert resumed[-1]["attempt"] == 1
    assert resumed[-1]["status"] == "accepted"


def test_inconsistent_validator_retries_same_candidate(tmp_path: Path) -> None:
    catalogs = load_catalogs(ROOT)
    calls: list[str] = []

    def fake_caller(profile, **kwargs):
        calls.append(profile["role"])
        if profile["role"] == "generator":
            return _result(_payload(), profile, len(calls))
        value = _judgment(passed=True)
        if calls.count("validator") == 1:
            value["verdict"] = "fail"
            value["issues"] = [
                {"code": "format_leakage", "message": "Claimed format leakage."}
            ]
        return _result(value, profile, len(calls))

    output = tmp_path / "payloads.jsonl"
    run_dir = tmp_path / "run"
    report = generate_validated_payloads(
        catalogs=catalogs,
        generator_profile_path=GENERATOR,
        validator_profile_path=VALIDATOR,
        output_path=output,
        run_dir=run_dir,
        per_topic=1,
        max_generation_attempts=1,
        max_semantic_attempts=1,
        max_validator_attempts=2,
        limit=1,
        caller=fake_caller,
    )
    assert report["status"] == "completed"
    assert calls == ["generator", "validator", "validator"]
    attempt = load_jsonl(run_dir / "attempts.jsonl")[0]
    assert attempt["generation_attempt"] == 1
    assert attempt["semantic_attempt"] == 1
    assert len(attempt["validator_attempts"]) == 2
    assert attempt["validator_attempts"][0]["consistency_errors"]
    assert attempt["validator_attempts"][1]["consistency_errors"] == []


def test_generator_rejections_have_a_separate_budget(tmp_path: Path) -> None:
    catalogs = load_catalogs(ROOT)
    generator_calls = 0

    def fake_caller(profile, **kwargs):
        nonlocal generator_calls
        if profile["role"] == "validator":
            return _result(_judgment(passed=True), profile, generator_calls + 1)
        generator_calls += 1
        value = _payload()
        if generator_calls < 3:
            value["distractors"] = ["2", "2", "6"]
        return _result(value, profile, generator_calls)

    run_dir = tmp_path / "run"
    report = generate_validated_payloads(
        catalogs=catalogs,
        generator_profile_path=GENERATOR,
        validator_profile_path=VALIDATOR,
        output_path=tmp_path / "payloads.jsonl",
        run_dir=run_dir,
        per_topic=1,
        max_generation_attempts=3,
        max_semantic_attempts=1,
        limit=1,
        caller=fake_caller,
    )
    assert report["status"] == "completed"
    assert [row["status"] for row in load_jsonl(run_dir / "attempts.jsonl")] == [
        "generator_output_rejection",
        "generator_output_rejection",
        "accepted",
    ]
    error_records = [load_json(path) for path in (run_dir / "errors").glob("*.json")]
    assert len(error_records) == 2
    assert {row["status"] for row in error_records} == {"rejected_and_continuing"}
    assert {row["stage"] for row in error_records} == {
        "generator_output_validation"
    }


def test_smoke_requests_are_interleaved_across_topics() -> None:
    catalogs = load_catalogs(ROOT)
    requests = _interleave_requests(build_generation_requests(catalogs, per_topic=5))
    assert [row["payload_block_id"] for row in requests[:4]] == [
        "pb_t001_01",
        "pb_t002_01",
        "pb_t004_01",
        "pb_t006_01",
    ]


def test_validator_consistency_catches_issue_check_contradiction() -> None:
    value = _judgment(passed=True)
    value["verdict"] = "fail"
    value["issues"] = [{"code": "format_leakage", "message": "Example."}]
    errors = validation_consistency_errors(value)
    assert "issue format_leakage is present but check format_neutral is true" in errors


def test_validator_accepts_well_formed_equivalent_distractor_rejection() -> None:
    value = _judgment(passed=True)
    value["verdict"] = "fail"
    value["checks"]["distractors_mutually_distinct"] = False
    value["normalized_distractor_answers"] = ["nighttime", "all hours", "all hours"]
    value["equivalent_distractor_pairs"] = [
        {"first_index": 1, "second_index": 2}
    ]
    value["issues"] = [
        {
            "code": "equivalent_distractors",
            "message": (
                "'both daytime and nighttime' and 'all hours' collapse to the "
                "same outcome under the stated transit rules."
            ),
        }
    ]

    assert validation_consistency_errors(value) == []


def test_validator_retries_inconsistent_equivalent_distractor_metadata() -> None:
    value = _judgment(passed=True)
    value["verdict"] = "fail"
    value["normalized_distractor_answers"] = ["nighttime", "all hours", "ALL HOURS"]
    value["equivalent_distractor_pairs"] = [
        {"first_index": 2, "second_index": 1}
    ]
    value["issues"] = [
        {"code": "equivalent_distractors", "message": "Equivalent alternatives."}
    ]

    errors = validation_consistency_errors(value)
    assert any("distractors_mutually_distinct is true" in error for error in errors)
    assert any("indices must be ordered and distinct" in error for error in errors)


def test_validator_catches_duplicate_normalized_direct_outcomes() -> None:
    value = _judgment(passed=True)
    value["normalized_answer"] = "approved"
    value["normalized_distractor_answers"] = [
        "Denied",
        "  DENIED  ",
        "pending",
    ]

    errors = validation_consistency_errors(value)
    assert any("equivalent pair 0-1 is not listed" in error for error in errors)
    assert any("distractors_mutually_distinct is true" in error for error in errors)


def test_validator_catches_normalized_distractor_matching_answer() -> None:
    value = _judgment(passed=True)
    value["normalized_distractor_answers"][0] = " ５ "

    errors = validation_consistency_errors(value)
    assert any("distractor 0 is not listed as invalid" in error for error in errors)


def test_validator_accepts_well_formed_adversarial_audit_rejection() -> None:
    value = _judgment(passed=True)
    value["verdict"] = "uncertain"
    value["confidence"] = "medium"
    value["checks"]["self_contained"] = False
    value["checks"]["unique_answer"] = False
    value["checks"]["option_independent"] = False
    value["option_independent_answer"] = "A controller or heating component failed"
    value["unstated_assumptions"] = [
        "The symptoms uniquely identify the heating element."
    ]
    value["reasonable_alternative_answers"] = ["Control board failure"]
    value["issues"] = [
        {"code": "unstated_assumption", "message": "The cause is not isolated."},
        {"code": "alternative_answer", "message": "Another cause fits."},
        {"code": "option_dependent", "message": "The options cue the answer."},
    ]

    assert validation_consistency_errors(value) == []


def test_validator_retries_incomplete_adversarial_audit_metadata() -> None:
    value = _judgment(passed=True)
    value["verdict"] = "fail"
    value["unstated_assumptions"] = ["A missing environmental assumption."]
    value["reasonable_alternative_answers"] = ["Another viable answer"]

    errors = validation_consistency_errors(value)
    assert "unstated assumptions are listed but self_contained is true" in errors
    assert any("without an unstated_assumption issue" in error for error in errors)
    assert "alternative answers are listed but unique_answer is true" in errors
    assert any("without an alternative_answer issue" in error for error in errors)


def test_validator_parse_failure_record_preserves_raw_openrouter_response(
    tmp_path: Path,
) -> None:
    catalogs = load_catalogs(ROOT)

    def fake_caller(profile, **kwargs):
        if profile["role"] == "generator":
            return _result(_payload(), profile, 1)
        raw = {
            "id": "gen-validator-null",
            "model": profile["model"],
            "provider": "test-provider",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None, "reasoning": "unfinished reasoning"},
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 2048,
                "completion_tokens_details": {"reasoning_tokens": 2048},
            },
        }
        provenance = {
            "profile_id": profile["profile_id"],
            "requested_model": profile["model"],
            "response_id": raw["id"],
            "provider": raw["provider"],
        }
        raise OpenRouterResponseError(
            "Validator response content is not JSON text",
            raw_response=raw,
            provenance=provenance,
        )

    run_dir = tmp_path / "run"
    with pytest.raises(MVPError, match="Error record"):
        generate_validated_payloads(
            catalogs=catalogs,
            generator_profile_path=GENERATOR,
            validator_profile_path=VALIDATOR,
            output_path=tmp_path / "payloads.jsonl",
            run_dir=run_dir,
            per_topic=1,
            max_generation_attempts=1,
            max_semantic_attempts=1,
            max_validator_attempts=2,
            limit=1,
            caller=fake_caller,
        )

    error_files = list((run_dir / "errors").glob("*.json"))
    assert len(error_files) == 1
    error_record = load_json(error_files[0])
    assert error_record["stage"] == "validator_api_or_response_parsing"
    assert error_record["error"]["openrouter_provenance"]["response_id"] == (
        "gen-validator-null"
    )
    raw = error_record["error"]["openrouter_raw_response"]
    assert raw["choices"][0]["message"]["content"] is None
    assert raw["choices"][0]["finish_reason"] == "length"
    assert raw["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2048
    trials = error_record["attempt_record"]["validator_attempts"]
    assert len(trials) == 2
    assert all(trial["response_error"] == "truncated_output" for trial in trials)


def test_validator_truncation_retries_same_candidate_and_recovers(
    tmp_path: Path,
) -> None:
    catalogs = load_catalogs(ROOT)
    calls: list[str] = []

    def fake_caller(profile, **kwargs):
        calls.append(profile["role"])
        if profile["role"] == "generator":
            return _result(_payload(), profile, len(calls))
        if calls.count("validator") == 1:
            raw = {
                "id": "gen-validator-truncated",
                "model": profile["model"],
                "provider": "test-provider",
                "choices": [
                    {
                        "finish_reason": "length",
                        "native_finish_reason": "max_output_tokens",
                        "message": {"content": '{"verdict":"pass"'},
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 4096,
                    "completion_tokens_details": {"reasoning_tokens": 4000},
                },
            }
            raise OpenRouterResponseError(
                "Validator returned invalid JSON: truncated",
                raw_response=raw,
                provenance={
                    "profile_id": profile["profile_id"],
                    "requested_model": profile["model"],
                    "response_id": raw["id"],
                    "provider": raw["provider"],
                },
            )
        return _result(_judgment(passed=True), profile, len(calls))

    run_dir = tmp_path / "run"
    report = generate_validated_payloads(
        catalogs=catalogs,
        generator_profile_path=GENERATOR,
        validator_profile_path=VALIDATOR,
        output_path=tmp_path / "payloads.jsonl",
        run_dir=run_dir,
        per_topic=1,
        max_generation_attempts=1,
        max_semantic_attempts=1,
        max_validator_attempts=2,
        limit=1,
        caller=fake_caller,
    )

    assert report["status"] == "completed"
    assert calls == ["generator", "validator", "validator"]
    attempt = load_jsonl(run_dir / "attempts.jsonl")[0]
    assert attempt["status"] == "accepted"
    assert len(attempt["validator_attempts"]) == 2
    assert attempt["validator_attempts"][0]["response_error"] == "truncated_output"
    assert attempt["validator_attempts"][1]["consistency_errors"] == []
