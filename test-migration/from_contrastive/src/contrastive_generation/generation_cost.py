"""All-in authoring and verification cost estimation for generation runs."""

from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal
from typing import Any, Iterable

from .allocation import generator_for
from .rendering import load_prompt_template
from .utils import (
    CONTRASTIVE_ROOT,
    ContrastiveError,
    canonical_json,
    load_json,
)
from .verification import verification_schema


EXPECTED_EXTRACTION_OUTPUT_TOKENS = 256
EXPECTED_COMPARISON_OUTPUT_TOKENS = 128
WRAPPER_OVERHEAD_TOKENS = 512
PROVIDER_FRAMING_OVERHEAD_TOKENS = 1024
PAIR_SHARED_VERIFICATION_BODIES = 2


def _rough_tokens(text: str, language: str) -> int:
    return max(1, math.ceil(len(text) / (1.8 if language == "ja" else 4.0)))


def _prompt_overhead(
    stage: str,
    language: str,
    schema: dict[str, Any],
) -> int:
    path = (
        CONTRASTIVE_ROOT
        / "prompts"
        / f"verification_{stage}_{language}.txt"
    )
    system, user = load_prompt_template(path)
    for placeholder in (
        "<<CONTRACT>>",
        "<<AUTHORING_REQUIREMENTS>>",
        "<<RENDERED_ITEM>>",
        "<<EXPECTED_OPTION>>",
        "<<EXTRACTION_JSON>>",
    ):
        user = user.replace(placeholder, "")
    return _rough_tokens(
        system + "\n" + user + "\n" + canonical_json(schema),
        language,
    )


def _prompt_byte_upper(
    stage: str,
    language: str,
    schema: dict[str, Any],
) -> int:
    path = (
        CONTRASTIVE_ROOT
        / "prompts"
        / f"verification_{stage}_{language}.txt"
    )
    system, user = load_prompt_template(path)
    for placeholder in (
        "<<CONTRACT>>",
        "<<AUTHORING_REQUIREMENTS>>",
        "<<RENDERED_ITEM>>",
        "<<EXPECTED_OPTION>>",
        "<<EXTRACTION_JSON>>",
    ):
        user = user.replace(placeholder, "")
    return len(
        (
            system
            + "\n"
            + user
            + "\n"
            + canonical_json(schema)
        ).encode("utf-8")
    )


def validate_verification_context(
    twin: Any,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    """Conservatively gate verifier input plus output against context limits."""
    maxima: dict[str, int] = {}
    verification = generation_config["verification"]
    extraction_output = max(
        int(verification["extraction_max_output_tokens"]),
        int(verification["escalation"]["extraction_max_output_tokens"]),
    )
    comparison_output = int(
        verification["comparison_max_output_tokens"]
    )
    profiles = {
        "generator_a": twin.config_a["provider_profile"],
        "generator_b": twin.config_b["provider_profile"],
    }
    for identity in twin.identities:
        author_role = generator_for(identity[0], identity[1])
        verifier_role = (
            "generator_b" if author_role == "generator_a" else "generator_a"
        )
        job = (
            twin.jobs_a[identity]
            if author_role == "generator_a"
            else twin.jobs_b[identity]
        )
        author_config = (
            twin.config_a
            if author_role == "generator_a"
            else twin.config_b
        )
        language = str(job["language"])
        request = load_json(
            CONTRASTIVE_ROOT
            / str(job["generation_attempts"][0]["request_path"])
        )
        requirements_upper = len(
            str(request["user"]).encode("utf-8")
        )
        author_output = int(
            author_config["generation"]["max_output_tokens"]
        )
        extraction_total = (
            author_output
            + requirements_upper
            + _prompt_byte_upper(
                "extraction",
                language,
                verification_schema("extraction", language),
            )
            + WRAPPER_OVERHEAD_TOKENS
            + PROVIDER_FRAMING_OVERHEAD_TOKENS
            + extraction_output
        )
        comparison_total = (
            author_output
            + requirements_upper
            + _prompt_byte_upper(
                "comparison",
                language,
                verification_schema("comparison", language),
            )
            + WRAPPER_OVERHEAD_TOKENS
            + PROVIDER_FRAMING_OVERHEAD_TOKENS
            + extraction_output
            + comparison_output
        )
        required = max(extraction_total, comparison_total)
        profile = profiles[verifier_role]
        profile_id = str(profile["profile_id"])
        maxima[profile_id] = max(maxima.get(profile_id, 0), required)
        available = int(profile["capabilities"]["max_context_tokens"])
        if required > available:
            raise ContrastiveError(
                f"Verifier context upper bound {required} exceeds provider "
                f"{profile_id!r} limit {available}"
            )
    return {
        "status": "passed",
        "method": (
            "UTF-8 bytes upper-bound authored verifier prose and requirements; "
            "author/verifier output caps and fixed framing allowances are added "
            "as tokens."
        ),
        "provider_maximum_required_tokens": dict(sorted(maxima.items())),
    }


def _priced(
    input_tokens: int,
    output_tokens: int,
    profile: dict[str, Any],
) -> Decimal:
    return (
        Decimal(input_tokens)
        * Decimal(str(profile["input_usd_per_million"]))
        + Decimal(output_tokens)
        * Decimal(str(profile["output_usd_per_million"]))
    ) / Decimal(1_000_000)


def _profile_key(profile: dict[str, Any]) -> str:
    return str(profile["profile_id"])


def _scenario(
    *,
    work_items: Iterable[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
    ],
    verification: dict[str, Any],
    mode: str,
    transport: bool,
) -> dict[str, Any]:
    cost = Decimal(0)
    calls: Counter[str] = Counter()
    input_tokens: Counter[str] = Counter()
    output_tokens: Counter[str] = Counter()
    expected = mode in {"baseline", "generation_retry"}
    all_attempts = mode != "baseline"
    schema_attempts = int(verification["schema_retries"]) + 1
    escalation = verification["escalation"]
    verification_runs = 1 + (
        1 if mode == "maximum" and escalation["enabled"] else 0
    )

    for job, author_profile, verifier_profile, preflight_config in work_items:
        language = str(job["language"])
        attempts = (
            job["generation_attempts"]
            if all_attempts
            else job["generation_attempts"][:1]
        )
        author_output = int(
            preflight_config["generation"][
                "estimated_output_tokens"
                if expected
                else "max_output_tokens"
            ]
        )
        extraction_overhead = _prompt_overhead(
            "extraction",
            language,
            verification_schema("extraction", language),
        )
        comparison_overhead = _prompt_overhead(
            "comparison",
            language,
            verification_schema("comparison", language),
        )
        rendered_tokens = author_output + WRAPPER_OVERHEAD_TOKENS

        for attempt in attempts:
            request = load_json(
                CONTRASTIVE_ROOT / str(attempt["request_path"])
            )
            requirements_tokens = _rough_tokens(
                str(request["user"]),
                language,
            )
            author_input = int(
                attempt["preflight"][
                    "estimated_input_tokens"
                    if expected
                    else "context_input_token_upper_bound"
                ]
            )
            author_multiplier = (
                int(author_profile.get("api_attempts", 1))
                if transport
                else 1
            )
            author_key = _profile_key(author_profile)
            calls[author_key] += author_multiplier
            input_tokens[author_key] += author_input * author_multiplier
            output_tokens[author_key] += author_output * author_multiplier
            cost += (
                _priced(author_input, author_output, author_profile)
                * author_multiplier
            )

            for run_index in range(verification_runs):
                extraction_max = int(
                    verification["extraction_max_output_tokens"]
                    if run_index == 0
                    else escalation["extraction_max_output_tokens"]
                )
                extraction_output = (
                    min(
                        extraction_max,
                        EXPECTED_EXTRACTION_OUTPUT_TOKENS,
                    )
                    if expected
                    else extraction_max
                )
                comparison_max = int(
                    verification["comparison_max_output_tokens"]
                )
                comparison_output = (
                    min(
                        comparison_max,
                        EXPECTED_COMPARISON_OUTPUT_TOKENS,
                    )
                    if expected
                    else comparison_max
                )
                retry_multiplier = 1 if expected else schema_attempts
                transport_multiplier = (
                    int(verifier_profile.get("api_attempts", 1))
                    if transport
                    else 1
                )
                multiplier = retry_multiplier * transport_multiplier
                verifier_key = _profile_key(verifier_profile)
                extraction_input = (
                    rendered_tokens
                    + requirements_tokens
                    + extraction_overhead
                )
                comparison_input = (
                    rendered_tokens
                    + requirements_tokens
                    + extraction_output
                    + comparison_overhead
                )
                calls[verifier_key] += (
                    2 * multiplier * PAIR_SHARED_VERIFICATION_BODIES
                )
                input_tokens[verifier_key] += (
                    extraction_input + comparison_input
                ) * multiplier * PAIR_SHARED_VERIFICATION_BODIES
                output_tokens[verifier_key] += (
                    extraction_output + comparison_output
                ) * multiplier * PAIR_SHARED_VERIFICATION_BODIES
                cost += (
                    _priced(
                        extraction_input,
                        extraction_output,
                        verifier_profile,
                    )
                    + _priced(
                        comparison_input,
                        comparison_output,
                        verifier_profile,
                    )
                ) * multiplier * PAIR_SHARED_VERIFICATION_BODIES

    value = float(cost)
    if not math.isfinite(value):
        raise ContrastiveError(
            "Generation all-in cost exceeds the supported numeric range"
        )
    return {
        "estimated_usd": round(value, 6),
        "exact_usd": format(cost, "f"),
        "provider_calls": dict(sorted(calls.items())),
        "input_tokens": dict(sorted(input_tokens.items())),
        "output_tokens": dict(sorted(output_tokens.items())),
    }


def estimate_generation_cost(
    twin: Any,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    """Estimate author plus verifier exposure for the exact twin-run grid."""
    work_items = []
    profiles = {
        "generator_a": twin.config_a["provider_profile"],
        "generator_b": twin.config_b["provider_profile"],
    }
    for identity in twin.identities:
        author_role = generator_for(identity[0], identity[1])
        verifier_role = (
            "generator_b" if author_role == "generator_a" else "generator_a"
        )
        author_job = (
            twin.jobs_a[identity]
            if author_role == "generator_a"
            else twin.jobs_b[identity]
        )
        author_config = (
            twin.config_a
            if author_role == "generator_a"
            else twin.config_b
        )
        work_items.append(
            (
                author_job,
                profiles[author_role],
                profiles[verifier_role],
                author_config,
            )
        )
    verification = generation_config["verification"]
    baseline = _scenario(
        work_items=work_items,
        verification=verification,
        mode="baseline",
        transport=False,
    )
    generation_retry = _scenario(
        work_items=work_items,
        verification=verification,
        mode="generation_retry",
        transport=False,
    )
    maximum = _scenario(
        work_items=work_items,
        verification=verification,
        mode="maximum",
        transport=False,
    )
    transport = _scenario(
        work_items=work_items,
        verification=verification,
        mode="maximum",
        transport=True,
    )
    return {
        "pricing_status": "all_in_author_and_cross_verification",
        "verification_context_authorization": (
            validate_verification_context(twin, generation_config)
        ),
        "assumptions": {
            "expected_extraction_output_tokens": (
                EXPECTED_EXTRACTION_OUTPUT_TOKENS
            ),
            "expected_comparison_output_tokens": (
                EXPECTED_COMPARISON_OUTPUT_TOKENS
            ),
            "wrapper_overhead_tokens": WRAPPER_OVERHEAD_TOKENS,
            "provider_framing_overhead_tokens": (
                PROVIDER_FRAMING_OVERHEAD_TOKENS
            ),
            "purpose_masked_pair_shared_verification_bodies_per_job": (
                PAIR_SHARED_VERIFICATION_BODIES
            ),
            "baseline": (
                "One author attempt and one extraction/comparison verification "
                "pass for each of the two purpose-masked pair-shared bodies."
            ),
            "maximum_output_retry_scenario": (
                "Every planned author attempt reaches verification; every schema "
                "retry and enabled escalation uses configured maximum outputs."
            ),
            "transport_retry_risk_scenario": (
                "The maximum-output scenario multiplied by each provider's "
                "declared transport attempts."
            ),
        },
        "baseline": baseline,
        "generation_retry_scenario": generation_retry,
        "maximum_output_retry_scenario": maximum,
        "transport_retry_risk_scenario": transport,
    }


def authorize_generation_budget(
    cost: dict[str, Any],
    generation_config: dict[str, Any],
    profiles: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    budget = generation_config["budget"]
    profile_rows = list(profiles)
    if (
        any(
            float(profile["input_usd_per_million"]) == 0
            or float(profile["output_usd_per_million"]) == 0
            for profile in profile_rows
            if profile["provider"] != "mock"
        )
        and not budget["allow_zero_pricing"]
    ):
        raise ContrastiveError(
            "A paid-provider rate is zero; accurate rates or an explicit "
            "budget.allow_zero_pricing=true declaration is required"
        )
    basis = str(budget["cost_basis"])
    selected = Decimal(str(cost[basis]["exact_usd"]))
    maximum = Decimal(str(budget["max_usd"]))
    if selected > maximum:
        raise ContrastiveError(
            f"All-in generation cost scenario {basis} estimates "
            f"${format(selected, 'f')}, above the authorized budget of "
            f"${format(maximum, 'f')}"
        )
    return {
        "status": "passed",
        "basis": basis,
        "estimated_usd": float(cost[basis]["estimated_usd"]),
        "exact_estimated_usd": format(selected, "f"),
        "authorized_max_usd": float(maximum),
        "allow_zero_pricing": bool(budget["allow_zero_pricing"]),
    }
