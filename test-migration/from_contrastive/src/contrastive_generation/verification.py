"""Deterministic QC and two-call blind cross-verification."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from .preflight import FULLWIDTH_DIGIT_RE, LATIN_RE, text_cue_terms
from .providers import (
    BaseProvider,
    MalformedProviderResponse,
    ProviderResponse,
    ProviderTransportError,
)
from .rendering import load_prompt_template
from .utils import CONTRASTIVE_ROOT, ContrastiveError, load_json


EXTRACTION_SCHEMA_PATH = (
    CONTRASTIVE_ROOT
    / "schemas"
    / "contrastive_verification_extraction.schema.json"
)
COMPARISON_SCHEMA_PATH = (
    CONTRASTIVE_ROOT
    / "schemas"
    / "contrastive_verification_comparison.schema.json"
)
EXTRACTION_SCHEMA = load_json(EXTRACTION_SCHEMA_PATH)
COMPARISON_SCHEMA = load_json(COMPARISON_SCHEMA_PATH)
VERIFICATION_SCHEMAS = {
    ("extraction", "en"): EXTRACTION_SCHEMA,
    ("comparison", "en"): COMPARISON_SCHEMA,
    ("extraction", "ja"): load_json(
        CONTRASTIVE_ROOT
        / "schemas"
        / "contrastive_verification_extraction.ja.schema.json"
    ),
    ("comparison", "ja"): load_json(
        CONTRASTIVE_ROOT
        / "schemas"
        / "contrastive_verification_comparison.ja.schema.json"
    ),
}
JAPANESE_VERIFICATION_KEYS = {
    "extraction": {
        "内容のみで回答可能": "is_answerable_from_content",
        "独立に選んだ選択肢": "independently_selected_option",
        "答えの漏えい": "leaked_answer_signal",
        "作成要件に一致": "matches_authoring_requirements",
        "文体が適切": "register_appropriate",
        "誤答選択肢が妥当": "distractors_plausible",
        "誤答選択肢の所見": "distractor_notes",
        "文体の所見": "register_notes",
    },
    "comparison": {
        "合格": "pass",
        "選択肢一致": "option_match",
        "回答可能": "answerable",
        "漏えいなし": "no_leak",
        "作成要件に一致": "matches_authoring_requirements",
        "文体が適切": "register_appropriate",
        "誤答選択肢が妥当": "distractors_plausible",
        "問題点": "issues",
    },
}

ANSWER_LEAK_PHRASES = {
    "en": (
        "the answer is",
        "correct answer is",
        "correct option is",
        "right option is",
        "right choice is",
    ),
    "ja": (
        "答えは",
        "正解は",
        "正しい選択肢は",
        "解答は",
    ),
}
JAPANESE_SCRIPT_RE = re.compile(r"[ぁ-んァ-ヶ一-龯]")
LATIN_TOKEN_RE = re.compile(r"[A-Za-z]+")

ProviderCaller = Callable[
    [
        BaseProvider,
        str,
        str,
        float,
        float,
        int,
        int,
        dict[str, Any],
        str,
    ],
    tuple[ProviderResponse, str],
]


def parse_structured_response(
    text: str,
    schema: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Parse one JSON object and validate it against its exact response schema."""
    stripped = text.strip()
    fence = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence:
        stripped = fence.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ContrastiveError(f"{label} returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ContrastiveError(f"{label} response must be a JSON object")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        raise ContrastiveError(f"{label} failed response schema: {details}")
    return value


def verification_schema(stage: str, language: str) -> dict[str, Any]:
    """Return the language-matched model-visible verifier response schema."""
    try:
        return VERIFICATION_SCHEMAS[(stage, language)]
    except KeyError as exc:
        raise ContrastiveError(
            f"Unsupported verification schema surface: {stage}/{language}"
        ) from exc


def validate_verification_surfaces() -> None:
    """Fail if Japanese verifier prose or user-defined schema names leak Latin."""
    for stage in ("extraction", "comparison"):
        path = (
            CONTRASTIVE_ROOT
            / "prompts"
            / f"verification_{stage}_ja.txt"
        )
        system, user = load_prompt_template(path)
        prose = re.sub(
            r"<<[A-Z0-9_]+>>",
            "",
            system + "\n" + user,
        )
        latin_hits = sorted(set(LATIN_RE.findall(prose)))
        if latin_hits:
            raise ContrastiveError(
                f"Japanese verification prompt contains Latin prose: "
                f"{stage} {latin_hits}"
            )
        schema = verification_schema(stage, "ja")
        property_hits = sorted(
            name
            for name in schema["properties"]
            if LATIN_RE.search(str(name))
        )
        if property_hits:
            raise ContrastiveError(
                f"Japanese verification schema contains Latin property names: "
                f"{stage} {property_hits}"
            )


def _canonical_verification_result(
    stage: str,
    language: str,
    value: dict[str, Any],
) -> dict[str, Any]:
    if language == "en":
        return value
    mapping = JAPANESE_VERIFICATION_KEYS[stage]
    return {mapping[key]: item for key, item in value.items()}


def content_fields(
    payload: dict[str, Any],
    response_schema: dict[str, Any],
) -> tuple[dict[str, str], str, str, str]:
    """Return localized slot values and answer-key field names."""
    properties = response_schema["properties"]
    values_name = (
        "slot_values" if "slot_values" in properties else "各項目の内容"
    )
    answer_name = "answer_key" if "answer_key" in properties else "解答記号"
    values = payload.get(values_name)
    answer = payload.get(answer_name)
    if not isinstance(values, dict) or not isinstance(answer, str):
        raise ContrastiveError("Generated output lacks localized content fields")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in values.items()):
        raise ContrastiveError("Generated slot values must be strings")
    return dict(values), answer, values_name, answer_name


def contract_type(response_schema: dict[str, Any]) -> str:
    properties = response_schema["properties"]
    answer_name = "answer_key" if "answer_key" in properties else "解答記号"
    return (
        "multiple_choice"
        if "enum" in properties[answer_name]
        else "free_response"
    )


def answer_key_ordinal(
    payload: dict[str, Any],
    response_schema: dict[str, Any],
) -> int | None:
    _, answer, _, answer_name = content_fields(payload, response_schema)
    answer_schema = response_schema["properties"][answer_name]
    allowed = answer_schema.get("enum")
    if allowed is None:
        return None
    try:
        return list(allowed).index(answer) + 1
    except ValueError as exc:
        raise ContrastiveError("Generated answer key is outside the declared enum") from exc


def deterministic_qc(
    text: str,
    response_schema: dict[str, Any],
    language: str,
    cue_lexicon: dict[str, Any],
    *,
    japanese_surface: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate generated structured content before making paid verifier calls."""
    checks: dict[str, Any] = {
        "nonempty": bool(text.strip()),
        "schema_valid": False,
        "slot_values_nonempty": False,
        "cue_free": False,
        "japanese_surface": language != "ja",
        "answer_signal_free": False,
    }
    issues: list[str] = []
    if not checks["nonempty"]:
        issues.append("empty provider text")
        return {"pass": False, "checks": checks, "issues": issues}, None
    try:
        payload = parse_structured_response(
            text,
            response_schema,
            label="generator",
        )
        checks["schema_valid"] = True
        slot_values, _, _, _ = content_fields(payload, response_schema)
    except ContrastiveError as exc:
        issues.append(str(exc))
        return {"pass": False, "checks": checks, "issues": issues}, None
    empty_slots = sorted(
        key for key, value in slot_values.items() if not value.strip()
    )
    checks["slot_values_nonempty"] = not empty_slots
    if empty_slots:
        issues.append(f"empty generated slots: {empty_slots}")
    combined = "\n".join(slot_values.values())
    cue_occurrences = text_cue_terms(
        combined,
        language,
        cue_lexicon,
        scope="generated_content",
    )
    checks["cue_free"] = not cue_occurrences
    checks["cue_hits"] = cue_occurrences
    if cue_occurrences:
        issues.append(f"frozen cue terms found: {cue_occurrences}")
    normalized = unicodedata.normalize("NFKC", combined)
    if language == "ja":
        japanese_script_chars = len(JAPANESE_SCRIPT_RE.findall(normalized))
        latin_letters = len(LATIN_RE.findall(normalized))
        denominator = latin_letters + japanese_script_chars
        latin_share = latin_letters / denominator if denominator else 0.0
        latin_tokens = sorted(set(LATIN_TOKEN_RE.findall(normalized)))
        fullwidth_digits = sorted(set(FULLWIDTH_DIGIT_RE.findall(combined)))
        min_japanese_chars = int(japanese_surface["min_japanese_chars"])
        max_latin_share = float(japanese_surface["max_latin_share"])
        enough_japanese = japanese_script_chars >= min_japanese_chars
        acceptable_latin_share = latin_share <= max_latin_share
        checks["japanese_surface"] = (
            enough_japanese
            and acceptable_latin_share
            and not fullwidth_digits
        )
        checks["latin_tokens"] = latin_tokens
        checks["latin_share"] = latin_share
        checks["japanese_script_chars"] = japanese_script_chars
        checks["fullwidth_digit_hits"] = fullwidth_digits
        checks["japanese_surface_policy"] = {
            "min_japanese_chars": min_japanese_chars,
            "max_latin_share": max_latin_share,
        }
        if not enough_japanese:
            issues.append(
                "Japanese output has insufficient Japanese script characters: "
                f"{japanese_script_chars} < {min_japanese_chars}"
            )
        if not acceptable_latin_share:
            issues.append(
                "Japanese output ASCII-Latin share exceeds the configured "
                f"maximum: {latin_share:.6f} > {max_latin_share:.6f}"
            )
        if fullwidth_digits:
            issues.append(
                "Japanese output contains full-width digits: "
                f"{fullwidth_digits}"
            )
    folded = normalized.casefold()
    leak_hits = [
        phrase
        for phrase in ANSWER_LEAK_PHRASES[language]
        if unicodedata.normalize("NFKC", phrase).casefold() in folded
    ]
    checks["answer_signal_free"] = not leak_hits
    checks["answer_signal_hits"] = leak_hits
    if leak_hits:
        issues.append(f"answer-revealing phrases found: {leak_hits}")
    passed = all(
        checks[name]
        for name in (
            "nonempty",
            "schema_valid",
            "slot_values_nonempty",
            "cue_free",
            "japanese_surface",
            "answer_signal_free",
        )
    )
    return {"pass": passed, "checks": checks, "issues": issues}, payload


def _render_prompt(
    kind: str,
    language: str,
    replacements: dict[str, str],
) -> tuple[str, str]:
    path = (
        CONTRASTIVE_ROOT
        / "prompts"
        / f"verification_{kind}_{language}.txt"
    )
    system, user = load_prompt_template(path)
    for name, value in replacements.items():
        user = user.replace(f"<<{name}>>", value)
    unresolved = sorted(set(re.findall(r"<<[A-Z0-9_]+>>", user)))
    if unresolved:
        raise ContrastiveError(
            f"Unresolved verification prompt placeholders: {unresolved}"
        )
    return system, user


def _seed(base: int, offset: int, retry: int = 0) -> int:
    return (int(base) + offset + retry) % 2147483648


def _comparison_consistent(
    comparison: dict[str, Any],
    extraction: dict[str, Any],
    expected_ordinal: int | None,
) -> bool:
    expected_option_match = (
        True
        if expected_ordinal is None
        else extraction["independently_selected_option"] == expected_ordinal
    )
    expected_answerable = bool(extraction["is_answerable_from_content"])
    expected_no_leak = not bool(extraction["leaked_answer_signal"])
    expected_requirements = bool(
        extraction["matches_authoring_requirements"]
    )
    expected_register = bool(extraction["register_appropriate"])
    expected_distractors = bool(extraction["distractors_plausible"])
    expected_pass = all(
        (
            expected_option_match,
            expected_answerable,
            expected_no_leak,
            expected_requirements,
            expected_register,
            expected_distractors,
        )
    )
    return (
        comparison["option_match"] is expected_option_match
        and comparison["answerable"] is expected_answerable
        and comparison["no_leak"] is expected_no_leak
        and comparison["matches_authoring_requirements"]
        is expected_requirements
        and comparison["register_appropriate"] is expected_register
        and comparison["distractors_plausible"] is expected_distractors
        and comparison["pass"] is expected_pass
    )


def verify_candidate(
    *,
    provider: BaseProvider,
    rendered_item: str,
    authoring_requirements: str,
    language: str,
    response_schema: dict[str, Any],
    generated_payload: dict[str, Any],
    generation_seed: int,
    config: dict[str, Any],
    call_provider: ProviderCaller,
) -> tuple[dict[str, Any], list[str]]:
    """Run blind extraction and comparison, escalating once when configured."""
    expected_ordinal = answer_key_ordinal(generated_payload, response_schema)
    kind = contract_type(response_schema)
    contract_label = {
        ("en", "multiple_choice"): "four-option multiple-choice",
        ("en", "free_response"): "free-response",
        ("ja", "multiple_choice"): "四つの選択肢がある選択式",
        ("ja", "free_response"): "選択肢のない自由回答形式",
    }[(language, kind)]
    sessions: list[str] = []
    extraction_schema = verification_schema("extraction", language)
    comparison_schema = verification_schema("comparison", language)
    schema_retries = int(config.get("schema_retries", 1))
    temperature = float(config.get("temperature", 0))
    top_p = float(config.get("top_p", 1))

    def run_once(
        label: str,
        *,
        seed_offset: int,
        extraction_max_tokens: int,
    ) -> dict[str, Any]:
        extraction: dict[str, Any] | None = None
        extraction_errors: list[str] = []
        extraction_system, extraction_user = _render_prompt(
            "extraction",
            language,
            {
                "CONTRACT": contract_label,
                "AUTHORING_REQUIREMENTS": authoring_requirements,
                "RENDERED_ITEM": rendered_item,
            },
        )
        for retry in range(schema_retries + 1):
            try:
                response, session = call_provider(
                    provider,
                    extraction_system,
                    extraction_user,
                    temperature,
                    top_p,
                    extraction_max_tokens,
                    _seed(generation_seed, seed_offset, retry),
                    extraction_schema,
                    f"{label}_extraction_r{retry}",
                )
                sessions.append(session)
                raw_extraction = parse_structured_response(
                    response.text,
                    extraction_schema,
                    label="verification extraction",
                )
                extraction = _canonical_verification_result(
                    "extraction",
                    language,
                    raw_extraction,
                )
                break
            except MalformedProviderResponse as exc:
                session = getattr(exc, "session_id", None)
                if session:
                    sessions.append(str(session))
                extraction_errors.append(str(exc))
            except ProviderTransportError:
                # Infrastructure failure is not evidence against candidate
                # content and must never be converted into a schema rejection.
                raise
            except ContrastiveError as exc:
                extraction_errors.append(str(exc))
        if extraction is None:
            return {
                "pass": False,
                "schema_valid": False,
                "stage": "extraction",
                "schema_errors": extraction_errors,
            }

        comparison: dict[str, Any] | None = None
        comparison_errors: list[str] = []
        expected = (
            ("該当なし" if language == "ja" else "none")
            if expected_ordinal is None
            else str(expected_ordinal)
        )
        comparison_system, comparison_user = _render_prompt(
            "comparison",
            language,
            {
                "CONTRACT": contract_label,
                "AUTHORING_REQUIREMENTS": authoring_requirements,
                "RENDERED_ITEM": rendered_item,
                "EXTRACTION_JSON": json.dumps(
                    raw_extraction,
                    ensure_ascii=False,
                ),
                "EXPECTED_OPTION": expected,
            },
        )
        for retry in range(schema_retries + 1):
            try:
                response, session = call_provider(
                    provider,
                    comparison_system,
                    comparison_user,
                    temperature,
                    top_p,
                    int(config.get("comparison_max_output_tokens", 1024)),
                    _seed(generation_seed, seed_offset + 10_000, retry),
                    comparison_schema,
                    f"{label}_comparison_r{retry}",
                )
                sessions.append(session)
                raw_candidate = parse_structured_response(
                    response.text,
                    comparison_schema,
                    label="verification comparison",
                )
                candidate = _canonical_verification_result(
                    "comparison",
                    language,
                    raw_candidate,
                )
                if not _comparison_consistent(
                    candidate,
                    extraction,
                    expected_ordinal,
                ):
                    raise ContrastiveError(
                        "verification comparison contradicts the blind extraction"
                    )
                comparison = candidate
                break
            except MalformedProviderResponse as exc:
                session = getattr(exc, "session_id", None)
                if session:
                    sessions.append(str(session))
                comparison_errors.append(str(exc))
            except ProviderTransportError:
                # Preserve the accepted author candidate for verification resume.
                raise
            except ContrastiveError as exc:
                comparison_errors.append(str(exc))
        if comparison is None:
            return {
                "pass": False,
                "schema_valid": False,
                "stage": "comparison",
                "extracted": extraction,
                "schema_errors": comparison_errors,
            }
        return {
            **comparison,
            "schema_valid": True,
            "extracted": extraction,
            "expected_option": expected_ordinal,
        }

    initial = run_once(
        "initial",
        seed_offset=1_000_003,
        extraction_max_tokens=int(
            config.get("extraction_max_output_tokens", 2048)
        ),
    )
    escalation = config.get("escalation") or {}
    if initial.get("pass") or not escalation.get("enabled", True):
        return initial, sessions
    escalated = run_once(
        "escalated",
        seed_offset=2_000_003,
        extraction_max_tokens=int(
            escalation.get(
                "extraction_max_output_tokens",
                config.get("extraction_max_output_tokens", 2048),
            )
        ),
    )
    escalated["escalation"] = {
        "applied": True,
        "initial": initial,
    }
    return escalated, sessions
