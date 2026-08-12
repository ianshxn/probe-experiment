"""Model-request construction and fail-closed pre-generation audits."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

from jsonschema import Draft202012Validator

from .catalog import ANSWER_KEY_CANONICAL, TopicRecord
from .content_specs import (
    ContentSpecRecord,
    ModelContentSpec,
    ModelTopic,
    response_schema_for_spec,
)
from .rendering import load_prompt_template
from .utils import (
    CONTRASTIVE_ROOT,
    ContrastiveError,
    canonical_json,
    hash_object,
    load_json,
    load_yaml,
    local_reference,
    resolve_local_path,
    sha256_file,
)


PLACEHOLDER_RE = re.compile(r"<<([^<>\r\n]+)>>")
LATIN_RE = re.compile(r"[A-Za-z]")
FULLWIDTH_DIGIT_RE = re.compile(r"[０-９]")
REQUIRED_PLACEHOLDERS = {
    "TOPIC_LABEL",
    "DOMAIN",
    "REGISTER",
    "ROLE_STRUCTURE",
    "MODEL_GUIDANCE",
    "SLOT_DESCRIPTIONS",
    "OPTION_CONTRACT",
    "ANSWER_KEY_CONTRACT",
    "SLOTS",
}

# This exact set is deliberate. Catalogs are run-specific inputs and are recorded
# separately; these are the executable/scaffolding files whose approved bundle is
# expected not to drift between planning and submission.
LOCKED_FILES = {
    "pyproject.toml",
    "uv.lock",
    "prompts/content_generation_en.txt",
    "prompts/content_generation_ja.txt",
    "prompts/content_scaffolding.yaml",
    "prompts/generation_cue_lexicon.yaml",
    "prompts/pre_generation_semantic_approval.json",
    "prompts/verification_comparison_en.txt",
    "prompts/verification_comparison_ja.txt",
    "prompts/verification_extraction_en.txt",
    "prompts/verification_extraction_ja.txt",
    "schemas/contrastive_generation_run.schema.json",
    "schemas/contrastive_generation_result.schema.json",
    "schemas/contrastive_generation_state.schema.json",
    "schemas/contrastive_run_lease.schema.json",
    "schemas/contrastive_verification_comparison.schema.json",
    "schemas/contrastive_verification_comparison.ja.schema.json",
    "schemas/contrastive_verification_extraction.schema.json",
    "schemas/contrastive_verification_extraction.ja.schema.json",
    "schemas/frame_calibration_report.schema.json",
    "schemas/cue_balanced_development_augmentation.schema.json",
    "schemas/cue_invariant_confirmatory_controls.schema.json",
    "schemas/format_matched_controls.schema.json",
    "schemas/heldout_comprehension_report.schema.json",
    "schemas/model_request.schema.json",
    "schemas/container.schema.json",
    "schemas/containers.schema.json",
    "schemas/lexical_controls.schema.json",
    "schemas/naturalistic_transfer.schema.json",
    "schemas/purpose_frames.schema.json",
    "schemas/pre_generation_approval.schema.json",
    "schemas/pre_generation_job.schema.json",
    "schemas/pre_generation_lock.schema.json",
    "schemas/pre_generation_plan.schema.json",
    "schemas/pre_generation_report.schema.json",
    "schemas/pre_generation_run.schema.json",
    "schemas/provider.schema.json",
    "schemas/topics.schema.json",
    "validation/lexical_controls.yaml",
    "validation/cue_balanced_development_augmentation.yaml",
    "validation/cue_invariant_confirmatory_controls.yaml",
    "validation/format_matched_controls.yaml",
    "validation/naturalistic_transfer.yaml",
    "src/contrastive_generation/catalog.py",
    "src/contrastive_generation/__init__.py",
    "src/contrastive_generation/__main__.py",
    "src/contrastive_generation/allocation.py",
    "src/contrastive_generation/calibration.py",
    "src/contrastive_generation/content_specs.py",
    "src/contrastive_generation/generation_cost.py",
    "src/contrastive_generation/lease.py",
    "src/contrastive_generation/planning.py",
    "src/contrastive_generation/post_generation.py",
    "src/contrastive_generation/pre_generation.py",
    "src/contrastive_generation/preflight.py",
    "src/contrastive_generation/providers.py",
    "src/contrastive_generation/publication.py",
    "src/contrastive_generation/rendering.py",
    "src/contrastive_generation/runner.py",
    "src/contrastive_generation/utils.py",
    "src/contrastive_generation/verification.py",
}

# Two cue screens over one frozen lexicon: the broad request screen, and the
# narrower screen applied to generated task text. See the lexicon file for why
# they differ.
CUE_SCOPES = ("languages", "generated_content")

# Regular English inflection accepted after a frozen cue stem, so that a screen
# built from stems catches "tests"/"deployed"/"evaluations" the way the Japanese
# substring screen already catches every form of 評価.
EN_CUE_INFLECTION = r"(?:s|es|ed|ing|d)?"

SEMANTIC_BUNDLE_FILES = {
    "prompts/content_generation_en.txt",
    "prompts/content_generation_ja.txt",
    "prompts/content_scaffolding.yaml",
}

# Gates whose evidence a run may proceed without, provided the run config records
# who waived it and why. A waiver only removes the production block: every hash
# binding, schema check, and provenance comparison still runs, and a waived gate
# is reported as "waived" rather than "passed" so no downstream artifact can be
# mistaken for one backed by evidence.
WAIVABLE_GATES = frozenset(
    {
        "semantic_approval",
        "frame_only",
        "heldout_comprehension",
    }
)

# Any locked implementation/runtime input could alter parsing or model-visible
# requests indirectly. Derive this set from the global lock, excluding only the
# separately hashed prose bundle and the approval record itself (which would
# create a circular hash).
REQUEST_PROJECTION_FILES = (
    LOCKED_FILES
    - SEMANTIC_BUNDLE_FILES
    - {"prompts/pre_generation_semantic_approval.json"}
)

# JSON-Schema keywords whose string values are protocol vocabulary rather than
# model-facing semantic text. References are separately restricted to resolvable
# local JSON Pointers.
SCHEMA_PROTOCOL_ONLY_KEYS = {
    "$anchor",
    "$dynamicAnchor",
    "$id",
    "$recursiveAnchor",
    "$schema",
    "$vocabulary",
    "contentEncoding",
    "contentMediaType",
    "format",
    "type",
}
SCHEMA_REFERENCE_KEYS = {"$dynamicRef", "$recursiveRef", "$ref"}
SCHEMA_NAME_MAP_KEYS = {
    "$defs",
    "definitions",
    "dependencies",
    "dependentRequired",
    "dependentSchemas",
    "patternProperties",
    "properties",
}
SCHEMA_INSTANCE_VALUE_KEYS = {"const", "default", "enum", "examples"}
SCHEMA_ALLOWED_KEYS = (
    SCHEMA_PROTOCOL_ONLY_KEYS
    | SCHEMA_REFERENCE_KEYS
    | SCHEMA_NAME_MAP_KEYS
    | SCHEMA_INSTANCE_VALUE_KEYS
    | {
        "$comment",
        "additionalItems",
        "additionalProperties",
        "allOf",
        "anyOf",
        "contains",
        "contentSchema",
        "deprecated",
        "description",
        "else",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "if",
        "items",
        "maxContains",
        "maximum",
        "maxItems",
        "maxLength",
        "maxProperties",
        "minContains",
        "minimum",
        "minItems",
        "minLength",
        "minProperties",
        "multipleOf",
        "not",
        "oneOf",
        "pattern",
        "prefixItems",
        "propertyNames",
        "readOnly",
        "required",
        "then",
        "title",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
        "writeOnly",
    }
)


@dataclass(frozen=True)
class PreflightBundle:
    scaffolding: dict[str, Any]
    cue_lexicon: dict[str, Any]
    approval: dict[str, Any]
    lock: dict[str, Any]
    audit: dict[str, Any]
    notifications: tuple[dict[str, str], ...]


def _validate_json(
    schema_name: str,
    value: dict[str, Any],
    label: str,
) -> None:
    schema = load_json(CONTRASTIVE_ROOT / "schemas" / schema_name)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors[:12]
        )
        raise ContrastiveError(f"Invalid {label}: {details}")


def semantic_bundle_sha256() -> str:
    """Hash the exact bilingual prose bundle to which humans attest."""
    return hash_object({
        relative: sha256_file(CONTRASTIVE_ROOT / relative)
        for relative in sorted(SEMANTIC_BUNDLE_FILES)
    })


def request_projection_sha256() -> str:
    """Hash all implementation modules that can project model-visible meaning."""
    return hash_object({
        relative: sha256_file(CONTRASTIVE_ROOT / relative)
        for relative in sorted(REQUEST_PROJECTION_FILES)
    })


def _validate_scaffolding(value: dict[str, Any], path: Path) -> None:
    if value.get("schema_version") != 1:
        raise ContrastiveError(f"Unsupported content scaffolding schema in {path}")
    languages = value.get("languages")
    if not isinstance(languages, dict) or set(languages) != {"en", "ja"}:
        raise ContrastiveError(
            f"Content scaffolding {path} must contain exactly en and ja views"
        )
    for language, view in languages.items():
        if not isinstance(view, dict):
            raise ContrastiveError(
                f"Content scaffolding {language} view must be a mapping"
            )
        for field in (
            "slot_descriptions",
            "option_contracts",
            "answer_key_contracts",
        ):
            mapping = view.get(field)
            if not isinstance(mapping, dict) or not mapping:
                raise ContrastiveError(
                    f"Content scaffolding {language}.{field} must be a nonempty mapping"
                )
            if any(not isinstance(item, str) or not item for item in mapping.values()):
                raise ContrastiveError(
                    f"Content scaffolding {language}.{field} contains empty prose"
                )


def _validate_lexicon(value: dict[str, Any], path: Path) -> None:
    if value.get("schema_version") != 1:
        raise ContrastiveError(f"Unsupported cue lexicon schema in {path}")
    for scope in CUE_SCOPES:
        languages = value.get(scope)
        if not isinstance(languages, dict) or set(languages) != {"en", "ja"}:
            raise ContrastiveError(
                f"Cue lexicon {path} must contain exactly en and ja term lists "
                f"under {scope!r}"
            )
        for language, terms in languages.items():
            if (
                not isinstance(terms, list)
                or not terms
                or any(not isinstance(term, str) or not term for term in terms)
                or len(terms) != len(set(terms))
            ):
                raise ContrastiveError(
                    f"Cue lexicon {path} has an invalid {scope}/{language} term list"
                )
    # The generated-content screen narrows the request screen; it may never admit
    # a term the request screen rejects, or the two scopes would disagree about
    # what counts as a cue.
    for language in ("en", "ja"):
        extra = set(value["generated_content"][language]) - set(
            value["languages"][language]
        )
        if extra:
            raise ContrastiveError(
                f"Cue lexicon {path} generated_content/{language} contains terms "
                f"absent from the request screen: {sorted(extra)}"
            )


def _prompt_placeholders(path: Path) -> set[str]:
    raw = path.read_text(encoding="utf-8")
    if raw.count("\nUSER\n") != 1 or not raw.startswith("SYSTEM\n"):
        raise ContrastiveError(
            f"Prompt must contain exactly one SYSTEM and one USER section: {path}"
        )
    placeholders = set(PLACEHOLDER_RE.findall(raw))
    if placeholders != REQUIRED_PLACEHOLDERS:
        raise ContrastiveError(
            f"Prompt {path} placeholder set differs from the approved contract: "
            f"missing={sorted(REQUIRED_PLACEHOLDERS - placeholders)}, "
            f"extra={sorted(placeholders - REQUIRED_PLACEHOLDERS)}"
        )
    stripped = PLACEHOLDER_RE.sub("", raw)
    if "<<" in stripped or ">>" in stripped:
        raise ContrastiveError(f"Prompt {path} contains a malformed placeholder")
    return placeholders


def gate_waiver(
    config: dict[str, Any],
    gate: str,
) -> dict[str, Any] | None:
    """Return the recorded waiver for a gate, or None when it is not waived."""
    if gate not in WAIVABLE_GATES:
        raise ContrastiveError(f"Gate {gate!r} is not waivable")
    waiver = config.get("waived_gates", {}).get(gate)
    if waiver is None:
        return None
    try:
        waived_at = datetime.fromisoformat(
            str(waiver["waived_at"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ContrastiveError(
            f"Waiver for {gate} has an invalid ISO-8601 waived_at timestamp"
        ) from exc
    if waived_at.tzinfo is None:
        raise ContrastiveError(
            f"Waiver for {gate} must record waived_at with a UTC offset"
        )
    return {
        "waived_by": str(waiver["waived_by"]),
        "waived_at": str(waiver["waived_at"]),
        "rationale": str(waiver["rationale"]),
    }


def validate_scaffolding_bundle(config: dict[str, Any]) -> PreflightBundle:
    """Validate bilingual parity, semantic review, and the exact hash lock."""
    paths = config["paths"]
    prompt_paths = {
        language: resolve_local_path(paths["prompts"][language])
        for language in ("en", "ja")
    }
    placeholder_sets = {
        language: _prompt_placeholders(path)
        for language, path in prompt_paths.items()
    }
    if placeholder_sets["en"] != placeholder_sets["ja"]:
        raise ContrastiveError(
            "English and Japanese content-generation prompts expose different "
            "placeholder sets"
        )

    scaffolding_path = resolve_local_path(paths["scaffolding"])
    lexicon_path = resolve_local_path(paths["cue_lexicon"])
    approval_path = resolve_local_path(paths["semantic_approval"])
    lock_path = resolve_local_path(paths["scaffolding_lock"])
    active_paths = {
        "prompts/content_generation_en.txt": prompt_paths["en"],
        "prompts/content_generation_ja.txt": prompt_paths["ja"],
        "prompts/content_scaffolding.yaml": scaffolding_path,
        "prompts/generation_cue_lexicon.yaml": lexicon_path,
        "prompts/pre_generation_semantic_approval.json": approval_path,
        "prompts/pre_generation.lock.json": lock_path,
    }
    for expected, actual in active_paths.items():
        if local_reference(actual) != expected:
            raise ContrastiveError(
                f"Configuration attempts to replace locked pre-generation input "
                f"{expected} with {local_reference(actual)}"
            )
    scaffolding = load_yaml(scaffolding_path)
    cue_lexicon = load_yaml(lexicon_path)
    approval = load_json(approval_path)
    lock = load_json(lock_path)
    _validate_scaffolding(scaffolding, scaffolding_path)
    _validate_lexicon(cue_lexicon, lexicon_path)
    _validate_json(
        "pre_generation_approval.schema.json",
        approval,
        f"semantic approval {approval_path}",
    )
    _validate_json(
        "pre_generation_lock.schema.json",
        lock,
        f"scaffolding lock {lock_path}",
    )

    locked = set(lock["files"])
    if locked != LOCKED_FILES:
        raise ContrastiveError(
            "Pre-generation lock file list differs from the active bundle: "
            f"missing={sorted(LOCKED_FILES - locked)}, "
            f"extra={sorted(locked - LOCKED_FILES)}"
        )
    for relative, expected in sorted(lock["files"].items()):
        path = resolve_local_path(relative)
        if not path.is_file():
            raise ContrastiveError(f"Locked pre-generation file is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ContrastiveError(
                f"Locked pre-generation file changed: {relative}; "
                f"expected {expected}, got {actual}"
            )

    bundle_hash = semantic_bundle_sha256()
    if approval["scaffolding_sha256"] != bundle_hash:
        raise ContrastiveError(
            "Semantic-equivalence approval targets a different bilingual scaffold "
            f"bundle: expected {bundle_hash}"
        )
    projection_hash = request_projection_sha256()
    if approval["request_projection_sha256"] != projection_hash:
        raise ContrastiveError(
            "Semantic-equivalence approval targets different model-visible "
            f"request-projection code: expected {projection_hash}"
        )
    if lock["semantic_equivalence_review"] != approval["status"]:
        raise ContrastiveError(
            "Scaffolding lock and semantic approval disagree on review status"
        )
    if approval["status"] == "approved":
        try:
            approved_at = datetime.fromisoformat(
                str(approval["approved_at"]).replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ContrastiveError(
                "Approved semantic review has an invalid ISO-8601 timestamp"
            ) from exc
        if approved_at.tzinfo is None:
            raise ContrastiveError(
                "Approved semantic review timestamp must include a UTC offset"
            )
    notifications: list[dict[str, str]] = []
    waiver = gate_waiver(config, "semantic_approval")
    if approval["status"] != "approved":
        if config["mode"] == "production" and waiver is None:
            raise ContrastiveError(
                "Production planning requires approved bilingual semantic equivalence"
            )
        if waiver is None:
            notifications.append({
                "level": "warning",
                "code": "semantic_equivalence_review_pending",
                "message": (
                    "Mechanical bilingual checks passed, but researcher/native-speaker "
                    "semantic approval remains pending; production planning is disabled."
                ),
            })
        else:
            notifications.append({
                "level": "warning",
                "code": "semantic_equivalence_review_waived",
                "message": (
                    "Researcher/native-speaker semantic approval is "
                    f"{approval['status']} and was waived by "
                    f"{waiver['waived_by']} on {waiver['waived_at']}. Bilingual "
                    "semantic equivalence is unverified for this run and every "
                    "artifact derived from it."
                ),
            })

    audit = {
        "placeholder_parity": "passed",
        "placeholder_set": sorted(REQUIRED_PLACEHOLDERS),
        "scaffolding_hash_lock": "passed",
        "semantic_bundle_sha256": bundle_hash,
        "request_projection_sha256": projection_hash,
        "semantic_equivalence_review": approval["status"],
        "lock_version": lock["lock_version"],
        "lock_sha256": sha256_file(lock_path),
        "locked_files": len(lock["files"]),
    }
    if waiver is not None and approval["status"] != "approved":
        audit["semantic_equivalence_review_waiver"] = waiver
    return PreflightBundle(
        scaffolding=scaffolding,
        cue_lexicon=cue_lexicon,
        approval=approval,
        lock=lock,
        audit=audit,
        notifications=tuple(notifications),
    )


def _answer_contract(
    spec: ModelContentSpec,
    scaffolding_view: dict[str, Any],
) -> str:
    mode = str(spec.response_contract["answer_key"])
    try:
        base = str(scaffolding_view["answer_key_contracts"][mode])
    except KeyError as exc:
        raise ContrastiveError(
            f"No localized answer-key instruction for contract {mode!r}"
        ) from exc
    canonical_mode = ANSWER_KEY_CANONICAL.get(mode, mode)
    if canonical_mode != "required":
        return base
    allowed = list(spec.response_contract["allowed_keys"])
    separator = "、" if spec.language == "ja" else ", "
    prefix = "許可された記号：" if spec.language == "ja" else "Permitted symbols: "
    return f"{base} {prefix}{separator.join(map(str, allowed))}。"


def render_model_request(
    prompt_path: Path,
    topic: ModelTopic,
    spec: ModelContentSpec,
    scaffolding: dict[str, Any],
    generation: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Render an exact request without accepting a container or fixed template."""
    if type(topic) is not ModelTopic or type(spec) is not ModelContentSpec:
        raise ContrastiveError(
            "Request construction accepts only exact ModelTopic and "
            "ModelContentSpec projections"
        )
    system, user_template = load_prompt_template(prompt_path)
    try:
        view = scaffolding["languages"][spec.language]
        option_contract = view["option_contracts"][
            spec.response_contract["type"]
        ]
    except KeyError as exc:
        raise ContrastiveError(
            f"Scaffolding lacks a {spec.language} response instruction for "
            f"{spec.response_contract['type']!r}"
        ) from exc
    if spec.language == "ja":
        descriptions = "\n".join(
            f"・{name}：{description}"
            for name, description in spec.slot_descriptions
        )
        slots = "、".join(spec.slots)
    else:
        descriptions = "\n".join(
            f"- {name}: {description}"
            for name, description in spec.slot_descriptions
        )
        slots = ", ".join(spec.slots)
    replacements = {
        "TOPIC_LABEL": topic.topic_display_name,
        "DOMAIN": topic.domain_display_name,
        "REGISTER": spec.register,
        "ROLE_STRUCTURE": spec.role_structure,
        "MODEL_GUIDANCE": spec.model_guidance,
        "SLOT_DESCRIPTIONS": descriptions,
        "OPTION_CONTRACT": str(option_contract),
        "ANSWER_KEY_CONTRACT": _answer_contract(spec, view),
        "SLOTS": slots,
    }
    user = user_template
    for name, value in replacements.items():
        user = user.replace(f"<<{name}>>", value)
    unresolved = PLACEHOLDER_RE.findall(user)
    if unresolved or "<<" in user or ">>" in user:
        raise ContrastiveError(
            f"Rendered request contains unresolved placeholders: {unresolved}"
        )
    request = {
        "system": system,
        "user": user,
        "response_schema": response_schema_for_spec(spec),
        "generation": {
            "temperature": generation["temperature"],
            "top_p": generation["top_p"],
            "max_output_tokens": generation["max_output_tokens"],
            "seed": seed,
        },
    }
    _validate_json("model_request.schema.json", request, "model request")
    return request


def model_visible_strings(request: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Extract every model-visible semantic string from the request schema."""
    result: list[tuple[str, str]] = [
        ("system", str(request["system"])),
        ("user", str(request["user"])),
    ]
    root_schema = request["response_schema"]
    visited: set[int] = set()

    def pointer_token(value: str) -> str:
        return value.replace("~", "~0").replace("/", "~1")

    def validate_local_reference(reference: str, pointer: str) -> None:
        if not reference.startswith("#"):
            raise ContrastiveError(
                f"External JSON-Schema reference is forbidden at {pointer}: "
                f"{reference!r}"
            )
        fragment = unquote(reference[1:])
        if not fragment:
            return
        if not fragment.startswith("/"):
            raise ContrastiveError(
                f"JSON-Schema anchor references are unsupported at {pointer}: "
                f"{reference!r}"
            )
        target: Any = root_schema
        for encoded in fragment[1:].split("/"):
            if re.search(r"~(?![01])", encoded):
                raise ContrastiveError(
                    f"Invalid JSON Pointer escape in reference at {pointer}: "
                    f"{reference!r}"
                )
            token = encoded.replace("~1", "/").replace("~0", "~")
            try:
                if isinstance(target, dict):
                    target = target[token]
                elif isinstance(target, list):
                    if not re.fullmatch(r"0|[1-9][0-9]*", token):
                        raise KeyError(token)
                    target = target[int(token)]
                else:
                    raise KeyError(token)
            except (KeyError, IndexError, ValueError) as exc:
                raise ContrastiveError(
                    f"Unresolvable local JSON-Schema reference at {pointer}: "
                    f"{reference!r}"
                ) from exc
        if not isinstance(target, (dict, bool)):
            raise ContrastiveError(
                f"Local JSON-Schema reference does not target a schema at "
                f"{pointer}: {reference!r}"
            )

    def collect_instance_value(value: Any, pointer: str) -> None:
        if isinstance(value, str):
            result.append((pointer, value))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect_instance_value(child, f"{pointer}/{index}")
        elif isinstance(value, dict):
            for name, child in value.items():
                escaped = pointer_token(str(name))
                result.append((f"{pointer}/{escaped}", str(name)))
                collect_instance_value(child, f"{pointer}/{escaped}")

    def visit_schema(value: Any, pointer: str) -> None:
        if isinstance(value, str):
            result.append((pointer, value))
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit_schema(child, f"{pointer}/{index}")
            return
        if isinstance(value, bool):
            return
        if not isinstance(value, dict):
            return
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        for key, child in value.items():
            escaped_key = pointer_token(str(key))
            child_pointer = f"{pointer}/{escaped_key}"
            if key not in SCHEMA_ALLOWED_KEYS:
                raise ContrastiveError(
                    f"Unsupported JSON-Schema keyword at {child_pointer}; "
                    "the semantic audit fails closed on unknown extensions"
                )
            if key in SCHEMA_PROTOCOL_ONLY_KEYS:
                continue
            if key in SCHEMA_REFERENCE_KEYS:
                if not isinstance(child, str):
                    raise ContrastiveError(
                        f"JSON-Schema reference must be a string at {child_pointer}"
                    )
                validate_local_reference(child, child_pointer)
                continue
            if key in SCHEMA_INSTANCE_VALUE_KEYS:
                collect_instance_value(child, child_pointer)
                continue
            if key in SCHEMA_NAME_MAP_KEYS:
                if not isinstance(child, dict):
                    raise ContrastiveError(
                        f"JSON-Schema {key} must be an object at {child_pointer}"
                    )
                for name, subschema in child.items():
                    escaped_name = pointer_token(str(name))
                    name_pointer = f"{child_pointer}/{escaped_name}"
                    result.append((name_pointer, str(name)))
                    if key in {"dependentRequired"} or (
                        key == "dependencies" and isinstance(subschema, list)
                    ):
                        collect_instance_value(subschema, name_pointer)
                    else:
                        visit_schema(subschema, name_pointer)
                continue
            visit_schema(child, child_pointer)

    visit_schema(root_schema, "/response_schema")
    return tuple(result)


def cue_hits(
    request: dict[str, Any],
    language: str,
    cue_lexicon: dict[str, Any],
) -> list[dict[str, str]]:
    terms = cue_lexicon["languages"][language]
    hits: list[dict[str, str]] = []
    for pointer, raw in model_visible_strings(request):
        text = unicodedata.normalize("NFKC", raw)
        if language == "en":
            folded = text.casefold()
            for term in terms:
                pattern = re.compile(
                    rf"(?<![a-z]){re.escape(str(term).casefold())}(?![a-z])"
                )
                if pattern.search(folded):
                    hits.append({"path": pointer, "term": str(term)})
        else:
            for term in terms:
                if unicodedata.normalize("NFKC", str(term)) in text:
                    hits.append({"path": pointer, "term": str(term)})
    return hits


def text_cue_terms(
    text: str,
    language: str,
    cue_lexicon: dict[str, Any],
    scope: str = "languages",
) -> list[str]:
    """Return every frozen cue occurrence in one renderer-owned text.

    `scope` selects the broad request screen (default) or the narrower
    `generated_content` screen applied to model-authored task text.
    """
    if scope not in CUE_SCOPES:
        raise ContrastiveError(f"Unknown cue screen scope {scope!r}")
    normalized = unicodedata.normalize("NFKC", text)
    terms = cue_lexicon[scope][language]
    hits: list[str] = []
    if language == "en":
        folded = normalized.casefold()
        for term in terms:
            # English inflects; Japanese does not. Matching the bare stem only
            # let regular forms ("evaluations", "deployed", "tests") through the
            # screen, so the two languages were filtered at different strengths.
            # Irregular forms that drop a stem vowel ("evaluating") stay
            # enumerated in the lexicon.
            pattern = re.compile(
                rf"(?<![a-z]){re.escape(str(term).casefold())}"
                rf"{EN_CUE_INFLECTION}(?![a-z])"
            )
            hits.extend(str(term) for _ in pattern.finditer(folded))
    else:
        for term in terms:
            normalized_term = unicodedata.normalize("NFKC", str(term))
            hits.extend(
                str(term)
                for _ in range(normalized.count(normalized_term))
            )
    return hits


def japanese_language_hits(request: dict[str, Any]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for pointer, raw in model_visible_strings(request):
        normalized = unicodedata.normalize("NFKC", raw)
        match = LATIN_RE.search(normalized)
        if match:
            hits.append({
                "path": pointer,
                "value": match.group(0),
                "reason": "Latin letter",
            })
        digit = FULLWIDTH_DIGIT_RE.search(raw)
        if digit:
            hits.append({
                "path": pointer,
                "value": digit.group(0),
                "reason": "full-width digit",
            })
    return hits


def leakage_hits(
    request: dict[str, Any],
    spec: ContentSpecRecord,
    topic: TopicRecord | None = None,
) -> list[dict[str, str]]:
    semantic_text = "\n".join(value for _, value in model_visible_strings(request))
    folded = semantic_text.casefold()
    private_values: list[tuple[str, str]] = [
        *(("pair_id", pair_id) for pair_id in spec.pair_ids),
        ("content_spec_id", spec.content_spec_id),
    ]
    if topic is not None:
        private_values.append(("topic_id", topic.topic_id))
        public_topic_text = "\n".join(
            (
                topic.topic_display_name,
                topic.domain_display_name,
            )
        ).casefold()
        for field, value in (
            ("topic_label", topic.topic_label),
            ("domain", topic.domain),
        ):
            # Machine labels can be ordinary content words (for example,
            # "proofreading"). They are not leaks when that same text is an
            # intentional part of the allowlisted localized topic projection.
            if value.casefold() not in public_topic_text:
                private_values.append((field, value))
        if topic.generation_guidance:
            private_values.append(
                ("generation_guidance", topic.generation_guidance)
            )
    for container in spec.containers:
        private_values.extend([
            ("container_id", container.container_id),
            ("class", str(container.data["class"])),
            ("display_name", str(container.data["display_name"])),
            ("design_notes", str(container.data["design_notes"])),
            ("body_template", str(container.data["body_template"])),
            (
                "purpose_frame.text",
                str(container.data["purpose_frame"]["text"]),
            ),
        ])
    hits: list[dict[str, str]] = []
    for field, value in private_values:
        if not value:
            continue
        if field == "class":
            found = re.search(
                rf"(?<![a-z]){re.escape(value.casefold())}(?![a-z])",
                folded,
            )
        else:
            found = value.casefold() in folded
        if found:
            hits.append({"field": field, "value": value})
    return hits


def estimate_input_tokens(request: dict[str, Any], language: str) -> int:
    semantic = "\n".join(value for _, value in model_visible_strings(request))
    schema_size = len(canonical_json(request["response_schema"]))
    divisor = 1.8 if language == "ja" else 4.0
    return max(1, math.ceil((len(semantic) + schema_size) / divisor))


def context_input_token_upper_bound(
    request: dict[str, Any],
    overhead_tokens: int,
) -> int:
    """Conservative byte bound plus declared provider/chat framing overhead."""
    return len(canonical_json(request).encode("utf-8")) + int(overhead_tokens)


def audit_model_request(
    request: dict[str, Any],
    spec: ContentSpecRecord,
    cue_lexicon: dict[str, Any],
    topic: TopicRecord | None = None,
    context_overhead_tokens: int = 0,
) -> dict[str, Any]:
    """Fail closed on the exact request artifact."""
    cues = cue_hits(request, spec.language, cue_lexicon)
    if cues:
        raise ContrastiveError(
            f"Model request for {spec.content_spec_id}/{spec.language} contains "
            f"frozen generation cues: {cues[:8]}"
        )
    language_hits = (
        japanese_language_hits(request) if spec.language == "ja" else []
    )
    if language_hits:
        raise ContrastiveError(
            f"Japanese model request for {spec.content_spec_id} contains "
            f"non-Japanese surface text: {language_hits[:8]}"
        )
    leaks = leakage_hits(request, spec, topic)
    if leaks:
        raise ContrastiveError(
            f"Model request for {spec.content_spec_id}/{spec.language} leaks "
            f"renderer/private material: {leaks[:8]}"
        )
    return {
        "placeholder_audit": "passed",
        "cue_audit": "passed",
        "language_audit": "passed",
        "leakage_audit": "passed",
        "estimated_input_tokens": estimate_input_tokens(
            request, spec.language
        ),
        "context_input_token_upper_bound": context_input_token_upper_bound(
            request, context_overhead_tokens
        ),
    }


def validate_registered_corpus(
    english_topics: Iterable[TopicRecord],
    english_specs: tuple[ContentSpecRecord, ...],
) -> dict[str, Any]:
    topics = tuple(english_topics)
    if len(topics) != 40:
        raise ContrastiveError(
            f"Registered contrastive corpus requires 40 topics, got {len(topics)}"
        )
    domain_counts: dict[str, int] = {}
    for topic in topics:
        domain_counts[topic.domain] = domain_counts.get(topic.domain, 0) + 1
    if len(domain_counts) != 8 or set(domain_counts.values()) != {5}:
        raise ContrastiveError(
            "Registered contrastive corpus requires eight domains with five topics "
            f"each, got {domain_counts}"
        )
    if len(english_specs) != 10:
        raise ContrastiveError(
            f"Registered contrastive corpus requires 10 content specs, "
            f"got {len(english_specs)}"
        )
    container_ids = [
        container.container_id
        for spec in english_specs
        for container in spec.containers
    ]
    if sorted(container_ids) != [f"c{number:02d}" for number in range(1, 41)]:
        raise ContrastiveError(
            "Registered contrastive corpus container identifiers differ from c01-c40"
        )
    return {
        "status": "passed",
        "topics": 40,
        "domains": 8,
        "topics_per_domain": 5,
        "content_specs": 10,
        "containers": 40,
    }


def validate_renderer_cue_contract(
    english_specs: tuple[ContentSpecRecord, ...],
    japanese_specs: tuple[ContentSpecRecord, ...],
    cue_lexicon: dict[str, Any],
) -> dict[str, Any]:
    """Keep the pair-shared body free of purpose-attribution cue terms."""
    containers = {
        "en": {
            container.container_id: container
            for spec in english_specs
            for container in spec.containers
        },
        "ja": {
            container.container_id: container
            for spec in japanese_specs
            for container in spec.containers
        },
    }
    checked = 0
    for language in ("en", "ja"):
        for container_id, container in sorted(containers[language].items()):
            hits = text_cue_terms(
                str(container.data["body_template"]), language, cue_lexicon
            )
            checked += 1
            if hits:
                raise ContrastiveError(
                    f"Matched container body {container_id}/{language} contains "
                    f"purpose-attribution cues outside the frame: {hits}"
                )
    return {
        "status": "passed",
        "matched_bodies_checked": checked,
        "body_cue_hits": 0,
    }
