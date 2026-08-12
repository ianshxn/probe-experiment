"""Deterministic v2 pre-generation planning and integrity verification."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import threading
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from jsonschema import Draft202012Validator

from .catalog import ContrastiveCatalog, TopicRecord, load_catalog
from .content_specs import (
    ContentSpecRecord,
    catalog_semantic_review_payload,
    catalog_semantic_review_sha256,
    compile_content_specs,
    model_topic,
    validate_bilingual_content_specs,
)
from .planning import load_provider_profile, stable_seed
from .preflight import (
    LOCKED_FILES,
    WAIVABLE_GATES,
    PreflightBundle,
    audit_model_request,
    gate_waiver,
    render_model_request,
    validate_registered_corpus,
    validate_renderer_cue_contract,
    validate_scaffolding_bundle,
)
from .rendering import prerender_template
from .utils import (
    CONTRASTIVE_ROOT,
    ContrastiveError,
    atomic_write_json,
    atomic_write_jsonl,
    hash_object,
    load_json,
    load_yaml,
    local_reference,
    resolve_local_path,
    sha256_bytes,
    sha256_file,
    sha256_text,
    utc_now,
)


PRE_GENERATION_PATH = Path(__file__).resolve()
PREFLIGHT_PATH = PRE_GENERATION_PATH.with_name("preflight.py")
CONTENT_SPECS_PATH = PRE_GENERATION_PATH.with_name("content_specs.py")
RENDERER_PATH = PRE_GENERATION_PATH.with_name("rendering.py")
JOB_SCHEMA_PATH = CONTRASTIVE_ROOT / "schemas" / "pre_generation_job.schema.json"
PLAN_SCHEMA_PATH = CONTRASTIVE_ROOT / "schemas" / "pre_generation_plan.schema.json"
REPORT_SCHEMA_PATH = (
    CONTRASTIVE_ROOT / "schemas" / "pre_generation_report.schema.json"
)
REQUEST_SCHEMA_PATH = CONTRASTIVE_ROOT / "schemas" / "model_request.schema.json"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
PLACEHOLDER_VALUE_RE = re.compile(r"<[^<>\r\n]+>")
ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x1f\x7f]")
ROUTE_RE = re.compile(
    r"^/?[A-Za-z0-9_~.-]+(?:/[A-Za-z0-9_~.-]+)*$"
)
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(?:api[_-]?key|x[_-]?api[_-]?key|key|access[_-]?token|"
    r"token|client[_-]?secret|secret|password|credential|authorization|"
    r"auth|signature|sig)\s*[=:]\s*[^&\s]+"
)
BEARER_SECRET_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-=]{8,}")
KNOWN_SECRET_PREFIX_RE = re.compile(
    r"(?i)(?:\bsk-(?:or-v1-)?[A-Za-z0-9_-]{8,}|"
    r"\bnvapi-[A-Za-z0-9_-]{8,}|"
    r"\bAIza[A-Za-z0-9_-]{8,})"
)
MUTABLE_REVISION_RE = re.compile(
    r"(?i)(?:^|[-_:/.])(latest|stable|preview|current|default|auto)"
    r"(?:$|[-_:/.])"
)
SENSITIVE_URL_QUERY_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "token",
    "secret",
    "password",
    "credential",
    "credentials",
    "authorization",
    "auth",
    "key",
    "client_secret",
    "x_api_key",
    "signature",
    "sig",
}
SENSITIVE_PROFILE_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "token",
    "secret",
    "password",
    "credential",
    "credentials",
}
ALLOWED_PRE_GENERATION_PROVIDER_KEYS = {
    "profile_id",
    "provider",
    "model",
    "model_revision",
    "structured_output_mode",
    "schema_dialect",
    "auth",
    "api_key_env",
    "credential_env",
    "project",
    "location",
    "base_url",
    "route",
    "timeout_seconds",
    "api_attempts",
    "max_concurrency",
    "circuit_breaker_failures",
    "circuit_breaker_cooldown_seconds",
    "thinking_level",
    "reasoning_effort",
    "input_usd_per_million",
    "output_usd_per_million",
    "capabilities",
}


@dataclass(frozen=True)
class PreGenerationArtifacts:
    resolved_config: dict[str, Any]
    plan: dict[str, Any]
    jobs: tuple[dict[str, Any], ...]
    requests: dict[str, dict[str, Any]]
    report: dict[str, Any]
    notifications: tuple[dict[str, str], ...]
    approvals: dict[str, Any]
    compiled_specs: dict[str, Any]


class VerifiedGenerationHandoff:
    """Verified boundary for exact requests plus a frozen adapter control plane."""

    def __init__(
        self,
        run_dir: Path,
        plan_sha256: str,
        plan_file_sha256: str,
        resolved_config_sha256: str,
        manifest_sha256: str,
        resolved_config_file_sha256: str,
        manifest_file_sha256: str,
        adapter_profile: dict[str, Any],
        provider_profile_sha256: str,
        provider_execution_sha256: str,
        jobs: list[dict[str, Any]],
    ) -> None:
        self._run_dir = run_dir
        self._plan_sha256 = plan_sha256
        self._plan_file_sha256 = plan_file_sha256
        self._resolved_config_sha256 = resolved_config_sha256
        self._manifest_sha256 = manifest_sha256
        self._resolved_config_file_sha256 = resolved_config_file_sha256
        self._manifest_file_sha256 = manifest_file_sha256
        self._adapter_profile = copy.deepcopy(adapter_profile)
        self._provider_profile_sha256 = provider_profile_sha256
        self._provider_execution_sha256 = provider_execution_sha256
        self._jobs = {str(job["job_id"]): copy.deepcopy(job) for job in jobs}
        self._released: set[tuple[str, int]] = set()
        self._next_attempt = {
            str(job["job_id"]): 0 for job in jobs
        }
        self._release_lock = threading.Lock()

    @property
    def job_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._jobs))

    def job_metadata(self, job_id: str) -> dict[str, Any]:
        try:
            return copy.deepcopy(self._jobs[job_id])
        except KeyError as exc:
            raise ContrastiveError(
                f"Unknown generation job in verified handoff: {job_id}"
            ) from exc

    def _load_verified_attempt(
        self,
        job_id: str,
        attempt_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Rehash one frozen attempt immediately before adapter release."""
        job = self.job_metadata(job_id)
        existing_outputs = [
            path
            for path in _job_release_collision_paths(job, attempt_index)
            if _path_entry_exists(path)
        ]
        if existing_outputs:
            raise ContrastiveError(
                f"Generation output already exists for {job_id}; explicit resume "
                f"is not implemented: {existing_outputs}"
            )
        plan, plan_file_sha256 = _load_json_snapshot(
            self._run_dir / "plan.json"
        )
        if (
            plan_file_sha256 != self._plan_file_sha256
            or
            plan.get("plan_sha256") != self._plan_sha256
            or _plan_hash(plan) != self._plan_sha256
            or plan.get("resolved_config_sha256")
            != self._resolved_config_sha256
            or plan.get("manifest_sha256") != self._manifest_sha256
        ):
            raise ContrastiveError(
                "Immutable plan changed after handoff verification"
            )
        if (
            sha256_file(self._run_dir / "resolved_config.json")
            != self._resolved_config_file_sha256
        ):
            raise ContrastiveError(
                "Resolved configuration changed after handoff verification"
            )
        if (
            sha256_file(self._run_dir / "manifest.jsonl")
            != self._manifest_file_sha256
        ):
            raise ContrastiveError(
                "Generation manifest changed after handoff verification"
            )
        attempts = {
            int(row["attempt_index"]): row
            for row in job["generation_attempts"]
        }
        if attempt_index not in attempts:
            raise ContrastiveError(
                f"Unknown preplanned generation attempt for {job_id}: "
                f"{attempt_index}"
            )
        attempt = copy.deepcopy(attempts[attempt_index])
        request_path = resolve_local_path(attempt["request_path"])
        try:
            request_path.relative_to(
                (self._run_dir / "requests").resolve()
            )
        except ValueError as exc:
            raise ContrastiveError(
                f"Request path escapes the verified run: {request_path}"
            ) from exc
        request, request_file_sha256 = _load_json_snapshot(request_path)
        expected_request_file_sha256 = sha256_text(
            json.dumps(
                request,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        if request_file_sha256 != expected_request_file_sha256:
            raise ContrastiveError(
                f"Request changed after handoff verification (file bytes): "
                f"{job_id}/{attempt_index}"
            )
        _validate(
            "model_request.schema.json",
            request,
            f"submission request {job_id}/{attempt_index}",
        )
        if hash_object(request) != attempt["request_sha256"]:
            raise ContrastiveError(
                f"Request changed after handoff verification: "
                f"{job_id}/{attempt_index}"
            )
        if (
            job["provider_profile_sha256"]
            != self._provider_profile_sha256
            or job["provider_execution_sha256"]
            != self._provider_execution_sha256
        ):
            raise ContrastiveError(
                f"Provider identity changed after handoff verification: {job_id}"
            )
        return copy.deepcopy(request), attempt

    def _release_request(
        self,
        job_id: str,
        attempt_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reserve and verify an attempt exactly once, including concurrently."""
        release_key = (job_id, attempt_index)
        with self._release_lock:
            try:
                valid_indices = {
                    int(row["attempt_index"])
                    for row in self._jobs[job_id]["generation_attempts"]
                }
            except KeyError as exc:
                raise ContrastiveError(
                    f"Unknown generation job in verified handoff: {job_id}"
                ) from exc
            if attempt_index not in valid_indices:
                raise ContrastiveError(
                    f"Unknown preplanned generation attempt for {job_id}: "
                    f"{attempt_index}"
                )
            if release_key in self._released:
                raise ContrastiveError(
                    f"Generation attempt {job_id}/{attempt_index} was already "
                    "released by this handoff; implicit replay is forbidden"
                )
            expected_index = self._next_attempt[job_id]
            if attempt_index != expected_index:
                raise ContrastiveError(
                    f"Generation attempts must be released in order for "
                    f"{job_id}: expected {expected_index}, got {attempt_index}"
                )
            self._released.add(release_key)
        try:
            result = self._load_verified_attempt(job_id, attempt_index)
        except BaseException:
            with self._release_lock:
                self._released.discard(release_key)
            raise
        with self._release_lock:
            self._next_attempt[job_id] = attempt_index + 1
        return result

    def submission_for(
        self,
        job_id: str,
        attempt_index: int = 0,
    ) -> dict[str, Any]:
        """Return the verified adapter envelope without resolving any secret."""
        request, attempt = self._release_request(job_id, attempt_index)
        return {
            "job_id": job_id,
            "attempt_id": attempt["attempt_id"],
            "attempt_index": attempt_index,
            "request_sha256": attempt["request_sha256"],
            "generation_execution_sha256": attempt[
                "generation_execution_sha256"
            ],
            "provider_profile_sha256": self._provider_profile_sha256,
            "provider_execution_sha256": self._provider_execution_sha256,
            "adapter_profile": copy.deepcopy(self._adapter_profile),
            "request": request,
        }

def _validate(
    schema_name: str,
    value: dict[str, Any],
    label: str,
) -> None:
    def validate_finite(current: Any, pointer: str = "<root>") -> None:
        if isinstance(current, float) and not math.isfinite(current):
            raise ContrastiveError(
                f"Invalid {label}: {pointer} is not a finite number"
            )
        if isinstance(current, dict):
            for key, child in current.items():
                validate_finite(child, f"{pointer}/{key}")
        elif isinstance(current, list):
            for index, child in enumerate(current):
                validate_finite(child, f"{pointer}/{index}")

    validate_finite(value)
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


def validate_run_id(run_id: str) -> None:
    if (
        not RUN_ID_RE.fullmatch(run_id)
        or run_id in {".", ".."}
        or run_id.endswith(".")
        or "/" in run_id
        or "\\" in run_id
        or run_id.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES
    ):
        raise ContrastiveError(
            "Run identifier must be 1-128 letters, digits, dots, underscores, or "
            "hyphens; it cannot be path-like, end in a dot, or use a reserved "
            "Windows device name"
        )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_path_layout(config: dict[str, Any]) -> None:
    """Keep mutable run/output trees disjoint from scientific sources."""
    paths = config["paths"]
    runs_root = resolve_local_path(paths["runs"])
    output_root = resolve_local_path(paths["output"])
    allowed_runs_root = (CONTRASTIVE_ROOT / "runs").resolve()
    allowed_data_root = (CONTRASTIVE_ROOT / "data").resolve()
    if not _is_within(runs_root, allowed_runs_root):
        raise ContrastiveError(
            f"paths.runs must stay beneath {allowed_runs_root}, got {runs_root}"
        )
    if not _is_within(output_root, allowed_data_root):
        raise ContrastiveError(
            f"paths.output must stay beneath {allowed_data_root}, got {output_root}"
        )
    if _is_within(runs_root, output_root) or _is_within(
        output_root, runs_root
    ):
        raise ContrastiveError("Run and output roots must be disjoint")

    source_values = [
        *paths["topics"].values(),
        paths["containers"],
        *paths["prompts"].values(),
        paths["scaffolding"],
        paths["cue_lexicon"],
        paths["semantic_approval"],
        paths["frame_calibration_report"],
        paths["heldout_comprehension_report"],
        paths["scaffolding_lock"],
    ]
    for value in source_values:
        source = resolve_local_path(value)
        if _is_within(source, runs_root) or _is_within(source, output_root):
            raise ContrastiveError(
                f"Scientific source path overlaps a mutable run/output tree: {source}"
            )


def load_pre_generation_config(
    path: str | Path = "configs/pre_generation.yaml",
) -> dict[str, Any]:
    resolved = resolve_local_path(path)
    config = load_yaml(resolved)
    _validate(
        "pre_generation_run.schema.json",
        config,
        f"pre-generation configuration {resolved}",
    )
    validate_path_layout(config)
    if "provider_profile" in config:
        validate_pre_generation_provider(config["provider_profile"])
    return config


def _walk_profile(value: Any, pointer: str = "") -> list[tuple[str, str, Any]]:
    result: list[tuple[str, str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_pointer = f"{pointer}/{key}"
            result.append((child_pointer, str(key), child))
            result.extend(_walk_profile(child, child_pointer))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_walk_profile(child, f"{pointer}/{index}"))
    return result


def _reject_embedded_provider_secret(
    field: str,
    value: str,
) -> None:
    if CONTROL_CHARACTER_RE.search(value):
        raise ContrastiveError(
            f"Provider profile contains a control character in {field}"
        )
    if field in {"base_url", "route"}:
        try:
            parsed = urlsplit(value)
        except ValueError as exc:
            raise ContrastiveError(
                f"Provider profile has an invalid {field}: {exc}"
            ) from exc
        try:
            parsed.port
        except ValueError as exc:
            raise ContrastiveError(
                f"Provider profile has an invalid {field} port: {exc}"
            ) from exc
        if field == "base_url" and (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.fragment
        ):
            raise ContrastiveError(
                "Provider base_url must be an absolute HTTP(S) URL without "
                "a fragment"
            )
        if parsed.username is not None or parsed.password is not None:
            raise ContrastiveError(
                f"Provider profile stores URL userinfo in {field}; credentials "
                "must come from an environment-variable reference"
            )
        sensitive = sorted({
            key
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
            if key.casefold().replace("-", "_") in SENSITIVE_URL_QUERY_KEYS
        })
        if sensitive:
            raise ContrastiveError(
                f"Provider profile stores sensitive query parameters in {field}: "
                f"{sensitive}"
            )
        if field == "route":
            route_segments = value.strip("/").split("/")
            if (
                not ROUTE_RE.fullmatch(value)
                or value.startswith("//")
                or "\\" in value
                or parsed.scheme
                or parsed.netloc
                or parsed.query
                or parsed.fragment
                or any(segment in {".", ".."} for segment in route_segments)
            ):
                raise ContrastiveError(
                    "Provider route must be a safe relative path or opaque label"
                )
    if (
        SECRET_ASSIGNMENT_RE.search(value)
        or BEARER_SECRET_RE.search(value)
        or KNOWN_SECRET_PREFIX_RE.search(value)
    ):
        raise ContrastiveError(
            f"Provider profile stores secret-like content in {field}; use an "
            "environment-variable reference instead"
        )


def validate_pre_generation_provider(profile: dict[str, Any]) -> None:
    _validate("provider.schema.json", profile, "pre-generation provider profile")
    for pointer, key, value in _walk_profile(profile):
        normalized_key = key.casefold().replace("-", "_")
        if normalized_key in SENSITIVE_PROFILE_KEYS and value not in (None, ""):
            raise ContrastiveError(
                f"Provider profile stores a literal secret at {pointer}; use an "
                "environment-variable reference instead"
            )
        if isinstance(value, str) and PLACEHOLDER_VALUE_RE.search(value):
            raise ContrastiveError(
                f"Provider profile contains an unresolved placeholder at {pointer}"
            )
        if isinstance(value, str):
            _reject_embedded_provider_secret(key, value)
    unknown = sorted(set(profile) - ALLOWED_PRE_GENERATION_PROVIDER_KEYS)
    if unknown:
        raise ContrastiveError(
            "Pre-generation provider profile contains unsupported fields that "
            f"could bypass request auditing: {unknown}"
        )
    provider = str(profile["provider"])
    provider_specific_fields = {
        "auth",
        "api_key_env",
        "credential_env",
        "project",
        "location",
        "base_url",
        "route",
    }
    allowed_specific = {
        "mock": set(),
        "google": {
            "auth",
            "credential_env",
            "project",
            "location",
        },
        "openai_compatible": {
            "auth",
            "api_key_env",
            "base_url",
            "route",
        },
        "openrouter": {
            "auth",
            "api_key_env",
            "base_url",
            "route",
        },
        "nim": {
            "auth",
            "api_key_env",
            "base_url",
            "route",
        },
    }[provider]
    ambiguous = sorted(
        (set(profile) & provider_specific_fields) - allowed_specific
    )
    if ambiguous:
        raise ContrastiveError(
            f"Provider {provider!r} contains incompatible routing/auth fields: "
            f"{ambiguous}"
        )
    for field in (
        "model_revision",
        "structured_output_mode",
        "schema_dialect",
        "input_usd_per_million",
        "output_usd_per_million",
        "capabilities",
        "api_attempts",
        "max_concurrency",
        "circuit_breaker_failures",
        "circuit_breaker_cooldown_seconds",
    ):
        if field not in profile:
            raise ContrastiveError(
                f"Pre-generation provider profile must declare {field!r}"
            )
    capabilities = profile["capabilities"]
    if not isinstance(capabilities, dict):
        raise ContrastiveError("Provider capabilities must be a mapping")
    if capabilities.get("structured_output") is not True:
        raise ContrastiveError(
            "Pre-generation provider must support structured output"
        )
    structured_mode = str(profile["structured_output_mode"])
    allowed_modes = {
        "mock": {"native_json_schema"},
        "google": {"native_json_schema"},
        "openai_compatible": {"guided_json", "json_schema"},
        "openrouter": {"guided_json", "json_schema"},
        "nim": {"guided_json", "json_schema"},
    }[provider]
    if structured_mode not in allowed_modes:
        raise ContrastiveError(
            f"Provider {provider!r} must use native JSON-Schema enforcement; "
            f"got structured_output_mode={structured_mode!r}"
        )
    if capabilities.get("seed") not in {"native", "best_effort", "none"}:
        raise ContrastiveError(
            "Provider seed capability must be native, best_effort, or none"
        )
    if MUTABLE_REVISION_RE.search(str(profile["model_revision"])):
        raise ContrastiveError(
            "Provider model_revision appears to be a mutable alias; supply an "
            "immutable provider revision assertion"
        )
    auth = profile.get("auth")
    if provider != "mock" and auth not in {"api_key", "gcloud", "auto"}:
        raise ContrastiveError(
            "Non-mock provider profiles must declare auth as api_key, gcloud, "
            "or auto"
        )
    if (
        provider in {"openai_compatible", "openrouter", "nim"}
        and auth != "api_key"
    ):
        raise ContrastiveError(
            f"Provider {provider!r} requires auth=api_key with an explicit "
            "api_key_env reference"
        )
    if provider == "google" and auth not in {"gcloud", "auto"}:
        raise ContrastiveError(
            "Google provider profiles require auth=gcloud or auth=auto"
        )
    if auth == "api_key" and not profile.get("api_key_env"):
        raise ContrastiveError(
            "Provider auth=api_key requires an api_key_env reference"
        )
    for field in ("api_key_env", "credential_env"):
        value = profile.get(field)
        if value and not ENVIRONMENT_NAME_RE.fullmatch(str(value)):
            raise ContrastiveError(
                f"Provider {field} must be an environment-variable name"
            )
    if auth == "gcloud" and not profile.get("project"):
        raise ContrastiveError(
            "Provider auth=gcloud requires a literal project identifier"
        )
    if provider == "google" and not profile.get("credential_env"):
        raise ContrastiveError(
            "Google provider profiles require a credential_env reference so "
            "handoff can check ADC readiness"
        )
    if (
        provider == "google"
        and profile.get("credential_env")
        != "GOOGLE_APPLICATION_CREDENTIALS"
    ):
        raise ContrastiveError(
            "Google credential_env must be "
            "'GOOGLE_APPLICATION_CREDENTIALS'"
        )
    if provider == "google" and not (
        profile.get("project") and profile.get("location")
    ):
        raise ContrastiveError(
            "Google provider profiles require literal project and location "
            "identifiers"
        )
    if provider in {"openai_compatible", "openrouter", "nim"} and not (
        profile.get("base_url") and profile.get("route")
    ):
        raise ContrastiveError(
            f"Provider {provider!r} requires explicit base_url and route"
        )
    if provider != "mock" and profile.get("base_url"):
        if urlsplit(str(profile["base_url"])).scheme != "https":
            raise ContrastiveError(
                "Non-mock provider base_url must use HTTPS"
            )


def _provider_execution_payload(profile: dict[str, Any]) -> dict[str, Any]:
    """Select provider fields capable of changing scientific generation."""
    fields = (
        "provider",
        "model",
        "model_revision",
        "structured_output_mode",
        "schema_dialect",
        "project",
        "location",
        "base_url",
        "route",
        "thinking_level",
    )
    return {
        **{
            field: profile[field]
            for field in fields
            if field in profile
        },
        "structured_output": profile["capabilities"]["structured_output"],
        "seed_support": profile["capabilities"]["seed"],
    }


def _adapter_execution_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen non-secret control plane an adapter may consume."""
    fields = (
        "profile_id",
        "provider",
        "model",
        "model_revision",
        "structured_output_mode",
        "schema_dialect",
        "auth",
        "api_key_env",
        "credential_env",
        "project",
        "location",
        "base_url",
        "route",
        "timeout_seconds",
        "api_attempts",
        "max_concurrency",
        "circuit_breaker_failures",
        "circuit_breaker_cooldown_seconds",
        "thinking_level",
        "capabilities",
    )
    return {
        field: copy.deepcopy(profile[field])
        for field in fields
        if field in profile
    }


def _runtime_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = re.sub(r"[-_.]+", "-", str(raw_name)).casefold()
        version = str(distribution.version)
        if name in packages and packages[name] != version:
            raise ContrastiveError(
                f"Active environment has conflicting {name!r} versions"
            )
        packages[name] = version
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "version_info": list(sys.version_info[:3]),
            "cache_tag": str(sys.implementation.cache_tag),
            "executable_sha256": sha256_file(Path(sys.executable)),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
        "packages": dict(sorted(packages.items())),
    }


def with_pre_generation_provider(
    config: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    validate_pre_generation_provider(profile)
    resolved = copy.deepcopy(config)
    resolved["provider_profile"] = copy.deepcopy(profile)
    _validate(
        "pre_generation_run.schema.json",
        resolved,
        "resolved pre-generation configuration",
    )
    return resolved


def apply_pre_generation_overrides(
    config: dict[str, Any],
    *,
    topics: list[str] | None = None,
    content_specs: list[str] | None = None,
    languages: list[str] | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    if topics is not None:
        resolved["selection"]["topics"] = topics
    if content_specs is not None:
        resolved["selection"]["content_specs"] = content_specs
    if languages is not None:
        resolved["grid"]["languages"] = languages
    if mode is not None:
        resolved["mode"] = mode
    _validate(
        "pre_generation_run.schema.json",
        resolved,
        "pre-generation configuration with overrides",
    )
    return resolved


def _select[T](
    records: tuple[T, ...],
    requested: str | list[str],
    identity,
    label: str,
) -> tuple[T, ...]:
    by_id = {identity(record): record for record in records}
    if requested == "all":
        return tuple(sorted(records, key=identity))
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ContrastiveError(f"Unknown {label}: {', '.join(unknown)}")
    return tuple(by_id[value] for value in sorted(requested))


def _topic_identity(topic: TopicRecord) -> tuple[str, str, str]:
    return topic.topic_id, topic.topic_label, topic.domain


def _topic_view_payload(topic: TopicRecord) -> dict[str, Any]:
    return {
        "topic_id": topic.topic_id,
        "topic_label": topic.topic_label,
        "topic_display_name": topic.topic_display_name,
        "domain": topic.domain,
        "domain_display_name": topic.domain_display_name,
        "generation_guidance": topic.generation_guidance,
    }


def _validate_topic_views(
    english: ContrastiveCatalog,
    japanese: ContrastiveCatalog,
) -> None:
    en_identity = [_topic_identity(topic) for topic in english.topics]
    ja_identity = [_topic_identity(topic) for topic in japanese.topics]
    if en_identity != ja_identity:
        raise ContrastiveError(
            "English and Japanese topic views have different machine identities "
            "or ordering"
        )


def _normalized_config(
    config: dict[str, Any],
    selected_topics: tuple[TopicRecord, ...],
    selected_specs: tuple[ContentSpecRecord, ...],
    languages: tuple[str, ...],
) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["selection"]["topics"] = [row.topic_id for row in selected_topics]
    value["selection"]["content_specs"] = [
        row.content_spec_id for row in selected_specs
    ]
    value["grid"]["languages"] = list(languages)
    return value


def _production_grid_gate(
    config: dict[str, Any],
    *,
    selected_topic_count: int,
    selected_spec_count: int,
) -> list[dict[str, str]]:
    complete = (
        set(config["grid"]["languages"]) == {"en", "ja"}
        and selected_topic_count == 40
        and selected_spec_count == 10
        and int(config["grid"]["samples_per_combination"]) == 1
    )
    if config["mode"] == "production" and not complete:
        raise ContrastiveError(
            "Production planning requires all 40 topics, all 10 content specs, "
            "both languages, and exactly one sample per combination"
        )
    if complete:
        return []
    return [{
        "level": "notice",
        "code": "development_grid_incomplete",
        "message": (
            "This explicit development run uses an incomplete topic/spec/language "
            "grid and cannot be promoted as a production corpus."
        ),
    }]


def _provider_preflight(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if "provider_profile" not in config:
        raise ContrastiveError(
            "Pre-generation planning requires a provider profile"
        )
    profile = config["provider_profile"]
    validate_pre_generation_provider(profile)
    capabilities = profile["capabilities"]
    requested_output = int(config["generation"]["max_output_tokens"])
    estimated_output = int(config["generation"]["estimated_output_tokens"])
    if estimated_output > requested_output:
        raise ContrastiveError(
            f"estimated_output_tokens {estimated_output} cannot exceed "
            f"max_output_tokens {requested_output}"
        )
    if requested_output > int(capabilities["max_output_tokens"]):
        raise ContrastiveError(
            f"Configured max_output_tokens {requested_output} exceeds provider "
            f"capability {capabilities['max_output_tokens']}"
        )
    if int(config["generation"]["max_workers"]) > int(
        profile["max_concurrency"]
    ):
        raise ContrastiveError(
            f"Configured max_workers {config['generation']['max_workers']} exceeds "
            f"provider max_concurrency {profile['max_concurrency']}"
        )
    notifications: list[dict[str, str]] = []
    seed_support = str(capabilities["seed"])
    if seed_support == "none":
        if config["mode"] == "production":
            raise ContrastiveError(
                "Production provider must support native or best-effort seeds"
            )
        notifications.append({
            "level": "warning",
            "code": "provider_seed_unsupported",
            "message": (
                "The provider declares no seed support; deterministic job identity "
                "is preserved, but model sampling cannot be replayed."
            ),
        })
    elif seed_support == "best_effort":
        notifications.append({
            "level": "notice",
            "code": "provider_seed_best_effort",
            "message": (
                "The provider declares best-effort seed support rather than exact "
                "sampling replay."
            ),
        })
    credential_status = "not_required"
    referenced_variables = [
        str(profile[field])
        for field in ("api_key_env", "credential_env")
        if profile.get(field)
    ]
    missing_variables: list[str] = []
    for variable in referenced_variables:
        value = os.environ.get(variable)
        if not value:
            missing_variables.append(variable)
            continue
        if (
            variable == "GOOGLE_APPLICATION_CREDENTIALS"
            and not Path(value).is_file()
        ):
            missing_variables.append(variable)
    if referenced_variables:
        credential_status = "missing" if missing_variables else "available"
        if missing_variables:
            if config["mode"] == "production":
                raise ContrastiveError(
                    "Production provider environment-variable references are "
                    f"missing or unusable: {missing_variables}"
                )
            notifications.append({
                "level": "warning",
                "code": "provider_credential_unavailable",
                "message": (
                    f"Provider environment references {missing_variables} are "
                    "missing or unusable; "
                    "planning can finish, but submission cannot start."
                ),
            })
    return ({
        "status": "passed",
        "structured_output": "passed",
        "seed_support": capabilities["seed"],
        "max_context_tokens": capabilities["max_context_tokens"],
        "max_output_tokens": capabilities["max_output_tokens"],
        "credential_reference": credential_status,
        "pricing": "explicit",
    }, notifications)


def _compiled_spec_artifact(
    en_specs: tuple[ContentSpecRecord, ...],
    ja_specs: tuple[ContentSpecRecord, ...],
    semantic_review_projection: dict[str, Any],
) -> dict[str, Any]:
    def rows(specs: tuple[ContentSpecRecord, ...]) -> list[dict[str, Any]]:
        return [
            {
                "content_spec_id": spec.content_spec_id,
                "pair_ids": list(spec.pair_ids),
                "model_payload": spec.model_payload(),
                "model_payload_sha256": spec.sha256(),
                "derived_containers": [
                    {
                        "container_id": container.container_id,
                        "class": container.data["class"],
                    }
                    for container in spec.containers
                ],
            }
            for spec in specs
        ]

    return {
        "schema_version": 1,
        "description": (
            "Private compiled source artifact. Only model_payload is eligible for "
            "request projection; derived_containers are renderer metadata."
        ),
        "languages": {"en": rows(en_specs), "ja": rows(ja_specs)},
        "semantic_review_projection": semantic_review_projection,
        "semantic_review_projection_sha256": hash_object(
            semantic_review_projection
        ),
    }


def _derived_containers(
    spec: ContentSpecRecord,
    seed: int,
    output_root: Path,
    run_id: str,
    topic_id: str,
    sample_index: int,
    renderer_hash: str,
    container_catalog_hash: str,
) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    for container in spec.containers:
        rendered_template, _ = prerender_template(container, seed)
        output_name = (
            f"{topic_id}_{container.container_id}_{spec.language}_"
            f"s{sample_index:02d}.json"
        )
        row = {
            "container_id": container.container_id,
            "pair_id": str(container.data["pair_id"]),
            "frame_family_id": str(
                container.data["purpose_frame"]["family_id"]
            ),
            "explicitness": str(
                container.data["purpose_frame"]["explicitness"]
            ),
            "surface": str(container.data["purpose_frame"]["surface"]),
            "class": str(container.data["class"]),
            "body_template_sha256": sha256_text(
                str(container.data["body_template"])
            ),
            "purpose_frame_sha256": hash_object(
                container.data["purpose_frame"]
            ),
            "rendered_template_sha256": sha256_text(rendered_template),
            "output_path": local_reference(
                output_root / run_id / "rendered" / output_name
            ),
        }
        rows.append(row)
        hash_rows.append({
            key: value for key, value in row.items() if key != "output_path"
        })
    rows.sort(key=lambda row: row["container_id"])
    hash_rows.sort(key=lambda row: row["container_id"])
    derivation_hash = hash_object({
        "renderer_sha256": renderer_hash,
        "container_catalog_sha256": container_catalog_hash,
        "containers": hash_rows,
    })
    return rows, derivation_hash


def _generation_execution_hash(
    request_sha256: str,
    provider_execution_sha256: str,
) -> str:
    return hash_object({
        "request_sha256": request_sha256,
        "provider_execution_sha256": provider_execution_sha256,
    })


def _frozen_files(
    catalogs: dict[str, ContrastiveCatalog],
    config: dict[str, Any],
) -> dict[str, str]:
    paths: set[Path] = {
        PRE_GENERATION_PATH,
        PREFLIGHT_PATH,
        CONTENT_SPECS_PATH,
        RENDERER_PATH,
        JOB_SCHEMA_PATH,
        PLAN_SCHEMA_PATH,
        REPORT_SCHEMA_PATH,
        REQUEST_SCHEMA_PATH,
        CONTRASTIVE_ROOT / "prompts" / "pre_generation.lock.json",
        resolve_local_path(config["paths"]["frame_calibration_report"]),
        resolve_local_path(config["paths"]["heldout_comprehension_report"]),
    }
    paths.update(CONTRASTIVE_ROOT / relative for relative in LOCKED_FILES)
    for catalog in catalogs.values():
        paths.update({
            catalog.topic_path,
            catalog.container_path,
            catalog.purpose_frame_path,
            catalog.legacy_topic_path,
        })
        paths.update(path for _, path in catalog.container_paths)
    return {
        local_reference(path): sha256_file(path)
        for path in sorted(paths)
    }


def _validate_manifest_grid(
    jobs: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    expected_grid = {
        (
            str(topic_id),
            str(content_spec_id),
            str(language),
            sample_index,
        )
        for topic_id in config["selection"]["topics"]
        for content_spec_id in config["selection"]["content_specs"]
        for language in config["grid"]["languages"]
        for sample_index in range(
            int(config["grid"]["samples_per_combination"])
        )
    }
    actual_grid_rows = [
        (
            str(job["topic_id"]),
            str(job["content_spec_id"]),
            str(job["language"]),
            int(job["sample_index"]),
        )
        for job in jobs
    ]
    actual_grid = set(actual_grid_rows)
    if (
        actual_grid != expected_grid
        or len(actual_grid_rows) != len(expected_grid)
    ):
        raise ContrastiveError(
            "Generation manifest is not the exact configured Cartesian grid: "
            f"missing={sorted(expected_grid - actual_grid)[:8]}, "
            f"extra={sorted(actual_grid - expected_grid)[:8]}, "
            f"duplicate_rows={len(actual_grid_rows) - len(actual_grid)}"
        )


def _counts_summary(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    language_counts = Counter(str(row["language"]) for row in jobs)
    domain_counts = Counter(str(row["domain"]) for row in jobs)
    spec_counts = Counter(str(row["content_spec_id"]) for row in jobs)
    derived_class_counts = Counter(
        str(derived["class"])
        for job in jobs
        for derived in job["derived_containers"]
    )
    planned_attempts = sum(
        len(job["generation_attempts"]) for job in jobs
    )
    return {
        "topics": len({str(job["topic_id"]) for job in jobs}),
        "content_specs": len({
            str(job["content_spec_id"]) for job in jobs
        }),
        "generation_jobs": len(jobs),
        "planned_generation_attempts": planned_attempts,
        "request_artifacts": planned_attempts,
        "derived_items": sum(
            len(job["derived_containers"]) for job in jobs
        ),
        "by_language": dict(sorted(language_counts.items())),
        "by_domain": dict(sorted(domain_counts.items())),
        "by_content_spec": dict(sorted(spec_counts.items())),
        "derived_by_class": dict(sorted(derived_class_counts.items())),
    }


def _cost_summary(
    jobs: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    profile = config["provider_profile"]
    initial_attempts = [job["generation_attempts"][0] for job in jobs]
    all_attempts = [
        attempt
        for job in jobs
        for attempt in job["generation_attempts"]
    ]
    initial_input = sum(
        int(attempt["preflight"]["estimated_input_tokens"])
        for attempt in initial_attempts
    )
    planned_input = sum(
        int(attempt["preflight"]["estimated_input_tokens"])
        for attempt in all_attempts
    )
    planned_input_upper_bound = sum(
        int(attempt["preflight"]["context_input_token_upper_bound"])
        for attempt in all_attempts
    )
    expected_output_per_call = int(
        config["generation"]["estimated_output_tokens"]
    )
    maximum_output_per_call = int(
        config["generation"]["max_output_tokens"]
    )
    initial_expected_output = len(initial_attempts) * expected_output_per_call
    planned_expected_output = len(all_attempts) * expected_output_per_call
    planned_maximum_output = len(all_attempts) * maximum_output_per_call
    input_rate = Decimal(str(profile["input_usd_per_million"]))
    output_rate = Decimal(str(profile["output_usd_per_million"]))

    def priced(input_tokens: int, output_tokens: int) -> Decimal:
        return (
            Decimal(input_tokens) * input_rate
            + Decimal(output_tokens) * output_rate
        ) / Decimal(1_000_000)

    def cost_fields(value: Decimal) -> dict[str, Any]:
        try:
            display_value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ContrastiveError(
                "Pre-generation cost exceeds the supported numeric range"
            ) from exc
        if not math.isfinite(display_value):
            raise ContrastiveError(
                "Pre-generation cost exceeds the supported numeric range"
            )
        return {
            "estimated_usd": round(display_value, 6),
            "exact_usd": format(value, "f"),
        }

    baseline = priced(initial_input, initial_expected_output)
    planned_expected = priced(planned_input, planned_expected_output)
    maximum_retry_scenario = priced(
        planned_input_upper_bound, planned_maximum_output
    )
    transport_attempts = int(profile.get("api_attempts", 1))
    transport_risk_scenario = maximum_retry_scenario * transport_attempts
    return {
        "pricing_status": "explicit_profile_rates",
        "input_usd_per_million": float(input_rate),
        "output_usd_per_million": float(output_rate),
        "baseline": {
            "generation_calls": len(jobs),
            "estimated_input_tokens": initial_input,
            "estimated_output_tokens": initial_expected_output,
            **cost_fields(baseline),
        },
        "generation_retry_scenario": {
            "attempt_multiplier": 1
            + int(config["generation"]["max_regenerations"]),
            "planned_generation_calls": len(all_attempts),
            **cost_fields(planned_expected),
        },
        "maximum_output_retry_scenario": {
            "attempt_multiplier": 1
            + int(config["generation"]["max_regenerations"]),
            "planned_generation_calls": len(all_attempts),
            "input_token_upper_bound": planned_input_upper_bound,
            "maximum_output_tokens": planned_maximum_output,
            **cost_fields(maximum_retry_scenario),
        },
        "transport_retry_risk_scenario": {
            "transport_attempts": transport_attempts,
            "maximum_provider_submissions": (
                len(all_attempts) * transport_attempts
            ),
            **cost_fields(transport_risk_scenario),
            "note": (
                "Conservative exposure scenario, not a guaranteed billing ceiling; "
                "transport retries reuse one frozen attempt and provider billing "
                "varies. A completed malformed response consumes that attempt."
            ),
        },
    }


def _budget_audit(
    cost: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    profile = config["provider_profile"]
    budget = config["budget"]
    rates = (
        float(profile["input_usd_per_million"]),
        float(profile["output_usd_per_million"]),
    )
    if (
        profile["provider"] != "mock"
        and any(rate == 0 for rate in rates)
        and not budget["allow_zero_pricing"]
    ):
        raise ContrastiveError(
            "A paid-provider rate is zero; set accurate rates or explicitly set "
            "budget.allow_zero_pricing=true"
        )
    basis = str(budget["cost_basis"])
    selected_cost = Decimal(str(cost[basis]["exact_usd"]))
    maximum = Decimal(str(budget["max_usd"]))
    if selected_cost > maximum:
        raise ContrastiveError(
            f"Pre-generation cost scenario {basis} estimates "
            f"${format(selected_cost, 'f')}, above the authorized budget of "
            f"${format(maximum, 'f')}"
        )
    return {
        "status": "passed",
        "basis": basis,
        "estimated_usd": float(cost[basis]["estimated_usd"]),
        "exact_estimated_usd": format(selected_cost, "f"),
        "authorized_max_usd": float(maximum),
        "allow_zero_pricing": bool(budget["allow_zero_pricing"]),
    }


def _plan_identity_hash(plan: dict[str, Any]) -> str:
    return hash_object({
        key: value
        for key, value in plan.items()
        if key not in {
            "created_at",
            "plan_identity_sha256",
            "plan_sha256",
        }
    })


def _plan_hash(plan: dict[str, Any]) -> str:
    return hash_object({
        key: value
        for key, value in plan.items()
        if key != "plan_sha256"
    })


def _calibration_evidence_gate(
    config: dict[str, Any],
    english_catalog: ContrastiveCatalog,
    japanese_catalog: ContrastiveCatalog,
    approval: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    paths = config["paths"]
    specifications = (
        (
            "frame_only",
            "frame_calibration_report",
            "frame_calibration_report.schema.json",
        ),
        (
            "heldout_comprehension",
            "heldout_comprehension_report",
            "heldout_comprehension_report.schema.json",
        ),
    )
    reports: dict[str, Any] = {}
    notices: list[dict[str, str]] = []
    expected = {
        "purpose_frame_catalog_sha256": sha256_file(
            english_catalog.purpose_frame_path
        ),
        "container_catalog_sha256_en": (
            english_catalog.container_catalog_sha256()
        ),
        "container_catalog_sha256_ja": (
            japanese_catalog.container_catalog_sha256()
        ),
    }
    for label, path_key, schema_name in specifications:
        path = resolve_local_path(paths[path_key])
        report = load_json(path)
        _validate(schema_name, report, f"{label} calibration evidence {path}")
        status = str(report["status"])
        if status == "passed":
            if label == "frame_only" and not all(
                report["languages"][language]["pass"]
                and all(
                    report["languages"][language]["gates"].values()
                )
                for language in ("en", "ja")
            ):
                raise ContrastiveError(
                    "Passed frame-only calibration report contains a "
                    "non-passing language"
                )
            if label == "heldout_comprehension" and (
                len(report["models"]) < 2
                or not all(
                    model["pass"]
                    and all(
                        language["pass"]
                        and all(language["gates"].values())
                        for language in model["languages"].values()
                    )
                    for model in report["models"].values()
                )
            ):
                raise ContrastiveError(
                    "Passed held-out comprehension report lacks two fully "
                    "passing calibration models"
                )
            if label == "heldout_comprehension":
                provenance_expected = {
                    "catalog_semantics_sha256": approval[
                        "catalog_semantics_sha256"
                    ],
                    "request_projection_sha256": approval[
                        "request_projection_sha256"
                    ],
                    "generation_settings_sha256": hash_object(
                        config["generation"]
                    ),
                }
                provenance_mismatches = [
                    field
                    for field, expected_value in provenance_expected.items()
                    if report[field] != expected_value
                ]
                current_profile = config["provider_profile"]
                model_identity = {
                    "profile_id": current_profile["profile_id"],
                    "model": current_profile["model"],
                    "model_revision": current_profile["model_revision"],
                }
                if model_identity not in report["generator_models"]:
                    provenance_mismatches.append("generator_models")
                if provenance_mismatches:
                    raise ContrastiveError(
                        "Passed held-out comprehension evidence targets "
                        "different authoring/generator provenance: "
                        f"{provenance_mismatches}"
                    )
            mismatches = [
                field
                for field, value in expected.items()
                if report[field] != value
            ]
            if mismatches:
                raise ContrastiveError(
                    f"Passed {label} calibration evidence targets a different "
                    f"catalog: {mismatches}"
                )
        else:
            waiver = gate_waiver(config, label)
            if config["mode"] == "production" and waiver is None:
                raise ContrastiveError(
                    f"Production planning requires passed {label} calibration "
                    f"evidence; current status is {status!r}"
                )
            if waiver is None:
                notices.append(
                    {
                        "level": "warning",
                        "code": f"{label}_calibration_pending",
                        "message": (
                            f"{label} calibration evidence status is {status}; "
                            "production planning remains disabled."
                        ),
                    }
                )
            else:
                notices.append(
                    {
                        "level": "warning",
                        "code": f"{label}_calibration_waived",
                        "message": (
                            f"{label} calibration evidence status is {status} "
                            f"and was waived by {waiver['waived_by']} on "
                            f"{waiver['waived_at']}. This run and every "
                            "artifact derived from it carry no "
                            f"{label} evidence."
                        ),
                    }
                )
        reports[label] = {
            "status": status,
            "path": local_reference(path),
            "sha256": sha256_file(path),
        }
        if status != "passed":
            waiver = gate_waiver(config, label)
            if waiver is not None:
                reports[label]["waiver"] = waiver
    return reports, notices


def _assemble_pre_generation(
    config: dict[str, Any],
    run_id: str,
) -> PreGenerationArtifacts:
    validate_run_id(run_id)
    _validate(
        "pre_generation_run.schema.json",
        config,
        "pre-generation configuration",
    )
    validate_path_layout(config)
    provider_audit, provider_notifications = _provider_preflight(config)
    bundle = validate_scaffolding_bundle(config)

    english_catalog = load_catalog(
        config["paths"]["topics"]["en"],
        config["paths"]["containers"],
    )
    japanese_catalog = load_catalog(
        config["paths"]["topics"]["ja"],
        config["paths"]["containers"],
    )
    if english_catalog.topic_path.name.endswith(".ja.yaml"):
        raise ContrastiveError("English topic path points to a Japanese view")
    if not japanese_catalog.topic_path.name.endswith(".ja.yaml"):
        raise ContrastiveError("Japanese topic path does not point to a .ja.yaml view")
    _validate_topic_views(english_catalog, japanese_catalog)
    calibration_evidence, calibration_notifications = (
        _calibration_evidence_gate(
            config,
            english_catalog,
            japanese_catalog,
            bundle.approval,
        )
    )

    english_specs = compile_content_specs(
        english_catalog, bundle.scaffolding
    )
    japanese_specs = compile_content_specs(
        japanese_catalog, bundle.scaffolding
    )
    validate_bilingual_content_specs(english_specs, japanese_specs)
    catalog_semantics_hash = catalog_semantic_review_sha256(
        english_catalog,
        japanese_catalog,
        english_specs,
        japanese_specs,
    )
    if (
        bundle.approval["catalog_semantics_sha256"]
        != catalog_semantics_hash
    ):
        raise ContrastiveError(
            "Semantic-equivalence approval targets a different topic/content/"
            f"container semantic bundle: expected {catalog_semantics_hash}"
        )
    corpus_audit = validate_registered_corpus(
        english_catalog.topics, english_specs
    )
    renderer_cue_audit = validate_renderer_cue_contract(
        english_specs, japanese_specs, bundle.cue_lexicon
    )

    selected_topics_en = _select(
        english_catalog.topics,
        config["selection"]["topics"],
        lambda row: row.topic_id,
        "topic identifiers",
    )
    en_spec_selection = _select(
        english_specs,
        config["selection"]["content_specs"],
        lambda row: row.content_spec_id,
        "content specification identifiers",
    )
    languages = tuple(
        language for language in ("en", "ja")
        if language in set(config["grid"]["languages"])
    )
    grid_notifications = _production_grid_gate(
        config,
        selected_topic_count=len(selected_topics_en),
        selected_spec_count=len(en_spec_selection),
    )
    resolved_config = _normalized_config(
        config, selected_topics_en, en_spec_selection, languages
    )

    topics_by_language = {
        "en": {row.topic_id: row for row in english_catalog.topics},
        "ja": {row.topic_id: row for row in japanese_catalog.topics},
    }
    specs_by_language = {
        "en": {row.content_spec_id: row for row in english_specs},
        "ja": {row.content_spec_id: row for row in japanese_specs},
    }
    catalogs = {"en": english_catalog, "ja": japanese_catalog}
    prompt_paths = {
        language: resolve_local_path(
            resolved_config["paths"]["prompts"][language]
        )
        for language in languages
    }
    output_root = resolve_local_path(resolved_config["paths"]["output"])
    provider_profile_hash = hash_object(resolved_config["provider_profile"])
    provider_execution_hash = hash_object(
        _provider_execution_payload(resolved_config["provider_profile"])
    )
    renderer_hash = sha256_file(RENDERER_PATH)
    scaffolding_hash = sha256_file(
        resolve_local_path(resolved_config["paths"]["scaffolding"])
    )
    lock_hash = sha256_file(
        resolve_local_path(resolved_config["paths"]["scaffolding_lock"])
    )
    lexicon_hash = sha256_file(
        resolve_local_path(resolved_config["paths"]["cue_lexicon"])
    )
    base_seed = int(resolved_config["randomization"]["seed"])
    maximum_context = int(
        resolved_config["provider_profile"]["capabilities"][
            "max_context_tokens"
        ]
    )

    jobs: list[dict[str, Any]] = []
    requests: dict[str, dict[str, Any]] = {}
    selected_topic_ids = [row.topic_id for row in selected_topics_en]
    selected_spec_ids = [
        row.content_spec_id for row in en_spec_selection
    ]
    for topic_id in selected_topic_ids:
        for content_spec_id in selected_spec_ids:
            for language in languages:
                topic = topics_by_language[language][topic_id]
                spec = specs_by_language[language][content_spec_id]
                catalog = catalogs[language]
                for sample_index in range(
                    int(resolved_config["grid"]["samples_per_combination"])
                ):
                    identity = (
                        "contrastive_content",
                        topic_id,
                        content_spec_id,
                        language,
                        str(sample_index),
                    )
                    attempt_rows: list[dict[str, Any]] = []
                    attempt_requests: list[dict[str, Any]] = []
                    for attempt_index in range(
                        1
                        + int(
                            resolved_config["generation"][
                                "max_regenerations"
                            ]
                        )
                    ):
                        attempt_seed = (
                            stable_seed(base_seed, *identity)
                            if attempt_index == 0
                            else stable_seed(
                                base_seed,
                                *identity,
                                "regeneration",
                                str(attempt_index),
                            )
                        )
                        request = render_model_request(
                            prompt_paths[language],
                            model_topic(topic),
                            spec.model_projection(),
                            bundle.scaffolding,
                            resolved_config["generation"],
                            attempt_seed,
                        )
                        audit = audit_model_request(
                            request,
                            spec,
                            bundle.cue_lexicon,
                            topic,
                            int(
                                resolved_config["generation"][
                                    "context_overhead_tokens"
                                ]
                            ),
                        )
                        if (
                            int(audit["context_input_token_upper_bound"])
                            + int(
                                resolved_config["generation"][
                                    "max_output_tokens"
                                ]
                            )
                            > maximum_context
                        ):
                            raise ContrastiveError(
                                f"Request {topic_id}/{content_spec_id}/"
                                f"{language}/attempt-{attempt_index} exceeds "
                                "provider context: input upper bound "
                                f"{audit['context_input_token_upper_bound']} + "
                                "max output "
                                f"{resolved_config['generation']['max_output_tokens']} "
                                f"> {maximum_context}"
                            )
                        request_hash = hash_object(request)
                        generation_hash = _generation_execution_hash(
                            request_hash, provider_execution_hash
                        )
                        attempt_rows.append({
                            "attempt_index": attempt_index,
                            "seed": attempt_seed,
                            "request_sha256": request_hash,
                            "generation_execution_sha256": generation_hash,
                            "preflight": audit,
                        })
                        attempt_requests.append(request)
                    attempt_seeds = [
                        int(row["seed"]) for row in attempt_rows
                    ]
                    if len(attempt_seeds) != len(set(attempt_seeds)):
                        raise ContrastiveError(
                            f"Preplanned generation seeds collide for "
                            f"{topic_id}/{content_spec_id}/{language}/"
                            f"{sample_index}"
                        )
                    job_id = "contrastive_content_" + sha256_text(
                        "\x1f".join([
                            *identity,
                            str(
                                attempt_rows[0][
                                    "generation_execution_sha256"
                                ]
                            ),
                        ])
                    )[:20]
                    run_request_root = (
                        resolve_local_path(resolved_config["paths"]["runs"])
                        / run_id
                        / "requests"
                    )
                    generation_attempts = [
                        {
                            **row,
                            "attempt_id": f"{job_id}_a{index:02d}",
                            "request_path": local_reference(
                                run_request_root
                                / f"{job_id}_a{index:02d}.json"
                            ),
                            "candidate_output_path": local_reference(
                                output_root
                                / run_id
                                / "attempts"
                                / f"{job_id}_a{index:02d}.json"
                            ),
                        }
                        for index, row in enumerate(attempt_rows)
                    ]
                    for attempt, request in zip(
                        generation_attempts,
                        attempt_requests,
                        strict=True,
                    ):
                        requests[str(attempt["attempt_id"])] = request
                    content_name = (
                        f"{topic_id}_{content_spec_id}_{language}_"
                        f"s{sample_index:02d}.json"
                    )
                    container_catalog_hash = (
                        catalog.container_catalog_sha256()
                    )
                    derivation_seed = stable_seed(base_seed, *identity)
                    derived, derivation_hash = _derived_containers(
                        spec,
                        derivation_seed,
                        output_root,
                        run_id,
                        topic_id,
                        sample_index,
                        renderer_hash,
                        container_catalog_hash,
                    )
                    response_schema_hash = hash_object(
                        attempt_requests[0]["response_schema"]
                    )
                    job = {
                        "schema_version": 2,
                        "stage": "contrastive_content_generation",
                        "job_id": job_id,
                        "topic_id": topic_id,
                        "topic_label": topic.topic_label,
                        "domain": topic.domain,
                        "content_spec_id": content_spec_id,
                        "language": language,
                        "sample_index": sample_index,
                        "derivation_seed": derivation_seed,
                        "provider_profile_id": str(
                            resolved_config["provider_profile"]["profile_id"]
                        ),
                        "provider_profile_sha256": provider_profile_hash,
                        "provider_execution_sha256": provider_execution_hash,
                        "provider": str(
                            resolved_config["provider_profile"]["provider"]
                        ),
                        "model": str(
                            resolved_config["provider_profile"]["model"]
                        ),
                        "topic_catalog_sha256": sha256_file(
                            catalog.topic_path
                        ),
                        "topic_view_sha256": hash_object(
                            _topic_view_payload(topic)
                        ),
                        "container_catalog_sha256": container_catalog_hash,
                        "content_spec_sha256": spec.sha256(),
                        "prompt_template": local_reference(
                            prompt_paths[language]
                        ),
                        "prompt_template_sha256": sha256_file(
                            prompt_paths[language]
                        ),
                        "scaffolding_sha256": scaffolding_hash,
                        "scaffolding_lock_sha256": lock_hash,
                        "cue_lexicon_sha256": lexicon_hash,
                        "renderer_sha256": renderer_hash,
                        "response_schema_sha256": response_schema_hash,
                        "generation_attempts": generation_attempts,
                        "derivation_sha256": derivation_hash,
                        "content_output_path": local_reference(
                            output_root
                            / run_id
                            / "content"
                            / content_name
                        ),
                        "derived_containers": derived,
                    }
                    _validate(
                        "pre_generation_job.schema.json",
                        job,
                        f"planned job {job_id}",
                    )
                    jobs.append(job)

    jobs.sort(key=lambda row: (
        row["topic_id"],
        row["content_spec_id"],
        row["language"],
        row["sample_index"],
    ))
    _validate_manifest_grid(jobs, resolved_config)
    uniqueness_fields = (
        "job_id",
        "content_output_path",
    )
    for field in uniqueness_fields:
        values = [str(row[field]) for row in jobs]
        if len(values) != len(set(values)):
            raise ContrastiveError(
                f"Planner produced duplicate {field} values"
            )
    attempt_ids = [
        str(attempt["attempt_id"])
        for job in jobs
        for attempt in job["generation_attempts"]
    ]
    request_paths = [
        str(attempt["request_path"])
        for job in jobs
        for attempt in job["generation_attempts"]
    ]
    candidate_paths = [
        str(attempt["candidate_output_path"])
        for job in jobs
        for attempt in job["generation_attempts"]
    ]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ContrastiveError("Planner produced duplicate generation attempt IDs")
    if len(request_paths) != len(set(request_paths)):
        raise ContrastiveError("Planner produced duplicate request paths")
    if len(candidate_paths) != len(set(candidate_paths)):
        raise ContrastiveError(
            "Planner produced duplicate candidate-output paths"
        )
    derivation_paths = [
        str(derived["output_path"])
        for job in jobs
        for derived in job["derived_containers"]
    ]
    if len(derivation_paths) != len(set(derivation_paths)):
        raise ContrastiveError(
            "Planner produced duplicate derived-container output paths"
        )

    notifications = tuple([
        *bundle.notifications,
        *provider_notifications,
        *grid_notifications,
        *calibration_notifications,
    ])
    planned_attempts = [
        attempt
        for job in jobs
        for attempt in job["generation_attempts"]
    ]
    counts = _counts_summary(jobs)
    if counts["request_artifacts"] != len(requests):
        raise ContrastiveError(
            "Planner request-artifact count differs from its attempt schedule"
        )
    report = {
        "schema_version": 1,
        "stage": "contrastive_pre_generation",
        "mode": resolved_config["mode"],
        "status": "passed_with_notices" if notifications else "passed",
        "global_gates": {
            "C01_run_configuration_schema": "passed",
            "C02_topic_catalog_schema": "passed",
            "C03_container_catalog_schema": "passed",
            "C04_permanent_identifiers": "passed",
            "C05_bilingual_topic_invariants": "passed",
            "C06_bilingual_container_invariants": "passed",
            "C07_registered_corpus": corpus_audit,
            "C08_pair_closure": "passed",
            "C09_content_twin_compilation": "passed",
            "C10_production_grid": (
                "passed" if not grid_notifications else "development_override"
            ),
            "C11_selection_consistency": "passed",
            "C12_bilingual_placeholder_parity": "passed",
            "C13_japanese_semantic_surface": "passed",
            "C14_scaffolding_hash_lock": bundle.audit,
            "C15_semantic_equivalence_approval": {
                "status": bundle.approval["status"],
                "scaffolding_sha256": bundle.approval[
                    "scaffolding_sha256"
                ],
                "catalog_semantics_sha256": catalog_semantics_hash,
                "request_projection_sha256": bundle.approval[
                    "request_projection_sha256"
                ],
                **(
                    {"waiver": bundle.audit[
                        "semantic_equivalence_review_waiver"
                    ]}
                    if "semantic_equivalence_review_waiver" in bundle.audit
                    else {}
                ),
            },
            "C16_model_request_cue_screen": "passed",
            "C17_private_leakage_screen": "passed",
            "C18_template_blind_projection": "passed_by_capability_boundary",
            "C19_provider_capabilities": provider_audit,
            "C20_context_and_cost": "passed",
            "C21_job_schema_and_uniqueness": "passed",
            "C22_immutable_publication_contract": (
                "plan_is_completion_marker_and_post_write_verification_is_required"
            ),
            "C23_calibration_evidence": calibration_evidence,
            "C24_waived_gates": {
                gate: gate_waiver(config, gate)
                for gate in sorted(WAIVABLE_GATES)
                if gate_waiver(config, gate) is not None
            } or "none",
            "renderer_cue_contract": renderer_cue_audit,
        },
        "job_audits": {
            "requests_checked": len(planned_attempts),
            "placeholder_audit": "passed",
            "cue_audit": "passed",
            "language_audit": "passed",
            "leakage_audit": "passed",
            "context_audit": "passed",
            "maximum_estimated_input_tokens": max(
                int(row["preflight"]["estimated_input_tokens"])
                for row in planned_attempts
            ),
            "maximum_context_input_token_upper_bound": max(
                int(
                    row["preflight"]["context_input_token_upper_bound"]
                )
                for row in planned_attempts
            ),
        },
        "counts": counts,
    }
    _validate(
        "pre_generation_report.schema.json",
        report,
        "pre-generation report",
    )
    compiled_specs = _compiled_spec_artifact(
        english_specs,
        japanese_specs,
        catalog_semantic_review_payload(
            english_catalog,
            japanese_catalog,
            english_specs,
            japanese_specs,
        ),
    )
    frozen_files = _frozen_files(catalogs, resolved_config)
    request_hashes = {
        attempt["attempt_id"]: attempt["request_sha256"]
        for job in jobs
        for attempt in job["generation_attempts"]
    }
    cost_estimate = _cost_summary(jobs, resolved_config)
    budget_audit = _budget_audit(cost_estimate, resolved_config)
    report["global_gates"]["C20_context_and_cost"] = {
        "status": "passed",
        "context": "passed",
        "budget": budget_audit,
    }
    report["status"] = "passed_with_notices" if notifications else "passed"
    _validate(
        "pre_generation_report.schema.json",
        report,
        "pre-generation report after budget authorization",
    )
    plan = {
        "schema_version": 2,
        "stage": "contrastive_pre_generation",
        "run_id": run_id,
        "mode": resolved_config["mode"],
        "created_at": utc_now(),
        "resolved_config_sha256": hash_object(resolved_config),
        "manifest_sha256": hash_object(jobs),
        "request_bundle_sha256": hash_object(request_hashes),
        "preflight_report_sha256": hash_object(report),
        "notifications_sha256": hash_object(notifications),
        "approvals_sha256": hash_object(bundle.approval),
        "generation_bundle_sha256": hash_object({
            attempt["attempt_id"]: attempt[
                "generation_execution_sha256"
            ]
            for job in jobs
            for attempt in job["generation_attempts"]
        }),
        "derivation_bundle_sha256": hash_object({
            job["job_id"]: job["derivation_sha256"]
            for job in jobs
        }),
        "counts": counts,
        "selection": resolved_config["selection"],
        "grid": resolved_config["grid"],
        "provider_profile": {
            "profile_id": resolved_config["provider_profile"]["profile_id"],
            "provider": resolved_config["provider_profile"]["provider"],
            "model": resolved_config["provider_profile"]["model"],
            "model_revision": resolved_config["provider_profile"][
                "model_revision"
            ],
            "sha256": provider_profile_hash,
            "execution_sha256": provider_execution_hash,
        },
        "runtime_environment": _runtime_environment(),
        "cost_estimate": cost_estimate,
        "budget_authorization": budget_audit,
        "preflight": {
            "status": report["status"],
            "notifications": len(notifications),
            "report_path": "preflight_report.json",
        },
        "artifacts": {
            "frozen_files": frozen_files,
            "compiled_content_specs_sha256": hash_object(compiled_specs),
        },
    }
    plan["plan_identity_sha256"] = _plan_identity_hash(plan)
    plan["plan_sha256"] = _plan_hash(plan)
    _validate(
        "pre_generation_plan.schema.json",
        plan,
        "pre-generation plan",
    )
    return PreGenerationArtifacts(
        resolved_config=resolved_config,
        plan=plan,
        jobs=tuple(jobs),
        requests=requests,
        report=report,
        notifications=notifications,
        approvals=copy.deepcopy(bundle.approval),
        compiled_specs=compiled_specs,
    )


def build_pre_generation_plan(
    config: dict[str, Any],
    run_id: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    artifacts = _assemble_pre_generation(config, run_id)
    return artifacts.plan, list(artifacts.jobs), artifacts.requests


def write_pre_generation_plan(
    config: dict[str, Any],
    run_id: str,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    artifacts = _assemble_pre_generation(config, run_id)
    run_root = resolve_local_path(artifacts.resolved_config["paths"]["runs"])
    run_dir = resolve_local_path(run_root / run_id)
    plan_path = run_dir / "plan.json"

    def reuse_completed_run() -> tuple[
        Path,
        dict[str, Any],
        list[dict[str, Any]],
    ]:
        verify_pre_generation_run(run_dir)
        existing = load_json(plan_path)
        if (
            existing["plan_identity_sha256"]
            != artifacts.plan["plan_identity_sha256"]
        ):
            raise ContrastiveError(
                f"Run directory {run_dir} contains a different immutable plan"
            )
        manifest = _read_jsonl(run_dir / "manifest.jsonl")
        return run_dir, existing, manifest

    if plan_path.is_file():
        return reuse_completed_run()
    try:
        # Directory creation is the exclusive publication claim for a new run
        # namespace. Only its owner may write the multi-file artifact bundle.
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        # A competing writer may have completed between the first plan check
        # and our ownership attempt. Reuse only a fully verified completed run.
        if plan_path.is_file():
            return reuse_completed_run()
        raise ContrastiveError(
            f"Run directory {run_dir} is already being created or contains an "
            "incomplete plan; concurrent or implicit resume is forbidden"
        ) from exc
    atomic_write_json(
        run_dir / "resolved_config.json", artifacts.resolved_config
    )
    atomic_write_json(
        run_dir / "content_specs.json", artifacts.compiled_specs
    )
    request_dir = run_dir / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    for job_id, request in sorted(artifacts.requests.items()):
        atomic_write_json(request_dir / f"{job_id}.json", request)
    atomic_write_jsonl(run_dir / "manifest.jsonl", list(artifacts.jobs))
    atomic_write_json(run_dir / "preflight_report.json", artifacts.report)
    atomic_write_json(
        run_dir / "notifications.json", list(artifacts.notifications)
    )
    atomic_write_json(run_dir / "approvals.json", artifacts.approvals)
    # The plan is the completion marker and is deliberately published last.
    atomic_write_json(plan_path, artifacts.plan)
    verify_pre_generation_run(run_dir)
    return run_dir, artifacts.plan, list(artifacts.jobs)


def _snapshot_bytes(path: Path) -> tuple[bytes, str]:
    """Read one immutable byte snapshot and hash those exact bytes."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContrastiveError(f"Could not read {path}: {exc}") from exc
    return raw, sha256_bytes(raw)


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _load_json_snapshot(path: Path) -> tuple[Any, str]:
    raw, file_sha256 = _snapshot_bytes(path)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ContrastiveError(f"Could not read JSON {path}: {exc}") from exc
    return value, file_sha256


def _read_jsonl_snapshot(
    path: Path,
) -> tuple[list[dict[str, Any]], str]:
    raw, file_sha256 = _snapshot_bytes(path)
    try:
        rows = [
            json.loads(line, parse_constant=_reject_nonfinite_json)
            for line in raw.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ContrastiveError(f"Could not read JSONL {path}: {exc}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ContrastiveError(f"Expected JSON objects in {path}")
    return rows, file_sha256


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows, _ = _read_jsonl_snapshot(path)
    return rows


def _resolve_unaliased_output_path(value: str) -> Path:
    """Resolve an output and reject existing symlink/junction path components."""
    resolved = resolve_local_path(value)
    lexical = Path(value)
    if lexical.is_absolute():
        candidate = lexical
    else:
        candidate = CONTRASTIVE_ROOT / lexical
    try:
        relative = candidate.relative_to(CONTRASTIVE_ROOT)
    except ValueError as exc:
        raise ContrastiveError(
            f"Output path is not lexically inside the contrastive root: {value}"
        ) from exc
    if ".." in relative.parts:
        raise ContrastiveError(f"Output path contains traversal: {value}")
    current = CONTRASTIVE_ROOT
    for part in relative.parts:
        current = current / part
        if not os.path.lexists(current):
            continue
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.is_symlink() or is_junction():
            raise ContrastiveError(
                f"Output path crosses a symlink or junction: {value}"
            )
    return resolved


def _path_entry_exists(path: Path) -> bool:
    """Return true for normal entries and broken symlink/reparse aliases."""
    return os.path.lexists(path)


def _job_output_paths(job: dict[str, Any]) -> tuple[Path, ...]:
    return (
        _resolve_unaliased_output_path(job["content_output_path"]),
        *(
            _resolve_unaliased_output_path(row["candidate_output_path"])
            for row in job["generation_attempts"]
        ),
        *(
            _resolve_unaliased_output_path(row["output_path"])
            for row in job["derived_containers"]
        ),
    )


def _job_release_collision_paths(
    job: dict[str, Any],
    attempt_index: int,
) -> tuple[Path, ...]:
    """Paths that must still be empty before releasing one planned attempt."""
    return (
        _resolve_unaliased_output_path(job["content_output_path"]),
        *(
            _resolve_unaliased_output_path(row["output_path"])
            for row in job["derived_containers"]
        ),
        *(
            _resolve_unaliased_output_path(row["candidate_output_path"])
            for row in job["generation_attempts"]
            if int(row["attempt_index"]) >= attempt_index
        ),
    )


def verify_pre_generation_run(
    run_dir: str | Path,
) -> dict[str, Any]:
    """Revalidate every frozen source and exact request before any model call."""
    resolved_run_dir = resolve_local_path(run_dir)
    required = {
        "plan.json",
        "manifest.jsonl",
        "resolved_config.json",
        "content_specs.json",
        "preflight_report.json",
        "notifications.json",
        "approvals.json",
    }
    missing = sorted(
        name for name in required if not (resolved_run_dir / name).is_file()
    )
    if missing:
        raise ContrastiveError(
            f"Immutable run is incomplete at {resolved_run_dir}: missing {missing}"
        )
    plan = load_json(resolved_run_dir / "plan.json")
    _validate("pre_generation_plan.schema.json", plan, "stored plan")
    validate_run_id(str(plan["run_id"]))
    if plan["plan_sha256"] != _plan_hash(plan):
        raise ContrastiveError("Stored pre-generation plan hash is invalid")
    if plan["plan_identity_sha256"] != _plan_identity_hash(plan):
        raise ContrastiveError(
            "Stored pre-generation plan identity hash is invalid"
        )
    if plan["runtime_environment"] != _runtime_environment():
        raise ContrastiveError(
            "Runtime environment differs from the immutable pre-generation plan"
        )
    if resolved_run_dir.name != plan["run_id"]:
        raise ContrastiveError(
            "Run directory name differs from the immutable plan run_id"
        )

    config = load_json(resolved_run_dir / "resolved_config.json")
    _validate(
        "pre_generation_run.schema.json",
        config,
        "stored resolved configuration",
    )
    validate_path_layout(config)
    if hash_object(config) != plan["resolved_config_sha256"]:
        raise ContrastiveError("Stored resolved configuration was modified")
    validate_pre_generation_provider(config["provider_profile"])
    _provider_preflight(config)

    for relative, expected in plan["artifacts"]["frozen_files"].items():
        path = resolve_local_path(relative)
        if not path.is_file():
            raise ContrastiveError(f"Frozen source is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ContrastiveError(
                f"Frozen source changed: {relative}; expected {expected}, got {actual}"
            )

    bundle = validate_scaffolding_bundle(config)
    report = load_json(resolved_run_dir / "preflight_report.json")
    _validate(
        "pre_generation_report.schema.json",
        report,
        "stored pre-generation report",
    )
    if hash_object(report) != plan["preflight_report_sha256"]:
        raise ContrastiveError("Stored preflight report was modified")
    notifications = load_json(resolved_run_dir / "notifications.json")
    if hash_object(notifications) != plan["notifications_sha256"]:
        raise ContrastiveError("Stored notification record was modified")
    approvals = load_json(resolved_run_dir / "approvals.json")
    if hash_object(approvals) != plan["approvals_sha256"]:
        raise ContrastiveError("Stored approval record was modified")
    if approvals != bundle.approval:
        raise ContrastiveError(
            "Stored approval record differs from the frozen source approval"
        )
    compiled_specs = load_json(resolved_run_dir / "content_specs.json")
    if (
        hash_object(compiled_specs)
        != plan["artifacts"]["compiled_content_specs_sha256"]
    ):
        raise ContrastiveError("Stored compiled content specs were modified")

    jobs = _read_jsonl(resolved_run_dir / "manifest.jsonl")
    if hash_object(jobs) != plan["manifest_sha256"]:
        raise ContrastiveError("Stored generation manifest was modified")
    for job in jobs:
        _validate(
            "pre_generation_job.schema.json",
            job,
            f"stored job {job.get('job_id', '<unknown>')}",
        )
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise ContrastiveError("Stored manifest contains duplicate job IDs")
    _validate_manifest_grid(jobs, config)

    english_catalog = load_catalog(
        config["paths"]["topics"]["en"], config["paths"]["containers"]
    )
    japanese_catalog = load_catalog(
        config["paths"]["topics"]["ja"], config["paths"]["containers"]
    )
    english_specs = compile_content_specs(
        english_catalog, bundle.scaffolding
    )
    japanese_specs = compile_content_specs(
        japanese_catalog, bundle.scaffolding
    )
    validate_bilingual_content_specs(english_specs, japanese_specs)
    expected_compiled_specs = _compiled_spec_artifact(
        english_specs,
        japanese_specs,
        catalog_semantic_review_payload(
            english_catalog,
            japanese_catalog,
            english_specs,
            japanese_specs,
        ),
    )
    if compiled_specs != expected_compiled_specs:
        raise ContrastiveError(
            "Stored compiled content specs cannot be reproduced from sources"
        )
    catalog_semantics_hash = catalog_semantic_review_sha256(
        english_catalog,
        japanese_catalog,
        english_specs,
        japanese_specs,
    )
    if approvals["catalog_semantics_sha256"] != catalog_semantics_hash:
        raise ContrastiveError(
            "Stored semantic approval does not target the current catalog bundle"
        )
    topic_maps = {
        "en": {row.topic_id: row for row in english_catalog.topics},
        "ja": {row.topic_id: row for row in japanese_catalog.topics},
    }
    spec_maps = {
        "en": {row.content_spec_id: row for row in english_specs},
        "ja": {row.content_spec_id: row for row in japanese_specs},
    }
    catalog_maps = {"en": english_catalog, "ja": japanese_catalog}
    request_dir = resolved_run_dir / "requests"
    expected_request_names = {
        f"{attempt['attempt_id']}.json"
        for job in jobs
        for attempt in job["generation_attempts"]
    }
    actual_request_names = (
        {path.name for path in request_dir.iterdir()}
        if request_dir.is_dir()
        else set()
    )
    if actual_request_names != expected_request_names:
        raise ContrastiveError(
            "Immutable request file set differs from the manifest: "
            f"missing={sorted(expected_request_names - actual_request_names)}, "
            f"extra={sorted(actual_request_names - expected_request_names)}"
        )

    verified_hashes: dict[str, str] = {}
    verified_generation: dict[str, str] = {}
    verified_derivations: dict[str, str] = {}
    provider_profile_hash = hash_object(config["provider_profile"])
    provider_execution_hash = hash_object(
        _provider_execution_payload(config["provider_profile"])
    )
    scaffolding_hash = sha256_file(
        resolve_local_path(config["paths"]["scaffolding"])
    )
    lock_hash = sha256_file(
        resolve_local_path(config["paths"]["scaffolding_lock"])
    )
    lexicon_hash = sha256_file(
        resolve_local_path(config["paths"]["cue_lexicon"])
    )
    renderer_hash = sha256_file(RENDERER_PATH)
    for job in jobs:
        language = str(job["language"])
        identity = (
            "contrastive_content",
            str(job["topic_id"]),
            str(job["content_spec_id"]),
            language,
            str(job["sample_index"]),
        )
        expected_derivation_seed = stable_seed(
            int(config["randomization"]["seed"]), *identity
        )
        if int(job["derivation_seed"]) != expected_derivation_seed:
            raise ContrastiveError(
                f"Stored derivation seed is invalid: {job['job_id']}"
            )
        spec = spec_maps[language][job["content_spec_id"]]
        topic = topic_maps[language][job["topic_id"]]
        catalog = catalog_maps[language]
        if job["topic_label"] != topic.topic_label or job["domain"] != topic.domain:
            raise ContrastiveError(
                f"Topic metadata is invalid: {job['job_id']}"
            )
        if job["topic_catalog_sha256"] != sha256_file(catalog.topic_path):
            raise ContrastiveError(
                f"Topic-catalog hash is invalid: {job['job_id']}"
            )
        if job["topic_view_sha256"] != hash_object(
            _topic_view_payload(topic)
        ):
            raise ContrastiveError(
                f"Topic-view hash is invalid: {job['job_id']}"
            )
        if job["content_spec_sha256"] != spec.sha256():
            raise ContrastiveError(
                f"Content-spec hash is invalid: {job['job_id']}"
            )
        expected_prompt_path = resolve_local_path(
            config["paths"]["prompts"][language]
        )
        if resolve_local_path(job["prompt_template"]) != expected_prompt_path:
            raise ContrastiveError(
                f"Prompt path is invalid: {job['job_id']}"
            )
        if job["prompt_template_sha256"] != sha256_file(expected_prompt_path):
            raise ContrastiveError(
                f"Prompt hash is invalid: {job['job_id']}"
            )
        fixed_hashes = {
            "scaffolding_sha256": scaffolding_hash,
            "scaffolding_lock_sha256": lock_hash,
            "cue_lexicon_sha256": lexicon_hash,
            "renderer_sha256": renderer_hash,
        }
        for field, expected in fixed_hashes.items():
            if job[field] != expected:
                raise ContrastiveError(
                    f"{field} is invalid: {job['job_id']}"
                )
        if job["provider_profile_sha256"] != provider_profile_hash:
            raise ContrastiveError(
                f"Provider-profile hash is invalid: {job['job_id']}"
            )
        if job["provider_execution_sha256"] != provider_execution_hash:
            raise ContrastiveError(
                f"Provider-execution hash is invalid: {job['job_id']}"
            )
        provider_fields = {
            "provider_profile_id": config["provider_profile"]["profile_id"],
            "provider": config["provider_profile"]["provider"],
            "model": config["provider_profile"]["model"],
        }
        for field, expected in provider_fields.items():
            if job[field] != expected:
                raise ContrastiveError(
                    f"{field} is invalid: {job['job_id']}"
                )
        attempts = job["generation_attempts"]
        expected_attempt_count = (
            1 + int(config["generation"]["max_regenerations"])
        )
        if len(attempts) != expected_attempt_count:
            raise ContrastiveError(
                f"Generation-attempt count is invalid: {job['job_id']}"
            )
        if [
            int(attempt["attempt_index"]) for attempt in attempts
        ] != list(range(expected_attempt_count)):
            raise ContrastiveError(
                f"Generation-attempt indices are not contiguous: "
                f"{job['job_id']}"
            )
        attempt_seeds = [int(attempt["seed"]) for attempt in attempts]
        if len(attempt_seeds) != len(set(attempt_seeds)):
            raise ContrastiveError(
                f"Generation-attempt seeds are not unique: {job['job_id']}"
            )
        expected_primary_generation = ""
        for attempt in attempts:
            attempt_index = int(attempt["attempt_index"])
            expected_attempt_seed = (
                expected_derivation_seed
                if attempt_index == 0
                else stable_seed(
                    int(config["randomization"]["seed"]),
                    *identity,
                    "regeneration",
                    str(attempt_index),
                )
            )
            if int(attempt["seed"]) != expected_attempt_seed:
                raise ContrastiveError(
                    f"Generation-attempt seed is invalid: "
                    f"{job['job_id']}/{attempt_index}"
                )
            expected_attempt_id = (
                f"{job['job_id']}_a{attempt_index:02d}"
            )
            if attempt["attempt_id"] != expected_attempt_id:
                raise ContrastiveError(
                    f"Generation-attempt identifier is invalid: "
                    f"{job['job_id']}/{attempt_index}"
                )
            request_path = resolve_local_path(attempt["request_path"])
            try:
                request_path.relative_to(
                    (resolved_run_dir / "requests").resolve()
                )
            except ValueError as exc:
                raise ContrastiveError(
                    "Request path escapes the run request directory: "
                    f"{request_path}"
                ) from exc
            expected_request_path = local_reference(
                resolved_run_dir
                / "requests"
                / f"{expected_attempt_id}.json"
            )
            if attempt["request_path"] != expected_request_path:
                raise ContrastiveError(
                    f"Generation-attempt request path is invalid: "
                    f"{expected_attempt_id}"
                )
            request = load_json(request_path)
            _validate(
                "model_request.schema.json",
                request,
                f"stored request {expected_attempt_id}",
            )
            request_hash = hash_object(request)
            if request_hash != attempt["request_sha256"]:
                raise ContrastiveError(
                    f"Stored request was modified: {expected_attempt_id}"
                )
            audit = audit_model_request(
                request,
                spec,
                bundle.cue_lexicon,
                topic,
                int(config["generation"]["context_overhead_tokens"]),
            )
            if audit != attempt["preflight"]:
                raise ContrastiveError(
                    f"Stored request audit is invalid: {expected_attempt_id}"
                )
            if (
                int(audit["context_input_token_upper_bound"])
                + int(config["generation"]["max_output_tokens"])
                > int(
                    config["provider_profile"]["capabilities"][
                        "max_context_tokens"
                    ]
                )
            ):
                raise ContrastiveError(
                    f"Stored request exceeds provider context: "
                    f"{expected_attempt_id}"
                )
            expected_request = render_model_request(
                resolve_local_path(job["prompt_template"]),
                model_topic(topic),
                spec.model_projection(),
                bundle.scaffolding,
                config["generation"],
                expected_attempt_seed,
            )
            if expected_request != request:
                raise ContrastiveError(
                    f"Stored request cannot be reproduced: "
                    f"{expected_attempt_id}"
                )
            if job["response_schema_sha256"] != hash_object(
                request["response_schema"]
            ):
                raise ContrastiveError(
                    f"Response-schema hash is invalid: {expected_attempt_id}"
                )
            expected_generation = _generation_execution_hash(
                request_hash, provider_execution_hash
            )
            if (
                expected_generation
                != attempt["generation_execution_sha256"]
            ):
                raise ContrastiveError(
                    f"Generation execution hash is invalid: "
                    f"{expected_attempt_id}"
                )
            output_root = resolve_local_path(config["paths"]["output"])
            expected_candidate_path = local_reference(
                output_root
                / plan["run_id"]
                / "attempts"
                / f"{expected_attempt_id}.json"
            )
            if (
                attempt["candidate_output_path"]
                != expected_candidate_path
            ):
                raise ContrastiveError(
                    f"Candidate-output path is invalid: {expected_attempt_id}"
                )
            verified_hashes[expected_attempt_id] = request_hash
            verified_generation[expected_attempt_id] = expected_generation
            if attempt_index == 0:
                expected_primary_generation = expected_generation
        expected_job_id = "contrastive_content_" + sha256_text(
            "\x1f".join([*identity, expected_primary_generation])
        )[:20]
        if job["job_id"] != expected_job_id:
            raise ContrastiveError(
                f"Stored job identifier is invalid: {job['job_id']}"
            )
        if (
            job["container_catalog_sha256"]
            != catalog.container_catalog_sha256()
        ):
            raise ContrastiveError(
                f"Container-catalog hash is invalid: {job['job_id']}"
            )
        output_root = resolve_local_path(config["paths"]["output"])
        expected_derived, expected_derivation = _derived_containers(
            spec,
            expected_derivation_seed,
            output_root,
            plan["run_id"],
            str(job["topic_id"]),
            int(job["sample_index"]),
            renderer_hash,
            str(job["container_catalog_sha256"]),
        )
        if expected_derived != job["derived_containers"]:
            raise ContrastiveError(
                f"Derived-container records are invalid: {job['job_id']}"
            )
        if expected_derivation != job["derivation_sha256"]:
            raise ContrastiveError(
                f"Derivation hash is invalid: {job['job_id']}"
            )
        content_name = (
            f"{job['topic_id']}_{job['content_spec_id']}_{language}_"
            f"s{int(job['sample_index']):02d}.json"
        )
        expected_content_path = local_reference(
            output_root
            / plan["run_id"]
            / "content"
            / content_name
        )
        if job["content_output_path"] != expected_content_path:
            raise ContrastiveError(
                f"Content output path is invalid: {job['job_id']}"
            )
        verified_derivations[job["job_id"]] = expected_derivation
    if hash_object(verified_hashes) != plan["request_bundle_sha256"]:
        raise ContrastiveError("Request bundle hash is invalid")
    if (
        hash_object(verified_generation)
        != plan["generation_bundle_sha256"]
    ):
        raise ContrastiveError("Generation bundle hash is invalid")
    if (
        hash_object(verified_derivations)
        != plan["derivation_bundle_sha256"]
    ):
        raise ContrastiveError("Derivation bundle hash is invalid")
    expected_counts = _counts_summary(jobs)
    if expected_counts["request_artifacts"] != len(verified_hashes):
        raise ContrastiveError(
            "Verified request count differs from the attempt schedule"
        )
    if plan["counts"] != expected_counts:
        raise ContrastiveError("Stored plan counts differ from the manifest")
    if report["counts"] != expected_counts:
        raise ContrastiveError("Stored report counts differ from the manifest")
    if (
        plan["mode"] != config["mode"]
        or plan["selection"] != config["selection"]
        or plan["grid"] != config["grid"]
    ):
        raise ContrastiveError(
            "Stored plan mode/selection/grid differs from resolved configuration"
        )
    expected_provider_plan = {
        "profile_id": config["provider_profile"]["profile_id"],
        "provider": config["provider_profile"]["provider"],
        "model": config["provider_profile"]["model"],
        "model_revision": config["provider_profile"]["model_revision"],
        "sha256": provider_profile_hash,
        "execution_sha256": provider_execution_hash,
    }
    if plan["provider_profile"] != expected_provider_plan:
        raise ContrastiveError(
            "Stored plan provider summary cannot be reconstructed"
        )
    expected_cost = _cost_summary(jobs, config)
    if plan["cost_estimate"] != expected_cost:
        raise ContrastiveError(
            "Stored plan cost estimate cannot be reconstructed"
        )
    expected_budget = _budget_audit(expected_cost, config)
    if plan["budget_authorization"] != expected_budget:
        raise ContrastiveError(
            "Stored plan budget authorization cannot be reconstructed"
        )
    expected_preflight_summary = {
        "status": report["status"],
        "notifications": len(notifications),
        "report_path": "preflight_report.json",
    }
    if plan["preflight"] != expected_preflight_summary:
        raise ContrastiveError(
            "Stored plan preflight summary cannot be reconstructed"
        )
    all_attempts = [
        attempt
        for job in jobs
        for attempt in job["generation_attempts"]
    ]
    expected_job_audits = {
        "requests_checked": len(all_attempts),
        "placeholder_audit": "passed",
        "cue_audit": "passed",
        "language_audit": "passed",
        "leakage_audit": "passed",
        "context_audit": "passed",
        "maximum_estimated_input_tokens": max(
            int(row["preflight"]["estimated_input_tokens"])
            for row in all_attempts
        ),
        "maximum_context_input_token_upper_bound": max(
            int(row["preflight"]["context_input_token_upper_bound"])
            for row in all_attempts
        ),
    }
    if report["job_audits"] != expected_job_audits:
        raise ContrastiveError(
            "Stored report request-audit summary cannot be reconstructed"
        )
    actual_derived = expected_counts["derived_items"]
    actual_attempts = expected_counts["planned_generation_attempts"]

    return {
        "status": "verified",
        "run_id": plan["run_id"],
        "plan_sha256": plan["plan_sha256"],
        "generation_jobs": len(jobs),
        "requests": len(verified_hashes),
        "planned_generation_attempts": actual_attempts,
        "derived_items": actual_derived,
    }


def prepare_generation_handoff(
    run_dir: str | Path,
) -> VerifiedGenerationHandoff:
    """Globally verify a run before the first call and prepare TOCTOU checks."""
    resolved_run_dir = resolve_local_path(run_dir)
    verification = verify_pre_generation_run(resolved_run_dir)
    plan, plan_file_sha256 = _load_json_snapshot(
        resolved_run_dir / "plan.json"
    )
    config, resolved_config_file_sha256 = _load_json_snapshot(
        resolved_run_dir / "resolved_config.json"
    )
    jobs, manifest_file_sha256 = _read_jsonl_snapshot(
        resolved_run_dir / "manifest.jsonl"
    )
    if (
        plan.get("plan_sha256") != verification["plan_sha256"]
        or _plan_hash(plan) != verification["plan_sha256"]
    ):
        raise ContrastiveError(
            "Immutable plan changed while preparing generation handoff"
        )
    if hash_object(config) != plan["resolved_config_sha256"]:
        raise ContrastiveError(
            "Resolved configuration changed while preparing generation handoff"
        )
    if hash_object(jobs) != plan["manifest_sha256"]:
        raise ContrastiveError(
            "Generation manifest changed while preparing generation handoff"
        )
    provider_audit, _ = _provider_preflight(config)
    if provider_audit["credential_reference"] == "missing":
        raise ContrastiveError(
            "Generation handoff requires the referenced provider credential "
            "environment variable"
        )
    existing_outputs = [
        path
        for job in jobs
        for path in _job_output_paths(job)
        if _path_entry_exists(path)
    ]
    if existing_outputs:
        raise ContrastiveError(
            "Generation handoff refuses existing output paths because explicit "
            f"resume provenance is not implemented: {existing_outputs[:8]}"
        )
    return VerifiedGenerationHandoff(
        resolved_run_dir,
        str(verification["plan_sha256"]),
        plan_file_sha256,
        str(plan["resolved_config_sha256"]),
        str(plan["manifest_sha256"]),
        resolved_config_file_sha256,
        manifest_file_sha256,
        _adapter_execution_profile(config["provider_profile"]),
        hash_object(config["provider_profile"]),
        hash_object(_provider_execution_payload(config["provider_profile"])),
        jobs,
    )
