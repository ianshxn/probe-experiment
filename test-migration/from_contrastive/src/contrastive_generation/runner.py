"""Resumable generation, cross-verification, rendering, and publication."""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from .allocation import generator_for, verifier_for
from .catalog import ContainerRecord, load_catalog
from .generation_cost import (
    authorize_generation_budget,
    estimate_generation_cost,
)
from .lease import RunLease
from .pre_generation import (
    VerifiedGenerationHandoff,
    _adapter_execution_profile,
    _plan_hash,
    _provider_execution_payload,
    verify_pre_generation_run,
)
from .providers import (
    BaseProvider,
    MalformedProviderResponse,
    ProviderError,
    ProviderResponse,
    ProviderTransportError,
    make_provider,
)
from .publication import (
    PublicationConflict,
    atomic_create_json,
    render_item,
    resolve_unaliased_output_path,
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
    sha256_file,
    sha256_text,
    utc_now,
)
from .verification import (
    deterministic_qc,
    validate_verification_surfaces,
    verify_candidate,
)


Identity = tuple[str, str, str, int]
ProviderFactory = Callable[[str, dict[str, Any]], BaseProvider]
STATE_SCHEMA_PATH = (
    CONTRASTIVE_ROOT / "schemas" / "contrastive_generation_state.schema.json"
)
RUN_SCHEMA_PATH = (
    CONTRASTIVE_ROOT / "schemas" / "contrastive_generation_run.schema.json"
)
RESULT_SCHEMA_PATH = (
    CONTRASTIVE_ROOT
    / "schemas"
    / "contrastive_generation_result.schema.json"
)


@dataclass(frozen=True)
class TwinRuns:
    run_dir_a: Path
    run_dir_b: Path
    config_a: dict[str, Any]
    config_b: dict[str, Any]
    jobs_a: dict[Identity, dict[str, Any]]
    jobs_b: dict[Identity, dict[str, Any]]

    @property
    def identities(self) -> tuple[Identity, ...]:
        return tuple(sorted(self.jobs_a))


def _validate_schema(
    value: dict[str, Any],
    schema_path: Path,
    label: str,
) -> None:
    errors = sorted(
        Draft202012Validator(load_json(schema_path)).iter_errors(value),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors[:12]
        )
        raise ContrastiveError(f"Invalid {label}: {details}")


def load_generation_run_config(
    path: str | Path = "configs/generation_run.yaml",
) -> dict[str, Any]:
    resolved = resolve_local_path(path)
    value = load_yaml(resolved)
    _validate_schema(value, RUN_SCHEMA_PATH, f"generation config {resolved}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ContrastiveError(f"Could not read JSONL {path}: {exc}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ContrastiveError(f"Expected JSON objects in {path}")
    return rows


def _identity(job: dict[str, Any]) -> Identity:
    return (
        str(job["topic_id"]),
        str(job["content_spec_id"]),
        str(job["language"]),
        int(job["sample_index"]),
    )


def _identity_map(
    jobs: list[dict[str, Any]],
    *,
    label: str,
) -> dict[Identity, dict[str, Any]]:
    result: dict[Identity, dict[str, Any]] = {}
    for job in jobs:
        identity = _identity(job)
        if identity in result:
            raise ContrastiveError(f"Duplicate identity in {label}: {identity}")
        result[identity] = job
    return result


def validate_twin_runs(
    run_dir_a: str | Path,
    run_dir_b: str | Path,
    *,
    production: bool = False,
) -> TwinRuns:
    """Verify two preflight runs and enforce an identical non-provider grid."""
    first = resolve_local_path(run_dir_a)
    second = resolve_local_path(run_dir_b)
    verify_pre_generation_run(first)
    if second != first:
        verify_pre_generation_run(second)
    config_a = load_json(first / "resolved_config.json")
    config_b = load_json(second / "resolved_config.json")
    comparable_a = copy.deepcopy(config_a)
    comparable_b = copy.deepcopy(config_b)
    comparable_a.pop("provider_profile", None)
    comparable_b.pop("provider_profile", None)
    if comparable_a != comparable_b:
        raise ContrastiveError(
            "Preflight run configurations are not grid twins; every field "
            "except provider_profile must be identical"
        )
    jobs_a = _identity_map(
        _read_jsonl(first / "manifest.jsonl"),
        label="run A",
    )
    jobs_b = _identity_map(
        _read_jsonl(second / "manifest.jsonl"),
        label="run B",
    )
    if set(jobs_a) != set(jobs_b):
        missing_a = sorted(set(jobs_b) - set(jobs_a))
        missing_b = sorted(set(jobs_a) - set(jobs_b))
        raise ContrastiveError(
            "Preflight manifests do not cover identical identity grids: "
            f"missing_from_a={missing_a[:8]}, missing_from_b={missing_b[:8]}"
        )
    provider_a = config_a["provider_profile"]
    provider_b = config_b["provider_profile"]
    if first == second and str(provider_a["provider"]) != "mock":
        raise ContrastiveError(
            "Single-run self-verification is allowed only for the mock provider"
        )
    if production:
        if first == second:
            raise ContrastiveError(
                "Production generation requires two distinct preflight runs"
            )
        if config_a.get("mode") != "production" or config_b.get("mode") != "production":
            raise ContrastiveError(
                "Production generation requires production-mode preflight runs"
            )
        if hash_object(provider_a) == hash_object(provider_b):
            raise ContrastiveError(
                "Production generation requires distinct author/verifier profiles"
            )
        if "mock" in {provider_a["provider"], provider_b["provider"]}:
            raise ContrastiveError("Production generation forbids mock providers")
    return TwinRuns(first, second, config_a, config_b, jobs_a, jobs_b)


def _resume_handoff(run_dir: Path) -> VerifiedGenerationHandoff:
    """Prepare the locked handoff without its fresh-run-only global output gate."""
    verification = verify_pre_generation_run(run_dir)
    plan_path = run_dir / "plan.json"
    config_path = run_dir / "resolved_config.json"
    manifest_path = run_dir / "manifest.jsonl"
    plan = load_json(plan_path)
    config = load_json(config_path)
    jobs = _read_jsonl(manifest_path)
    if _plan_hash(plan) != verification["plan_sha256"]:
        raise ContrastiveError("Immutable plan changed while preparing handoff")
    if hash_object(config) != plan["resolved_config_sha256"]:
        raise ContrastiveError("Resolved configuration changed while preparing handoff")
    if hash_object(jobs) != plan["manifest_sha256"]:
        raise ContrastiveError("Manifest changed while preparing handoff")
    return VerifiedGenerationHandoff(
        run_dir,
        str(verification["plan_sha256"]),
        sha256_file(plan_path),
        str(plan["resolved_config_sha256"]),
        str(plan["manifest_sha256"]),
        sha256_file(config_path),
        sha256_file(manifest_path),
        _adapter_execution_profile(config["provider_profile"]),
        hash_object(config["provider_profile"]),
        hash_object(_provider_execution_payload(config["provider_profile"])),
        jobs,
    )


class GenerationRunner:
    """Execute one verified pair of pre-generation runs."""

    def __init__(
        self,
        config: dict[str, Any] | str | Path,
        *,
        provider_factory: ProviderFactory = make_provider,
    ) -> None:
        if isinstance(config, (str, Path)):
            config = load_generation_run_config(config)
        else:
            config = copy.deepcopy(config)
            _validate_schema(config, RUN_SCHEMA_PATH, "generation configuration")
        self.config = config
        self.twin = validate_twin_runs(
            config["run_dir_a"],
            config["run_dir_b"],
            production=bool(config["production"]),
        )
        validate_verification_surfaces()
        self.cost_estimate = estimate_generation_cost(self.twin, self.config)
        self.budget_authorization = authorize_generation_budget(
            self.cost_estimate,
            self.config,
            (
                self.twin.config_a["provider_profile"],
                self.twin.config_b["provider_profile"],
            ),
        )
        self.run_dir = self.twin.run_dir_a
        self.state_dir = self.run_dir / "state"
        self.session_dir = self.run_dir / "sessions"
        self.generation_policy_path = self.run_dir / "generation_policy.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.profiles = {
            "generator_a": self.twin.config_a["provider_profile"],
            "generator_b": self.twin.config_b["provider_profile"],
        }
        self.providers = {
            role: provider_factory(role, profile)
            for role, profile in self.profiles.items()
        }
        self.profile_hashes = {
            role: hash_object(profile) for role, profile in self.profiles.items()
        }
        semaphores: dict[str, threading.BoundedSemaphore] = {}
        for role, profile_hash in self.profile_hashes.items():
            semaphores.setdefault(
                profile_hash,
                threading.BoundedSemaphore(
                    int(self.profiles[role]["max_concurrency"])
                ),
            )
        self._provider_semaphores = semaphores
        self._failure_counts: Counter[str] = Counter()
        self._circuits: dict[str, dict[str, Any]] = {}
        self._provider_lock = threading.Lock()
        self._manifest_lock = threading.Lock()
        self._current_job = threading.local()
        self.handoffs = {
            "generator_a": _resume_handoff(self.twin.run_dir_a),
            "generator_b": (
                _resume_handoff(self.twin.run_dir_b)
                if self.twin.run_dir_b != self.twin.run_dir_a
                else None
            ),
        }
        if self.handoffs["generator_b"] is None:
            self.handoffs["generator_b"] = self.handoffs["generator_a"]
        self.catalogs = {
            language: load_catalog(
                self.twin.config_a["paths"]["topics"][language],
                self.twin.config_a["paths"]["containers"],
            )
            for language in ("en", "ja")
            if language
            in {
                identity[2] for identity in self.twin.identities
            }
        }
        self.containers: dict[str, dict[str, ContainerRecord]] = {
            language: {
                container.container_id: container
                for container in catalog.containers
            }
            for language, catalog in self.catalogs.items()
        }
        self.cue_lexicon = load_yaml(
            resolve_local_path(self.twin.config_a["paths"]["cue_lexicon"])
        )

    def _generation_policy(self) -> dict[str, Any]:
        handoff_a = self.handoffs["generator_a"]
        handoff_b = self.handoffs["generator_b"]
        return {
            "schema_version": 1,
            "run_a": {
                "path": local_reference(self.twin.run_dir_a),
                "plan_sha256": str(
                    handoff_a._plan_sha256  # type: ignore[attr-defined]
                ),
            },
            "run_b": {
                "path": local_reference(self.twin.run_dir_b),
                "plan_sha256": str(
                    handoff_b._plan_sha256  # type: ignore[attr-defined]
                ),
            },
            "production": bool(self.config["production"]),
            "verification": copy.deepcopy(self.config["verification"]),
            "cost_estimate": copy.deepcopy(self.cost_estimate),
            "budget_authorization": copy.deepcopy(
                self.budget_authorization
            ),
        }

    def _generation_evidence_exists(self) -> bool:
        return (
            any(self.state_dir.glob("*.json"))
            or any(self.session_dir.iterdir())
            or (self.run_dir / "stimulus_manifest.jsonl").exists()
            or (self.run_dir / "generation_result.json").exists()
        )

    def _bind_generation_policy(self) -> str:
        active = self._generation_policy()
        if self.generation_policy_path.is_file():
            stored = load_json(self.generation_policy_path)
            if stored != active:
                raise ContrastiveError(
                    "Generation policy differs from the policy already bound "
                    f"to {self.generation_policy_path}"
                )
        else:
            if self._generation_evidence_exists():
                raise ContrastiveError(
                    "Generation state or output exists without a bound "
                    "generation_policy.json; use fresh run IDs rather than "
                    "backfilling provenance"
                )
            atomic_create_json(self.generation_policy_path, active)
        return hash_object(active)

    def _jobs_for(
        self,
        identity: Identity,
    ) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
        author_role = generator_for(identity[0], identity[1])
        verifier_role = verifier_for(identity[0], identity[1])
        author = (
            self.twin.jobs_a[identity]
            if author_role == "generator_a"
            else self.twin.jobs_b[identity]
        )
        verifier = (
            self.twin.jobs_a[identity]
            if verifier_role == "generator_a"
            else self.twin.jobs_b[identity]
        )
        return author_role, verifier_role, author, verifier

    def _state_path(self, job_id: str) -> Path:
        return self.state_dir / f"{job_id}.json"

    def _new_state(
        self,
        job: dict[str, Any],
        author_role: str,
        verifier_role: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "job_id": job["job_id"],
            "generator_role": author_role,
            "verifier_role": verifier_role,
            "status": "pending",
            "attempts": [],
            "selected_attempt_index": None,
            "content_output_path": None,
            "content_sha256": None,
            "derived_outputs": [],
            "pending_verification": None,
            "updated_at": utc_now(),
        }

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        _validate_schema(
            state,
            STATE_SCHEMA_PATH,
            f"state for {state['job_id']}",
        )
        atomic_write_json(self._state_path(str(state["job_id"])), state)

    def _load_state(
        self,
        job: dict[str, Any],
        author_role: str,
        verifier_role: str,
    ) -> dict[str, Any]:
        path = self._state_path(str(job["job_id"]))
        if not path.is_file():
            return self._new_state(job, author_role, verifier_role)
        state = load_json(path)
        _validate_schema(state, STATE_SCHEMA_PATH, f"state {path}")
        if (
            state["job_id"] != job["job_id"]
            or state["generator_role"] != author_role
            or state["verifier_role"] != verifier_role
        ):
            raise ContrastiveError(f"State identity disagrees with job: {path}")
        seen = [
            int(attempt["attempt_index"]) for attempt in state["attempts"]
        ]
        if seen != list(range(len(seen))):
            raise ContrastiveError(
                f"State attempts are not a contiguous prefix for {job['job_id']}"
            )
        for attempt in state["attempts"]:
            candidate = resolve_unaliased_output_path(attempt["candidate_path"])
            if (
                not candidate.is_file()
                or sha256_file(candidate) != attempt["candidate_sha256"]
            ):
                raise ContrastiveError(
                    f"Recorded candidate output changed for {job['job_id']}"
                )
            sessions = [
                str(value) for value in attempt["session_ids"]
            ]
            if (
                set(sessions) != set(attempt["session_sha256s"])
                or self._session_sha256s(sessions)
                != attempt["session_sha256s"]
            ):
                raise ContrastiveError(
                    f"Recorded provider sessions changed for {job['job_id']}"
                )
        pending = state["pending_verification"]
        if pending is not None:
            if state["status"] != "verification_pending":
                raise ContrastiveError(
                    f"Job {job['job_id']} has pending verification under "
                    f"incompatible status {state['status']!r}"
                )
            if int(pending["attempt_index"]) != len(state["attempts"]):
                raise ContrastiveError(
                    f"Job {job['job_id']} pending-verification index is not the "
                    "next contiguous attempt"
                )
            candidate = resolve_unaliased_output_path(pending["candidate_path"])
            if (
                not candidate.is_file()
                or sha256_file(candidate) != pending["candidate_sha256"]
            ):
                raise ContrastiveError(
                    f"Pending-verification candidate changed for {job['job_id']}"
                )
            completed_pairs = [
                str(row["pair_id"])
                for row in pending["completed_verifications"]
            ]
            expected_pairs = {
                str(row["pair_id"])
                for row in job["derived_containers"]
            }
            if (
                len(completed_pairs) != len(set(completed_pairs))
                or not set(completed_pairs) <= expected_pairs
            ):
                raise ContrastiveError(
                    f"Pending-verification progress is invalid for "
                    f"{job['job_id']}"
                )
            pending_sessions = [
                str(value) for value in pending["session_ids"]
            ]
            if (
                set(pending_sessions) != set(
                    pending["session_sha256s"]
                )
                or self._session_sha256s(pending_sessions)
                != pending["session_sha256s"]
            ):
                raise ContrastiveError(
                    f"Pending provider sessions changed for {job['job_id']}"
                )
        elif state["status"] == "verification_pending":
            raise ContrastiveError(
                f"Job {job['job_id']} is verification_pending without a candidate"
            )
        if state["status"] == "accepted":
            content = resolve_unaliased_output_path(state["content_output_path"])
            if (
                not content.is_file()
                or sha256_file(content) != state["content_sha256"]
            ):
                raise ContrastiveError(
                    f"Accepted content output changed for {job['job_id']}"
                )
            for derived in state["derived_outputs"]:
                output = resolve_unaliased_output_path(derived["output_path"])
                if (
                    not output.is_file()
                    or sha256_file(output) != derived["output_sha256"]
                ):
                    raise ContrastiveError(
                        f"Accepted derived output changed for {job['job_id']}"
                    )
        return state

    def _set_handoff_resume(
        self,
        role: str,
        job_id: str,
        next_attempt: int,
    ) -> None:
        handoff = self.handoffs[role]
        # The locked handoff is intentionally fresh-run-only at its public
        # boundary. Its internal next-index guard is advanced only after the
        # persisted prefix above has been hash-validated.
        with handoff._release_lock:  # type: ignore[attr-defined]
            handoff._next_attempt[job_id] = next_attempt  # type: ignore[attr-defined]

    def _session_id(
        self,
        job: dict[str, Any],
        attempt_index: int,
        role: str,
    ) -> str:
        safe_role = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in role
        )
        return (
            f"{job['job_id']}__a{attempt_index:02d}__{safe_role}__"
            f"{uuid.uuid4().hex[:12]}"
        )

    def _record_provider_success(self, profile_hash: str) -> None:
        with self._provider_lock:
            self._failure_counts[profile_hash] = 0
            self._circuits.pop(profile_hash, None)

    def _record_provider_failure(
        self,
        profile_hash: str,
        provider: BaseProvider,
        exc: ProviderError,
    ) -> None:
        if not isinstance(exc, ProviderTransportError):
            # Completed but malformed model output is a scientific attempt
            # outcome, not evidence that the provider transport is unavailable.
            # It also interrupts a sequence of transport failures.
            self._record_provider_success(profile_hash)
            return
        threshold = int(provider.config.get("circuit_breaker_failures", 0) or 0)
        if threshold <= 0:
            return
        with self._provider_lock:
            self._failure_counts[profile_hash] += 1
            count = self._failure_counts[profile_hash]
            if count >= threshold:
                self._circuits[profile_hash] = {
                    "opened_monotonic": time.monotonic(),
                    "half_open": False,
                    "reason": (
                        f"{count} consecutive provider failures; "
                        f"last error: {exc}"
                    ),
                }

    def _call_provider(
        self,
        provider: BaseProvider,
        system: str,
        user: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        seed: int,
        response_schema: dict[str, Any],
        role: str,
    ) -> tuple[ProviderResponse, str]:
        job = self._current_job.job
        attempt_index = self._current_job.attempt_index
        session_id = self._session_id(job, attempt_index, role)
        directory = self.session_dir / session_id
        directory.mkdir(parents=True, exist_ok=False)
        provider_role = next(
            name for name, candidate in self.providers.items() if candidate is provider
        )
        profile_hash = self.profile_hashes[provider_role]
        meta = {
            "session_id": session_id,
            "job_id": job["job_id"],
            "attempt_index": attempt_index,
            "call_role": role,
            "provider_role": provider_role,
            "provider_profile_sha256": profile_hash,
            "language": job["language"],
            "created_at": utc_now(),
        }
        request_record = {
            "system": system,
            "user": user,
            "generation": {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_output_tokens,
                "seed": seed,
            },
            "response_schema": response_schema,
        }
        atomic_write_json(directory / "meta.json", meta)
        atomic_write_json(directory / "request.json", request_record)
        blocked_reason: str | None = None
        with self._provider_lock:
            circuit = self._circuits.get(profile_hash)
            if circuit is not None:
                cooldown = float(
                    provider.config.get(
                        "circuit_breaker_cooldown_seconds",
                        30,
                    )
                )
                elapsed = time.monotonic() - float(
                    circuit["opened_monotonic"]
                )
                if elapsed < cooldown:
                    blocked_reason = (
                        f"{circuit['reason']}; retry after "
                        f"{max(0.0, cooldown - elapsed):.3f}s"
                    )
                elif bool(circuit["half_open"]):
                    blocked_reason = (
                        f"{circuit['reason']}; a half-open recovery call "
                        "is already in progress"
                    )
                else:
                    circuit["half_open"] = True
        if blocked_reason is not None:
            error = ProviderTransportError(
                "Provider circuit is open; no call was made: "
                + blocked_reason
            )
            atomic_write_json(
                directory / "error.json",
                {"type": type(error).__name__, "message": str(error)},
            )
            raise error
        try:
            with self._provider_semaphores[profile_hash]:
                response = provider.generate(
                    system,
                    user,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_output_tokens,
                    seed=seed,
                    response_schema=response_schema,
                )
        except ProviderError as exc:
            self._record_provider_failure(profile_hash, provider, exc)
            setattr(exc, "session_id", session_id)
            atomic_write_json(
                directory / "error.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "request_payload": exc.request_payload,
                    "response_payload": exc.response_payload,
                    "response_headers": exc.response_headers,
                },
            )
            raise
        self._record_provider_success(profile_hash)
        atomic_write_json(
            directory / "response.json",
            {
                "text": response.text,
                "request_id": response.request_id,
                "requested_model": response.requested_model,
                "resolved_model": response.resolved_model,
                "route": response.route,
                "usage": response.usage,
                "request_payload": response.request_payload,
                "response_payload": response.response_payload,
            },
        )
        atomic_write_json(
            directory / "summary.json",
            {
                "status": "completed",
                "response_text_sha256": sha256_text(response.text),
                "completed_at": utc_now(),
            },
        )
        return response, session_id

    def _candidate_artifact(
        self,
        *,
        job: dict[str, Any],
        envelope: dict[str, Any],
        response: ProviderResponse | None,
        session_id: str,
        parsed: dict[str, Any] | None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "job_id": job["job_id"],
            "attempt_id": envelope["attempt_id"],
            "attempt_index": envelope["attempt_index"],
            "request_sha256": envelope["request_sha256"],
            "session_id": session_id,
            "provider": (
                {
                    "request_id": response.request_id,
                    "requested_model": response.requested_model,
                    "resolved_model": response.resolved_model,
                    "route": response.route,
                    "usage": response.usage,
                }
                if response is not None
                else None
            ),
            "raw_text": response.text if response is not None else "",
            "parsed_output": parsed,
            "error": error,
            "created_at": utc_now(),
        }

    def _session_sha256s(
        self,
        session_ids: list[str],
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        for session_id in session_ids:
            directory = self.session_dir / session_id
            if not directory.is_dir():
                raise ContrastiveError(
                    f"Provider session directory is missing: {session_id}"
                )
            files = sorted(
                path for path in directory.iterdir() if path.is_file()
            )
            if not files:
                raise ContrastiveError(
                    f"Provider session is missing audit files: {session_id}"
                )
            result[session_id] = hash_object(
                {
                    path.name: sha256_file(path)
                    for path in files
                }
            )
        return result

    def _publish_candidate(
        self,
        attempt_plan: dict[str, Any],
        artifact: dict[str, Any],
    ) -> tuple[str, str]:
        path = resolve_unaliased_output_path(
            attempt_plan["candidate_output_path"]
        )
        atomic_create_json(path, artifact)
        return str(path), sha256_file(path)

    def _render_outputs(
        self,
        job: dict[str, Any],
        slot_values: dict[str, str],
        answer_key: str,
        content_sha256: str,
    ) -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
        language = str(job["language"])
        result: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
        for derived in job["derived_containers"]:
            container = self.containers[language][derived["container_id"]]
            template, _ = prerender_template(
                container,
                int(job["derivation_seed"]),
            )
            if sha256_text(template) != derived["rendered_template_sha256"]:
                raise ContrastiveError(
                    f"Container drift since planning for {job['job_id']}/"
                    f"{derived['container_id']}"
                )
            text = render_item(template, slot_values)
            payload = {
                "schema_version": 1,
                "job_id": job["job_id"],
                "topic_id": job["topic_id"],
                "content_spec_id": job["content_spec_id"],
                "pair_id": derived["pair_id"],
                "frame_family_id": derived["frame_family_id"],
                "explicitness": derived["explicitness"],
                "surface": derived["surface"],
                "language": language,
                "sample_index": job["sample_index"],
                "container_id": derived["container_id"],
                "class": derived["class"],
                "text": text,
                "answer_key": answer_key,
                "content_sha256": content_sha256,
                "derivation_sha256": job["derivation_sha256"],
            }
            path = resolve_unaliased_output_path(derived["output_path"])
            result.append((derived, payload, path))
        return result

    @staticmethod
    def _assert_absent(paths: list[Path]) -> None:
        conflicts = [path for path in paths if os.path.lexists(path)]
        if conflicts:
            raise PublicationConflict(
                f"Publication destinations already exist: {conflicts}"
            )

    def _publish_accepted(
        self,
        *,
        job: dict[str, Any],
        state: dict[str, Any],
        attempt_index: int,
        payload: dict[str, Any],
        verification: dict[str, Any],
        candidate_path: str,
        candidate_sha256: str,
    ) -> None:
        from .verification import content_fields

        slot_values, answer_key, _, _ = content_fields(
            payload,
            job_request_schema(job, attempt_index),
        )
        content_path = resolve_unaliased_output_path(job["content_output_path"])
        content_payload = {
            "schema_version": 1,
            "job_id": job["job_id"],
            "topic_id": job["topic_id"],
            "topic_label": job["topic_label"],
            "domain": job["domain"],
            "content_spec_id": job["content_spec_id"],
            "language": job["language"],
            "sample_index": job["sample_index"],
            "selected_attempt_index": attempt_index,
            "candidate_path": candidate_path,
            "candidate_sha256": candidate_sha256,
            "slot_values": slot_values,
            "answer_key": answer_key,
            "verification": verification,
        }
        rendered_preview = self._render_outputs(
            job,
            slot_values,
            answer_key,
            "<pending>",
        )
        self._assert_absent(
            [content_path, *[row[2] for row in rendered_preview]]
        )
        atomic_create_json(content_path, content_payload)
        content_hash = sha256_file(content_path)
        rendered = self._render_outputs(
            job,
            slot_values,
            answer_key,
            content_hash,
        )
        derived_state: list[dict[str, Any]] = []
        for derived, rendered_payload, path in rendered:
            atomic_create_json(path, rendered_payload)
            derived_state.append(
                {
                    "container_id": derived["container_id"],
                    "output_path": str(path),
                    "output_sha256": sha256_file(path),
                }
            )
        state.update(
            {
                "status": "accepted",
                "selected_attempt_index": attempt_index,
                "content_output_path": str(content_path),
                "content_sha256": content_hash,
                "derived_outputs": derived_state,
            }
        )
        self._save_state(state)

    def _complete_pending_verification(
        self,
        *,
        job: dict[str, Any],
        state: dict[str, Any],
        verifier_role: str,
    ) -> str:
        """Resume or complete verification without regenerating author content."""
        pending = state["pending_verification"]
        if pending is None:
            raise ContrastiveError(
                f"Job {job['job_id']} has no pending verification candidate"
            )
        attempt_index = int(pending["attempt_index"])
        attempt_plan = job["generation_attempts"][attempt_index]
        candidate_path = resolve_unaliased_output_path(
            pending["candidate_path"]
        )
        candidate = load_json(candidate_path)
        parsed = candidate.get("parsed_output")
        if not isinstance(parsed, dict):
            raise ContrastiveError(
                f"Pending-verification candidate is not parsed for {job['job_id']}"
            )
        response_schema = job_request_schema(job, attempt_index)
        author_request = load_json(
            resolve_local_path(attempt_plan["request_path"])
        )
        self._current_job.job = job
        self._current_job.attempt_index = attempt_index
        completed = pending["completed_verifications"]
        completed_pairs = {
            str(row["pair_id"]) for row in completed
        }
        for pair_id, rendered_item in self._verification_renders(
            job,
            parsed,
            response_schema,
        ):
            if pair_id in completed_pairs:
                continue
            try:
                judgment, verifier_sessions = verify_candidate(
                    provider=self.providers[verifier_role],
                    rendered_item=rendered_item,
                    authoring_requirements=str(author_request["user"]),
                    language=str(job["language"]),
                    response_schema=response_schema,
                    generated_payload=parsed,
                    generation_seed=(
                        int(attempt_plan["seed"])
                        + int(pair_id[1:]) * 100_003
                    ),
                    config=self.config["verification"],
                    call_provider=self._call_provider,
                )
            except ProviderTransportError as exc:
                state["last_error"] = (
                    "Verification infrastructure failure; author candidate "
                    f"and completed pair checks are preserved for resume: {exc}"
                )
                self._save_state(state)
                raise
            completed.append(
                {
                    "pair_id": pair_id,
                    "purpose_frame_masked": True,
                    "judgment": judgment,
                }
            )
            pending["session_ids"].extend(verifier_sessions)
            pending["session_sha256s"] = self._session_sha256s(
                [str(value) for value in pending["session_ids"]]
            )
            completed_pairs.add(pair_id)
            self._save_state(state)

        verification = {
            "pass": all(
                row["judgment"].get("pass") for row in completed
            ),
            "purpose_frame_masked": True,
            "pair_results": list(completed),
        }
        sessions = [str(value) for value in pending["session_ids"]]
        outcome = (
            "accepted" if verification.get("pass") else "verification_rejected"
        )
        state["attempts"].append(
            {
                "attempt_index": attempt_index,
                "outcome": outcome,
                "session_ids": sessions,
                "session_sha256s": self._session_sha256s(sessions),
                "candidate_path": str(pending["candidate_path"]),
                "candidate_sha256": str(pending["candidate_sha256"]),
                "deterministic_qc": pending["deterministic_qc"],
                "verification": verification,
                "provider": pending["provider"],
            }
        )
        state["pending_verification"] = None
        state.pop("last_error", None)
        if outcome == "accepted":
            self._publish_accepted(
                job=job,
                state=state,
                attempt_index=attempt_index,
                payload=parsed,
                verification=verification,
                candidate_path=str(pending["candidate_path"]),
                candidate_sha256=str(pending["candidate_sha256"]),
            )
            return "accepted"
        state["status"] = "verification_rejected_advance"
        self._save_state(state)
        return "verification_rejected"

    def _process_job(self, identity: Identity) -> dict[str, Any]:
        author_role, verifier_role, job, _ = self._jobs_for(identity)
        state = self._load_state(job, author_role, verifier_role)
        if state["status"] in {"accepted", "exhausted"}:
            return state
        if state["pending_verification"] is not None:
            pending_index = int(
                state["pending_verification"]["attempt_index"]
            )
            self._set_handoff_resume(
                author_role,
                str(job["job_id"]),
                pending_index + 1,
            )
            if (
                self._complete_pending_verification(
                    job=job,
                    state=state,
                    verifier_role=verifier_role,
                )
                == "accepted"
            ):
                return state
        next_attempt = len(state["attempts"])
        self._assert_absent(
            [
                resolve_unaliased_output_path(job["content_output_path"]),
                *[
                    resolve_unaliased_output_path(row["output_path"])
                    for row in job["derived_containers"]
                ],
                *[
                    resolve_unaliased_output_path(row["candidate_output_path"])
                    for row in job["generation_attempts"]
                    if int(row["attempt_index"]) >= next_attempt
                ],
            ]
        )
        self._set_handoff_resume(author_role, str(job["job_id"]), next_attempt)
        handoff = self.handoffs[author_role]
        attempts = job["generation_attempts"]
        for attempt_index in range(next_attempt, len(attempts)):
            state["status"] = "pending"
            state.pop("last_error", None)
            self._save_state(state)
            envelope = handoff.submission_for(job["job_id"], attempt_index)
            request = envelope["request"]
            attempt_plan = attempts[attempt_index]
            self._current_job.job = job
            self._current_job.attempt_index = attempt_index
            author_session = ""
            try:
                author_response, author_session = self._call_provider(
                    self.providers[author_role],
                    str(request["system"]),
                    str(request["user"]),
                    float(request["generation"]["temperature"]),
                    float(request["generation"]["top_p"]),
                    int(request["generation"]["max_output_tokens"]),
                    int(request["generation"]["seed"]),
                    request["response_schema"],
                    "author",
                )
            except ProviderTransportError as exc:
                state["last_error"] = str(exc)
                self._save_state(state)
                raise
            except MalformedProviderResponse as exc:
                error_session = next(
                    (
                        path.name
                        for path in sorted(
                            self.session_dir.glob(
                                f"{job['job_id']}__a{attempt_index:02d}__author__*"
                            )
                        )
                    ),
                    "provider-malformed",
                )
                artifact = self._candidate_artifact(
                    job=job,
                    envelope=envelope,
                    response=None,
                    session_id=error_session,
                    parsed=None,
                    error=str(exc),
                )
                candidate_path, candidate_hash = self._publish_candidate(
                    attempt_plan,
                    artifact,
                )
                state["attempts"].append(
                    {
                        "attempt_index": attempt_index,
                        "outcome": "malformed",
                        "session_ids": [error_session],
                        "session_sha256s": self._session_sha256s(
                            [error_session]
                        ),
                        "candidate_path": candidate_path,
                        "candidate_sha256": candidate_hash,
                        "deterministic_qc": None,
                        "verification": None,
                        "provider": None,
                    }
                )
                state["status"] = "malformed_advance"
                self._save_state(state)
                continue
            qc, parsed = deterministic_qc(
                author_response.text,
                request["response_schema"],
                str(job["language"]),
                self.cue_lexicon,
                japanese_surface=self.config["verification"]["japanese_surface"],
            )
            artifact = self._candidate_artifact(
                job=job,
                envelope=envelope,
                response=author_response,
                session_id=author_session,
                parsed=parsed,
            )
            candidate_path, candidate_hash = self._publish_candidate(
                attempt_plan,
                artifact,
            )
            if parsed is None:
                outcome = "malformed"
                status = "malformed_advance"
                verification = None
                sessions = [author_session]
            elif not qc["pass"]:
                outcome = "qc_rejected"
                status = "qc_rejected_advance"
                verification = None
                sessions = [author_session]
            else:
                state["status"] = "verification_pending"
                state["pending_verification"] = {
                    "attempt_index": attempt_index,
                    "session_ids": [author_session],
                    "session_sha256s": self._session_sha256s(
                        [author_session]
                    ),
                    "candidate_path": candidate_path,
                    "candidate_sha256": candidate_hash,
                    "deterministic_qc": qc,
                    "provider": {
                        "request_id": author_response.request_id,
                        "requested_model": author_response.requested_model,
                        "resolved_model": author_response.resolved_model,
                        "route": author_response.route,
                        "usage": author_response.usage,
                    },
                    "completed_verifications": [],
                }
                self._save_state(state)
                if (
                    self._complete_pending_verification(
                        job=job,
                        state=state,
                        verifier_role=verifier_role,
                    )
                    == "accepted"
                ):
                    return state
                continue
            attempt_state = {
                "attempt_index": attempt_index,
                "outcome": outcome,
                "session_ids": sessions,
                "session_sha256s": self._session_sha256s(sessions),
                "candidate_path": candidate_path,
                "candidate_sha256": candidate_hash,
                "deterministic_qc": qc,
                "verification": verification,
                "provider": {
                    "request_id": author_response.request_id,
                    "requested_model": author_response.requested_model,
                    "resolved_model": author_response.resolved_model,
                    "route": author_response.route,
                    "usage": author_response.usage,
                },
            }
            state["attempts"].append(attempt_state)
            state["status"] = status
            if outcome == "accepted":
                self._publish_accepted(
                    job=job,
                    state=state,
                    attempt_index=attempt_index,
                    payload=parsed,
                    verification=verification,
                    candidate_path=candidate_path,
                    candidate_sha256=candidate_hash,
                )
                return state
            self._save_state(state)
        state["status"] = "exhausted"
        self._save_state(state)
        return state

    def _verification_renders(
        self,
        job: dict[str, Any],
        payload: dict[str, Any],
        response_schema: dict[str, Any],
    ) -> list[tuple[str, str]]:
        from .verification import content_fields

        slot_values, _, _, _ = content_fields(payload, response_schema)
        representatives: dict[str, dict[str, Any]] = {}
        for row in sorted(
            job["derived_containers"],
            key=lambda value: value["container_id"],
        ):
            representatives.setdefault(str(row["pair_id"]), row)
        result: list[tuple[str, str]] = []
        for pair_id, row in sorted(representatives.items()):
            container = self.containers[str(job["language"])][
                row["container_id"]
            ]
            body = str(container.data["body_template"])
            if body.count("{{PURPOSE_FRAME}}") != 1:
                raise ContrastiveError(
                    f"Container drift before verification for {job['job_id']}"
                )
            purpose_masked = body.replace("{{PURPOSE_FRAME}}", "").strip()
            result.append(
                (pair_id, render_item(purpose_masked, slot_values))
            )
        if len(result) != 2:
            raise ContrastiveError(
                f"Job {job['job_id']} must expose two pair-shared bodies "
                "for verification"
            )
        return result

    def _write_manifest(self, states: list[dict[str, Any]]) -> None:
        rows = []
        for state in sorted(states, key=lambda row: row["job_id"]):
            if state["status"] != "accepted":
                continue
            rows.append(
                {
                    "job_id": state["job_id"],
                    "generator_role": state["generator_role"],
                    "verifier_role": state["verifier_role"],
                    "selected_attempt_index": state["selected_attempt_index"],
                    "content_output_path": state["content_output_path"],
                    "content_sha256": state["content_sha256"],
                    "derived_outputs": state["derived_outputs"],
                }
            )
        path = self.run_dir / "stimulus_manifest.jsonl"
        with self._manifest_lock:
            if path.is_file():
                existing = _read_jsonl(path)
                if existing != rows:
                    raise ContrastiveError(
                        "Existing stimulus manifest differs from reconstructed "
                        "accepted states"
                    )
                return
            atomic_write_jsonl(path, rows)

    def run(self) -> dict[str, Any]:
        """Acquire the pair lease, process all identities, and publish a manifest."""
        lease = RunLease(
            self.twin.run_dir_a,
            self.twin.run_dir_b,
            timeout_seconds=float(self.config["lease_timeout_seconds"]),
        )
        states: list[dict[str, Any]] = []
        errors: list[BaseException] = []
        with lease:
            generation_policy_sha256 = self._bind_generation_policy()
            for provider in {
                id(provider): provider for provider in self.providers.values()
            }.values():
                provider.check_credentials()
            with ThreadPoolExecutor(
                max_workers=int(self.config["max_workers"])
            ) as executor:
                futures = {
                    executor.submit(self._process_job, identity): identity
                    for identity in self.twin.identities
                }
                for future in as_completed(futures):
                    try:
                        states.append(future.result())
                    except BaseException as exc:
                        errors.append(exc)
            if errors:
                raise errors[0]
            self._write_manifest(states)
            stimulus_manifest_path = (
                self.run_dir / "stimulus_manifest.jsonl"
            )
            state_bundle = [
                {
                    "job_id": str(state["job_id"]),
                    "sha256": sha256_file(
                        self._state_path(str(state["job_id"]))
                    ),
                }
                for state in sorted(
                    states,
                    key=lambda row: str(row["job_id"]),
                )
            ]
            counts = Counter(state["status"] for state in states)
            result = {
                "schema_version": 1,
                "status": (
                    "completed"
                    if len(states) == len(self.twin.identities)
                    else "partial"
                ),
                "run_dir_a": str(self.twin.run_dir_a),
                "run_dir_b": str(self.twin.run_dir_b),
                "jobs": len(states),
                "counts": dict(sorted(counts.items())),
                "stimulus_manifest": str(stimulus_manifest_path),
                "stimulus_manifest_sha256": sha256_file(
                    stimulus_manifest_path
                ),
                "state_bundle_sha256": hash_object(state_bundle),
                "generation_policy": local_reference(
                    self.generation_policy_path
                ),
                "generation_policy_sha256": generation_policy_sha256,
                "cost_estimate": self.cost_estimate,
                "budget_authorization": self.budget_authorization,
                "completed_at": utc_now(),
            }
            _validate_schema(
                result,
                RESULT_SCHEMA_PATH,
                "generation result",
            )
            result_path = self.run_dir / "generation_result.json"
            if result_path.is_file():
                stored = load_json(result_path)
                _validate_schema(
                    stored,
                    RESULT_SCHEMA_PATH,
                    "stored generation result",
                )
                comparable_result = {
                    key: value
                    for key, value in result.items()
                    if key != "completed_at"
                }
                comparable_stored = {
                    key: value
                    for key, value in stored.items()
                    if key != "completed_at"
                }
                if comparable_stored != comparable_result:
                    raise ContrastiveError(
                        "Existing generation result differs from reconstructed "
                        "terminal state"
                    )
                return stored
            atomic_create_json(result_path, result)
            return result


def job_request_schema(
    job: dict[str, Any],
    attempt_index: int,
) -> dict[str, Any]:
    attempt = next(
        row
        for row in job["generation_attempts"]
        if int(row["attempt_index"]) == attempt_index
    )
    request = load_json(resolve_local_path(attempt["request_path"]))
    return request["response_schema"]
