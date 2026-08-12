"""Small deterministic helpers shared by v2 pre-generation planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .utils import (
    CONTRASTIVE_ROOT,
    ContrastiveError,
    load_json,
    load_yaml,
    resolve_local_path,
    sha256_text,
)


def _validate_provider(value: dict[str, Any], label: str) -> None:
    schema = load_json(CONTRASTIVE_ROOT / "schemas" / "provider.schema.json")
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


def load_provider_profile(path: str | Path) -> dict[str, Any]:
    """Load a provider profile without resolving or persisting credentials."""
    resolved = resolve_local_path(path)
    profile = load_yaml(resolved)
    _validate_provider(profile, f"provider profile {resolved}")
    placeholders = [
        value
        for value in profile.values()
        if isinstance(value, str) and value.startswith("<") and value.endswith(">")
    ]
    if placeholders:
        raise ContrastiveError(
            f"Provider profile {resolved} still contains example placeholders"
        )
    return profile


def stable_seed(base_seed: int, *parts: str) -> int:
    """Derive one stable non-negative 31-bit seed from an identity tuple."""
    digest = sha256_text("\x1f".join([str(base_seed), *parts]))
    return int(digest[:8], 16) & 0x7FFFFFFF
