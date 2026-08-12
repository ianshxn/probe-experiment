"""JSON Schema validation for payloads and rendered rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .io import MVPError, PROJECT_ROOT, load_json


def validate_against_schema(
    value: dict[str, Any], schema_name: str, *, label: str
) -> None:
    schema_path = PROJECT_ROOT / "schemas" / schema_name
    validator = Draft202012Validator(load_json(schema_path))
    errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(
            f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        raise MVPError(f"Invalid {label}: {details}")

