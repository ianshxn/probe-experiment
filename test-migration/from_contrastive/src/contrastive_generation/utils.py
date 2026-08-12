"""Local deterministic I/O and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


CONTRASTIVE_ROOT = Path(__file__).resolve().parents[2]


class ContrastiveError(RuntimeError):
    """A user-actionable contrastive pipeline error."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_object(value: Any) -> str:
    return sha256_text(canonical_json(value))


def resolve_local_path(value: str | Path) -> Path:
    candidate = Path(value)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (CONTRASTIVE_ROOT / candidate).resolve()
    )
    try:
        resolved.relative_to(CONTRASTIVE_ROOT.resolve())
    except ValueError as exc:
        raise ContrastiveError(
            f"Contrastive paths must stay inside {CONTRASTIVE_ROOT}: {value}"
        ) from exc
    return resolved


def local_reference(path: Path) -> str:
    return path.resolve().relative_to(CONTRASTIVE_ROOT.resolve()).as_posix()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContrastiveError(f"Could not read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContrastiveError(f"Expected a YAML mapping in {path}")
    return value


def load_json(path: Path) -> Any:
    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_nonstandard_constant,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ContrastiveError(f"Could not read JSON {path}: {exc}") from exc


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
