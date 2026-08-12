"""Deterministic rendering and atomic no-clobber publication."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .utils import CONTRASTIVE_ROOT, ContrastiveError


SLOT_RE = re.compile(r"\{\{([^{}]+)\}\}")


class PublicationConflict(ContrastiveError):
    """A publication destination already exists and must not be overwritten."""


def render_item(template: str, slot_values: dict[str, Any]) -> str:
    """Substitute the exact declared slots into one pre-rendered container."""
    needed = [
        slot for slot in SLOT_RE.findall(template) if slot not in {"N", "番号"}
    ]
    expected = set(needed)
    actual = set(slot_values)
    if actual != expected:
        raise ContrastiveError(
            f"Slot keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    rendered = SLOT_RE.sub(
        lambda match: str(slot_values[match.group(1)]),
        template,
    )
    if SLOT_RE.search(rendered):
        raise ContrastiveError("Rendered item contains unresolved slots")
    return rendered


def resolve_unaliased_output_path(value: str | Path) -> Path:
    """Resolve an output and reject existing symlink/junction path components."""
    lexical = Path(value)
    candidate = lexical if lexical.is_absolute() else CONTRASTIVE_ROOT / lexical
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
    # Do not pass an already-rooted Windows path back through Path.resolve().
    # Concurrent hard-link publication can make pathlib surface an equivalent
    # ``\\?\`` spelling on a later resolution, which then fails a purely
    # lexical relative-to comparison against the ordinary root spelling.
    return Path(os.path.abspath(candidate))


def _atomic_create_bytes(path: Path, payload: bytes) -> None:
    destination = resolve_unaliased_output_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = resolve_unaliased_output_path(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise PublicationConflict(
                f"Publication destination already exists: {destination}"
            ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_create_text(path: str | Path, value: str) -> None:
    """Atomically publish UTF-8 text iff the destination is absent."""
    _atomic_create_bytes(Path(path), value.encode("utf-8"))


def atomic_create_json(path: str | Path, value: Any) -> None:
    """Atomically publish canonical pretty JSON iff the destination is absent."""
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_create_bytes(Path(path), payload)
