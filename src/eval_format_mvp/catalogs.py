"""Load the intentionally small English-only MVP catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import MVPError, PROJECT_ROOT, load_yaml


@dataclass(frozen=True)
class Catalogs:
    topics: tuple[dict[str, Any], ...]
    purpose_pairs: tuple[dict[str, Any], ...]
    formats: dict[str, dict[str, Any]]

    @property
    def topic_ids(self) -> set[str]:
        return {str(row["topic_id"]) for row in self.topics}


def _unique(rows: tuple[dict[str, Any], ...], key: str, label: str) -> None:
    values = [str(row[key]) for row in rows]
    if len(values) != len(set(values)):
        raise MVPError(f"Duplicate {key} in {label}")


def load_catalogs(root: Path = PROJECT_ROOT) -> Catalogs:
    topics_value = load_yaml(root / "catalogs" / "topics.yaml")
    purpose_value = load_yaml(root / "catalogs" / "purpose_frames.yaml")
    formats_value = load_yaml(root / "catalogs" / "formats.yaml")

    topics = tuple(topics_value.get("topics", []))
    purpose_pairs = tuple(purpose_value.get("pairs", []))
    format_rows = tuple(formats_value.get("formats", []))
    if not topics or not purpose_pairs or len(format_rows) != 2:
        raise MVPError("MVP catalogs require topics, purpose pairs, and exactly two formats")
    _unique(topics, "topic_id", "topics")
    _unique(purpose_pairs, "pair_id", "purpose frames")
    _unique(format_rows, "format_variant_id", "formats")

    families = {str(row["family_id"]) for row in purpose_pairs}
    if families != {"ff01", "ff02", "ff03", "ff04", "ff05"}:
        raise MVPError(f"Purpose catalog must cover ff01-ff05, found {sorted(families)}")

    formats = {str(row["format"]): row for row in format_rows}
    if set(formats) != {"benchmark", "casual"}:
        raise MVPError("Format catalog must contain benchmark and casual")

    for row in purpose_pairs:
        if set(row.get("en", {})) != {"evaluation", "deployment"}:
            raise MVPError(f"Purpose pair {row.get('pair_id')} is incomplete")
    return Catalogs(topics=topics, purpose_pairs=purpose_pairs, formats=formats)

