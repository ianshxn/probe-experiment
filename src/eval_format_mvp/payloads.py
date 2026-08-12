"""Canonical-payload request construction and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .catalogs import Catalogs
from .io import MVPError, PROJECT_ROOT, load_jsonl, write_jsonl
from .schema import validate_against_schema


def validate_payload(row: dict[str, Any], catalogs: Catalogs) -> None:
    payload_id = str(row.get("payload_block_id", "<missing>"))
    validate_against_schema(row, "payload.schema.json", label=payload_id)
    if row["topic_id"] not in catalogs.topic_ids:
        raise MVPError(f"Payload {payload_id} uses unknown topic {row['topic_id']}")
    topic = next(item for item in catalogs.topics if item["topic_id"] == row["topic_id"])
    if row["task_family"] != topic["task_family"]:
        raise MVPError(
            f"Payload {payload_id} task family {row['task_family']} does not match "
            f"topic catalog value {topic['task_family']}"
        )
    answers = [str(row["correct_answer"]), *map(str, row["distractors"])]
    folded = [answer.strip().casefold() for answer in answers]
    if len(set(folded)) != 4:
        raise MVPError(f"Payload {payload_id} answer and distractors must be distinct")


def validate_payloads(rows: list[dict[str, Any]], catalogs: Catalogs) -> None:
    if not rows:
        raise MVPError("Payload file is empty")
    identifiers: set[str] = set()
    for row in rows:
        validate_payload(row, catalogs)
        payload_id = str(row["payload_block_id"])
        if payload_id in identifiers:
            raise MVPError(f"Duplicate payload_block_id {payload_id}")
        identifiers.add(payload_id)


def load_and_validate_payloads(path: Path, catalogs: Catalogs) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    validate_payloads(rows, catalogs)
    return rows


def build_generation_requests(
    catalogs: Catalogs,
    *,
    per_topic: int = 5,
    prompt_path: Path = PROJECT_ROOT / "prompts" / "payload_generation.txt",
) -> list[dict[str, Any]]:
    if per_topic < 1:
        raise MVPError("per_topic must be positive")
    prompt = prompt_path.read_text(encoding="utf-8")
    response_schema = __import__("json").loads(
        (PROJECT_ROOT / "schemas" / "generation_response.schema.json").read_text(
            encoding="utf-8"
        )
    )
    rows: list[dict[str, Any]] = []
    for topic in catalogs.topics:
        for sample_number in range(1, per_topic + 1):
            payload_id = f"pb_{topic['topic_id']}_{sample_number:02d}"
            rows.append(
                {
                    "request_id": f"request::{payload_id}",
                    "payload_block_id": payload_id,
                    "topic_id": topic["topic_id"],
                    "task_family": topic["task_family"],
                    "system": "You create controlled research stimuli and return valid JSON.",
                    "user": prompt.format(
                        topic_display_name=topic["display_name"],
                        topic_id=topic["topic_id"],
                        task_family=topic["task_family"],
                        sample_number=sample_number,
                    ).strip(),
                    "response_schema": response_schema,
                }
            )
    return rows


def write_generation_requests(path: Path, catalogs: Catalogs, per_topic: int) -> int:
    rows = build_generation_requests(catalogs, per_topic=per_topic)
    write_jsonl(path, rows)
    return len(rows)
