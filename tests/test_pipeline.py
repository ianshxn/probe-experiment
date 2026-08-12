from __future__ import annotations

import copy
from pathlib import Path

import pytest

from eval_format_mvp.catalogs import load_catalogs
from eval_format_mvp.io import MVPError, load_jsonl
from eval_format_mvp.payloads import build_generation_requests, validate_payloads
from eval_format_mvp.render import render_dataset
from eval_format_mvp.validate import validate_rendered_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_catalogs_and_request_grid() -> None:
    catalogs = load_catalogs(ROOT)
    assert len(catalogs.topics) == 16
    assert len(catalogs.purpose_pairs) == 10
    requests = build_generation_requests(catalogs, per_topic=5)
    assert len(requests) == 80
    assert len({row["payload_block_id"] for row in requests}) == 80


def test_topical_words_are_not_rejected_by_structural_validation() -> None:
    catalogs = load_catalogs(ROOT)
    payload = {
        "schema_version": 1,
        "payload_block_id": "pb_t001_99",
        "topic_id": "t001",
        "task_family": "classification",
        "context": (
            "A reviewer examines a benchmark program during code review before "
            "deployment and notices that it produces the wrong result."
        ),
        "question": "What kind of defect makes a program finish with the wrong result?",
        "correct_answer": "Logic error",
        "distractors": ["Syntax error", "Linker error", "Formatting error"],
    }
    validate_payloads([payload], catalogs)


def test_example_payloads_render_complete_deterministic_blocks() -> None:
    catalogs = load_catalogs(ROOT)
    payloads = load_jsonl(ROOT / "data" / "example_payloads.jsonl")
    validate_payloads(payloads, catalogs)
    first = render_dataset(payloads, catalogs)
    second = render_dataset(payloads, catalogs)
    assert first == second
    assert len(first) == 40
    report = validate_rendered_dataset(first, payloads, catalogs)
    assert report["status"] == "passed"
    assert report["payload_blocks"] == 10
    assert set(report["purpose_families"]) == {
        "ff01",
        "ff02",
        "ff03",
        "ff04",
        "ff05",
    }


def test_tampered_rendered_text_is_rejected() -> None:
    catalogs = load_catalogs(ROOT)
    payloads = load_jsonl(ROOT / "data" / "example_payloads.jsonl")
    rows = render_dataset(payloads, catalogs)
    tampered = copy.deepcopy(rows)
    tampered[0]["text"] += " changed"
    with pytest.raises(MVPError, match="reproduce deterministically"):
        validate_rendered_dataset(tampered, payloads, catalogs)


def test_incomplete_block_is_rejected() -> None:
    catalogs = load_catalogs(ROOT)
    payloads = load_jsonl(ROOT / "data" / "example_payloads.jsonl")
    rows = render_dataset(payloads, catalogs)
    with pytest.raises(MVPError, match="complete purpose x format 2x2"):
        validate_rendered_dataset(rows[1:], payloads, catalogs)
