from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
_spec = importlib.util.spec_from_file_location("compare_geometry_models", Path("results/compare_geometry_models.py"))
assert _spec is not None and _spec.loader is not None
_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compare)


def _geometry_payload(model: str, revision: str, items_sha: str) -> dict[str, object]:
    return {
        "status": "development_diagnostic",
        "model": {"name": model, "revision": revision},
        "input_artifacts": {"items_sha256": items_sha},
    }


def _equal_n_payload(model: str, revision: str, items_sha: str) -> dict[str, object]:
    return {
        "model": {"name": model, "revision": revision},
        "input_artifacts": {"items_sha256": items_sha},
    }


def test_geometry_provenance_accepts_expected_model_and_input(tmp_path: Path) -> None:
    model, revision = _compare.EXPECTED_MODELS["llama31_8b"]
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(_geometry_payload(model, revision, _compare.INPUT_SHA)))
    payload = _compare.load_geometry(path, "llama31_8b")
    assert payload["model"]["revision"] == revision  # type: ignore[index]


@pytest.mark.parametrize(
    "payload",
    [
        _geometry_payload("meta-llama/Llama-3.1-8B-Instruct", "wrong", _compare.INPUT_SHA),
        _geometry_payload("meta-llama/Llama-3.3-70B-Instruct", _compare.EXPECTED_MODELS["llama31_8b"][1], _compare.INPUT_SHA),
        _geometry_payload(*_compare.EXPECTED_MODELS["llama31_8b"], "wrong-input"),
    ],
)
def test_geometry_provenance_rejects_model_revision_or_input_mismatch(tmp_path: Path, payload: dict[str, object]) -> None:
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        _compare.load_geometry(path, "llama31_8b")


def test_equal_n_provenance_rejects_wrong_model(tmp_path: Path) -> None:
    model, revision = _compare.EXPECTED_MODELS["llama31_8b"]
    path = tmp_path / "equal_n.json"
    path.write_text(json.dumps(_equal_n_payload(model, revision, _compare.INPUT_SHA)))
    with pytest.raises(ValueError):
        _compare.load_equal_n(path, "llama31_70b")
