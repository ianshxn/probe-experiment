from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from results import validate_activation_archive as validator


MODEL = "meta-llama/Llama-3.1-70B-Instruct"
REVISION = "1605565b47bb9346c5515c34102e054115b4f98b"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_archive(tmp_path: Path) -> tuple[Path, Path]:
    rows = [{"item_id": f"item_{i:03d}", "purpose_family_id": "family"} for i in range(288)]
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    validator.EXPECTED_ITEMS_SHA256 = _sha256(items)

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    x = np.arange(288 * 2 * 3, dtype=np.float32).reshape(288, 2, 3)
    item_ids = np.asarray([row["item_id"] for row in rows], dtype=str)
    layers = np.asarray([0, 1], dtype=np.int16)
    np.savez_compressed(archive_dir / "activations.npz", X=x[:, 0], item_ids=item_ids)
    np.savez_compressed(archive_dir / "activations_all_layers.npz", X=x, item_ids=item_ids, layers=layers)
    (archive_dir / "meta.jsonl").write_text(items.read_text(encoding="utf-8"), encoding="utf-8")
    config = {
        "model": MODEL,
        "model_revision": REVISION,
        "items_sha256": validator.EXPECTED_ITEMS_SHA256,
        "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm",
        "position": "last_prompt_token_after_chat_template_before_generation",
        "dtype_stored": "float32",
        "add_special_tokens": False,
        "n_items": 288,
        "layers": [0, 1],
        "hidden_size": 3,
        "activations_sha256": _sha256(archive_dir / "activations.npz"),
        "activations_all_layers_sha256": _sha256(archive_dir / "activations_all_layers.npz"),
    }
    (archive_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return items, archive_dir / "activations.npz"


def test_validator_accepts_exact_archive(tmp_path: Path) -> None:
    items, activations = _make_archive(tmp_path)
    result = validator.validate(items=items, activations=activations, model=MODEL, revision=REVISION)
    assert result["status"] == "validated"
    assert result["shape"] == [288, 2, 3]


def test_validator_rejects_noncanonical_input_bytes(tmp_path: Path) -> None:
    items, activations = _make_archive(tmp_path)
    validator.EXPECTED_ITEMS_SHA256 = "canonical-but-different"
    try:
        validator.validate(items=items, activations=activations, model=MODEL, revision=REVISION)
    except ValueError as exc:
        assert "items hash is not canonical" in str(exc)
    else:
        raise AssertionError("validator accepted noncanonical input bytes")
