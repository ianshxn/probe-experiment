from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_ITEMS_SHA256 = "ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(
    *,
    items: Path,
    activations: Path,
    model: str,
    revision: str,
) -> dict[str, Any]:
    if not items.is_file():
        raise ValueError(f"missing items file: {items}")
    if not activations.is_file():
        raise ValueError(f"missing activation archive: {activations}")
    config_path = activations.parent / "config.json"
    meta_path = activations.parent / "meta.jsonl"
    all_layers_path = activations.parent / "activations_all_layers.npz"
    if not config_path.is_file() or not meta_path.is_file() or not all_layers_path.is_file():
        raise ValueError("archive is missing config.json, meta.jsonl, or activations_all_layers.npz")

    rows = read_jsonl(items)
    if len(rows) != 288:
        raise ValueError(f"expected 288 rendered rows, got {len(rows)}")
    items_sha256 = sha256_file(items)
    if items_sha256 != EXPECTED_ITEMS_SHA256:
        raise ValueError(f"items hash is not canonical: {items_sha256}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model") != model or config.get("model_revision") != revision:
        raise ValueError(f"model provenance mismatch: {config.get('model')} {config.get('model_revision')}")
    if config.get("items_sha256") != EXPECTED_ITEMS_SHA256:
        raise ValueError(f"config items hash mismatch: {config.get('items_sha256')}")
    required_config = {
        "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm",
        "position": "last_prompt_token_after_chat_template_before_generation",
        "dtype_stored": "float32",
        "add_special_tokens": False,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise ValueError(f"config {key} mismatch: {config.get(key)!r}")
    if config.get("n_items") != len(rows) or config.get("layers") != list(range(len(config.get("layers", [])))):
        raise ValueError("config does not describe a contiguous all-layer archive")

    expected_ids = [str(row["item_id"]) for row in rows]
    meta_rows = read_jsonl(meta_path)
    if meta_rows != rows:
        raise ValueError("archive meta.jsonl does not exactly match canonical rendered rows")

    with np.load(all_layers_path, allow_pickle=False) as archive:
        x = np.asarray(archive["X"])
        archive_ids = [str(value) for value in archive["item_ids"].tolist()]
        archive_layers = [int(value) for value in archive["layers"].tolist()]
    if x.ndim != 3 or x.shape[0] != len(rows) or x.shape[1] != len(archive_layers):
        raise ValueError(f"unexpected all-layer shape: {x.shape}")
    if archive_ids != expected_ids:
        raise ValueError("activation item_ids do not exactly match canonical rendered rows")
    if archive_layers != list(range(len(archive_layers))):
        raise ValueError("activation layers are not contiguous zero-based transformer blocks")
    if x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("stored activations are not finite float32")
    if config.get("hidden_size") != int(x.shape[2]) or config.get("layers") != archive_layers:
        raise ValueError("config and all-layer archive shapes disagree")

    activation_sha256 = sha256_file(activations)
    all_layers_sha256 = sha256_file(all_layers_path)
    if config.get("activations_sha256") != activation_sha256:
        raise ValueError("single-layer activation hash does not match config")
    if config.get("activations_all_layers_sha256") != all_layers_sha256:
        raise ValueError("all-layer activation hash does not match config")

    return {
        "schema_version": 1,
        "status": "validated",
        "model": model,
        "revision": revision,
        "items_sha256": items_sha256,
        "item_count": len(rows),
        "activation_archive": str(activations),
        "activation_sha256": activation_sha256,
        "all_layers_archive": str(all_layers_path),
        "all_layers_sha256": all_layers_sha256,
        "shape": list(x.shape),
        "layer_semantics": required_config["layer_semantics"],
        "position": required_config["position"],
        "dtype_stored": required_config["dtype_stored"],
        "scientific_boundary": "archive provenance only; no model or causal claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(items=args.items, activations=args.activations, model=args.model, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
