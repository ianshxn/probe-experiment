"""Extract one fixed last-token residual-stream activation per rendered prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import MVPError, load_jsonl, sha256_file, write_json, write_jsonl


def extract_activations(
    *,
    items_path: Path,
    output_dir: Path,
    model_name: str,
    model_revision: str | None,
    layer: int,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise MVPError(
            "Activation extraction requires the 'extract' optional dependencies"
        ) from exc

    rows = load_jsonl(items_path)
    if not rows:
        raise MVPError("Rendered item file is empty")
    if batch_size < 1 or max_length < 1:
        raise MVPError("batch_size and max_length must be positive")

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=model_revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    n_layers = int(model.config.num_hidden_layers)
    if not 0 <= layer < n_layers:
        raise MVPError(f"layer must be in [0, {n_layers - 1}], got {layer}")

    activations: list[Any] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": str(row["text"])}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in chunk
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            result = model(**encoded, output_hidden_states=True, use_cache=False)
            # hidden_states[0] is the embedding output. A zero-based transformer
            # block index L therefore maps to hidden_states[L + 1].
            hidden = result.hidden_states[layer + 1][:, -1, :]
            activations.append(hidden.float().cpu().numpy())

    matrix = np.concatenate(activations, axis=0)
    item_ids = np.asarray([str(row["item_id"]) for row in rows], dtype=str)
    output_dir.mkdir(parents=True, exist_ok=True)
    activation_path = output_dir / "activations.npz"
    np.savez_compressed(activation_path, X=matrix, item_ids=item_ids)
    write_jsonl(output_dir / "meta.jsonl", rows)
    config = {
        "schema_version": 1,
        "model": model_name,
        "model_revision": model_revision or "default_resolved_by_hub",
        "n_transformer_layers": n_layers,
        "layer": layer,
        "layer_semantics": "zero_based_transformer_block_output",
        "position": "last_prompt_token_after_chat_template",
        "dtype_stored": "float32",
        "n_items": len(rows),
        "hidden_size": int(matrix.shape[1]),
        "items_path": str(items_path),
        "items_sha256": sha256_file(items_path),
        "activations_sha256": sha256_file(activation_path),
    }
    write_json(output_dir / "config.json", config)
    return config

