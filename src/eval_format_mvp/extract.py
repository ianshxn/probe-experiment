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
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise MVPError("Expected a causal-LM exposing transformer blocks at .model.layers")
    n_layers = int(model.config.num_hidden_layers)
    if not 0 <= layer < n_layers:
        raise MVPError(f"layer must be in [0, {n_layers - 1}], got {layer}")

    # The chat template is rendered to text first, so it already carries the
    # model's control tokens; add_special_tokens must stay False or the
    # tokenizer prepends a duplicate begin-of-text.
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": str(row["text"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    unpadded = tokenizer(
        prompts, add_special_tokens=False, padding=False, truncation=False
    )["input_ids"]
    longest = max(len(ids) for ids in unpadded)
    if longest > max_length:
        raise MVPError(
            f"Longest prompt has {longest} tokens, exceeding max_length={max_length}. "
            "Raise --max-length intentionally; prompts are never truncated."
        )

    # A forward hook on the block reads its output residual stream directly,
    # before the final model norm that hidden_states applies at the last layer.
    captured: dict[str, Any] = {}

    def capture_last_token(_module, _inputs, output) -> None:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured["hidden"] = hidden[:, -1, :].detach()

    handle = model.model.layers[layer].register_forward_hook(capture_last_token)
    activations: list[Any] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                captured.clear()
                encoded = tokenizer(
                    prompts[start : start + batch_size],
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                )
                if not bool(torch.all(encoded["attention_mask"][:, -1] == 1)):
                    raise MVPError("Last position must be a real prompt token, not padding")
                encoded = {key: value.to(device) for key, value in encoded.items()}
                model.model(**encoded, use_cache=False, return_dict=True)
                if "hidden" not in captured:
                    raise MVPError(f"Forward hook did not fire for layer {layer}")
                activations.append(captured["hidden"].float().cpu().numpy())
    finally:
        handle.remove()

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
        "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm",
        "position": "last_prompt_token_after_chat_template_before_generation",
        "chat_template": "tokenizer.apply_chat_template(user_message, add_generation_prompt=True)",
        "add_special_tokens": False,
        "max_length": max_length,
        "max_prompt_tokens": longest,
        "dtype_stored": "float32",
        "n_items": len(rows),
        "hidden_size": int(matrix.shape[1]),
        "items_path": str(items_path),
        "items_sha256": sha256_file(items_path),
        "activations_sha256": sha256_file(activation_path),
    }
    write_json(output_dir / "config.json", config)
    return config

