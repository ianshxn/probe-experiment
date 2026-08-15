from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval_format_mvp.equal_n import run_equal_n

SEEDS = (1729, 2718, 31415, 4242, 8675309)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--activations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--revision", default="0e9e39f249a16976918f6564b8830bc894c89659")
    p.add_argument("--git-commit", default="working_tree")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.items.open() if line.strip()]
    archive = np.load(args.activations, allow_pickle=False)
    x = np.asarray(archive["X"], dtype=np.float32)
    layers = [int(value) for value in archive["layers"].tolist()]
    if x.ndim != 3 or x.shape[1] != len(layers):
        raise ValueError(f"expected N x L x D, got {x.shape}")
    if [str(v) for v in archive["item_ids"].tolist()] != [str(r["item_id"]) for r in rows]:
        raise ValueError("activation item_ids do not exactly match metadata")
    if not np.isfinite(x).all():
        raise ValueError("non-finite activation")
    layer_results = []
    for offset, layer in enumerate(layers):
        result = run_equal_n(x[:, offset, :], rows, seeds=SEEDS)
        result["layer"] = layer
        layer_results.append(result)
    output = {
        "schema_version": 1,
        "analysis": "equal_n_mixed_format_control_all_layers",
        "status": "development_diagnostic",
        "run_id": args.run_id,
        "git_commit": args.git_commit,
        "model": {"name": args.model, "revision": args.revision, "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm", "token_position": "last_prompt_token_after_chat_template_before_generation", "dtype_inference": "bfloat16", "dtype_stored": "float32"},
        "hidden_size": int(x.shape[2]),
        "input_artifacts": {"items": str(args.items), "items_sha256": sha256(args.items), "activations": str(args.activations), "activations_sha256": sha256(args.activations)},
        "model": {"name": "meta-llama/Llama-3.1-8B-Instruct", "revision": "0e9e39f249a16976918f6564b8830bc894c89659", "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm", "token_position": "last_prompt_token_after_chat_template_before_generation", "dtype_inference": "bfloat16", "dtype_stored": "float32"},
        "layers_results": layer_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
