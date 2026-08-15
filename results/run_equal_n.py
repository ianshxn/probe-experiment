from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval_format_mvp.equal_n import run_equal_n


SEEDS = (1729, 2718, 31415, 4242, 8675309)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.items.open() if line.strip()]
    archive = np.load(args.activations, allow_pickle=False)
    item_ids = [str(value) for value in archive["item_ids"].tolist()]
    if item_ids != [str(row["item_id"]) for row in rows]:
        raise ValueError("activation item_ids do not exactly match item metadata")
    x = np.asarray(archive["X"], dtype=np.float32)
    if x.ndim == 3:
        if args.layer is None:
            raise ValueError("--layer is required for an all-layer archive")
        x = x[:, args.layer, :]
    elif x.ndim != 2:
        raise ValueError(f"expected 2D or 3D activations, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("activation archive contains non-finite values")

    result = run_equal_n(x, rows, seeds=SEEDS)
    result["run_id"] = args.run_id
    result["git_commit"] = "4e64e93729026ce39c85e13800e1a40467e4f644"
    result["input_artifacts"] = {
        "items": str(args.items),
        "items_sha256": sha256(args.items),
        "activations": str(args.activations),
        "activations_sha256": sha256(args.activations),
    }
    result["model"] = {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "dtype_inference": "bfloat16",
        "dtype_stored": "float32",
        "layer": args.layer,
        "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm",
        "token_position": "last_prompt_token_after_chat_template_before_generation",
    }
    result["confirmatory_status"] = "development_diagnostic"
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
