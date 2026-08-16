from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eval_format_mvp.extract import extract_activations

ROOT = Path("/scratch/jppatton/langea/deploy/tae_2026")
ITEMS = ROOT / "rendered_items_canonical_lf.jsonl"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
REVISION = "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
OUT = Path("/scratch/jppatton/langea/runs/tae_2026/llama33_70b_parent/llama33_70b_20260816_rev6f6073b")
GIT_COMMIT = "0060d64"


def runner_sha256() -> str:
    digest = hashlib.sha256()
    with Path(__file__).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


config = extract_activations(
    items_path=ITEMS,
    output_dir=OUT,
    model_name=MODEL,
    model_revision=REVISION,
    layer=0,
    batch_size=1,
    max_length=2048,
    device_map="auto",
    all_layers=True,
)
(OUT / "sprint_provenance.json").write_text(json.dumps({
    "run_id": "llama33_70b_parent_20260816_rev6f6073b",
    "git_commit": GIT_COMMIT,
    "runner_sha256": runner_sha256(),
    "model": MODEL,
    "revision": REVISION,
    "items_sha256_expected": "ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022",
    "dtype_inference": "bfloat16",
    "dtype_stored": "float32",
    "all_layers": True,
    "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm",
    "token_position": "last_prompt_token_after_chat_template_before_generation",
    "config": config,
}, indent=2) + "\n")
