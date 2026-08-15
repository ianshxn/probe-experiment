from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/scratch/jppatton/langea/deploy/tae_2026")
sys.path.insert(0, str(ROOT))
from eval_format_mvp.extract import extract_activations  # noqa: E402

ITEMS = ROOT / "rendered_items.jsonl"
OUT = Path("/scratch/jppatton/langea/runs/tae_2026/llama31_70b/llama31_70b_20260815_rev1605565b")

config = extract_activations(
    items_path=ITEMS,
    output_dir=OUT,
    model_name="meta-llama/Llama-3.1-70B-Instruct",
    model_revision="1605565b47bb9346c5515c34102e054115b4f98b",
    layer=0,
    batch_size=1,
    max_length=2048,
    device_map="auto",
    all_layers=True,
)
(OUT / "sprint_provenance.json").write_text(json.dumps({
    "run_id": "llama31_70b_20260815_rev1605565b",
    "git_commit": "4e64e93729026ce39c85e13800e1a40467e4f644",
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "revision": "1605565b47bb9346c5515c34102e054115b4f98b",
    "items_sha256_expected": "ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022",
    "dtype_inference": "bfloat16",
    "dtype_stored": "float32",
    "all_layers": True,
    "layer_semantics": "zero_based_transformer_block_output_before_final_model_norm",
    "token_position": "last_prompt_token_after_chat_template_before_generation",
    "config": config,
}, indent=2) + "\n")
