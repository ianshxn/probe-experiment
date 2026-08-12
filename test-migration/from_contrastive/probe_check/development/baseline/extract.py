#!/usr/bin/env python3
"""Extract subject-model activations for the demo16_v2 contrastive items.

This answers CUE_LOCUS_ENGLISH_ASYMMETRY.md item 3: does the held-out reader
comprehension gate predict what a linear probe can recover? The gate measures
what a reader model can *say* about a prompt; the probe measures what a subject
model's activations *encode*. Those can diverge, and the whole redesign question
turns on whether they do.

The real subject models (Llama-3.3-70B, Llama-3.3-Swallow-70B, Stockmark-2-100B)
cannot run on this machine. A small stand-in is used instead. That makes this a
qualitative test of the dissociation, not a quantitative estimate for the real
subjects -- see probe_check/README.md for what it can and cannot support.

Standard library plus torch/transformers only, run from an ephemeral uv
environment, because pyproject.toml and uv.lock are inside the pre-generation
integrity boundary and must not gain dependencies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

PROBE_CHECK = Path(__file__).resolve().parents[2]
ROOT = PROBE_CHECK.parent
ITEMS = ROOT / "data/processed/demo16_v2/probe_items.jsonl"
FRAMES = ROOT / "purpose_frames.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--layers", default="0.25,0.5,0.75,1.0",
                    help="fractional depths to store")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--per-cell", type=int, default=None,
                    help="subsample N items per (pair, language, class)")
    ap.add_argument("--stratum", default=None, choices=["implicit", "explicit"])
    args = ap.parse_args()

    frames = {
        r["pair_id"]: r
        for r in yaml.safe_load(FRAMES.read_text(encoding="utf-8"))["pairs"]
    }
    rows = [json.loads(l) for l in ITEMS.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["source"] == "matched_generated"]
    rows.sort(key=lambda r: r["item_id"])
    if args.stratum:
        rows = [r for r in rows if r["explicitness"] == args.stratum]
    if args.per_cell:
        seen: dict[tuple, int] = {}
        kept = []
        for r in rows:
            k = (r["pair_id"], r["language"], r["class"])
            if seen.get(k, 0) < args.per_cell:
                seen[k] = seen.get(k, 0) + 1
                kept.append(r)
        rows = kept
    if args.limit:
        rows = rows[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    depths = [float(x) for x in args.layers.split(",")]
    layer_ids = sorted({max(1, min(n_layers, round(d * n_layers))) for d in depths})
    print(f"model={args.model} layers={n_layers} extracting={layer_ids}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    meta_f = (args.out / "meta.jsonl").open("w", encoding="utf-8", newline="")
    feats: dict[int, list[torch.Tensor]] = {i: [] for i in layer_ids}

    # Left-padded batching: with left padding the final column is the true last
    # token for every row, so the read-out position stays identical to the
    # unbatched case.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    with torch.inference_mode():
        for start in range(0, len(rows), args.batch):
            chunk = rows[start : start + args.batch]
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": r["text"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                for r in chunk
            ]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024)
            out = model(**enc, output_hidden_states=True)
            for i in layer_ids:
                feats[i].append(out.hidden_states[i][:, -1].to(torch.float32).clone())
            for r in chunk:
                pf = frames[r["pair_id"]]
                meta_f.write(json.dumps({
                    "item_id": r["item_id"], "pair_id": r["pair_id"],
                    "language": r["language"], "class": r["class"], "label": r["label"],
                    "explicitness": r["explicitness"], "cue_locus": pf["cue_locus"],
                    "frame_family_id": r["frame_family_id"], "surface": r["surface"],
                    "job_id": r["job_id"], "content_sha256": r["content_sha256"],
                    "content_spec_id": r["content_spec_id"], "topic_id": r["topic_id"],
                }, ensure_ascii=False) + "\n")
            meta_f.flush()
            print(f"  {min(start + args.batch, len(rows))}/{len(rows)}", flush=True)

    meta_f.close()
    feats = {i: [torch.cat(feats[i], dim=0)] for i in layer_ids}
    for i in layer_ids:
        torch.save(feats[i][0], args.out / f"acts_layer{i}.pt")
    (args.out / "config.json").write_text(json.dumps({
        "model": args.model, "n_layers": n_layers, "layers_stored": layer_ids,
        "n_items": len(rows), "dtype": "float32", "position": "last_token",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} items to {args.out}", flush=True)


if __name__ == "__main__":
    main()
