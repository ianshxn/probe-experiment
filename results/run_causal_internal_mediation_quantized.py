from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval_format_mvp.geometry import factorial_components
from run_causal_internal_mediation import capture_batches, orient_direction
from run_causal_transport import PROTOCOL_HASH, PROTOCOL_SEED, build_deltas, make_prompts, mapping

LAYERS = (5, 16)
READOUT_LAYERS = (16, 24, 31)
MODES = ("purpose_full", "purpose_main", "plus_A", "minus_A", "plus_C", "minus_C", "zero")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def projection_metric(scores: np.ndarray, rows: list[dict[str, Any]], readout: str) -> dict[str, Any]:
    categories = ("evaluation", "deployment") if readout == "purpose" else ("benchmark", "casual")
    labels = np.asarray([categories.index(str(row["intended_purpose"] if readout == "purpose" else row["format"])) for row in rows])
    prediction = (scores < 0).astype(int)
    signed = np.where(labels == 0, scores, -scores)
    return {"n": len(rows), "accuracy": float(np.mean(prediction == labels)), "mean_target_projection": float(np.mean(signed))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--quant-bits", type=int, choices=(4, 8), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["status"] != "protocol_frozen_after_endpoint_gate_failure_before_internal_inference":
        raise ValueError("internal protocol is not frozen")
    if protocol["model"] != args.model or protocol["revision"] != args.revision:
        raise ValueError("protocol model provenance mismatch")
    if protocol["input_sha256"] != sha256(args.items):
        raise ValueError("protocol input hash mismatch")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if args.quant_bits == 4:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
        quantization_name = "bitsandbytes-nf4-4bit"
    else:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        quantization_name = "bitsandbytes-int8-8bit"
    rows = [json.loads(line) for line in args.items.open(encoding="utf-8") if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=args.local_files_only)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, quantization_config=quant_cfg, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=args.device_map, local_files_only=args.local_files_only)
    model.eval()
    purpose_mapping, _ = mapping(["evaluation", "deployment"], tokenizer)
    prompts = {"purpose": make_prompts(rows, tokenizer, "purpose", purpose_mapping)}
    all_layers = tuple(sorted(set(LAYERS) | set(READOUT_LAYERS)))
    unsteered = capture_batches(model, tokenizer, prompts["purpose"], layers=all_layers, batch_size=args.batch_size, max_length=args.max_length)
    families = sorted({str(row["purpose_family_id"]) for row in rows})
    results: dict[str, Any] = {}
    row_family = np.asarray([str(row["purpose_family_id"]) for row in rows])
    for intervention_layer in LAYERS:
        x_intervention = unsteered[intervention_layer]
        for mode in MODES:
            for held in families:
                held_mask = row_family == held
                held_indices = np.flatnonzero(held_mask)
                deltas, meta = build_deltas(x_intervention, rows, intervention_layer, held, "purpose", mode)
                held_prompts = [prompts["purpose"][index] for index in held_indices]
                edited = capture_batches(model, tokenizer, held_prompts, layers=all_layers, batch_size=args.batch_size, deltas=deltas, target_layer=intervention_layer, max_length=args.max_length)
                train_mask = ~held_mask
                held_rows = [rows[index] for index in held_indices]
                readouts: dict[str, Any] = {}
                for readout_layer in READOUT_LAYERS:
                    comp_down = factorial_components(unsteered[readout_layer], rows)
                    direction = orient_direction(comp_down["A"], comp_down["families"], unsteered[readout_layer], rows, "purpose", train_mask)
                    readouts[f"L{readout_layer}::purpose"] = projection_metric(edited[readout_layer] @ direction, held_rows, "purpose")
                key = f"purpose::intervention_L{intervention_layer}::{mode}::held_{held}"
                results[key] = {"source_kind": "purpose", "intervention_layer": intervention_layer, "mode": mode, "held_family": held, "n": len(held_rows), "readouts": readouts, "meta": meta}
    payload = {
        "schema_version": 1,
        "status": "EXPLORATORY_QUANTIZED_INTERNAL_MEDIATION",
        "model": args.model,
        "revision": args.revision,
        "input_sha256": sha256(args.items),
        "protocol_hash": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "source_protocol_hash": PROTOCOL_HASH,
        "protocol_seed": PROTOCOL_SEED,
        "quantization": {"name": quantization_name, "bits": args.quant_bits, "library": "bitsandbytes", "config": quant_cfg.to_dict()},
        "intervention_layers": list(LAYERS),
        "downstream_readout_layers": list(READOUT_LAYERS),
        "modes": list(MODES),
        "n_results": len(results),
        "results": results,
        "scientific_boundary": "Exploratory quantized internal readout only; not BF16 evidence, not behavioral causal steering, and not a replacement for the frozen result.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.joinpath("results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
