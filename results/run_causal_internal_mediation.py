from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval_format_mvp.geometry import factorial_components
from run_causal_transport import PROTOCOL_HASH, PROTOCOL_SEED, build_deltas, make_prompts, mapping

LAYERS = (5, 16)
READOUT_LAYERS = (16, 24, 31)
BASE_MODES = ("purpose_full", "purpose_main", "format_full", "format_main", "plus_A", "minus_A", "plus_B", "minus_B", "plus_C", "minus_C", "zero")
RANDOM_SEEDS = tuple(range(2026081501, 2026081533))
MODES = BASE_MODES + tuple(f"random_{seed}" for seed in RANDOM_SEEDS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positive = scores[labels == 0]
    negative = scores[labels == 1]
    if len(positive) == 0 or len(negative) == 0:
        return None
    comparisons = (positive[:, None] > negative[None, :]).astype(float)
    comparisons += 0.5 * (positive[:, None] == negative[None, :])
    return float(comparisons.mean())


def projection_metric(scores: np.ndarray, rows: list[dict[str, Any]], readout: str) -> dict[str, Any]:
    categories = ("evaluation", "deployment") if readout == "purpose" else ("benchmark", "casual")
    labels = np.asarray([categories.index(str(row["intended_purpose"] if readout == "purpose" else row["format"])) for row in rows])
    prediction = (scores < 0).astype(int)
    signed = np.where(labels == 0, scores, -scores)
    by_family: dict[str, Any] = {}
    for family in sorted({str(row["purpose_family_id"]) for row in rows}):
        mask = np.asarray([str(row["purpose_family_id"]) == family for row in rows])
        by_family[family] = {
            "n": int(mask.sum()),
            "accuracy": float(np.mean(prediction[mask] == labels[mask])),
            "auc": auc(scores[mask], labels[mask]),
            "mean_target_projection": float(np.mean(signed[mask])),
        }
    return {
        "n": len(rows),
        "accuracy": float(np.mean(prediction == labels)),
        "auc": auc(scores, labels),
        "mean_target_projection": float(np.mean(signed)),
        "by_family": by_family,
    }


def capture_batches(model: Any, tokenizer: Any, prompts: list[str], *, layers: tuple[int, ...], batch_size: int, deltas: np.ndarray | None = None, target_layer: int | None = None, max_length: int = 2048) -> dict[int, np.ndarray]:
    import torch

    captures = {layer: [] for layer in layers}
    active_delta: list[np.ndarray | None] = [None]
    handles = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            captures[layer].append(hidden[:, -1, :].detach().float().cpu().numpy())
            if target_layer == layer and active_delta[0] is not None:
                update = torch.as_tensor(active_delta[0], device=hidden.device, dtype=hidden.dtype)
                edited = hidden.clone()
                edited[:, -1, :] = edited[:, -1, :] + update
                if isinstance(output, tuple):
                    return (edited, *output[1:])
                if isinstance(output, list):
                    return [edited, *output[1:]]
                return edited
            return output
        return hook

    try:
        for layer in layers:
            handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
        with torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                stop = min(start + batch_size, len(prompts))
                encoded = tokenizer(prompts[start:stop], add_special_tokens=False, return_tensors="pt", padding=True, truncation=False)
                if max(int(v) for v in encoded["attention_mask"].sum(dim=1).tolist()) > max_length:
                    raise ValueError("prompt exceeds frozen max_length")
                if not bool(torch.all(encoded["attention_mask"][:, -1] == 1)):
                    raise ValueError("last token is padding")
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                active_delta[0] = None if deltas is None else deltas[start:stop]
                model(**encoded, use_cache=False, return_dict=True)
                active_delta[0] = None
    finally:
        for handle in handles:
            handle.remove()
    return {layer: np.concatenate(values, axis=0) for layer, values in captures.items()}
def orient_direction(component: np.ndarray, block_families: np.ndarray, x: np.ndarray, rows: list[dict[str, Any]], readout: str, train_mask: np.ndarray) -> np.ndarray:
    row_families = np.asarray([str(row["purpose_family_id"]) for row in rows])
    train_families = set(row_families[train_mask].tolist())
    block_train = np.isin(np.asarray(block_families), list(train_families))
    direction = component[block_train].mean(0)
    if readout == "purpose":
        plus = np.asarray([str(row["intended_purpose"]) == "evaluation" for row in rows]) & train_mask
        minus = np.asarray([str(row["intended_purpose"]) == "deployment" for row in rows]) & train_mask
    else:
        plus = np.asarray([str(row["format"]) == "benchmark" for row in rows]) & train_mask
        minus = np.asarray([str(row["format"]) == "casual" for row in rows]) & train_mask
    empirical = x[plus].mean(0) - x[minus].mean(0)
    if float(np.dot(direction, empirical)) < 0:
        direction = -direction
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"zero downstream {readout} direction")
    return direction / norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in args.items.open() if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, local_files_only=args.local_files_only)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )
    model.eval()
    purpose_mapping, _ = mapping(["evaluation", "deployment"], tokenizer)
    format_mapping, _ = mapping(["benchmark", "casual"], tokenizer)
    prompts = {
        "purpose": make_prompts(rows, tokenizer, "purpose", purpose_mapping),
        "format": make_prompts(rows, tokenizer, "format", format_mapping),
    }
    all_layers = tuple(sorted(set(LAYERS) | set(READOUT_LAYERS)))
    unsteered = {kind: capture_batches(model, tokenizer, prompts[kind], layers=all_layers, batch_size=args.batch_size, max_length=args.max_length) for kind in prompts}
    families = sorted({str(row["purpose_family_id"]) for row in rows})
    results: dict[str, Any] = {}
    for source_kind in ("purpose", "format"):
        x_by_layer = unsteered[source_kind]
        for intervention_layer in LAYERS:
            x_intervention = x_by_layer[intervention_layer]
            for mode in MODES:
                rng = np.random.default_rng(int(mode.split("_", 1)[1])) if mode.startswith("random_") else None
                for held in families:
                    row_family = np.asarray([str(row["purpose_family_id"]) for row in rows])
                    held_mask = row_family == held
                    held_indices = np.flatnonzero(held_mask)
                    deltas, meta = build_deltas(x_intervention, rows, intervention_layer, held, source_kind, mode, rng=rng)
                    held_prompts = [prompts[source_kind][index] for index in held_indices]
                    edited = capture_batches(model, tokenizer, held_prompts, layers=all_layers, batch_size=args.batch_size, deltas=deltas, target_layer=intervention_layer, max_length=args.max_length)
                    train_mask = ~held_mask
                    held_rows = [rows[index] for index in held_indices]
                    readouts: dict[str, Any] = {}
                    for readout_layer in READOUT_LAYERS:
                        comp_down = factorial_components(x_by_layer[readout_layer], rows)
                        for readout in ("purpose", "format"):
                            direction = orient_direction(comp_down["A" if readout == "purpose" else "B"], comp_down["families"], x_by_layer[readout_layer], rows, readout, train_mask)
                            scores = edited[readout_layer] @ direction
                            readouts[f"L{readout_layer}::{readout}"] = projection_metric(scores, held_rows, readout)
                    key = f"{source_kind}::intervention_L{intervention_layer}::{mode}::held_{held}"
                    results[key] = {"source_kind": source_kind, "intervention_layer": intervention_layer, "mode": mode, "held_family": held, "n": len(held_rows), "readouts": readouts, "meta": meta}
    args.output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "status": "internal_causal_mediation_complete",
        "model": args.model,
        "revision": args.revision,
        "input_sha256": sha256(args.items),
        "protocol_hash": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "source_protocol_hash": PROTOCOL_HASH,
        "protocol_seed": PROTOCOL_SEED,
        "intervention_layers": list(LAYERS),
        "downstream_readout_layers": list(READOUT_LAYERS),
        "n_results": len(results),
        "results": results,
        "scientific_boundary": "Internal causal mediation only; no behavioral steering or direct-logit causal claim because baseline purpose endpoint was near chance.",
    }
    args.output.joinpath("results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
