from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from eval_format_mvp.causal import factorial_transport, unit_strength
from eval_format_mvp.geometry import factorial_components
from run_causal_internal_mediation import capture_batches
from run_causal_transport import LAYERS, PROTOCOL_HASH, PROTOCOL_SEED, make_prompts, mapping

READOUT_LAYERS = (16, 24, 31)
BASE_MODES = ("zero", "purpose_full", "purpose_main", "purpose_norm_matched", "purpose_wrong_c", "purpose_shuffled_c")
RANDOM_SEEDS = tuple(range(2026081501, 2026081533))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return None if na == 0.0 or nb == 0.0 else float(np.dot(a, b) / (na * nb))


def cells(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], int]:
    out = {}
    for i, row in enumerate(rows):
        out[(str(row["payload_block_id"]), str(row["intended_purpose"]), str(row["format"]))] = i
    return out


def metric_rows(pred: np.ndarray, source: np.ndarray, target: np.ndarray, scale: float) -> list[dict[str, float | None]]:
    out = []
    for p, s, t in zip(pred, source, target):
        error = float(np.linalg.norm(p - t))
        out.append({
            "raw_euclidean_error": error,
            "normalized_euclidean_error": error / scale if scale else None,
            "cosine_to_natural_target": cosine(p, t),
            "cosine_predicted_vs_natural_displacement": cosine(p - s, t - s),
        })
    return out


def fixed_shuffled_c(c: np.ndarray, a: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = c[rng.permutation(c.size)]
    norm_c, norm_shuffled = float(np.linalg.norm(c)), float(np.linalg.norm(shuffled))
    if norm_c == 0.0 or norm_shuffled == 0.0:
        return shuffled
    return shuffled * (norm_c / norm_shuffled)


def random_orthogonal_c(c: np.ndarray, a: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = rng.normal(size=c.shape)
    a_norm_sq = float(np.dot(a, a))
    if a_norm_sq:
        r = r - a * (float(np.dot(r, a)) / a_norm_sq)
    r_norm, c_norm = float(np.linalg.norm(r)), float(np.linalg.norm(c))
    if r_norm == 0.0 or c_norm == 0.0:
        return r
    return r * (c_norm / r_norm)


def make_deltas(x: np.ndarray, rows: list[dict[str, Any]], held: str, mode: str, *, shuffle_seed: int = 2026081616) -> np.ndarray:
    comp = factorial_components(x, rows)
    train_blocks = comp["families"] != held
    a = comp["A"][train_blocks].mean(0)
    b = comp["B"][train_blocks].mean(0)
    c = comp["C"][train_blocks].mean(0)
    train_a = comp["A"][train_blocks]
    train_c = comp["C"][train_blocks]
    shuffled_c = fixed_shuffled_c(c, a, shuffle_seed)
    row_family = np.asarray([str(row["purpose_family_id"]) for row in rows])
    held_indices = np.flatnonzero(row_family == held)
    out = np.zeros((len(held_indices), x.shape[1]), dtype=np.float64)
    for j, idx in enumerate(held_indices):
        p = 1.0 if str(rows[idx]["intended_purpose"]) == "evaluation" else -1.0
        f = 1.0 if str(rows[idx]["format"]) == "benchmark" else -1.0
        full_raw = factorial_transport(p, f, a, b, c, target="purpose", include_interaction=True)
        main_raw = factorial_transport(p, f, a, b, c, target="purpose", include_interaction=False)
        if mode == "zero":
            out[j] = 0.0
        elif mode == "purpose_full":
            unit, alpha = unit_strength(full_raw, train_a + f * train_c)
            out[j] = unit * alpha
        elif mode == "purpose_main":
            unit, alpha = unit_strength(main_raw, train_a)
            out[j] = unit * alpha
        elif mode == "purpose_norm_matched":
            unit, alpha = unit_strength(main_raw, train_a)
            main_delta = unit * alpha
            full_unit, full_alpha = unit_strength(full_raw, train_a + f * train_c)
            full_norm = float(np.linalg.norm(full_unit * full_alpha))
            main_norm = float(np.linalg.norm(main_delta))
            out[j] = main_delta * (full_norm / main_norm) if main_norm else 0.0
        elif mode == "purpose_wrong_c":
            wrong_raw = factorial_transport(p, f, a, b, c, target="purpose", include_interaction=True) + 4.0 * p * f * c
            wrong_unit, wrong_alpha = unit_strength(wrong_raw, train_a - f * train_c)
            out[j] = wrong_unit * wrong_alpha
        elif mode == "purpose_shuffled_c":
            shuffled_raw = -2.0 * p * (a + f * shuffled_c)
            shuffled_train = train_a + f * shuffled_c
            shuffled_unit, shuffled_alpha = unit_strength(shuffled_raw, shuffled_train)
            out[j] = shuffled_unit * shuffled_alpha
        elif mode.startswith("random_c_"):
            seed = int(mode.rsplit("_", 1)[1])
            random_c = random_orthogonal_c(c, a, seed)
            random_raw = -2.0 * p * (a + f * random_c)
            random_unit, random_alpha = unit_strength(random_raw, train_a + f * random_c)
            out[j] = random_unit * random_alpha
        else:
            raise ValueError(mode)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--random-bank", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["model"] != args.model or protocol["revision"] != args.revision or protocol["input_sha256"] != sha256(args.items):
        raise ValueError("frozen protocol provenance mismatch")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in args.items.open(encoding="utf-8") if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=args.device_map)
    model.eval()
    purpose_mapping, _ = mapping(["evaluation", "deployment"], tokenizer)
    prompts = make_prompts(rows, tokenizer, "purpose", purpose_mapping)
    all_layers = tuple(sorted(set(LAYERS) | set(READOUT_LAYERS)))
    natural = capture_batches(model, tokenizer, prompts, layers=all_layers, batch_size=args.batch_size, max_length=args.max_length)
    cell_index = cells(rows)
    families = sorted({str(row["purpose_family_id"]) for row in rows})
    modes = list(BASE_MODES) + ([f"random_c_{seed}" for seed in RANDOM_SEEDS] if args.random_bank else [])
    records: list[dict[str, Any]] = []
    row_family = np.asarray([str(row["purpose_family_id"]) for row in rows])
    for intervention_layer in LAYERS:
        x_intervention = natural[intervention_layer]
        for held in families:
            held_indices = np.flatnonzero(row_family == held)
            source_rows = [rows[i] for i in held_indices]
            target_indices = []
            wrong_format_indices = []
            for idx in held_indices:
                row = rows[idx]
                block, purpose, fmt = str(row["payload_block_id"]), str(row["intended_purpose"]), str(row["format"])
                target_purpose = "deployment" if purpose == "evaluation" else "evaluation"
                other_format = "casual" if fmt == "benchmark" else "benchmark"
                target_indices.append(cell_index[(block, target_purpose, fmt)])
                wrong_format_indices.append(cell_index[(block, purpose, other_format)])
            source_targets = np.asarray(target_indices, dtype=int)
            wrong_targets = np.asarray(wrong_format_indices, dtype=int)
            deltas_by_mode = {mode: make_deltas(x_intervention, rows, held, mode) for mode in modes}
            held_prompts = [prompts[i] for i in held_indices]
            train_mask = row_family != held
            for mode, deltas in deltas_by_mode.items():
                edited = capture_batches(model, tokenizer, held_prompts, layers=all_layers, batch_size=args.batch_size, deltas=deltas, target_layer=intervention_layer, max_length=args.max_length)
                for readout_layer in READOUT_LAYERS:
                    source = natural[readout_layer][held_indices]
                    target = natural[readout_layer][source_targets]
                    wrong_target = natural[readout_layer][wrong_targets]
                    scale = float(np.sqrt(np.mean(np.sum(natural[readout_layer][train_mask] ** 2, axis=1))))
                    metrics = metric_rows(edited[readout_layer], source, target, scale)
                    wrong_metrics = metric_rows(edited[readout_layer], source, wrong_target, scale)
                    for block_row, source_idx, metric, wrong_metric in zip(source_rows, held_indices, metrics, wrong_metrics):
                        records.append({"intervention_layer": int(intervention_layer), "readout_layer": int(readout_layer), "held_out_family": held, "payload_block_id": str(block_row["payload_block_id"]), "source_item_id": str(block_row["item_id"]), "mode": mode, "source_format": str(block_row["format"]), "source_purpose": str(block_row["intended_purpose"]), **metric, "wrong_format_target_raw_error": wrong_metric["raw_euclidean_error"], "wrong_format_target_cosine": wrong_metric["cosine_to_natural_target"]})
    payload = {
        "schema_version": 1,
        "status": "FINAL_BF16_INTERNAL_CAUSAL_NATURAL_TARGET_CONTROLS",
        "model": args.model,
        "revision": args.revision,
        "input_sha256": sha256(args.items),
        "protocol_hash": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "source_protocol_hash": PROTOCOL_HASH,
        "protocol_seed": PROTOCOL_SEED,
        "layers": list(LAYERS),
        "readout_layers": list(READOUT_LAYERS),
        "modes": modes,
        "random_bank": args.random_bank,
        "n_records": len(records),
        "records": records,
        "scientific_boundary": "BF16 held-out-family residual-stream counterfactual transport evaluated against natural matched downstream representations; no behavioral endpoint.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.joinpath("results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
