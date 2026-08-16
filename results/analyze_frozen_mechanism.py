from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/tae_2026/causal/causal_20260816_8b_bf16_natural_counterfactual_controls/results.json"
PROTOCOL = ROOT / "runs/tae_2026/causal/causal_20260816_8b_bf16_natural_counterfactual_controls/protocol.json"
EXECUTED_PROTOCOL = ROOT / "runs/tae_2026/causal/causal_20260816_internal_mediation_protocol/protocol.json"
ITEMS = ROOT / "runs/tae_2026/input/rendered_items_canonical_lf.jsonl"
GEOMETRY_PROTOCOL = ROOT / "runs/tae_2026/causal/causal_20260815_geometry_frozen/protocol.json"
ACTIVATION_CONFIG = ROOT / "probe results/llama31_8b_3f9ccbe890cb7a06/config.json"
RUNNER = ROOT / "results/run_causal_natural_counterfactual_controls.py"
ANALYSIS_SCRIPT = ROOT / "results/analyze_frozen_mechanism.py"
OUT = ROOT / "runs/tae_2026/causal/causal_20260816_8b_bf16_mechanism_frozen"
SEED = 2026081617
def load_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # The pre-causal geometry protocol has one redundant closing brace in
        # its historical artifact; preserve its hash and parse its content.
        repaired = text.replace("]}}\n  },\n  \"forward_intervention\"", "]}\n  },\n  \"forward_intervention\"", 1)
        return json.loads(repaired)
BOOTSTRAP_REPS = 10000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_ci(values: np.ndarray, units: list[str], seed: int = SEED) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    unique = list(dict.fromkeys(units))
    unit_values = np.asarray([values[np.asarray(units) == unit].mean() for unit in unique])
    rng = np.random.default_rng(seed)
    draws = np.asarray([rng.choice(unit_values, size=len(unit_values), replace=True).mean() for _ in range(BOOTSTRAP_REPS)])
    return {
        "n_observations": int(len(values)),
        "n_units": int(len(unit_values)),
        "unit": "payload_block",
        "mean": float(unit_values.mean()),
        "median_observation": float(np.median(values)),
        "unit_values": {unit: float(value) for unit, value in zip(unique, unit_values)},
        "bootstrap_seed": seed,
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_ci_95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
    }


def paired_ci(values: np.ndarray, units: list[str], seed: int = SEED) -> dict[str, Any]:
    return mean_ci(values, units, seed)


def record_map(records: list[dict[str, Any]], layer: int, mode: str) -> dict[str, dict[str, Any]]:
    return {r["source_item_id"]: r for r in records if r["intervention_layer"] == layer and r["readout_layer"] == 24 and r["mode"] == mode}


def family_map(records: list[dict[str, Any]]) -> dict[str, str]:
    return {r["source_item_id"]: r["held_out_family"] for r in records if r["readout_layer"] == 24 and r["mode"] == "zero" and r["intervention_layer"] == 5}


def block_map(records: list[dict[str, Any]]) -> dict[str, str]:
    return {r["source_item_id"]: r["payload_block_id"] for r in records if r["readout_layer"] == 24 and r["mode"] == "zero" and r["intervention_layer"] == 5}


def family_summary(values: dict[str, float], families: dict[str, str], blocks: dict[str, str]) -> list[dict[str, Any]]:
    out = []
    for family in sorted(set(families.values())):
        ids = [item for item, f in families.items() if f == family and item in values]
        out.append({"family_id": family, "n_payloads": len(set(blocks[i] for i in ids)), "n_cells": len(ids), "value": float(np.mean([values[i] for i in ids]))})
    return out


def sign_flip(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    signs = np.asarray([1 if v > 0 else -1 if v < 0 else 0 for v in values])
    positive = int(np.sum(signs > 0))
    # Exact two-sided sign test, ignoring zero values.
    nonzero = int(np.sum(signs != 0))
    tails = sum(math.comb(nonzero, k) for k in range(positive + 1)) / (2**nonzero)
    upper = sum(math.comb(nonzero, k) for k in range(positive, nonzero + 1)) / (2**nonzero)
    sign_test_two_sided = min(1.0, 2 * min(tails, upper))
    signed_sums = []
    observed = float(values.sum())
    for mask in range(1 << nonzero):
        s = 0.0
        j = 0
        for value in values:
            if value == 0:
                continue
            s += value if (mask >> j) & 1 else -value
            j += 1
        signed_sums.append(s)
    return {"n_families": n, "positive_families": positive, "fraction_positive": positive / n if n else None, "exact_two_sided_sign_test_p": float(sign_test_two_sided), "sign_flip_one_sided_p": float((1 + sum(s >= observed - 1e-15 for s in signed_sums)) / (len(signed_sums) + 1)), "sign_flip_observed_sum": observed, "sign_flip_null_count": len(signed_sums)}


def load() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source = json.loads(SOURCE.read_text())
    records = source["records"]
    rows = [json.loads(line) for line in ITEMS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return source, records, rows


def main() -> None:
    source, records, rows = load()
    families = family_map(records)
    blocks = block_map(records)
    items = sorted(families)
    full = {layer: record_map(records, layer, "purpose_full") for layer in (5, 16)}
    main = {layer: record_map(records, layer, "purpose_main") for layer in (5, 16)}
    zero = {layer: record_map(records, layer, "zero") for layer in (5, 16)}
    wrong = {layer: record_map(records, layer, "purpose_wrong_c") for layer in (5, 16)}
    norm = {layer: record_map(records, layer, "purpose_norm_matched") for layer in (5, 16)}
    shuffled = {layer: record_map(records, layer, "purpose_shuffled_c") for layer in (5, 16)}

    cbenefit: dict[int, dict[str, float]] = {}
    for layer in (5, 16):
        cbenefit[layer] = {item: main[layer][item]["raw_euclidean_error"] - full[layer][item]["raw_euclidean_error"] for item in items}
    contrast = {item: cbenefit[16][item] - cbenefit[5][item] for item in items}
    family_rows = []
    for family in sorted(set(families.values())):
        ids = [i for i in items if families[i] == family]
        family_rows.append({"family_id": family, "n_payloads": len(set(blocks[i] for i in ids)), "n_cells": len(ids), "L5_C_benefit": float(np.mean([cbenefit[5][i] for i in ids])), "L16_C_benefit": float(np.mean([cbenefit[16][i] for i in ids])), "L16_minus_L5": float(np.mean([contrast[i] for i in ids]))})
    family_contrasts = np.asarray([r["L16_minus_L5"] for r in family_rows])
    leave_one = [{"removed_family": row["family_id"], "remaining_families": [r["family_id"] for r in family_rows if r["family_id"] != row["family_id"]], "contrast": float(np.mean([contrast[i] for i in items if families[i] != row["family_id"]]))} for row in family_rows]

    primary = {
        "endpoint": "natural matched purpose-counterfactual target at downstream L24",
        "source_artifact": str(SOURCE),
        "family_table": family_rows,
        "overall": {"n_cells": len(items), "n_payload_blocks": len(set(blocks.values())), "L5_C_benefit": float(np.mean(list(cbenefit[5].values()))), "L16_C_benefit": float(np.mean(list(cbenefit[16].values()))), "L16_minus_L5": float(np.mean(list(contrast.values())))},
        "sign_consistency": sign_flip(family_contrasts),
        "leave_one_family_out": leave_one,
        "leave_one_family_out_range": [float(min(x["contrast"] for x in leave_one)), float(max(x["contrast"] for x in leave_one))],
    }

    # Fraction of natural source-target gap closed by C. Zero-mode error equals source-to-target distance.
    frac: dict[int, dict[str, float]] = {}
    frac_family = {}
    for layer in (5, 16):
        frac[layer] = {item: cbenefit[layer][item] / zero[layer][item]["raw_euclidean_error"] for item in items if zero[layer][item]["raw_euclidean_error"] > 0}
        frac_family[layer] = family_summary(frac[layer], families, blocks)
    frac_diff = {item: frac[16][item] - frac[5][item] for item in items if item in frac[5] and item in frac[16]}
    frac_effect = {"definition": "(distance(A-only,target)-distance(full,target))/distance(source,target)", "L5": mean_ci(np.asarray(list(frac[5].values())), [blocks[i] for i in frac[5]], SEED + 1), "L16": mean_ci(np.asarray(list(frac[16].values())), [blocks[i] for i in frac[16]], SEED + 2), "L16_minus_L5": paired_ci(np.asarray(list(frac_diff.values())), [blocks[i] for i in frac_diff], SEED + 3), "family_values": {"L5": frac_family[5], "L16": frac_family[16], "L16_minus_L5": family_summary(frac_diff, families, blocks)}}
    a_improvement = {layer: {item: zero[layer][item]["raw_euclidean_error"] - main[layer][item]["raw_euclidean_error"] for item in items} for layer in (5, 16)}
    ratio_values = {layer: [cbenefit[layer][i] / a_improvement[layer][i] for i in items if abs(a_improvement[layer][i]) > 1e-3 and np.sign(a_improvement[layer][i]) > 0] for layer in (5, 16)}
    frac_positive_denominator = {str(layer): float(np.mean([a_improvement[layer][i] > 0 for i in items])) for layer in (5, 16)}
    effect_sizes = {"fraction_gap_closed_by_C": frac_effect, "relative_to_A_only": {"status": "not_primary; denominators are sign-unstable", "fraction_positive_A_only_improvement": frac_positive_denominator, "stable_ratio_values_only_if_A_improvement_gt_1e-3": {str(layer): {"n": len(values), "median": float(np.median(values)) if values else None, "mean": float(np.mean(values)) if values else None} for layer, values in ratio_values.items()}}}

    # Format-stratified correct/wrong sign results and family detail.
    wrong_format = {"definition": "benefit = distance(A-only,target)-distance(intervention,target)", "rows": [], "family_rows": []}
    for layer in (5, 16):
        for fmt in ("benchmark", "casual"):
            ids = [i for i in items if records_for_item(records, i)["source_format"] == fmt]
            for mode, label in ((full[layer], "correct_sign_C"), (wrong[layer], "wrong_sign_C")):
                vals = [main[layer][i]["raw_euclidean_error"] - mode[i]["raw_euclidean_error"] for i in ids]
                wrong_format["rows"].append({"layer": layer, "format": fmt, "condition": label, "n_cells": len(vals), "mean_C_increment": float(np.mean(vals)), "median_C_increment": float(np.median(vals))})
                for family in sorted(set(families[i] for i in ids)):
                    fids = [i for i in ids if families[i] == family]
                    wrong_format["family_rows"].append({"layer": layer, "format": fmt, "condition": label, "family_id": family, "n_cells": len(fids), "mean_C_increment": float(np.mean([main[layer][i]["raw_euclidean_error"] - mode[i]["raw_euclidean_error"] for i in fids]))})

    # Selectivity compares improvement toward the correct purpose target with improvement toward wrong-format target.
    selectivity = {"definition": "(zero correct-target error - mode correct-target error) - (zero wrong-format-target error - mode wrong-format-target error)", "rows": [], "family_rows": []}
    modes = [(full, "purpose_full"), (main, "purpose_main"), (wrong, "purpose_wrong_c"), (norm, "purpose_norm_matched"), (shuffled, "purpose_shuffled_c")]
    random_modes = sorted({r["mode"] for r in records if r["mode"].startswith("random_c_")})
    for rm in random_modes:
        modes.append(({layer: record_map(records, layer, rm) for layer in (5, 16)}, rm))
    for layer in (5, 16):
        for mode_map, label in modes:
            vals = []
            for item in items:
                r0, rm = zero[layer][item], mode_map[layer][item]
                correct = r0["raw_euclidean_error"] - rm["raw_euclidean_error"]
                wrong_improvement = r0["wrong_format_target_raw_error"] - rm["wrong_format_target_raw_error"]
                vals.append((item, correct, wrong_improvement, correct - wrong_improvement))
            selectivity["rows"].append({"layer": layer, "mode": label, "n_cells": len(vals), "correct_target_improvement": float(np.mean([x[1] for x in vals])), "wrong_format_target_improvement": float(np.mean([x[2] for x in vals])), "selectivity": float(np.mean([x[3] for x in vals]))})
            if not label.startswith("random_c_"):
                for family in sorted(set(families.values())):
                    fv = [x for x in vals if families[x[0]] == family]
                    selectivity["family_rows"].append({"layer": layer, "mode": label, "family_id": family, "n_cells": len(fv), "correct_target_improvement": float(np.mean([x[1] for x in fv])), "wrong_format_target_improvement": float(np.mean([x[2] for x in fv])), "selectivity": float(np.mean([x[3] for x in fv]))})

    random_values = []
    for mode in random_modes:
        m5, m16 = record_map(records, 5, mode), record_map(records, 16, mode)
        random_values.append({"mode": mode, "L5_C_benefit": float(np.mean([main[5][i]["raw_euclidean_error"] - m5[i]["raw_euclidean_error"] for i in items])), "L16_C_benefit": float(np.mean([main[16][i]["raw_euclidean_error"] - m16[i]["raw_euclidean_error"] for i in items]))})
    observed = primary["overall"]["L16_minus_L5"]
    random_d = np.asarray([x["L16_C_benefit"] - x["L5_C_benefit"] for x in random_values])
    random_sd = float(random_d.std(ddof=0))
    random_controls = {"n": len(random_d), "seed_bank": [int(mode.rsplit("_", 1)[1]) for mode in random_modes], "observed_L16_minus_L5": observed, "values": [{**x, "L16_minus_L5": x["L16_C_benefit"] - x["L5_C_benefit"]} for x in random_values], "mean": float(random_d.mean()), "sd": random_sd, "sd_definition": "population SD across the fixed 32-control bank", "descriptive_standardized_effect": float((observed - random_d.mean()) / random_sd), "finite_bank_empirical_statistic": float((1 + np.sum(random_d >= observed)) / (1 + len(random_d))), "n_random_ge_observed": int(np.sum(random_d >= observed)), "interpretation": "descriptive fixed-bank comparison; not a Gaussian z-test or population-level p-value"}

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in (SOURCE, PROTOCOL, EXECUTED_PROTOCOL, ITEMS, GEOMETRY_PROTOCOL, ACTIVATION_CONFIG, RUNNER, ANALYSIS_SCRIPT)}
    provenance = {"bundle_status": "immutable consolidation bundle", "created_from_git_commit": git_commit, "model": source["model"], "revision": source["revision"], "dtype_inference": "BF16", "dtype_stored": "float32", "stimulus_sha256": source["input_sha256"], "stimulus_count": len(rows), "layer_selection": load_json(GEOMETRY_PROTOCOL)["layer_selection_rule"], "factorial_definition": "h_(p,f)=mu+p*A+f*B+p*f*C+epsilon", "fit_scope": "training purpose families only; held-out family activations are never used for directions or strength", "strength_rule": "unit raw residual direction; alpha equals one training-block projection standard deviation", "readout": "continued-forward natural matched target distance at downstream L24; no behavioral endpoint", "excluded_readout": "intervention-layer/readout-layer L16/L16 records are excluded because capture_batches records the readout before applying the target-layer hook; no claim uses that pair", "executed_protocol": str(EXECUTED_PROTOCOL.relative_to(ROOT)), "wrapper_protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_gate_note": "The historical batch passed the generic executed protocol; the natural-control wrapper protocol records the additional target/control declarations. The runner validated model, revision, and input SHA; this bundle binds both hashes explicitly.", "fixed_seeds": {"analysis_bootstrap": SEED, "random_controls": random_controls["seed_bank"], "shuffle_C": 2026081616}, "source_artifact_sha256": source_hashes, "scientific_boundary": "Internal causal counterfactual transport/mediation under controlled prompt-provided purpose×format context; not behavioral steering."}
    report = make_report(primary, effect_sizes, wrong_format, selectivity, random_controls, provenance)
    OUT.mkdir(parents=True, exist_ok=False)
    for name, obj in (("primary_results.json", primary), ("family_results.json", {"family_table": family_rows, "sign_consistency": primary["sign_consistency"], "leave_one_family_out": leave_one, "leave_one_family_out_range": primary["leave_one_family_out_range"]}), ("effect_sizes.json", effect_sizes), ("wrong_sign_by_format.json", wrong_format), ("selectivity.json", selectivity), ("random_controls.json", random_controls), ("provenance.json", provenance)):
        (OUT / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    (OUT / "protocol.json").write_text(json.dumps({"schema_version": 1, "bundle_id": "causal_20260816_8b_bf16_mechanism_frozen", "executed_protocol": str(EXECUTED_PROTOCOL.relative_to(ROOT)), "wrapper_protocol": str(PROTOCOL.relative_to(ROOT)), "model": source["model"], "revision": source["revision"], "input_sha256": source["input_sha256"], "layers": {"shared": 5, "interaction_heavy": 16, "readout": 24}, "estimand": "C benefit = distance(A-only,target)-distance(full A+fC,target)", "target": "natural matched purpose-counterfactual representation at L24 after continued forward", "controls": ["wrong-sign C", "norm-matched A-only", "coordinate-shuffled C", "32 fixed random orthogonal-C directions", "zero"], "direct_endpoint": "failed predeclared purpose gate; excluded from causal claim", "protocol_gate_note": "The historical batch executed the generic protocol; the wrapper protocol and runner hash bind the natural-target/control declarations used for this consolidation.", "scientific_boundary": provenance["scientific_boundary"]}, indent=2) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"bundle": str(OUT), "git_commit": git_commit, "family_table": family_rows, "effect_sizes": effect_sizes, "random_controls": random_controls}, indent=2))


def records_for_item(records: list[dict[str, Any]], item: str) -> dict[str, Any]:
    for r in records:
        if r["source_item_id"] == item and r["readout_layer"] == 24 and r["intervention_layer"] == 5 and r["mode"] == "zero":
            return r
    raise KeyError(item)


def make_report(primary: dict[str, Any], effects: dict[str, Any], wrong: dict[str, Any], selectivity: dict[str, Any], random: dict[str, Any], provenance: dict[str, Any]) -> str:
    lines = ["# Frozen 8B BF16 mechanism bundle", "", "## Claim boundary", "", "The BF16 Llama-3.1-8B result tests internal causal counterfactual transport under a controlled prompt-provided purpose×format variable. The direct purpose-logit endpoint failed its frozen gate; no behavioral steering or spontaneous evaluation-awareness claim is made. All reported causal distances use downstream L24; the same-layer L16/L16 records are excluded because the runner records the readout before applying the target-layer hook.", "", "## Family robustness", "", "| family | payloads | L5 C benefit | L16 C benefit | L16−L5 |", "|---|---:|---:|---:|---:|"]
    for r in primary["family_table"]:
        lines.append(f"| {r['family_id']} | {r['n_payloads']} | {r['L5_C_benefit']:+.6f} | {r['L16_C_benefit']:+.6f} | {r['L16_minus_L5']:+.6f} |")
    lines += ["", f"{primary['sign_consistency']['positive_families']}/5 family contrasts are positive; ff03 is negative. Exact two-sided sign-test p={primary['sign_consistency']['exact_two_sided_sign_test_p']:.4f}; exact sign-flip one-sided finite-family statistic={primary['sign_consistency']['sign_flip_one_sided_p']:.4f}. Leave-one-family-out range: [{primary['leave_one_family_out_range'][0]:+.6f}, {primary['leave_one_family_out_range'][1]:+.6f}].", "", "## Effect sizes", "", f"Fraction of natural source-target gap closed by C: L5 mean {effects['fraction_gap_closed_by_C']['L5']['mean']:+.6f}, L16 mean {effects['fraction_gap_closed_by_C']['L16']['mean']:+.6f}, paired L16−L5 {effects['fraction_gap_closed_by_C']['L16_minus_L5']['mean']:+.6f}. Bootstrap intervals are in `effect_sizes.json`.", f"Random-bank descriptive standardized effect: {random['descriptive_standardized_effect']:+.3f}; finite-bank empirical statistic {(random['finite_bank_empirical_statistic']):.4f} = (1+{random['n_random_ge_observed']})/(1+{random['n']}). This is not a Gaussian z-test or population-level p-value.", "", "## Wrong-sign by format", "", "The format-stratified table is in `wrong_sign_by_format.json`; it reports correct-sign and wrong-sign C increments relative to A-only for benchmark and casual source cells at L5 and L16.", "", "## Selectivity", "", "Selectivity is defined once as improvement toward the correct purpose target minus improvement toward the wrong-format target. All components and family rows are in `selectivity.json`.", "", "## Provenance", "", f"Frozen from commit `{provenance['created_from_git_commit']}`; source hashes and fit scope are in `provenance.json`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
