"""Equal-training-row mixed-format controls for the English 2x2 probe.

The unit held out by this module is ``purpose_family_id``.  A training payload
block contributes its evaluation/deployment pair from one format only in the
equal-N condition, so benchmark-only, casual-only, and equal-N mixed-format
training use identical row counts within every outer fold.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

EXPECTED_CELLS = {
    ("evaluation", "benchmark"),
    ("evaluation", "casual"),
    ("deployment", "benchmark"),
    ("deployment", "casual"),
}


def _validate_rows(rows: Sequence[dict[str, Any]]) -> None:
    item_ids: set[str] = set()
    by_block: dict[str, set[tuple[str, str]]] = defaultdict(set)
    block_families: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        item_id = str(row["item_id"])
        if item_id in item_ids:
            raise ValueError(f"duplicate item_id: {item_id}")
        item_ids.add(item_id)
        block = str(row["payload_block_id"])
        cell = (str(row["intended_purpose"]), str(row["format"]))
        by_block[block].add(cell)
        block_families[block].add(str(row["purpose_family_id"]))
        if "split_group_id" in row and str(row["split_group_id"]) != block:
            raise ValueError(f"split_group_id mismatch for block {block}")
        expected_label = 1 if row["intended_purpose"] == "evaluation" else 0
        if int(row["label"]) != expected_label:
            raise ValueError(f"label/purpose mismatch for item {item_id}")
    if any(cells != EXPECTED_CELLS for cells in by_block.values()):
        raise ValueError("every payload block must contain exactly one complete 2x2")
    if any(len(families) != 1 for families in block_families.values()):
        raise ValueError("each payload block must belong to one purpose family")


def _fit_score(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, c: float):
    if len(np.unique(y_train)) != 2:
        raise ValueError("training fold does not contain both purpose classes")
    scaler = StandardScaler()
    classifier = LogisticRegression(C=c, max_iter=3000, random_state=0)
    classifier.fit(scaler.fit_transform(x_train), y_train)
    scores = np.asarray(classifier.decision_function(scaler.transform(x_test)), dtype=float)
    return scores, (scores >= 0).astype(int)


def _metric(y: np.ndarray, scores: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    idx = np.flatnonzero(mask & np.isfinite(scores))
    if len(idx) == 0:
        return {"n": 0, "auc": None, "balanced_accuracy": None}
    yy, ss, pp = y[idx], scores[idx], predictions[idx]
    auc = float(roc_auc_score(yy, ss)) if len(np.unique(yy)) == 2 else None
    return {
        "n": int(len(idx)),
        "auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(yy, pp)) if len(np.unique(yy)) == 2 else None,
    }


def _paired(rows: Sequence[dict[str, Any]], scores: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for i, row in enumerate(rows):
        if mask[i] and np.isfinite(scores[i]):
            grouped[(str(row["payload_block_id"]), str(row["format"]))][str(row["intended_purpose"])] = float(scores[i])
    out: dict[str, Any] = {}
    for fmt in ("benchmark", "casual"):
        deltas = [
            values["evaluation"] - values["deployment"]
            for (block, current_fmt), values in grouped.items()
            if current_fmt == fmt and set(values) == {"evaluation", "deployment"}
        ]
        if deltas:
            d = np.asarray(deltas)
            out[fmt] = {
                "n_blocks": int(len(d)),
                "mean_eval_minus_deploy": float(d.mean()),
                "sign_consistency": float((d > 0).mean()),
            }
    return out


def _assignment(blocks: Sequence[str], seed: int) -> dict[str, str]:
    """Return a deterministic, balanced block-to-format assignment."""
    ordered = np.asarray(sorted(map(str, blocks)), dtype=object)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ordered))
    split = (len(ordered) + 1) // 2
    assignment: dict[str, str] = {}
    for rank, position in enumerate(perm):
        assignment[str(ordered[position])] = "benchmark" if rank < split else "casual"
    return assignment


def _fold_scores(
    x: np.ndarray,
    y: np.ndarray,
    rows: Sequence[dict[str, Any]],
    families: np.ndarray,
    held_family: str,
    train_selector: np.ndarray,
    test_selector: np.ndarray,
    c: float,
) -> tuple[np.ndarray, np.ndarray]:
    training = (families != held_family) & train_selector
    testing = (families == held_family) & test_selector
    scores = np.full(len(rows), np.nan, dtype=float)
    predictions = np.full(len(rows), -1, dtype=int)
    if testing.any():
        scores[testing], predictions[testing] = _fit_score(x[training], y[training], x[testing], c)
    return scores, predictions


def _regime_summary(
    rows: Sequence[dict[str, Any]],
    y: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    test_selector: np.ndarray,
    families: np.ndarray,
) -> dict[str, Any]:
    result = _metric(y, scores, predictions, test_selector)
    result["paired_eval_minus_deploy"] = _paired(rows, scores, test_selector)
    result["per_family"] = {
        str(family): _metric(y, scores, predictions, test_selector & (families == family))
        for family in sorted(set(families.tolist()))
        if (test_selector & (families == family)).any()
    }
    return result


def _family_bootstrap(
    seed_results: list[dict[str, Any]],
    families: Sequence[str],
    *,
    n_resamples: int = 1000,
    seed: int = 20260815,
) -> dict[str, Any]:
    """Conditional family bootstrap over fixed OOF fold predictions.

    The fitted fold models remain fixed; this is uncertainty over the
    held-out-family generalization units, not a refit bootstrap.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for regime in seed_results[0]["regimes"]:
        auc_samples: list[float] = []
        ba_samples: list[float] = []
        for _ in range(n_resamples):
            chosen = rng.choice(len(families), size=len(families), replace=True)
            auc_values: list[float] = []
            ba_values: list[float] = []
            for seed_result in seed_results:
                per_family = seed_result["regimes"][regime]["per_family"]
                auc_values.extend(per_family[families[index]]["auc"] for index in chosen)
                ba_values.extend(per_family[families[index]]["balanced_accuracy"] for index in chosen)
            auc_samples.append(float(np.nanmean(auc_values)))
            ba_samples.append(float(np.nanmean(ba_values)))
        out[regime] = {
            "n_resamples": n_resamples,
            "seed": seed,
            "unit": "purpose_family_id",
            "conditional_on_fixed_fold_fits": True,
            "auc_ci95": [float(np.percentile(auc_samples, 2.5)), float(np.percentile(auc_samples, 97.5))],
            "balanced_accuracy_ci95": [float(np.percentile(ba_samples, 2.5)), float(np.percentile(ba_samples, 97.5))],
        }
    return out


def run_equal_n(
    x: np.ndarray,
    rows: Sequence[dict[str, Any]],
    *,
    c: float = 0.1,
    seeds: Iterable[int] = (1729, 2718, 31415, 4242, 8675309),
) -> dict[str, Any]:
    """Run equal-N and reference regimes with family-held-out OOF scores."""
    x = np.asarray(x, dtype=float)
    _validate_rows(rows)
    if x.ndim != 2 or x.shape[0] != len(rows):
        raise ValueError("x must be a two-dimensional array aligned with rows")
    if c <= 0:
        raise ValueError("regularization C must be positive")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("at least one fixed assignment seed is required")
    y = np.asarray([int(row["label"]) for row in rows], dtype=int)
    families = np.asarray([str(row["purpose_family_id"]) for row in rows])
    formats = np.asarray([str(row["format"]) for row in rows])
    blocks = np.asarray([str(row["payload_block_id"]) for row in rows])
    all_rows = np.ones(len(rows), dtype=bool)
    families_sorted = sorted(set(families.tolist()))
    seed_results: list[dict[str, Any]] = []

    for seed in seeds:
        scores_by_regime = {
            name: np.full(len(rows), np.nan, dtype=float)
            for name in ("benchmark_only", "casual_only", "equal_n_mixed", "pooled", "benchmark_to_casual", "casual_to_benchmark", "benchmark_to_benchmark", "casual_to_casual")
        }
        predictions_by_regime = {name: np.full(len(rows), -1, dtype=int) for name in scores_by_regime}
        training_counts: dict[str, dict[str, int]] = {}
        assignments: dict[str, dict[str, str]] = {}
        for held in families_sorted:
            train_rows = families != held
            train_blocks = sorted(set(blocks[train_rows].tolist()))
            assigned = _assignment(train_blocks, seed + sum(map(ord, held)))
            assignments[held] = assigned
            benchmark = formats == "benchmark"
            casual = formats == "casual"
            equal = np.asarray([assigned.get(block, "") == fmt for block, fmt in zip(blocks, formats)], dtype=bool)
            selectors = {
                "benchmark_only": benchmark,
                "casual_only": casual,
                "equal_n_mixed": equal,
                "pooled": all_rows,
                "benchmark_to_casual": benchmark,
                "casual_to_benchmark": casual,
                "benchmark_to_benchmark": benchmark,
                "casual_to_casual": casual,
            }
            tests = {
                "benchmark_only": all_rows,
                "casual_only": all_rows,
                "equal_n_mixed": all_rows,
                "pooled": all_rows,
                "benchmark_to_casual": casual,
                "casual_to_benchmark": benchmark,
                "benchmark_to_benchmark": benchmark,
                "casual_to_casual": casual,
            }
            training_counts[held] = {
                name: int((train_rows & selector).sum())
                for name, selector in selectors.items()
            }
            for name, selector in selectors.items():
                fold_scores, fold_predictions = _fold_scores(
                    x, y, rows, families, held, selector, tests[name], c
                )
                test_mask = (families == held) & tests[name]
                scores_by_regime[name][test_mask] = fold_scores[test_mask]
                predictions_by_regime[name][test_mask] = fold_predictions[test_mask]

        seed_result: dict[str, Any] = {
            "seed": seed,
            "assignment_by_held_family": assignments,
            "training_row_counts_by_held_family": training_counts,
            "regimes": {
                name: _regime_summary(rows, y, scores, predictions, all_rows, families)
                for name, scores in scores_by_regime.items()
                for predictions in [predictions_by_regime[name]]
            },
        }
        seed_results.append(seed_result)

    first_counts = [
        tuple(seed_result["training_row_counts_by_held_family"][held][name] for name in ("benchmark_only", "casual_only", "equal_n_mixed"))
        for seed_result in seed_results
        for held in families_sorted
    ]
    equalized = all(counts[0] == counts[1] == counts[2] for counts in first_counts)
    aggregate: dict[str, Any] = {}
    regime_names = tuple(seed_results[0]["regimes"])
    for name in regime_names:
        metrics = [seed_result["regimes"][name] for seed_result in seed_results]
        aggregate[name] = {
            "auc_by_seed": [metric["auc"] for metric in metrics],
            "balanced_accuracy_by_seed": [metric["balanced_accuracy"] for metric in metrics],
            "mean_auc": float(np.nanmean([metric["auc"] for metric in metrics])),
            "mean_balanced_accuracy": float(np.nanmean([metric["balanced_accuracy"] for metric in metrics])),
            "per_seed": metrics,
        }
    family_bootstrap = _family_bootstrap(seed_results, families_sorted)


    return {
        "schema_version": 1,
        "analysis": "equal_n_mixed_format_control",
        "status": "development_diagnostic",
        "regularization_C": c,
        "seed_bank": list(seeds),
        "split_unit": "leave_one_purpose_family_out; payload blocks and eval/deploy pairs kept together",
        "counts": {
            "items": len(rows),
            "payload_blocks": int(len(set(blocks.tolist()))),
            "purpose_families": len(families_sorted),
            "hidden_size": int(x.shape[1]),
        },
        "equal_training_rows_first_three": equalized,
        "family_bootstrap": family_bootstrap,
        "seed_results": seed_results,
        "aggregate": aggregate,
    }
