"""Grouped standard-versus-decorrelated probe analysis for the English 2x2."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import MVPError, load_jsonl, write_json


def _analysis_imports():
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise MVPError("Probe analysis requires the 'analysis' optional dependencies") from exc
    return np, LogisticRegression, balanced_accuracy_score, roc_auc_score, StandardScaler


def _fit_score(X_train, y_train, X_test, *, c: float):
    np, LogisticRegression, _, _, StandardScaler = _analysis_imports()
    if len(set(y_train.tolist())) != 2:
        raise MVPError("A probe training fold does not contain both purpose classes")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)
    classifier = LogisticRegression(C=c, max_iter=3000, random_state=0)
    classifier.fit(train_scaled, y_train)
    scores = classifier.decision_function(test_scaled)
    return np.asarray(scores), (np.asarray(scores) >= 0).astype(int)


def _safe_auc(y, scores) -> float | None:
    _, _, _, roc_auc_score, _ = _analysis_imports()
    return float(roc_auc_score(y, scores)) if len(set(y.tolist())) == 2 else None


def _purpose_metrics(rows, y, predictions, scores, mask) -> dict[str, Any]:
    np, _, balanced_accuracy_score, _, _ = _analysis_imports()
    if int(mask.sum()) == 0:
        return {"n": 0}
    local_y = y[mask]
    local_predictions = predictions[mask]
    local_scores = scores[mask]
    result: dict[str, Any] = {
        "n": int(mask.sum()),
        "accuracy": float((local_predictions == local_y).mean()),
        "balanced_accuracy": float(
            balanced_accuracy_score(local_y, local_predictions)
        ),
        "purpose_auc": _safe_auc(local_y, local_scores),
    }
    by_cell: dict[str, Any] = {}
    purpose = np.asarray([row["intended_purpose"] for row in rows])
    format_name = np.asarray([row["format"] for row in rows])
    for purp in ("evaluation", "deployment"):
        for fmt in ("benchmark", "casual"):
            cell = mask & (purpose == purp) & (format_name == fmt)
            if cell.sum():
                by_cell[f"{purp}::{fmt}"] = {
                    "n": int(cell.sum()),
                    "percent_predicted_evaluation": float(predictions[cell].mean() * 100),
                    "mean_score": float(scores[cell].mean()),
                }
    result["by_cell"] = by_cell
    return result


def _paired_deltas(rows, scores, mask) -> dict[str, Any]:
    np, _, _, _, _ = _analysis_imports()
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for index, row in enumerate(rows):
        if mask[index]:
            grouped[(row["payload_block_id"], row["format"])][
                row["intended_purpose"]
            ] = float(scores[index])
    by_format: dict[str, Any] = {}
    for fmt in ("benchmark", "casual"):
        deltas = [
            values["evaluation"] - values["deployment"]
            for (block_id, format_name), values in grouped.items()
            if format_name == fmt and set(values) == {"evaluation", "deployment"}
        ]
        if deltas:
            values = np.asarray(deltas)
            by_format[fmt] = {
                "n_blocks": len(deltas),
                "mean_eval_minus_deploy_score": float(values.mean()),
                "proportion_positive": float((values > 0).mean()),
            }
    return by_format


def _oof_regime(X, rows, y, train_selector, test_selector, *, c: float):
    np, _, _, _, _ = _analysis_imports()
    families = np.asarray([row["purpose_family_id"] for row in rows])
    scores = np.full(len(rows), np.nan, dtype=float)
    predictions = np.full(len(rows), -1, dtype=int)
    evaluated = np.zeros(len(rows), dtype=bool)
    for held_family in sorted(set(families.tolist())):
        training = (families != held_family) & train_selector
        testing = (families == held_family) & test_selector
        if not testing.any():
            continue
        fold_scores, fold_predictions = _fit_score(
            X[training], y[training], X[testing], c=c
        )
        scores[testing] = fold_scores
        predictions[testing] = fold_predictions
        evaluated[testing] = True
    if not evaluated.any() or np.isnan(scores[evaluated]).any():
        raise MVPError("Grouped analysis did not produce complete out-of-fold scores")
    return scores, predictions, evaluated


def analyze_activations(
    *, items_path: Path, activations_path: Path, output_path: Path, c: float = 0.1
) -> dict[str, Any]:
    np, _, balanced_accuracy_score, _, _ = _analysis_imports()
    if c <= 0:
        raise MVPError("Regularization C must be positive")
    rows = load_jsonl(items_path)
    archive = np.load(activations_path, allow_pickle=False)
    X = np.asarray(archive["X"], dtype=float)
    activation_ids = [str(value) for value in archive["item_ids"].tolist()]
    item_ids = [str(row["item_id"]) for row in rows]
    if activation_ids != item_ids or X.shape[0] != len(rows):
        raise MVPError("Activation rows do not align exactly with rendered item IDs")

    y = np.asarray([int(row["label"]) for row in rows], dtype=int)
    purposes = np.asarray([row["intended_purpose"] for row in rows])
    formats = np.asarray([row["format"] for row in rows])
    all_rows = np.ones(len(rows), dtype=bool)
    standard_training = (
        ((purposes == "evaluation") & (formats == "benchmark"))
        | ((purposes == "deployment") & (formats == "casual"))
    )

    regimes: dict[str, Any] = {}
    for name, train_selector, test_selector in (
        ("standard_confounded", standard_training, all_rows),
        ("decorrelated", all_rows, all_rows),
        ("benchmark_to_casual", formats == "benchmark", formats == "casual"),
        ("casual_to_benchmark", formats == "casual", formats == "benchmark"),
    ):
        scores, predictions, evaluated = _oof_regime(
            X, rows, y, train_selector, test_selector, c=c
        )
        metrics = _purpose_metrics(rows, y, predictions, scores, evaluated)
        metrics["paired_eval_minus_deploy"] = _paired_deltas(rows, scores, evaluated)
        regimes[name] = metrics

    format_y = (formats == "benchmark").astype(int)
    format_scores, format_predictions, format_evaluated = _oof_regime(
        X, rows, format_y, all_rows, all_rows, c=c
    )
    result = {
        "schema_version": 1,
        "status": "development_diagnostic",
        "split_unit": "purpose_family_id_with_payload_blocks_kept_complete",
        "regularization_C": c,
        "counts": {
            "items": len(rows),
            "payload_blocks": len({row["payload_block_id"] for row in rows}),
            "purpose_families": len({row["purpose_family_id"] for row in rows}),
            "hidden_size": int(X.shape[1]),
        },
        "purpose_probe_regimes": regimes,
        "format_probe": {
            "n": int(format_evaluated.sum()),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    format_y[format_evaluated], format_predictions[format_evaluated]
                )
            ),
            "format_auc": _safe_auc(
                format_y[format_evaluated], format_scores[format_evaluated]
            ),
        },
    }
    write_json(output_path, result)
    return result

