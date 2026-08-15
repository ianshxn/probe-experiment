"""Cross-fitted 2x2 factorial geometry and transfer diagnostics."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if n == 0.0 else v / n


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    ua, ub = _unit(a), _unit(b)
    return None if ua is None or ub is None else float(np.dot(ua, ub))


def _principal_cosines(a: np.ndarray, b: np.ndarray, rank: int = 5) -> list[float]:
    if min(a.shape[0], b.shape[0]) == 0:
        return []
    qa, _ = np.linalg.qr(a.T)
    qb, _ = np.linalg.qr(b.T)
    singular = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return [float(value) for value in singular[:rank]]


def _validate_cells(rows: Sequence[dict[str, Any]]) -> dict[str, dict[tuple[str, str], int]]:
    cells: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    for index, row in enumerate(rows):
        key = (str(row["intended_purpose"]), str(row["format"]))
        block = str(row["payload_block_id"])
        if key in cells[block]:
            raise ValueError(f"duplicate factorial cell for block {block}: {key}")
        cells[block][key] = index
    expected = {("evaluation", "benchmark"), ("evaluation", "casual"), ("deployment", "benchmark"), ("deployment", "casual")}
    if any(set(block_cells) != expected for block_cells in cells.values()):
        raise ValueError("every payload block must contain exactly one complete 2x2 factorial")
    return cells


def factorial_components(x: np.ndarray, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct h_EB/h_EC/h_DB/h_DC and fixed A/B/C per payload block."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != len(rows) or not np.isfinite(x).all():
        raise ValueError("factorial activations must be finite N x D and align with rows")
    cells = _validate_cells(rows)
    block_ids = sorted(cells)
    values = {
        block: {
            "EB": x[cells[block][("evaluation", "benchmark")]],
            "EC": x[cells[block][("evaluation", "casual")]],
            "DB": x[cells[block][("deployment", "benchmark")]],
            "DC": x[cells[block][("deployment", "casual")]],
        }
        for block in block_ids
    }
    h = np.stack([[values[b][key] for key in ("EB", "EC", "DB", "DC")] for b in block_ids])
    delta_b = h[:, 0] - h[:, 2]
    delta_c = h[:, 1] - h[:, 3]
    return {
        "block_ids": block_ids,
        "families": np.asarray([str(rows[cells[b][("evaluation", "benchmark")]]["purpose_family_id"]) for b in block_ids]),
        "h": h,
        "delta_B": delta_b,
        "delta_C": delta_c,
        "A": (delta_b + delta_c) / 4.0,
        "B": ((h[:, 0] + h[:, 2]) - (h[:, 1] + h[:, 3])) / 4.0,
        "C": (delta_b - delta_c) / 4.0,
    }


def _fit_score(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    model = LogisticRegression(C=0.1, max_iter=3000, random_state=0)
    model.fit(scaler.fit_transform(train_x), train_y)
    return np.asarray(model.decision_function(scaler.transform(test_x)), dtype=float)


def _transfer_metrics(x: np.ndarray, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    y = np.asarray([int(row["label"]) for row in rows])
    family = np.asarray([str(row["purpose_family_id"]) for row in rows])
    fmt = np.asarray([str(row["format"]) for row in rows])
    scores = {name: np.full(len(rows), np.nan) for name in ("pooled", "benchmark_to_casual", "casual_to_benchmark")}
    for held in sorted(set(family)):
        train_family, test_family = family != held, family == held
        regimes = {
            "pooled": (np.ones(len(rows), dtype=bool), np.ones(len(rows), dtype=bool)),
            "benchmark_to_casual": (fmt == "benchmark", fmt == "casual"),
            "casual_to_benchmark": (fmt == "casual", fmt == "benchmark"),
        }
        for name, (train_selector, test_selector) in regimes.items():
            train, test = train_family & train_selector, test_family & test_selector
            scores[name][test] = _fit_score(x[train], y[train], x[test])
    out: dict[str, Any] = {}
    for name, score in scores.items():
        mask = np.isfinite(score)
        pred = (score >= 0).astype(int)
        out[name] = {
            "n": int(mask.sum()),
            "auc": float(roc_auc_score(y[mask], score[mask])),
            "balanced_accuracy": float(balanced_accuracy_score(y[mask], pred[mask])),
            "per_family": {
                held: {
                    "auc": float(roc_auc_score(y[m], score[m])),
                    "balanced_accuracy": float(balanced_accuracy_score(y[m], pred[m])),
                    "n": int(m.sum()),
                }
                for held in sorted(set(family))
                if (m := mask & (family == held)).sum() and len(np.unique(y[m])) == 2
            },
        }
    return out


def _geometry_fold(comp: dict[str, Any], held: str) -> dict[str, Any]:
    train, test = comp["families"] != held, comp["families"] == held
    db_train, dc_train = comp["delta_B"][train], comp["delta_C"][train]
    dirs = {
        "benchmark": db_train.mean(0),
        "casual": dc_train.mean(0),
        "pooled": (db_train.mean(0) + dc_train.mean(0)) / 2.0,
    }
    a_test, b_test, c_test = comp["A"][test].mean(0), comp["B"][test].mean(0), comp["C"][test].mean(0)
    ratio = float(np.linalg.norm(c_test) / np.linalg.norm(a_test)) if np.linalg.norm(a_test) else None
    projections = {}
    for name, direction in dirs.items():
        unit = _unit(direction)
        projections[name] = None if unit is None else float(np.dot(b_test, unit))
    return {
        "held_out_family": held,
        "n_test_blocks": int(test.sum()),
        "norm_A": float(np.linalg.norm(a_test)),
        "norm_B": float(np.linalg.norm(b_test)),
        "norm_C": float(np.linalg.norm(c_test)),
        "norm_C_over_norm_A": ratio,
        "delta_cosine": _cosine(comp["delta_B"][test].mean(0), comp["delta_C"][test].mean(0)),
        "purpose_direction_cosine": _cosine(dirs["benchmark"], dirs["casual"]),
        "principal_cosines": _principal_cosines(comp["delta_B"][test], comp["delta_C"][test]),
        "B_projection": projections,
    }


def analyze_geometry(x: np.ndarray, rows: Sequence[dict[str, Any]], *, layer: int | None = None) -> dict[str, Any]:
    comp = factorial_components(x, rows)
    folds = [_geometry_fold(comp, held) for held in sorted(set(comp["families"]))]
    scalar_keys = ("norm_A", "norm_B", "norm_C", "norm_C_over_norm_A", "delta_cosine", "purpose_direction_cosine")
    summary = {
        key: float(np.nanmean([fold[key] for fold in folds if fold[key] is not None]))
        if any(fold[key] is not None for fold in folds) else None
        for key in scalar_keys
    }
    result = {"layer": layer, "summary": summary, "per_held_family": folds}
    return result


def synthetic_factorial_case(*, a: float, b: float, c: float, seed: int = 7, n_blocks: int = 24, dimension: int = 8) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    x: list[np.ndarray] = []
    for block in range(n_blocks):
        family = f"family_{block % 6}"
        mu = rng.normal(scale=0.05, size=dimension)
        for purpose, p in (("evaluation", 1.0), ("deployment", -1.0)):
            for fmt, f in (("benchmark", 1.0), ("casual", -1.0)):
                rows.append({"payload_block_id": f"block_{block}", "purpose_family_id": family, "intended_purpose": purpose, "format": fmt, "label": int(p == 1), "item_id": f"{block}_{purpose}_{fmt}"})
                x.append(mu + p * a * np.eye(1, dimension, 0)[0] + f * b * np.eye(1, dimension, 1)[0] + p * f * c * np.eye(1, dimension, 2)[0] + rng.normal(scale=0.005, size=dimension))
    return np.asarray(x), rows
