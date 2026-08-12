"""Shared cue-invariant projection for development and fresh confirmation.

The nuisance basis is learned only from matched cue swaps in the spent
format-matched development bank.  Intended purpose, payload block, cue family,
and language are held fixed inside every difference.  The smallest SVD basis
explaining at least 90% of squared singular-value mass is removed before the
explicit-only purpose classifier is fitted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from analyze import apply_stats, lang_stats, pick_C


VARIANCE_FRACTION = 0.90
FAMILIES = ("explicit_lexeme", "implicit_cue")
LANGUAGES = ("en", "ja")
PURPOSES = ("evaluation", "deployment")


def read_metadata(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def metadata_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        key: np.array([row[key] for row in rows])
        for key in (
            "control_block_id",
            "cue_family",
            "intended_purpose",
            "language",
            "lexical_cue",
            "surface",
        )
    }


def cue_difference_matrix(
    x_control: np.ndarray,
    rows: list[dict[str, Any]],
    keep: np.ndarray | None = None,
) -> np.ndarray:
    """Matched evaluation-cue minus deployment-cue activation differences."""

    arrays = metadata_arrays(rows)
    if keep is None:
        keep = np.ones(len(rows), dtype=bool)
    differences: list[np.ndarray] = []
    blocks = sorted(set(arrays["control_block_id"][keep].tolist()))
    for block in blocks:
        for family in FAMILIES:
            for language in LANGUAGES:
                for purpose in PURPOSES:
                    base = (
                        keep
                        & (arrays["control_block_id"] == block)
                        & (arrays["cue_family"] == family)
                        & (arrays["language"] == language)
                        & (arrays["intended_purpose"] == purpose)
                    )
                    evaluation = base & (arrays["lexical_cue"] == "evaluation")
                    deployment = base & (arrays["lexical_cue"] == "deployment")
                    if evaluation.sum() != 1 or deployment.sum() != 1:
                        raise ValueError(
                            "cue-invariant development controls require one exact "
                            "cue swap per block/family/language/purpose cell"
                        )
                    differences.append(
                        x_control[evaluation][0] - x_control[deployment][0]
                    )
    if not differences:
        raise ValueError("no matched cue differences are available")
    return np.stack(differences)


def nuisance_basis(
    x_control: np.ndarray,
    rows: list[dict[str, Any]],
    keep: np.ndarray | None = None,
    variance_fraction: float = VARIANCE_FRACTION,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not 0 < variance_fraction <= 1:
        raise ValueError("variance_fraction must lie in (0, 1]")
    differences = cue_difference_matrix(x_control, rows, keep)
    _, singular_values, right = np.linalg.svd(differences, full_matrices=False)
    mass = singular_values * singular_values
    cumulative = np.cumsum(mass) / mass.sum()
    rank = int(np.searchsorted(cumulative, variance_fraction) + 1)
    basis = right[:rank].T
    return basis, {
        "method": "matched_cue_difference_svd",
        "variance_fraction": variance_fraction,
        "rank": rank,
        "difference_vectors": int(len(differences)),
        "variance_explained": float(cumulative[rank - 1]),
    }


def compact_mean_basis(
    x_control: np.ndarray,
    rows: list[dict[str, Any]],
    keep: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Four-axis development comparator: cue family by language."""

    arrays = metadata_arrays(rows)
    if keep is None:
        keep = np.ones(len(rows), dtype=bool)
    differences = []
    for family in FAMILIES:
        for language in LANGUAGES:
            base = keep & (arrays["cue_family"] == family) & (
                arrays["language"] == language
            )
            evaluation = base & (arrays["lexical_cue"] == "evaluation")
            deployment = base & (arrays["lexical_cue"] == "deployment")
            differences.append(
                x_control[evaluation].mean(0) - x_control[deployment].mean(0)
            )
    basis, _ = np.linalg.qr(np.stack(differences).T, mode="reduced")
    return basis, {
        "method": "four_mean_cue_axes",
        "rank": int(basis.shape[1]),
        "difference_vectors": 4,
    }


def project(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return x.copy()
    return x - (x @ basis) @ basis.T


def fit_probe(
    x_fit_all: np.ndarray,
    data: dict[str, np.ndarray],
    x_development_controls: np.ndarray,
    development_rows: list[dict[str, Any]],
    *,
    control_keep: np.ndarray | None = None,
    method: str = "svd90",
) -> dict[str, Any]:
    """Fit the frozen explicit-only classifier and its nuisance projection."""

    fit_mask = data["expl"] == "explicit"
    stats = lang_stats(x_fit_all[fit_mask], data["lang"][fit_mask])
    x_fit_standard = apply_stats(x_fit_all, data["lang"], stats)
    control_language = np.array([row["language"] for row in development_rows])
    x_control_standard = apply_stats(
        x_development_controls, control_language, stats
    )
    if method == "svd90":
        basis, basis_record = nuisance_basis(
            x_control_standard,
            development_rows,
            control_keep,
            VARIANCE_FRACTION,
        )
    elif method == "axes4":
        basis, basis_record = compact_mean_basis(
            x_control_standard, development_rows, control_keep
        )
    elif method == "baseline":
        basis = np.zeros((x_fit_all.shape[1], 0), dtype=x_fit_all.dtype)
        basis_record = {"method": "none", "rank": 0, "difference_vectors": 0}
    else:
        raise ValueError(f"unknown cue-invariance method: {method}")
    x_projected = project(x_fit_standard, basis)
    selected_c, inner_bacc = pick_C(x_projected, data, fit_mask)
    classifier = LogisticRegression(max_iter=5000, C=selected_c).fit(
        x_projected[fit_mask], data["y"][fit_mask]
    )
    fit_score = classifier.decision_function(x_projected[fit_mask])
    return {
        "basis": basis,
        "basis_record": basis_record,
        "classifier": classifier,
        "fit_mask": fit_mask,
        "fit_score": fit_score,
        "inner_cv_bacc": float(inner_bacc),
        "selected_C": float(selected_c),
        "stats": stats,
        "x_fit_projected": x_projected,
    }


def score_controls(
    fitted: dict[str, Any],
    x_control: np.ndarray,
    rows: list[dict[str, Any]],
) -> np.ndarray:
    language = np.array([row["language"] for row in rows])
    standardized = apply_stats(x_control, language, fitted["stats"])
    return fitted["classifier"].decision_function(
        project(standardized, fitted["basis"])
    )
