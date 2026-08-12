"""Stronger development estimators for cue-invariant purpose probing.

This module was authored after the five-context development result left one
Meta/Japanese explicit-lexeme cell above the frozen 0.25-SD bound.  All recipes
below are therefore post-result development candidates.  They may be selected
only on spent development data and require a later untouched confirmation bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from analyze import apply_stats, lang_stats, pick_C
from cue_balanced import _loss_and_gradient
from cue_invariant import FAMILIES, LANGUAGES, PURPOSES, metadata_arrays


@dataclass(frozen=True)
class ConstrainedRecipe:
    name: str
    control_mass: float
    cue_penalty: float
    hard_families: tuple[str, ...] = ()


# Declared together before the second five-context development run.
#
# The first two test whether the previous soft penalty was simply too weak.
# The third exactly removes the canonical explicit-lexeme swap subspace while
# retaining a soft all-cue penalty.  The fourth exactly removes the row space of
# every observed cue swap.  No candidate changes the frozen gate thresholds.
CONSTRAINED_RECIPES = (
    ConstrainedRecipe("balanced_penalty_16", 1.0, 16.0),
    ConstrainedRecipe("balanced_penalty_64", 1.0, 64.0),
    ConstrainedRecipe(
        "hard_explicit_soft4", 1.0, 4.0, ("explicit_lexeme",)
    ),
    ConstrainedRecipe("hard_all", 1.0, 0.0, tuple(FAMILIES)),
)


# Declared together after round 1 showed that hard_all passed all cue-context
# gates but missed one Meta/Japanese core cell by 0.0059.  This final spent-data
# round maps the cue-control/core-generalization tradeoff without changing any
# gate.  A passing candidate still requires untouched confirmation.
CONSTRAINED_RECIPES_ROUND2 = (
    ConstrainedRecipe("balanced_penalty_256", 1.0, 256.0),
    ConstrainedRecipe(
        "hard_explicit_soft16", 1.0, 16.0, ("explicit_lexeme",)
    ),
    ConstrainedRecipe(
        "hard_explicit_soft64", 1.0, 64.0, ("explicit_lexeme",)
    ),
    ConstrainedRecipe("hard_all_core_only", 0.0, 0.0, tuple(FAMILIES)),
    ConstrainedRecipe("hard_all_control_quarter", 0.25, 0.0, tuple(FAMILIES)),
)


def cue_differences(
    x_control: np.ndarray,
    rows: list[dict[str, Any]],
    families: tuple[str, ...] = tuple(FAMILIES),
) -> np.ndarray:
    """Return exact evaluation-cue minus deployment-cue differences."""

    arrays = metadata_arrays(rows)
    differences: list[np.ndarray] = []
    for block in sorted(set(arrays["control_block_id"].tolist())):
        for family in families:
            for language in LANGUAGES:
                for purpose in PURPOSES:
                    base = (
                        (arrays["control_block_id"] == block)
                        & (arrays["cue_family"] == family)
                        & (arrays["language"] == language)
                        & (arrays["intended_purpose"] == purpose)
                    )
                    evaluation = base & (arrays["lexical_cue"] == "evaluation")
                    deployment = base & (arrays["lexical_cue"] == "deployment")
                    if evaluation.sum() != 1 or deployment.sum() != 1:
                        raise ValueError(
                            "hard cue constraints require one exact cue swap per "
                            "block/family/language/purpose cell"
                        )
                    differences.append(
                        x_control[evaluation][0] - x_control[deployment][0]
                    )
    if not differences:
        raise ValueError("no cue differences are available for the constraint")
    return np.stack(differences).astype(np.float64, copy=False)


def rowspace_basis(differences: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Numerically exact orthonormal basis for the observed cue row space."""

    _, singular, right = np.linalg.svd(differences, full_matrices=False)
    tolerance = float(
        np.finfo(np.float64).eps
        * max(differences.shape)
        * singular[0]
    )
    rank = int(np.sum(singular > tolerance))
    if rank < 1:
        raise ValueError("cue-difference matrix has zero numerical rank")
    basis = right[:rank]
    return basis, {
        "difference_vectors": int(len(differences)),
        "rank": rank,
        "tolerance": tolerance,
        "largest_singular_value": float(singular[0]),
        "smallest_retained_singular_value": float(singular[rank - 1]),
        "nullspace_dimension": int(differences.shape[1] - rank),
    }


def project_null(x: np.ndarray, basis: np.ndarray | None) -> np.ndarray:
    if basis is None or not len(basis):
        return x
    return x - (x @ basis.T) @ basis


def prepare_basis(
    x_fit_all: np.ndarray,
    data: dict[str, np.ndarray],
    x_control: np.ndarray,
    control_rows: list[dict[str, Any]],
    hard_families: tuple[str, ...],
    *,
    fit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    explicit = data["expl"] == "explicit"
    active = explicit if fit_mask is None else np.asarray(fit_mask, dtype=bool) & explicit
    stats = lang_stats(x_fit_all[active], data["lang"][active])
    languages = np.array([row["language"] for row in control_rows])
    standardized = apply_stats(x_control, languages, stats).astype(
        np.float64, copy=False
    )
    return rowspace_basis(cue_differences(standardized, control_rows, hard_families))


def fit_constrained(
    x_fit_all: np.ndarray,
    data: dict[str, np.ndarray],
    x_control: np.ndarray,
    control_rows: list[dict[str, Any]],
    recipe: ConstrainedRecipe,
    *,
    fit_mask: np.ndarray | None = None,
    selected_c: float | None = None,
    prepared_basis: tuple[np.ndarray, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    explicit = data["expl"] == "explicit"
    active = explicit if fit_mask is None else np.asarray(fit_mask, dtype=bool) & explicit
    if active.sum() < 10 or len(np.unique(data["y"][active])) != 2:
        raise ValueError("constrained fitting requires both explicit classes")

    stats = lang_stats(x_fit_all[active], data["lang"][active])
    x_fit = apply_stats(x_fit_all, data["lang"], stats).astype(np.float64, copy=False)
    control_language = np.array([row["language"] for row in control_rows])
    x_controls = apply_stats(x_control, control_language, stats).astype(
        np.float64, copy=False
    )
    constraint: dict[str, Any] | None = None
    basis: np.ndarray | None = None
    if recipe.hard_families:
        if prepared_basis is None:
            prepared_basis = rowspace_basis(
                cue_differences(x_controls, control_rows, recipe.hard_families)
            )
        basis, constraint = prepared_basis
        x_fit = project_null(x_fit, basis)
        x_controls = project_null(x_controls, basis)
        constraint = {
            **constraint,
            "families": list(recipe.hard_families),
            "maximum_projected_training_difference_norm": float(
                np.linalg.norm(
                    cue_differences(x_controls, control_rows, recipe.hard_families),
                    axis=1,
                ).max()
            ),
        }

    differences = cue_differences(x_controls, control_rows)
    y_control = np.array(
        [1 if row["intended_purpose"] == "evaluation" else 0 for row in control_rows],
        dtype=np.float64,
    )
    if selected_c is None:
        selected_c, inner_bacc = pick_C(x_fit_all, data, active)
    else:
        inner_bacc = float("nan")
    l2_strength = 1.0 / (float(selected_c) * int(active.sum()))
    theta0 = np.zeros(x_fit_all.shape[1] + 1, dtype=np.float64)

    stages = [recipe.cue_penalty]
    if recipe.cue_penalty > 16.0:
        stages = [16.0, recipe.cue_penalty]
    if recipe.cue_penalty > 64.0:
        stages = [16.0, 64.0, recipe.cue_penalty]
    stage_records: list[dict[str, Any]] = []
    result = None
    for cue_penalty in stages:
        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            return _loss_and_gradient(
                theta,
                x_fit[active],
                data["y"][active].astype(np.float64),
                x_controls,
                y_control,
                differences,
                control_mass=recipe.control_mass,
                cue_penalty=cue_penalty,
                l2_strength=l2_strength,
            )

        result = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 4000, "ftol": 1e-11, "gtol": 1e-7, "maxls": 60},
        )
        stage_records.append(
            {
                "cue_penalty": cue_penalty,
                "iterations": int(result.nit),
                "objective": float(result.fun),
                "gradient_max_abs": float(np.max(np.abs(result.jac))),
            }
        )
        if not result.success:
            raise RuntimeError(
                f"{recipe.name} optimization failed at continuation penalty "
                f"{cue_penalty}: {result.status} {result.message}; "
                f"iterations={result.nit}; gradient_max_abs="
                f"{float(np.max(np.abs(result.jac))):.6g}"
            )
        theta0 = result.x
    assert result is not None
    weight = result.x[:-1]
    intercept = float(result.x[-1])
    fit_score = x_fit[active] @ weight + intercept
    return {
        "recipe": {
            "name": recipe.name,
            "control_mass": recipe.control_mass,
            "cue_penalty": recipe.cue_penalty,
            "hard_families": list(recipe.hard_families),
        },
        "weight": weight,
        "intercept": intercept,
        "stats": stats,
        "basis": basis,
        "constraint": constraint,
        "fit_mask": active,
        "fit_score": fit_score,
        "selected_C": float(selected_c),
        "inner_cv_bacc": float(inner_bacc),
        "l2_strength": float(l2_strength),
        "optimization": {
            "iterations": int(result.nit),
            "objective": float(result.fun),
            "gradient_max_abs": float(np.max(np.abs(result.jac))),
            "continuation_stages": stage_records,
        },
    }


def score_constrained(
    estimator: dict[str, Any],
    x: np.ndarray,
    language: np.ndarray,
) -> np.ndarray:
    standardized = apply_stats(x, language, estimator["stats"]).astype(
        np.float64, copy=False
    )
    projected = project_null(standardized, estimator.get("basis"))
    return projected @ estimator["weight"] + estimator["intercept"]
