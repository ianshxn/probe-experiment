"""Cue-balanced linear purpose estimators for spent-bank development.

Unlike SVD nuisance removal, these estimators target the classifier's score
change under a matched cue swap directly.  The objective is convex:

    explicit purpose log loss
    + control_mass * cue-balanced control purpose log loss
    + cue_penalty * mean((difference @ weight) ** 2)
    + L2 penalty.

The matched differences hold payload block, intended purpose, cue family, and
language fixed.  All fitting and tuning performed with this module is
development-only until a new untouched bank is frozen and scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from analyze import apply_stats, lang_stats, pick_C
from cue_invariant import cue_difference_matrix


@dataclass(frozen=True)
class Recipe:
    name: str
    control_mass: float
    cue_penalty: float


# Declared before running the reciprocal-bank comparison.  The penalty is the
# mean squared change in raw decision score under an exact matched cue swap.
RECIPES = (
    Recipe("cue_balanced", control_mass=1.0, cue_penalty=0.0),
    Recipe("cue_penalty", control_mass=0.0, cue_penalty=1.0),
    Recipe("balanced_penalty_1", control_mass=1.0, cue_penalty=1.0),
    Recipe("balanced_penalty_4", control_mass=1.0, cue_penalty=4.0),
)


def _logistic_loss(z: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def _loss_and_gradient(
    theta: np.ndarray,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_control: np.ndarray,
    y_control: np.ndarray,
    differences: np.ndarray,
    *,
    control_mass: float,
    cue_penalty: float,
    l2_strength: float,
) -> tuple[float, np.ndarray]:
    weight = theta[:-1]
    intercept = float(theta[-1])
    fit_score = x_fit @ weight + intercept
    fit_error = expit(fit_score) - y_fit
    value = _logistic_loss(fit_score, y_fit)
    grad_weight = x_fit.T @ fit_error / len(x_fit)
    grad_intercept = float(fit_error.mean())

    if control_mass:
        control_score = x_control @ weight + intercept
        control_error = expit(control_score) - y_control
        value += control_mass * _logistic_loss(control_score, y_control)
        grad_weight += control_mass * (
            x_control.T @ control_error / len(x_control)
        )
        grad_intercept += control_mass * float(control_error.mean())

    if cue_penalty:
        cue_change = differences @ weight
        value += cue_penalty * float(np.mean(cue_change * cue_change))
        grad_weight += (
            2.0
            * cue_penalty
            * (differences.T @ cue_change)
            / len(differences)
        )

    value += 0.5 * l2_strength * float(weight @ weight)
    grad_weight += l2_strength * weight
    gradient = np.concatenate([grad_weight, np.array([grad_intercept])])
    return value, gradient


def fit_estimator(
    x_fit_all: np.ndarray,
    data: dict[str, np.ndarray],
    x_control: np.ndarray,
    control_rows: list[dict[str, Any]],
    recipe: Recipe,
    *,
    fit_mask: np.ndarray | None = None,
    selected_c: float | None = None,
) -> dict[str, Any]:
    """Fit one development estimator with training-only language scaling."""

    explicit = data["expl"] == "explicit"
    if fit_mask is None:
        fit_mask = explicit
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool) & explicit
    if fit_mask.sum() < 10 or len(np.unique(data["y"][fit_mask])) != 2:
        raise ValueError("cue-balanced fitting requires both explicit classes")

    stats = lang_stats(x_fit_all[fit_mask], data["lang"][fit_mask])
    x_fit_standard = apply_stats(x_fit_all, data["lang"], stats).astype(
        np.float64, copy=False
    )
    control_language = np.array([row["language"] for row in control_rows])
    x_control_standard = apply_stats(
        x_control, control_language, stats
    ).astype(np.float64, copy=False)
    differences = cue_difference_matrix(x_control_standard, control_rows).astype(
        np.float64, copy=False
    )
    y_control = np.array(
        [1 if row["intended_purpose"] == "evaluation" else 0 for row in control_rows],
        dtype=np.float64,
    )
    if selected_c is None:
        selected_c, inner_bacc = pick_C(x_fit_all, data, fit_mask)
    else:
        inner_bacc = float("nan")
    # Match sklearn's sample-count scaling closely enough to retain the existing
    # grouped-C interpretation while using an anisotropic cue penalty.
    l2_strength = 1.0 / (float(selected_c) * int(fit_mask.sum()))
    theta0 = np.zeros(x_fit_all.shape[1] + 1, dtype=np.float64)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return _loss_and_gradient(
            theta,
            x_fit_standard[fit_mask],
            data["y"][fit_mask].astype(np.float64),
            x_control_standard,
            y_control,
            differences,
            control_mass=recipe.control_mass,
            cue_penalty=recipe.cue_penalty,
            l2_strength=l2_strength,
        )

    result = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 750, "ftol": 1e-11, "gtol": 1e-7, "maxls": 40},
    )
    if not result.success:
        raise RuntimeError(
            f"{recipe.name} optimization failed: {result.status} {result.message}"
        )
    weight = result.x[:-1]
    intercept = float(result.x[-1])
    fit_score = x_fit_standard[fit_mask] @ weight + intercept
    return {
        "recipe": {
            "name": recipe.name,
            "control_mass": recipe.control_mass,
            "cue_penalty": recipe.cue_penalty,
        },
        "weight": weight,
        "intercept": intercept,
        "stats": stats,
        "fit_mask": fit_mask,
        "fit_score": fit_score,
        "selected_C": float(selected_c),
        "inner_cv_bacc": float(inner_bacc),
        "l2_strength": float(l2_strength),
        "optimization": {
            "iterations": int(result.nit),
            "objective": float(result.fun),
            "gradient_max_abs": float(np.max(np.abs(result.jac))),
        },
    }


def score(estimator: dict[str, Any], x: np.ndarray, language: np.ndarray) -> np.ndarray:
    standardized = apply_stats(x, language, estimator["stats"])
    return standardized @ estimator["weight"] + estimator["intercept"]
