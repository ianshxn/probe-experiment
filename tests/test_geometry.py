from __future__ import annotations

import numpy as np

from eval_format_mvp.geometry import analyze_geometry, factorial_components, synthetic_factorial_case


def test_factorial_recovery_a_only() -> None:
    x, rows = synthetic_factorial_case(a=2.0, b=0.0, c=0.0)
    comp = factorial_components(x, rows)
    assert np.linalg.norm(comp["A"].mean(0)) > 0.9
    assert np.linalg.norm(comp["C"].mean(0)) < 0.05


def test_factorial_recovery_b_only() -> None:
    x, rows = synthetic_factorial_case(a=0.0, b=2.0, c=0.0)
    comp = factorial_components(x, rows)
    assert np.linalg.norm(comp["B"].mean(0)) > 0.9
    assert np.linalg.norm(comp["A"].mean(0)) < 0.05
    assert np.linalg.norm(comp["C"].mean(0)) < 0.05


def test_factorial_recovery_c_only() -> None:
    x, rows = synthetic_factorial_case(a=0.0, b=0.0, c=2.0)
    comp = factorial_components(x, rows)
    assert np.linalg.norm(comp["C"].mean(0)) > 0.9
    assert np.linalg.norm(comp["A"].mean(0)) < 0.05


def test_factorial_recovery_additive_cases() -> None:
    x, rows = synthetic_factorial_case(a=1.5, b=1.0, c=0.0)
    summary = analyze_geometry(x, rows)["summary"]
    assert summary["norm_A"] > 0.5
    assert summary["norm_B"] > 0.3
    assert summary["norm_C"] < 0.1

    x, rows = synthetic_factorial_case(a=1.5, b=0.0, c=1.0)
    summary = analyze_geometry(x, rows)["summary"]
    assert summary["norm_A"] > 0.5
    assert summary["norm_C"] > 0.3
    assert summary["norm_C_over_norm_A"] > 0.3


def test_factorial_null_has_no_systematic_components() -> None:
    rng = np.random.default_rng(123)
    rows: list[dict[str, object]] = []
    x: list[np.ndarray] = []
    for block in range(24):
        for purpose, label in (("evaluation", 1), ("deployment", 0)):
            for fmt in ("benchmark", "casual"):
                rows.append({"payload_block_id": f"b{block}", "purpose_family_id": f"f{block % 6}", "intended_purpose": purpose, "format": fmt, "label": label})
                x.append(rng.normal(size=8))
    summary = analyze_geometry(np.asarray(x), rows)["summary"]
    assert summary["norm_A"] < 1.0
    assert summary["norm_C"] < 1.0
