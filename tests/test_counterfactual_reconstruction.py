from __future__ import annotations

import numpy as np

from eval_format_mvp.geometry import synthetic_factorial_case
from results.run_counterfactual_reconstruction import analyze_space


def test_full_factorial_transport_beats_main_effect_on_synthetic_interaction() -> None:
    x, rows = synthetic_factorial_case(a=1.0, b=0.5, c=0.8, seed=19, n_blocks=24, dimension=8)
    records, summaries = analyze_space(x[:, None, :], rows, [0], projection_name="synthetic")
    purpose = [record for record in records if record["direction"].startswith("purpose_")]
    full = np.mean([record["raw_euclidean_error"] for record in purpose if record["method"] == "full"])
    main = np.mean([record["raw_euclidean_error"] for record in purpose if record["method"] == "main"])
    assert full < main
    assert summaries[0]["by_direction"]["purpose_db_to_eb"]["C_benefit_raw_error"] > 0.0


def test_main_effect_only_is_exact_when_interaction_is_absent() -> None:
    x, rows = synthetic_factorial_case(a=1.0, b=0.5, c=0.0, seed=23, n_blocks=24, dimension=8)
    records, _ = analyze_space(x[:, None, :], rows, [0], projection_name="synthetic")
    purpose = [record for record in records if record["direction"].startswith("purpose_")]
    full = np.mean([record["raw_euclidean_error"] for record in purpose if record["method"] == "full"])
    main = np.mean([record["raw_euclidean_error"] for record in purpose if record["method"] == "main"])
    assert abs(full - main) < 0.02
