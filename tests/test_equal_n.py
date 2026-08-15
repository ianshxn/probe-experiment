from __future__ import annotations

import numpy as np

from eval_format_mvp.equal_n import _assignment, run_equal_n


def _rows(n_blocks: int = 24, n_families: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for block_index in range(n_blocks):
        family = f"family_{block_index % n_families}"
        for purpose, label in (("evaluation", 1), ("deployment", 0)):
            for fmt in ("benchmark", "casual"):
                rows.append(
                    {
                        "item_id": f"item_{block_index}_{purpose}_{fmt}",
                        "payload_block_id": f"block_{block_index}",
                        "purpose_family_id": family,
                        "intended_purpose": purpose,
                        "format": fmt,
                        "label": label,
                    }
                )
    return rows


def test_equal_n_assignment_is_balanced_and_reproducible() -> None:
    assignment = _assignment([f"b{i}" for i in range(11)], 1729)
    assert assignment == _assignment([f"b{i}" for i in range(11)], 1729)
    assert abs(sum(value == "benchmark" for value in assignment.values()) - sum(value == "casual" for value in assignment.values())) <= 1


def test_shared_purpose_signal_transfers_but_null_does_not() -> None:
    rows = _rows()
    rng = np.random.default_rng(41)
    purpose = np.asarray([1.0 if row["label"] == 1 else -1.0 for row in rows])
    x_signal = purpose[:, None] + rng.normal(scale=0.15, size=(len(rows), 3))
    signal = run_equal_n(x_signal, rows, seeds=(1729, 2718))
    assert signal["equal_training_rows_first_three"]
    assert signal["aggregate"]["benchmark_to_casual"]["mean_auc"] > 0.9
    assert signal["aggregate"]["casual_to_benchmark"]["mean_auc"] > 0.9

    x_null = rng.normal(size=(len(rows), 4))
    null = run_equal_n(x_null, rows, seeds=(1729, 2718))
    assert 0.35 < null["aggregate"]["benchmark_to_casual"]["mean_auc"] < 0.65
    assert 0.35 < null["aggregate"]["casual_to_benchmark"]["mean_auc"] < 0.65


def test_equal_n_training_counts_expose_sample_size_confound() -> None:
    rows = _rows(n_blocks=25)
    rng = np.random.default_rng(9)
    x = rng.normal(size=(len(rows), 2))
    result = run_equal_n(x, rows, seeds=(4242,))
    for counts in result["seed_results"][0]["training_row_counts_by_held_family"].values():
        assert counts["benchmark_only"] == counts["casual_only"] == counts["equal_n_mixed"]
        assert counts["pooled"] == 2 * counts["equal_n_mixed"]
