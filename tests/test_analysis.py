from __future__ import annotations

from pathlib import Path

import numpy as np

from eval_format_mvp.analyze import analyze_activations
from eval_format_mvp.catalogs import load_catalogs
from eval_format_mvp.io import load_jsonl, write_jsonl
from eval_format_mvp.render import render_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_format_shortcut_and_decorrelated_recovery(tmp_path: Path) -> None:
    catalogs = load_catalogs(ROOT)
    payloads = load_jsonl(ROOT / "data" / "example_payloads.jsonl")
    rows = render_dataset(payloads, catalogs)
    items_path = tmp_path / "items.jsonl"
    write_jsonl(items_path, rows)

    rng = np.random.default_rng(7)
    X = rng.normal(scale=0.05, size=(len(rows), 8))
    for index, row in enumerate(rows):
        purpose_sign = 1.0 if row["intended_purpose"] == "evaluation" else -1.0
        format_sign = 1.0 if row["format"] == "benchmark" else -1.0
        X[index, 0] += purpose_sign
        # Redundant format dimensions make the confounded training direction
        # prefer the shortcut. Decorrelated training can still cancel them.
        X[index, 1:4] += 3.0 * format_sign
    activation_path = tmp_path / "activations.npz"
    np.savez_compressed(
        activation_path,
        X=X,
        item_ids=np.asarray([row["item_id"] for row in rows], dtype=str),
    )
    result = analyze_activations(
        items_path=items_path,
        activations_path=activation_path,
        output_path=tmp_path / "result.json",
        c=0.1,
    )
    standard = result["purpose_probe_regimes"]["standard_confounded"]
    decorrelated = result["purpose_probe_regimes"]["decorrelated"]
    assert decorrelated["balanced_accuracy"] > 0.95
    assert standard["by_cell"]["deployment::benchmark"][
        "percent_predicted_evaluation"
    ] > 50
    assert standard["by_cell"]["evaluation::casual"][
        "percent_predicted_evaluation"
    ] < 50
    assert result["format_probe"]["balanced_accuracy"] > 0.95
