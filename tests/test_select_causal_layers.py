from __future__ import annotations

import json
import sys
from pathlib import Path

from results.select_causal_layers import main


def test_selection_uses_geometry_only_and_excludes_endpoints(tmp_path: Path, monkeypatch) -> None:
    geometry = {
        "input_artifacts": {"items_sha256": "ebac61dbe7b6bcf89bb8195d92447619ca4821adc2b4de6b1d4bc98058f44022"},
        "model": {"name": "test", "revision": "test"},
        "layers": [0, 1, 2, 3, 4],
        "layers_results": [
            {"layer": 0, "summary": {"norm_C_over_norm_A": 0.01}, "transfer": {"pooled": {"auc": 0.99}}},
            {"layer": 1, "summary": {"norm_C_over_norm_A": 0.60}, "transfer": {"pooled": {"auc": 0.81}}},
            {"layer": 2, "summary": {"norm_C_over_norm_A": 0.20}, "transfer": {"pooled": {"auc": 0.90}}},
            {"layer": 3, "summary": {"norm_C_over_norm_A": 0.80}, "transfer": {"pooled": {"auc": 0.85}}},
            {"layer": 4, "summary": {"norm_C_over_norm_A": 0.01}, "transfer": {"pooled": {"auc": 0.99}}},
        ],
    }
    geometry_path = tmp_path / "geometry.json"
    output_path = tmp_path / "protocol" / "protocol.json"
    geometry_path.write_text(json.dumps(geometry))
    monkeypatch.setattr(sys, "argv", ["select_causal_layers", "--geometry", str(geometry_path), "--output", str(output_path), "--run-id", "test", "--git-commit", "test"])
    main()
    protocol = json.loads(output_path.read_text())
    assert protocol["selected_layers"] == {"shared": 2, "interaction": 3}
    assert 0 not in protocol["selected_layers"].values()
    assert 4 not in protocol["selected_layers"].values()
    assert protocol["status"] == "protocol_frozen_before_endpoint_and_steering"
    assert len(protocol["protocol_sha256"]) == 64
