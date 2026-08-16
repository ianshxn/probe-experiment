from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval_format_mvp.causal import (  # noqa: E402
    direct_candidate_scores,
    factorial_transport,
    raw_effect_from_standardized,
    residual_intervention,
    unit_strength,
)
from results.run_causal_baseline import _auc


def test_baseline_auc_uses_candidate_margin_and_handles_ties() -> None:
    scores = np.asarray([[3.0, 0.0], [2.0, 1.0], [0.0, 2.0], [1.0, 3.0]])
    targets = np.asarray([0, 0, 1, 1])
    assert _auc(scores, targets) == 1.0
    tied = np.asarray([[1.0, 1.0], [0.0, 0.0]])
    assert _auc(tied, np.asarray([0, 1])) == 0.5


class _Block(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + 1.0


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Block(), _Block()])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()

    def forward(self, inputs: torch.Tensor):
        hidden = self.model(inputs)
        return type("Output", (), {"logits": hidden})()


def test_factorial_transport_interaction_ablation() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 2.0])
    c = np.array([3.0, 4.0])
    assert np.allclose(factorial_transport(1, 1, a, b, c, target="purpose"), -2 * (a + c))
    assert np.allclose(factorial_transport(1, 1, a, b, c, target="purpose", include_interaction=False), -2 * a)
    assert np.allclose(factorial_transport(-1, -1, a, b, c, target="format"), 2 * (b - c))


def test_standardized_effect_is_converted_to_raw_space() -> None:
    assert np.allclose(raw_effect_from_standardized(np.array([2.0, -1.0]), np.array([3.0, 4.0])), [6.0, -4.0])


def test_hook_edits_only_target_layer_once_and_zero_strength_is_baseline() -> None:
    model = _FakeModel()
    inputs = torch.zeros((1, 3, 2))
    baseline = model(inputs).logits.detach().clone()
    with residual_intervention(model, 1, np.array([2.0, -1.0])) as calls:
        steered = model(inputs).logits.detach().clone()
    assert calls == [1]
    assert torch.allclose(steered[:, -1, :] - baseline[:, -1, :], torch.tensor([[2.0, -1.0]]))
    assert torch.allclose(steered[:, 0, :], baseline[:, 0, :])
    with residual_intervention(model, 1, np.zeros(2)):
        zero = model(inputs).logits.detach()
    assert torch.allclose(zero, baseline)


def test_strength_and_candidate_logits_are_finite() -> None:
    unit, alpha = unit_strength(np.array([1.0, 0.0]), np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]))
    assert np.allclose(unit, [1.0, 0.0])
    assert alpha > 0
    outputs = type("Output", (), {"logits": torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]])})()
    assert np.allclose(direct_candidate_scores(outputs, [0, 2]), [[3.0, 5.0]])
