"""Protocol-owned causal transport primitives.

These utilities intentionally separate raw residual-space vectors from the
standardized probe space and provide testable hook/readout contracts. They do
not select layers or infer causal effects.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


def raw_effect_from_standardized(effect: np.ndarray, training_scale: np.ndarray) -> np.ndarray:
    effect = np.asarray(effect, dtype=float)
    scale = np.asarray(training_scale, dtype=float)
    if effect.shape != scale.shape or not np.isfinite(effect).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("effect and positive finite training scales must have identical shapes")
    return effect * scale


def unit_strength(direction: np.ndarray, training_components: np.ndarray) -> tuple[np.ndarray, float]:
    direction = np.asarray(direction, dtype=float)
    components = np.asarray(training_components, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0 or components.ndim != 2 or components.shape[1] != direction.size:
        raise ValueError("direction must be nonzero and match component dimensionality")
    unit = direction / norm
    alpha = float(np.std(components @ unit, ddof=1))
    if not np.isfinite(alpha) or alpha == 0.0:
        raise ValueError("training projection strength is undefined")
    return unit, alpha


def factorial_transport(p: float, f: float, a: np.ndarray, b: np.ndarray, c: np.ndarray, *, target: str, include_interaction: bool = True) -> np.ndarray:
    if p not in (-1, 1) or f not in (-1, 1) or target not in {"purpose", "format"}:
        raise ValueError("effect codes must be +/-1 and target must be purpose or format")
    a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
    if not (a.shape == b.shape == c.shape):
        raise ValueError("factorial vectors must have equal shapes")
    if target == "purpose":
        return -2.0 * p * (a + f * c if include_interaction else a)
    return -2.0 * f * (b + p * c if include_interaction else b)


@contextmanager
def residual_intervention(model: Any, layer: int, delta: np.ndarray, *, token_index: int = -1) -> Iterator[list[int]]:
    """Install one layer hook that edits exactly one residual token."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("model must expose model.layers")
    calls: list[int] = []
    delta_array = np.asarray(delta, dtype=float)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        calls.append(1)
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.shape[-1] != delta_array.size:
            raise ValueError("intervention dimensionality does not match residual stream")
        import torch
        update = torch.as_tensor(delta_array, device=hidden.device, dtype=hidden.dtype)
        edited = hidden.clone()
        edited[:, token_index, :] = edited[:, token_index, :] + update
        if isinstance(output, tuple):
            return (edited, *output[1:])
        if isinstance(output, list):
            return [edited, *output[1:]]
        return edited

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield calls
    finally:
        handle.remove()


def direct_candidate_scores(outputs: Any, candidate_token_ids: Sequence[int]) -> np.ndarray:
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
    values = logits[:, -1, :]
    ids = np.asarray(list(candidate_token_ids), dtype=int)
    if ids.ndim != 1 or len(ids) < 2 or len(set(ids.tolist())) != len(ids):
        raise ValueError("candidate token IDs must be distinct and contain at least two entries")
    selected = values[:, ids]
    import torch
    if not bool(torch.isfinite(selected.detach().float()).all().item()):
        raise ValueError("candidate logits are not finite")
    return selected.detach().float().cpu().numpy()


def validate_candidate_mapping(tokenizer: Any, candidates: Mapping[str, str]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for label, text in candidates.items():
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"candidate {label!r} is not exactly one continuation token")
        ids[label] = int(encoded[0])
    if len(set(ids.values())) != len(ids):
        raise ValueError("candidate mapping contains duplicate token IDs")
    return ids
