"""Thin per-tensor top-k sparsification and communication accounting."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from upmf.types import TensorMap


def topk_sparsify(
    delta: Mapping[str, torch.Tensor], fraction: float
) -> tuple[TensorMap, int]:
    """Keep the largest magnitudes per tensor and count value-plus-index bytes."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    sparse: TensorMap = {}
    transmitted_bytes = 0
    for name, tensor in delta.items():
        flattened = tensor.detach().cpu().flatten()
        count = min(flattened.numel(), max(1, math.ceil(flattened.numel() * fraction)))
        indices = torch.topk(flattened.abs(), count, sorted=False).indices
        sparse_flat = torch.zeros_like(flattened)
        sparse_flat[indices] = flattened[indices]
        sparse[name] = sparse_flat.reshape(tensor.shape)
        transmitted_bytes += count * (tensor.element_size() + 8)
    return sparse, transmitted_bytes


class TopKCompressor:
    """Configured callable wrapper used directly by the UPMF coordinator."""

    def __init__(self, fraction: float) -> None:
        self.fraction = fraction

    def __call__(self, delta: TensorMap) -> tuple[TensorMap, int]:
        """Compress a model delta using the configured retained fraction."""
        return topk_sparsify(delta, self.fraction)
