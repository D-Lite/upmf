"""Core formula tests for UPMF aggregation and top-k compression."""

from __future__ import annotations

import torch

from upmf.types import ClientUpdate
from upmf.upmf_components.aggregator import StalenessTolerantAggregator
from upmf.upmf_components.compressor import topk_sparsify


def _update(client_id: int, values: list[float], arrival: float) -> ClientUpdate:
    tensor = torch.tensor(values)
    return ClientUpdate(
        client_id=client_id,
        source_version=0,
        delta={"weight": tensor},
        num_samples=10,
        assigned_steps=2,
        completed_steps=2,
        start_time=0.0,
        arrival_time=arrival,
        uplink_bytes=tensor.numel() * tensor.element_size(),
        dense_uplink_bytes=tensor.numel() * tensor.element_size(),
    )


def _aggregator(cutoff: float = 5.0) -> StalenessTolerantAggregator:
    return StalenessTolerantAggregator(
        {
            "expected_round_time": 1.0,
            "staleness_rate": 1.0,
            "divergence_rate": 2.0,
            "hard_staleness_cutoff": cutoff,
        }
    )


def test_later_update_receives_lower_weight() -> None:
    decision = _aggregator().aggregate(
        [_update(0, [1.0, 0.0], 1.0), _update(1, [1.0, 0.0], 3.0)]
    )
    assert decision.normalized_weights[0] > decision.normalized_weights[1]


def test_divergent_update_is_discounted() -> None:
    decision = _aggregator().aggregate(
        [
            _update(0, [1.0, 0.0], 1.0),
            _update(1, [1.0, 0.0], 1.0),
            _update(2, [0.0, 1.0], 1.0),
        ]
    )
    assert decision.normalized_weights[0] > decision.normalized_weights[2]


def test_hard_cutoff_drops_extreme_staleness() -> None:
    decision = _aggregator(cutoff=2.0).aggregate(
        [_update(0, [1.0], 1.0), _update(1, [1.0], 5.0)]
    )
    assert [update.client_id for update in decision.dropped] == [1]


def test_topk_keeps_largest_values_and_counts_sparse_bytes() -> None:
    sparse, byte_count = topk_sparsify(
        {"weight": torch.tensor([1.0, -5.0, 3.0, 2.0])}, 0.5
    )
    assert torch.equal(sparse["weight"], torch.tensor([0.0, -5.0, 3.0, 0.0]))
    assert byte_count == 2 * (4 + 8)
