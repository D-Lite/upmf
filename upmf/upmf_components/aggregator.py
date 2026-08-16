"""Staleness- and divergence-aware UPMF update aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from upmf.coordinator_base import cosine_distance, weighted_average
from upmf.types import ClientUpdate, TensorMap


@dataclass
class AggregationDecision:
    """Accepted/dropped updates and the final normalized aggregation evidence."""

    aggregate: TensorMap | None
    accepted: list[ClientUpdate]
    dropped: list[ClientUpdate]
    normalized_weights: list[float]


class StalenessTolerantAggregator:
    """Downweight late or divergent updates, dropping only beyond a hard cutoff."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.expected_round_time = float(config["expected_round_time"])
        self.staleness_rate = float(config["staleness_rate"])
        self.divergence_rate = float(config["divergence_rate"])
        self.hard_cutoff = float(config["hard_staleness_cutoff"])
        if self.expected_round_time <= 0 or self.hard_cutoff < 0:
            raise ValueError("round time must be positive and cutoff non-negative")

    def staleness(self, update: ClientUpdate) -> float:
        """Return lateness in excess multiples of the expected round duration."""
        return max(0.0, update.latency / self.expected_round_time - 1.0)

    def aggregate(self, updates: list[ClientUpdate]) -> AggregationDecision:
        """Calculate the dissertation's explicit two-factor update weighting.

        For accepted update i:

        raw_i = n_i * exp(-staleness_rate * s_i)
                    / (1 + divergence_rate * max(0, d_i))

        where n_i is local sample count, s_i is excess latency measured in
        expected-round multiples, and d_i is cosine distance from the
        sample-size-weighted provisional aggregate. Raw weights are normalized.
        """
        accepted = [
            update for update in updates if self.staleness(update) <= self.hard_cutoff
        ]
        dropped = [
            update for update in updates if self.staleness(update) > self.hard_cutoff
        ]
        if not accepted:
            return AggregationDecision(None, [], dropped, [])
        provisional = weighted_average(
            accepted, [float(update.num_samples) for update in accepted]
        )
        raw_weights: list[float] = []
        for update in accepted:
            update.divergence = cosine_distance(update.delta, provisional)
            staleness_discount = np.exp(
                -self.staleness_rate * self.staleness(update)
            )
            divergence_discount = 1.0 / (
                1.0 + self.divergence_rate * max(0.0, update.divergence)
            )
            raw_weights.append(
                float(update.num_samples * staleness_discount * divergence_discount)
            )
        total = sum(raw_weights)
        normalized = [weight / total for weight in raw_weights]
        final = weighted_average(accepted, normalized)
        return AggregationDecision(final, accepted, dropped, normalized)
