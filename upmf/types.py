"""Typed records shared by clients, strategies, metrics, and checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


TensorMap = dict[str, torch.Tensor]


@dataclass(frozen=True)
class ClientSystemProfile:
    """Fixed simulated capacity and reliability characteristics of one client."""

    client_id: int
    compute_speed_multiplier: float
    availability_probability: float
    network_delay_mean: float
    network_delay_sigma: float


@dataclass
class ClientUpdate:
    """A model delta together with all timing and accounting provenance."""

    client_id: int
    source_version: int
    delta: TensorMap
    num_samples: int
    assigned_steps: int
    completed_steps: int
    start_time: float
    arrival_time: float
    uplink_bytes: int
    dense_uplink_bytes: int
    divergence: float = 0.0

    @property
    def latency(self) -> float:
        """Return simulated response latency from dispatch to arrival."""
        return self.arrival_time - self.start_time


@dataclass
class RoundResult:
    """Strategy output consumed uniformly by the runner and metric tracker."""

    accepted_updates: list[ClientUpdate] = field(default_factory=list)
    dropped_updates: list[ClientUpdate] = field(default_factory=list)
    total_steps: int = 0
    wasted_steps: int = 0
    uplink_bytes: int = 0
    downlink_bytes: int = 0
    virtual_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
