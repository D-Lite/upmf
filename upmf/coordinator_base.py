"""Shared coordinator contracts and model-update utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from upmf.client import train_client
from upmf.heterogeneity import simulated_latency
from upmf.model import model_bytes
from upmf.types import ClientSystemProfile, ClientUpdate, RoundResult, TensorMap
from upmf.utils import derive_seed


def tensor_map_bytes(delta: Mapping[str, torch.Tensor]) -> int:
    """Count bytes in a dense tensor map."""
    return sum(value.numel() * value.element_size() for value in delta.values())


def weighted_average(
    updates: Sequence[ClientUpdate], weights: Sequence[float]
) -> TensorMap:
    """Return the normalized weighted average delta for accepted client updates."""
    if not updates or len(updates) != len(weights):
        raise ValueError("updates and weights must be non-empty and equally sized")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("aggregation weights must sum to a positive value")
    return {
        name: sum(
            update.delta[name] * (float(weight) / total)
            for update, weight in zip(updates, weights)
        )
        for name in updates[0].delta
    }


def apply_delta(model: nn.Module, delta: Mapping[str, torch.Tensor]) -> None:
    """Apply a CPU model delta to the current global model in place."""
    current = model.state_dict()
    model.load_state_dict(
        {
            name: current[name] + delta[name].to(current[name].device)
            for name in current
        }
    )


def cosine_distance(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
) -> float:
    """Compute cosine distance between two flattened model updates."""
    first_vector = torch.cat([value.detach().cpu().flatten() for value in first.values()])
    second_vector = torch.cat(
        [value.detach().cpu().flatten() for value in second.values()]
    )
    denominator = torch.linalg.vector_norm(first_vector) * torch.linalg.vector_norm(
        second_vector
    )
    if float(denominator) == 0.0:
        return 0.0
    similarity = torch.dot(first_vector, second_vector) / denominator
    return float((1.0 - similarity.clamp(-1.0, 1.0)).item())


class BaseCoordinator(ABC):
    """Base class enforcing one client trainer and common byte/timing semantics."""

    def __init__(
        self,
        model: nn.Module,
        clients: Sequence[Dataset[Any]],
        profiles: Sequence[ClientSystemProfile],
        config: Mapping[str, Any],
        device: torch.device,
        seed: int,
    ) -> None:
        self.model = model
        self.clients = list(clients)
        self.profiles = list(profiles)
        self.config = config
        self.device = device
        self.seed = seed
        self.virtual_time = 0.0
        self.model_version = 0
        self.pending_updates: list[ClientUpdate] = []

    def sample_clients(self, round_number: int, namespace: str = "shared") -> list[int]:
        """Select a deterministic fixed fraction without depending on prior draws."""
        count = max(
            1,
            int(np.ceil(len(self.clients) * float(self.config["training"]["client_fraction"]))),
        )
        rng = np.random.default_rng(
            derive_seed(self.seed, f"selection:{namespace}:{round_number}")
        )
        return sorted(rng.choice(len(self.clients), size=count, replace=False).tolist())

    def train_update(
        self,
        client_id: int,
        round_number: int,
        assigned_steps: int,
        completed_steps: int,
        start_time: float,
        network_delay: float,
        transform: Callable[[TensorMap], tuple[TensorMap, int]] | None = None,
    ) -> ClientUpdate:
        """Run shared local SGD and construct a uniformly accounted arrival event."""
        result = train_client(
            self.model,
            self.clients[client_id],
            completed_steps,
            int(self.config["training"]["batch_size"]),
            float(self.config["training"]["learning_rate"]),
            float(self.config["training"]["momentum"]),
            self.device,
            derive_seed(self.seed, f"training:{round_number}:{client_id}"),
        )
        dense_bytes = tensor_map_bytes(result.delta)
        delta, uplink_bytes = (
            transform(result.delta) if transform is not None else (result.delta, dense_bytes)
        )
        latency = simulated_latency(
            self.profiles[client_id], completed_steps, network_delay
        )
        return ClientUpdate(
            client_id=client_id,
            source_version=self.model_version,
            delta=delta,
            num_samples=len(self.clients[client_id]),
            assigned_steps=assigned_steps,
            completed_steps=result.completed_steps,
            start_time=start_time,
            arrival_time=start_time + latency,
            uplink_bytes=uplink_bytes,
            dense_uplink_bytes=dense_bytes,
        )

    @property
    def downlink_model_bytes(self) -> int:
        """Return dense global-model bytes sent to each invited client."""
        return model_bytes(self.model)

    def state_dict(self) -> dict[str, Any]:
        """Serialize coordinator state needed for exact round-boundary resume."""
        return {
            "virtual_time": self.virtual_time,
            "model_version": self.model_version,
            "pending_updates": self.pending_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore coordinator state from a trusted checkpoint."""
        self.virtual_time = float(state["virtual_time"])
        self.model_version = int(state["model_version"])
        self.pending_updates = list(state["pending_updates"])

    @abstractmethod
    def run_round(self, round_number: int) -> RoundResult:
        """Execute one virtual coordination round and update the global model."""
