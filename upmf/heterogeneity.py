"""Deterministic virtual systems heterogeneity for simulated participants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from upmf.types import ClientSystemProfile
from upmf.utils import derive_seed


@dataclass(frozen=True)
class ParticipationState:
    """Round-specific availability and network delay for one fixed client."""

    available: bool
    network_delay: float


def generate_client_profiles(
    num_clients: int,
    severity_config: Mapping[str, Any],
    seed: int,
) -> list[ClientSystemProfile]:
    """Generate fixed client traits from documented, configurable presets.

    These distributions are plausible experimental assumptions; they are not
    fitted estimates of Nigerian grid or broadband infrastructure.
    """
    rng = np.random.default_rng(derive_seed(seed, "systems-profiles"))
    speeds = rng.lognormal(
        mean=float(severity_config["speed_lognormal_mean"]),
        sigma=float(severity_config["speed_lognormal_sigma"]),
        size=num_clients,
    )
    availability = 1.0 - float(severity_config["dropout_probability"])
    return [
        ClientSystemProfile(
            client_id=client_id,
            compute_speed_multiplier=float(max(speed, np.finfo(float).eps)),
            availability_probability=availability,
            network_delay_mean=float(severity_config["network_delay_mean"]),
            network_delay_sigma=float(severity_config["network_delay_sigma"]),
        )
        for client_id, speed in enumerate(speeds)
    ]


def sample_participation(
    profile: ClientSystemProfile, seed: int, round_number: int
) -> ParticipationState:
    """Sample a paired client/round outcome independent of strategy control flow."""
    namespace = f"participation:{round_number}:{profile.client_id}"
    rng = np.random.default_rng(derive_seed(seed, namespace))
    available = bool(rng.random() < profile.availability_probability)
    delay = max(
        0.0,
        float(rng.normal(profile.network_delay_mean, profile.network_delay_sigma)),
    )
    return ParticipationState(available=available, network_delay=delay)


def simulated_latency(
    profile: ClientSystemProfile, completed_steps: int, network_delay: float
) -> float:
    """Convert completed work and network delay into event-driven virtual latency."""
    if completed_steps < 0 or network_delay < 0:
        raise ValueError("steps and network delay must be non-negative")
    compute_time = completed_steps / profile.compute_speed_multiplier
    return float(compute_time + network_delay)


def steps_before_deadline(
    profile: ClientSystemProfile,
    assigned_steps: int,
    network_delay: float,
    deadline: float,
) -> int:
    """Estimate partial SGD work completed before a coordinator deadline."""
    compute_budget = max(0.0, deadline - network_delay)
    possible = int(np.floor(compute_budget * profile.compute_speed_multiplier))
    return min(assigned_steps, max(0, possible))
