"""Minimal EWMA participant state feeding the adaptive scheduler."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class ParticipantProfiler:
    """Maintain compact, serializable observations for each simulated client."""

    def __init__(self, decay: float, default_steps: int) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.default_steps = default_steps
        self._profiles: dict[int, dict[str, float]] = {}

    def _default(self) -> dict[str, float]:
        return {
            "latency": float(max(1, self.default_steps)),
            "completed_steps": float(self.default_steps),
            "divergence": 0.0,
            "availability": 1.0,
        }

    def get_profile(self, client_id: int) -> dict[str, float]:
        """Return a copy so callers cannot mutate stored profiling evidence."""
        return dict(self._profiles.get(client_id, self._default()))

    def update_profile(
        self,
        client_id: int,
        latency: float,
        steps: int,
        divergence: float,
        available: bool = True,
    ) -> None:
        """Blend a new observation into the client's exponentially weighted state."""
        previous = self.get_profile(client_id)
        observation = {
            "latency": float(latency),
            "completed_steps": float(steps),
            "divergence": float(divergence),
            "availability": float(available),
        }
        self._profiles[client_id] = {
            key: self.decay * previous[key] + (1.0 - self.decay) * value
            for key, value in observation.items()
        }

    def mark_unavailable(self, client_id: int) -> None:
        """Record dropout without inventing latency, work, or divergence values."""
        previous = self.get_profile(client_id)
        previous["availability"] *= self.decay
        self._profiles[client_id] = previous

    def state_dict(self) -> dict[str, Any]:
        """Return all helper state required for checkpoint resumption."""
        return {"profiles": deepcopy(self._profiles)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore helper state from a trusted checkpoint."""
        self._profiles = {
            int(client_id): dict(values)
            for client_id, values in state["profiles"].items()
        }
