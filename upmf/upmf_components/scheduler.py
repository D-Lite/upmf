"""Availability-aware, non-starving UPMF round scheduling."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from upmf.upmf_components.profiler import ParticipantProfiler


class AdaptiveRoundScheduler:
    """Select clients and allocate bounded work from recent participant evidence."""

    def __init__(
        self,
        num_clients: int,
        profiler: ParticipantProfiler,
        config: Mapping[str, Any],
    ) -> None:
        self.num_clients = num_clients
        self.profiler = profiler
        self.config = config
        self.last_selected = {client_id: 0 for client_id in range(num_clients)}

    def _scores(self) -> np.ndarray:
        profiles = [
            self.profiler.get_profile(client_id)
            for client_id in range(self.num_clients)
        ]
        speeds = np.asarray(
            [
                profile["completed_steps"] / max(profile["latency"], 1e-12)
                for profile in profiles
            ]
        )
        normalized_speeds = speeds / max(float(speeds.max()), 1e-12)
        availability = np.asarray(
            [profile["availability"] for profile in profiles]
        )
        return (
            float(self.config["availability_weight"]) * availability
            + float(self.config["speed_weight"]) * normalized_speeds
        )

    def select(
        self, round_number: int, count: int, rng: np.random.Generator
    ) -> list[int]:
        """Sample reliable clients while forcing bounded-wait anti-starvation."""
        count = min(max(1, count), self.num_clients)
        starvation_rounds = int(self.config["starvation_rounds"])
        overdue = sorted(
            (
                client_id
                for client_id, last_round in self.last_selected.items()
                if round_number - last_round >= starvation_rounds
            ),
            key=lambda client_id: self.last_selected[client_id],
        )
        selected = overdue[:count]
        candidates = [
            client_id
            for client_id in range(self.num_clients)
            if client_id not in selected
        ]
        remaining = count - len(selected)
        if remaining:
            scores = self._scores()[candidates]
            score_probabilities = scores / scores.sum()
            floor = float(self.config["exploration_floor"])
            probabilities = (
                floor / len(candidates)
                + (1.0 - floor) * score_probabilities
            )
            chosen = rng.choice(
                candidates, size=remaining, replace=False, p=probabilities
            )
            selected.extend(int(client_id) for client_id in chosen)
        for client_id in selected:
            self.last_selected[client_id] = round_number
        return sorted(selected)

    def allocate_steps(
        self,
        client_ids: list[int],
        default_steps: int,
        min_steps: int,
        max_steps: int,
    ) -> dict[int, int]:
        """Scale work by profiled throughput relative to the invited median.

        Throughput is EWMA completed steps divided by EWMA latency. Allocation
        is `round(default_steps * throughput / median_throughput)`, clamped to
        configured minimum and maximum bounds.
        """
        throughputs = {
            client_id: self.profiler.get_profile(client_id)["completed_steps"]
            / max(self.profiler.get_profile(client_id)["latency"], 1e-12)
            for client_id in client_ids
        }
        median = max(float(np.median(list(throughputs.values()))), 1e-12)
        return {
            client_id: int(
                np.clip(
                    round(default_steps * throughput / median),
                    min_steps,
                    max_steps,
                )
            )
            for client_id, throughput in throughputs.items()
        }

    def state_dict(self) -> dict[str, Any]:
        """Return selection history needed to preserve fairness after resume."""
        return {"last_selected": dict(self.last_selected)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore selection history from a checkpoint."""
        self.last_selected = {
            int(client_id): int(round_number)
            for client_id, round_number in state["last_selected"].items()
        }
