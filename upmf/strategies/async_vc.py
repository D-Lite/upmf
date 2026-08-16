"""Statistics-unaware asynchronous volunteer-computing baseline."""

from __future__ import annotations

from upmf.coordinator_base import BaseCoordinator, apply_delta, weighted_average
from upmf.heterogeneity import sample_participation
from upmf.types import RoundResult


class AsyncVCCoordinator(BaseCoordinator):
    """Aggregate all arrivals in each virtual budget window with equal weights."""

    def run_round(self, round_number: int) -> RoundResult:
        """Dispatch to available volunteers and process arrivals without waiting."""
        budget = float(self.config["strategies"]["async_vc"]["round_budget"])
        cutoff = float(self.config["strategies"]["async_vc"]["hard_cutoff"])
        assigned_steps = int(self.config["training"]["default_local_steps"])
        start = self.virtual_time
        window_end = start + budget
        total_steps = 0
        uplink_bytes = 0
        downlink_bytes = 0
        volunteers: list[int] = []

        for client_id in range(len(self.clients)):
            participation = sample_participation(
                self.profiles[client_id], self.seed, round_number
            )
            if not participation.available:
                continue
            volunteers.append(client_id)
            update = self.train_update(
                client_id,
                round_number,
                assigned_steps,
                assigned_steps,
                start,
                participation.network_delay,
            )
            self.pending_updates.append(update)
            total_steps += update.completed_steps
            uplink_bytes += update.uplink_bytes
            downlink_bytes += self.downlink_model_bytes

        accepted = [
            update
            for update in self.pending_updates
            if update.arrival_time <= window_end
        ]
        remaining = [
            update
            for update in self.pending_updates
            if update.arrival_time > window_end
        ]
        dropped = [
            update for update in remaining if window_end - update.start_time > cutoff
        ]
        self.pending_updates = [
            update for update in remaining if window_end - update.start_time <= cutoff
        ]
        if round_number == int(self.config["training"]["rounds"]):
            dropped.extend(self.pending_updates)
            self.pending_updates = []
        if accepted:
            apply_delta(self.model, weighted_average(accepted, [1.0] * len(accepted)))
            self.model_version += 1
        self.virtual_time = window_end
        return RoundResult(
            accepted_updates=accepted,
            dropped_updates=dropped,
            total_steps=total_steps,
            wasted_steps=sum(update.completed_steps for update in dropped),
            uplink_bytes=uplink_bytes,
            downlink_bytes=downlink_bytes,
            virtual_time=self.virtual_time,
            metadata={"volunteer_clients": volunteers},
        )
