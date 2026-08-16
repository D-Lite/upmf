"""Synchronous, sample-size-weighted FedAvg baseline."""

from __future__ import annotations

from upmf.coordinator_base import BaseCoordinator, apply_delta, weighted_average
from upmf.heterogeneity import sample_participation, simulated_latency
from upmf.types import RoundResult


class FedAvgSyncCoordinator(BaseCoordinator):
    """Wait for sampled clients until a fixed virtual timeout, then average."""

    def run_round(self, round_number: int) -> RoundResult:
        """Execute one synchronous round with timeout-based update dropping."""
        invited = self.sample_clients(round_number)
        timeout = float(self.config["strategies"]["fedavg_sync"]["timeout"])
        assigned_steps = int(self.config["training"]["default_local_steps"])
        start = self.virtual_time
        accepted = []
        dropped = []
        latencies: list[float] = []
        total_steps = 0
        uplink_bytes = 0
        for client_id in invited:
            participation = sample_participation(
                self.profiles[client_id], self.seed, round_number
            )
            if not participation.available:
                continue
            update = self.train_update(
                client_id,
                round_number,
                assigned_steps,
                assigned_steps,
                start,
                participation.network_delay,
            )
            latency = simulated_latency(
                self.profiles[client_id],
                assigned_steps,
                participation.network_delay,
            )
            latencies.append(min(latency, timeout))
            total_steps += update.completed_steps
            uplink_bytes += update.uplink_bytes
            if latency <= timeout:
                accepted.append(update)
            else:
                dropped.append(update)

        if accepted:
            aggregate = weighted_average(
                accepted, [float(update.num_samples) for update in accepted]
            )
            apply_delta(self.model, aggregate)
            self.model_version += 1
        round_duration = max(latencies, default=0.0)
        self.virtual_time = start + round_duration
        return RoundResult(
            accepted_updates=accepted,
            dropped_updates=dropped,
            total_steps=total_steps,
            wasted_steps=sum(update.completed_steps for update in dropped),
            uplink_bytes=uplink_bytes,
            downlink_bytes=len(invited) * self.downlink_model_bytes,
            virtual_time=self.virtual_time,
            metadata={"invited_clients": invited},
        )
