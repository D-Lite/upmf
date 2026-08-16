"""UPMF coordination composed from the reduced-scope components."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from upmf.coordinator_base import BaseCoordinator, apply_delta
from upmf.heterogeneity import sample_participation
from upmf.types import RoundResult
from upmf.upmf_components.aggregator import StalenessTolerantAggregator
from upmf.upmf_components.compressor import TopKCompressor
from upmf.upmf_components.profiler import ParticipantProfiler
from upmf.upmf_components.scheduler import AdaptiveRoundScheduler
from upmf.utils import derive_seed


class UPMFCoordinator(BaseCoordinator):
    """Coordinate adaptive work and tolerant aggregation under virtual time."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        upmf_config = self.config["strategies"]["upmf"]
        scheduler_config = upmf_config["scheduler"]
        self.profiler = ParticipantProfiler(
            float(scheduler_config["ewma_decay"]),
            int(self.config["training"]["default_local_steps"]),
        )
        self.scheduler = AdaptiveRoundScheduler(
            len(self.clients), self.profiler, scheduler_config
        )
        self.aggregator = StalenessTolerantAggregator(upmf_config["aggregator"])
        self.compressor = TopKCompressor(
            float(upmf_config["compressor"]["topk_fraction"])
        )

    def run_round(self, round_number: int) -> RoundResult:
        """Schedule clients, queue sparse arrivals, and apply tolerant aggregation."""
        count = max(
            1,
            int(
                np.ceil(
                    len(self.clients)
                    * float(self.config["training"]["client_fraction"])
                )
            ),
        )
        rng = np.random.default_rng(
            derive_seed(self.seed, f"upmf-scheduler:{round_number}")
        )
        invited = self.scheduler.select(round_number, count, rng)
        steps = self.scheduler.allocate_steps(
            invited,
            int(self.config["training"]["default_local_steps"]),
            int(self.config["training"]["min_local_steps"]),
            int(self.config["training"]["max_local_steps"]),
        )
        start = self.virtual_time
        window_end = start + float(self.config["strategies"]["upmf"]["round_budget"])
        total_steps = 0
        uplink_bytes = 0
        for client_id in invited:
            participation = sample_participation(
                self.profiles[client_id], self.seed, round_number
            )
            if not participation.available:
                self.profiler.mark_unavailable(client_id)
                continue
            update = self.train_update(
                client_id,
                round_number,
                steps[client_id],
                steps[client_id],
                start,
                participation.network_delay,
                self.compressor,
            )
            self.pending_updates.append(update)
            total_steps += update.completed_steps
            uplink_bytes += update.uplink_bytes

        arrived = [
            update
            for update in self.pending_updates
            if update.arrival_time <= window_end
        ]
        self.pending_updates = [
            update
            for update in self.pending_updates
            if update.arrival_time > window_end
        ]
        decision = self.aggregator.aggregate(arrived)
        if round_number == int(self.config["training"]["rounds"]):
            decision.dropped.extend(self.pending_updates)
            self.pending_updates = []
        if decision.aggregate is not None:
            apply_delta(self.model, decision.aggregate)
            self.model_version += 1
        for update in decision.accepted:
            self.profiler.update_profile(
                update.client_id,
                update.latency,
                update.completed_steps,
                update.divergence,
            )
        for update in decision.dropped:
            self.profiler.update_profile(
                update.client_id,
                update.latency,
                update.completed_steps,
                update.divergence,
            )
        self.virtual_time = window_end
        return RoundResult(
            accepted_updates=decision.accepted,
            dropped_updates=decision.dropped,
            total_steps=total_steps,
            wasted_steps=sum(update.completed_steps for update in decision.dropped),
            uplink_bytes=uplink_bytes,
            downlink_bytes=len(invited) * self.downlink_model_bytes,
            virtual_time=self.virtual_time,
            metadata={"invited_clients": invited, "assigned_steps": steps},
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize base, profiler, and fairness state for exact resumption."""
        return {
            **super().state_dict(),
            "profiler": self.profiler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore all UPMF state from a trusted checkpoint."""
        super().load_state_dict(state)
        self.profiler.load_state_dict(dict(state["profiler"]))
        self.scheduler.load_state_dict(state["scheduler"])
