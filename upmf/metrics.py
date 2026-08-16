"""Round-level metric accumulation and run summary generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RoundMetrics:
    """Persisted measurements for one completed coordination round."""

    round: int
    simulated_time: float
    accuracy: float | None
    test_loss: float | None
    total_steps: int
    wasted_steps: int
    wasted_compute_fraction: float
    uplink_bytes: int
    downlink_bytes: int
    communication_bytes: int
    real_elapsed_seconds: float


class MetricTracker:
    """Accumulate monotonic run metrics and detect first target attainment."""

    def __init__(self, target_accuracy: float) -> None:
        self.target_accuracy = target_accuracy
        self.records: list[RoundMetrics] = []
        self.total_steps = 0
        self.wasted_steps = 0
        self.uplink_bytes = 0
        self.downlink_bytes = 0
        self.target_round: int | None = None
        self.target_time: float | None = None

    def record(
        self,
        round_number: int,
        simulated_time: float,
        added_steps: int,
        added_wasted_steps: int,
        added_uplink_bytes: int,
        added_downlink_bytes: int,
        real_elapsed_seconds: float,
        accuracy: float | None = None,
        test_loss: float | None = None,
    ) -> RoundMetrics:
        """Add one round while preserving cumulative accounting invariants."""
        if added_steps < 0 or added_wasted_steps < 0:
            raise ValueError("step increments must be non-negative")
        self.total_steps += added_steps
        self.wasted_steps += added_wasted_steps
        if self.wasted_steps > self.total_steps:
            raise ValueError("cumulative wasted steps cannot exceed performed steps")
        self.uplink_bytes += added_uplink_bytes
        self.downlink_bytes += added_downlink_bytes
        wasted_fraction = (
            self.wasted_steps / self.total_steps if self.total_steps else 0.0
        )
        record = RoundMetrics(
            round=round_number,
            simulated_time=simulated_time,
            accuracy=accuracy,
            test_loss=test_loss,
            total_steps=self.total_steps,
            wasted_steps=self.wasted_steps,
            wasted_compute_fraction=wasted_fraction,
            uplink_bytes=self.uplink_bytes,
            downlink_bytes=self.downlink_bytes,
            communication_bytes=self.uplink_bytes + self.downlink_bytes,
            real_elapsed_seconds=real_elapsed_seconds,
        )
        self.records.append(record)
        if (
            accuracy is not None
            and self.target_round is None
            and accuracy >= self.target_accuracy
        ):
            self.target_round = round_number
            self.target_time = simulated_time
        return record

    def state_dict(self) -> dict[str, Any]:
        """Return JSON- and checkpoint-friendly accumulated metric state."""
        return {
            "target_accuracy": self.target_accuracy,
            "records": [asdict(record) for record in self.records],
            "total_steps": self.total_steps,
            "wasted_steps": self.wasted_steps,
            "uplink_bytes": self.uplink_bytes,
            "downlink_bytes": self.downlink_bytes,
            "target_round": self.target_round,
            "target_time": self.target_time,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "MetricTracker":
        """Restore accumulated metrics exactly after an interrupted run."""
        tracker = cls(float(state["target_accuracy"]))
        tracker.records = [RoundMetrics(**record) for record in state["records"]]
        tracker.total_steps = int(state["total_steps"])
        tracker.wasted_steps = int(state["wasted_steps"])
        tracker.uplink_bytes = int(state["uplink_bytes"])
        tracker.downlink_bytes = int(state["downlink_bytes"])
        tracker.target_round = state["target_round"]
        tracker.target_time = state["target_time"]
        return tracker

    def summary(self) -> dict[str, Any]:
        """Return final scalar metrics suitable for the matrix summary."""
        final = self.records[-1] if self.records else None
        evaluated = [record for record in self.records if record.accuracy is not None]
        final_accuracy = evaluated[-1].accuracy if evaluated else None
        return {
            "final_accuracy": final_accuracy,
            "simulated_time": final.simulated_time if final else 0.0,
            "wasted_compute_fraction": (
                final.wasted_compute_fraction if final else 0.0
            ),
            "communication_bytes": final.communication_bytes if final else 0,
            "uplink_bytes": final.uplink_bytes if final else 0,
            "downlink_bytes": final.downlink_bytes if final else 0,
            "target_round": self.target_round,
            "target_time": self.target_time,
        }
