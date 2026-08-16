"""Core behavior tests for profiling and adaptive scheduling."""

from __future__ import annotations

import numpy as np

from upmf.upmf_components.profiler import ParticipantProfiler
from upmf.upmf_components.scheduler import AdaptiveRoundScheduler


CONFIG = {
    "availability_weight": 0.65,
    "speed_weight": 0.35,
    "exploration_floor": 0.1,
    "starvation_rounds": 2,
}


def test_profiler_uses_expected_ewma() -> None:
    profiler = ParticipantProfiler(decay=0.5, default_steps=4)
    profiler.update_profile(0, latency=2.0, steps=8, divergence=0.4)
    profile = profiler.get_profile(0)
    assert profile["latency"] == 3.0
    assert profile["completed_steps"] == 6.0
    assert profile["divergence"] == 0.2


def test_scheduler_scales_steps_by_profiled_throughput() -> None:
    profiler = ParticipantProfiler(decay=0.0, default_steps=5)
    profiler.update_profile(0, latency=1.0, steps=10, divergence=0.0)
    profiler.update_profile(1, latency=10.0, steps=1, divergence=0.0)
    scheduler = AdaptiveRoundScheduler(2, profiler, CONFIG)
    allocation = scheduler.allocate_steps([0, 1], 5, 2, 10)
    assert allocation[0] == 10
    assert allocation[1] == 2


def test_scheduler_forces_non_starvation() -> None:
    profiler = ParticipantProfiler(decay=0.5, default_steps=5)
    scheduler = AdaptiveRoundScheduler(4, profiler, CONFIG)
    rng = np.random.default_rng(3)
    selected = set()
    for round_number in range(1, 9):
        selected.update(scheduler.select(round_number, 1, rng))
    assert selected == {0, 1, 2, 3}
