"""Tests for configured systems-severity behavior."""

from __future__ import annotations

import numpy as np

from upmf.heterogeneity import generate_client_profiles, sample_participation


LOW = {
    "speed_lognormal_mean": 0.0,
    "speed_lognormal_sigma": 0.1,
    "dropout_probability": 0.05,
    "network_delay_mean": 0.1,
    "network_delay_sigma": 0.02,
}
HIGH = {
    "speed_lognormal_mean": 0.0,
    "speed_lognormal_sigma": 1.0,
    "dropout_probability": 0.3,
    "network_delay_mean": 1.0,
    "network_delay_sigma": 0.6,
}


def test_high_severity_has_more_speed_dispersion() -> None:
    low = generate_client_profiles(500, LOW, 4)
    high = generate_client_profiles(500, HIGH, 4)
    assert np.std([p.compute_speed_multiplier for p in high]) > np.std(
        [p.compute_speed_multiplier for p in low]
    )


def test_high_severity_has_more_dropout_and_delay() -> None:
    low_profile = generate_client_profiles(1, LOW, 9)[0]
    high_profile = generate_client_profiles(1, HIGH, 9)[0]
    low = [sample_participation(low_profile, 5, round_) for round_ in range(500)]
    high = [sample_participation(high_profile, 5, round_) for round_ in range(500)]
    assert sum(not state.available for state in high) > sum(
        not state.available for state in low
    )
    assert np.mean([state.network_delay for state in high]) > np.mean(
        [state.network_delay for state in low]
    )


def test_participation_is_reproducible() -> None:
    profile = generate_client_profiles(1, HIGH, 2)[0]
    assert sample_participation(profile, 7, 3) == sample_participation(profile, 7, 3)
