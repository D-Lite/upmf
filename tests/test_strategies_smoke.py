"""Small end-to-end checks for baselines and exact round-boundary resume."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from upmf.coordinator_base import BaseCoordinator
from upmf.experiment_matrix import expand_matrix
from upmf.runner import IntentionalInterruption, run_experiment
from upmf.strategies.async_vc import AsyncVCCoordinator
from upmf.strategies.fedavg_sync import FedAvgSyncCoordinator
from upmf.strategies.upmf import UPMFCoordinator


def _datasets() -> tuple[TensorDataset, TensorDataset]:
    generator = torch.Generator().manual_seed(10)
    train_images = torch.randn(80, 1, 28, 28, generator=generator)
    train_labels = torch.arange(10).repeat(8)
    test_images = torch.randn(20, 1, 28, 28, generator=generator)
    test_labels = torch.arange(10).repeat(2)
    train = TensorDataset(train_images, train_labels)
    test = TensorDataset(test_images, test_labels)
    train.targets = train_labels  # type: ignore[attr-defined]
    test.targets = test_labels  # type: ignore[attr-defined]
    return train, test


def _config(results: Path, strategy: str = "fedavg_sync") -> dict:
    return {
        "project": {"data_dir": "data", "results_dir": str(results), "device": "cpu"},
        "dataset": {"name": "mnist", "num_clients": 4},
        "model": {
            "conv1_channels": 2,
            "conv2_channels": 2,
            "kernel_size": 3,
            "num_classes": 10,
        },
        "training": {
            "rounds": 2,
            "client_fraction": 0.5,
            "batch_size": 4,
            "test_batch_size": 10,
            "learning_rate": 0.01,
            "momentum": 0.0,
            "default_local_steps": 1,
            "min_local_steps": 1,
            "max_local_steps": 2,
        },
        "systems": {
            "low": {
                "speed_lognormal_mean": 0.0,
                "speed_lognormal_sigma": 0.01,
                "dropout_probability": 0.0,
                "network_delay_mean": 0.01,
                "network_delay_sigma": 0.0,
            }
        },
        "strategies": {
            "fedavg_sync": {"timeout": 100.0},
            "async_vc": {"round_budget": 100.0, "hard_cutoff": 200.0},
            "upmf": {
                "round_budget": 100.0,
                "scheduler": {
                    "ewma_decay": 0.7,
                    "availability_weight": 0.65,
                    "speed_weight": 0.35,
                    "exploration_floor": 0.1,
                    "starvation_rounds": 5,
                },
                "aggregator": {
                    "expected_round_time": 100.0,
                    "staleness_rate": 0.35,
                    "divergence_rate": 1.0,
                    "hard_staleness_cutoff": 4.0,
                },
                "compressor": {"topk_fraction": 0.5},
            },
        },
        "metrics": {"evaluation_interval": 1, "target_accuracy": 0.9},
        "checkpoint": {"every_rounds": 1},
        "run": {
            "strategy": strategy,
            "systems_severity": "low",
            "stats_severity": "low",
            "alpha": 100.0,
            "seed": 1,
        },
    }


@pytest.mark.parametrize("strategy", ["fedavg_sync", "async_vc", "upmf"])
def test_baseline_smoke(tmp_path: Path, strategy: str) -> None:
    train, test = _datasets()
    summary = run_experiment(
        _config(tmp_path / strategy, strategy),
        train_dataset=train,
        test_dataset=test,
    )
    assert summary["rounds"] == 2
    assert summary["communication_bytes"] > 0


def test_interrupted_fedavg_matches_uninterrupted(tmp_path: Path) -> None:
    train, test = _datasets()
    uninterrupted = run_experiment(
        _config(tmp_path / "whole"), train_dataset=train, test_dataset=test
    )
    resumed_config = _config(tmp_path / "resumed")
    with pytest.raises(IntentionalInterruption):
        run_experiment(
            deepcopy(resumed_config),
            train_dataset=train,
            test_dataset=test,
            stop_after_round=1,
        )
    resumed = run_experiment(
        resumed_config, train_dataset=train, test_dataset=test
    )
    keys = [
        "final_accuracy",
        "simulated_time",
        "wasted_compute_fraction",
        "communication_bytes",
        "target_round",
        "target_time",
    ]
    assert {key: resumed[key] for key in keys} == {
        key: uninterrupted[key] for key in keys
    }


def test_all_strategies_inherit_the_shared_client_training_path() -> None:
    assert FedAvgSyncCoordinator.train_update is BaseCoordinator.train_update
    assert AsyncVCCoordinator.train_update is BaseCoordinator.train_update
    assert UPMFCoordinator.train_update is BaseCoordinator.train_update


def test_canonical_matrix_has_24_unique_runs() -> None:
    configs = expand_matrix("configs/experiment_matrix.yaml")
    identities = [
        (
            config["run"]["strategy"],
            config["run"]["systems_severity"],
            config["run"]["stats_severity"],
            config["run"]["seed"],
        )
        for config in configs
    ]
    assert len(identities) == 24
    assert len(set(identities)) == 24


@pytest.mark.parametrize("strategy", ["async_vc", "upmf"])
def test_pending_final_updates_are_counted_as_wasted(
    tmp_path: Path, strategy: str
) -> None:
    train, test = _datasets()
    config = _config(tmp_path / f"pending-{strategy}", strategy)
    config["strategies"][strategy]["round_budget"] = 0.001
    summary = run_experiment(config, train_dataset=train, test_dataset=test)
    assert summary["wasted_compute_fraction"] == 1.0
