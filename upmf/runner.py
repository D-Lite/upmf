"""Execution of one deterministic, checkpointed experiment run."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from upmf.checkpoint import load_checkpoint, remove_checkpoint, save_checkpoint
from upmf.config import config_hash, implementation_hash, run_id
from upmf.data import load_mnist, partition_dirichlet
from upmf.heterogeneity import generate_client_profiles
from upmf.metrics import MetricTracker
from upmf.model import build_model
from upmf.strategies.async_vc import AsyncVCCoordinator
from upmf.strategies.fedavg_sync import FedAvgSyncCoordinator
from upmf.utils import derive_seed, resolve_device, set_global_seed

LOGGER = logging.getLogger(__name__)


class IntentionalInterruption(RuntimeError):
    """Test-only signal raised after a valid round checkpoint is persisted."""


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _write_records(path: Path, tracker: MetricTracker) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame([vars(record) for record in tracker.records]).to_csv(temp, index=False)
    os.replace(temp, path)


def _coordinator(
    strategy: str,
    model: torch.nn.Module,
    clients: list[Dataset[Any]],
    profiles: list[Any],
    config: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> Any:
    if strategy == "fedavg_sync":
        return FedAvgSyncCoordinator(model, clients, profiles, config, device, seed)
    if strategy == "async_vc":
        return AsyncVCCoordinator(model, clients, profiles, config, device, seed)
    if strategy == "upmf":
        from upmf.strategies.upmf import UPMFCoordinator

        return UPMFCoordinator(model, clients, profiles, config, device, seed)
    raise ValueError(f"Unknown strategy: {strategy}")


def run_experiment(
    config: dict[str, Any],
    fresh: bool = False,
    train_dataset: Dataset[Any] | None = None,
    test_dataset: Dataset[Any] | None = None,
    stop_after_round: int | None = None,
) -> dict[str, Any]:
    """Run or resume one experiment and persist round and scalar outputs."""
    identifier = run_id(config)
    digest = config_hash(config)
    results_dir = Path(config["project"]["results_dir"])
    checkpoint_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{identifier}.csv"
    summary_path = results_dir / f"{identifier}_summary.json"
    if fresh:
        remove_checkpoint(checkpoint_dir, identifier)
        for path in (csv_path, summary_path):
            if path.exists():
                path.unlink()

    seed = int(config["run"]["seed"])
    set_global_seed(seed)
    device = resolve_device(str(config["project"]["device"]))
    if train_dataset is None or test_dataset is None:
        train_dataset, test_dataset = load_mnist(
            config["project"]["data_dir"],
            config["dataset"].get("train_subset"),
            config["dataset"].get("test_subset"),
            derive_seed(seed, "dataset-subset"),
        )
    clients = partition_dirichlet(
        train_dataset,
        int(config["dataset"]["num_clients"]),
        float(config["run"]["alpha"]),
        derive_seed(seed, "partition"),
    )
    set_global_seed(derive_seed(seed, "model"))
    model = build_model(config["model"])
    profiles = generate_client_profiles(
        len(clients),
        config["systems"][config["run"]["systems_severity"]],
        seed,
    )
    coordinator = _coordinator(
        config["run"]["strategy"], model, clients, profiles, config, device, seed
    )
    tracker = MetricTracker(float(config["metrics"]["target_accuracy"]))
    next_round = 1
    elapsed_offset = 0.0

    checkpoint = load_checkpoint(checkpoint_dir, identifier, digest)
    if checkpoint is not None:
        payload, status = checkpoint
        model.load_state_dict(payload["model_state"])
        coordinator.load_state_dict(payload["coordinator_state"])
        tracker = MetricTracker.from_state_dict(payload["metrics_state"])
        next_round = int(payload["next_round"])
        elapsed_offset = float(payload["real_elapsed_seconds"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if status.get("status") == "completed" and summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        LOGGER.info("resuming at round %s", next_round, extra={"run_id": identifier})

    started = time.perf_counter()
    total_rounds = int(config["training"]["rounds"])
    evaluation_interval = int(config["metrics"]["evaluation_interval"])
    from upmf.client import evaluate_model

    for round_number in range(next_round, total_rounds + 1):
        result = coordinator.run_round(round_number)
        should_evaluate = (
            round_number % evaluation_interval == 0 or round_number == total_rounds
        )
        loss: float | None = None
        accuracy: float | None = None
        if should_evaluate:
            loss, accuracy = evaluate_model(
                model,
                test_dataset,
                int(config["training"]["test_batch_size"]),
                device,
            )
        elapsed = elapsed_offset + (time.perf_counter() - started)
        tracker.record(
            round_number,
            result.virtual_time,
            result.total_steps,
            result.wasted_steps,
            result.uplink_bytes,
            result.downlink_bytes,
            elapsed,
            accuracy,
            loss,
        )
        generation = round_number
        payload = {
            "generation": generation,
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "coordinator_state": coordinator.state_dict(),
            "metrics_state": tracker.state_dict(),
            "next_round": round_number + 1,
            "real_elapsed_seconds": elapsed,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }
        status = {
            "generation": generation,
            "config_hash": digest,
            "run_id": identifier,
            "status": "running",
            "last_completed_round": round_number,
        }
        save_checkpoint(checkpoint_dir, identifier, payload, status)
        _write_records(csv_path, tracker)
        LOGGER.info(
            "completed round %s/%s accuracy=%s",
            round_number,
            total_rounds,
            f"{accuracy:.4f}" if accuracy is not None else "-",
            extra={"run_id": identifier},
        )
        if stop_after_round == round_number:
            raise IntentionalInterruption(f"stopped after round {round_number}")

    summary = {
        "run_id": identifier,
        "config_hash": digest,
        "implementation_hash": implementation_hash(),
        **config["run"],
        **tracker.summary(),
        "rounds": total_rounds,
        "device": str(device),
        "config": config,
    }
    _atomic_json(summary_path, summary)
    final_payload = {
        **payload,
        "generation": total_rounds,
        "next_round": total_rounds + 1,
    }
    save_checkpoint(
        checkpoint_dir,
        identifier,
        final_payload,
        {
            "generation": total_rounds,
            "config_hash": digest,
            "run_id": identifier,
            "status": "completed",
            "last_completed_round": total_rounds,
        },
    )
    return summary
