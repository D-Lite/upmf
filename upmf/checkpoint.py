"""Crash-safe checkpoint persistence for unattended experiment runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def save_checkpoint(
    checkpoint_dir: str | Path,
    run_id: str,
    payload: dict[str, Any],
    status: dict[str, Any],
) -> None:
    """Atomically publish tensor state followed by its human-readable manifest."""
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pt_path = directory / f"{run_id}.pt"
    temp_path = directory / f"{run_id}.pt.tmp"
    with temp_path.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, pt_path)
    _atomic_json(directory / f"{run_id}.json", status)


def load_checkpoint(
    checkpoint_dir: str | Path, run_id: str, expected_config_hash: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Load a matching checkpoint pair or return None when neither file exists."""
    directory = Path(checkpoint_dir)
    pt_path = directory / f"{run_id}.pt"
    json_path = directory / f"{run_id}.json"
    if not pt_path.exists() and not json_path.exists():
        return None
    if not pt_path.exists() or not json_path.exists():
        raise RuntimeError(f"Incomplete checkpoint pair for {run_id}")
    with json_path.open("r", encoding="utf-8") as handle:
        status = json.load(handle)
    if status.get("config_hash") != expected_config_hash:
        raise RuntimeError(f"Checkpoint configuration mismatch for {run_id}")
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    if payload.get("generation") != status.get("generation"):
        raise RuntimeError(f"Checkpoint generation mismatch for {run_id}")
    return payload, status


def remove_checkpoint(checkpoint_dir: str | Path, run_id: str) -> None:
    """Remove generated state for one run when the user explicitly requests fresh."""
    directory = Path(checkpoint_dir)
    for suffix in (".pt", ".json", ".pt.tmp", ".json.tmp"):
        path = directory / f"{run_id}{suffix}"
        if path.exists():
            path.unlink()
