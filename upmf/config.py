"""Configuration loading and validation for reproducible experiments."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject malformed top-level content."""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def deep_merge(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge configuration overrides without mutating either input."""
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail early when a resolved run configuration violates core invariants."""
    required = {
        "project",
        "dataset",
        "model",
        "training",
        "systems",
        "strategies",
        "metrics",
        "checkpoint",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    if int(config["dataset"]["num_clients"]) < 2:
        raise ValueError("dataset.num_clients must be at least 2")
    if int(config["training"]["rounds"]) < 1:
        raise ValueError("training.rounds must be positive")
    fraction = float(config["training"]["client_fraction"])
    if not 0.0 < fraction <= 1.0:
        raise ValueError("training.client_fraction must be in (0, 1]")
    topk = float(config["strategies"]["upmf"]["compressor"]["topk_fraction"])
    if not 0.0 < topk <= 1.0:
        raise ValueError("topk_fraction must be in (0, 1]")


def resolved_run_config(
    base: Mapping[str, Any],
    strategy: str,
    systems_severity: str,
    stats_severity: str,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """Attach run dimensions to a validated copy of the base configuration."""
    config = copy.deepcopy(dict(base))
    if strategy not in {"upmf", "fedavg_sync", "async_vc"}:
        raise ValueError(f"Unknown strategy: {strategy}")
    if systems_severity not in config["systems"]:
        raise ValueError(f"Unknown systems severity: {systems_severity}")
    config["run"] = {
        "strategy": strategy,
        "systems_severity": systems_severity,
        "stats_severity": stats_severity,
        "alpha": float(alpha),
        "seed": int(seed),
    }
    validate_config(config)
    return config


def implementation_hash() -> str:
    """Fingerprint simulator source so stale checkpoints cannot cross code changes."""
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(str(path.relative_to(package_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def config_hash(config: Mapping[str, Any]) -> str:
    """Digest resolved configuration and implementation for safe resumption."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded)
    digest.update(implementation_hash().encode())
    return digest.hexdigest()[:16]


def run_id(config: Mapping[str, Any]) -> str:
    """Build the canonical filesystem-safe identifier for one experiment run."""
    run = config["run"]
    return (
        f"{run['strategy']}_{run['systems_severity']}_"
        f"{run['stats_severity']}_seed{run['seed']}"
    )
