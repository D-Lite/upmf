"""Deterministic expansion and failure-isolated execution of the 24-run matrix."""

from __future__ import annotations

import json
import logging
import os
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from upmf.config import load_yaml, resolved_run_config, run_id
from upmf.runner import run_experiment

LOGGER = logging.getLogger(__name__)


class MatrixRunError(RuntimeError):
    """Raised after all entries are attempted when one or more remain failed."""


def expand_matrix(matrix_path: str | Path) -> list[dict[str, Any]]:
    """Resolve every strategy/severity/seed combination in stable order."""
    path = Path(matrix_path)
    matrix = load_yaml(path)
    base_path = Path(matrix["base_config"])
    if not base_path.exists():
        base_path = path.resolve().parent.parent / base_path
    base = load_yaml(base_path)
    runs = [
        resolved_run_config(base, strategy, systems, stats, alpha, seed)
        for strategy, systems, (stats, alpha), seed in product(
            matrix["strategies"],
            matrix["systems_severities"],
            matrix["statistical_severities"].items(),
            matrix["seeds"],
        )
    ]
    identifiers = [run_id(config) for config in runs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("experiment matrix contains duplicate run identifiers")
    return runs


def _atomic_summary(path: Path, records: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(records).to_csv(temp, index=False)
    os.replace(temp, path)


def run_matrix(matrix_path: str | Path, fresh: bool = False) -> list[dict[str, Any]]:
    """Run all entries sequentially, preserving progress across isolated failures."""
    configs = expand_matrix(matrix_path)
    if len(configs) != 24:
        raise ValueError(f"Canonical reduced matrix must contain 24 runs, got {len(configs)}")
    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    for config in configs:
        identifier = run_id(config)
        try:
            LOGGER.info("starting run", extra={"run_id": identifier})
            summary = run_experiment(config, fresh=fresh)
            summaries.append(summary)
            LOGGER.info("run completed", extra={"run_id": identifier})
        except Exception as exc:
            failures.append(identifier)
            LOGGER.exception("run failed: %s", exc, extra={"run_id": identifier})
            results_dir = Path(config["project"]["results_dir"])
            failure_path = results_dir / f"{identifier}_failure.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            temp = failure_path.with_suffix(".json.tmp")
            with temp.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"run_id": identifier, "error": repr(exc), "config": config},
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            os.replace(temp, failure_path)
    results_dir = Path(configs[0]["project"]["results_dir"])
    persisted_summaries: list[dict[str, Any]] = []
    for summary in summaries:
        summary_path = results_dir / f"{summary['run_id']}_summary.json"
        with summary_path.open("r", encoding="utf-8") as handle:
            persisted_summaries.append(json.load(handle))
    records = [
        {key: value for key, value in summary.items() if key != "config"}
        for summary in sorted(persisted_summaries, key=lambda item: item["run_id"])
    ]
    _atomic_summary(results_dir / "summary.csv", records)
    if failures:
        raise MatrixRunError(
            f"{len(failures)} matrix runs failed: {', '.join(failures)}"
        )
    if len(summaries) != 24:
        raise MatrixRunError(f"Expected 24 completed summaries, got {len(summaries)}")
    return persisted_summaries
