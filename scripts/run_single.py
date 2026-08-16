#!/usr/bin/env python3
"""CLI for one resumable strategy/condition/seed experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upmf.config import load_yaml, resolved_run_config
from upmf.runner import run_experiment
from upmf.utils import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse the dimensions required to identify exactly one run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--strategy", choices=["upmf", "fedavg_sync", "async_vc"], required=True
    )
    parser.add_argument("--systems", choices=["low", "high"], required=True)
    parser.add_argument("--stats", choices=["low", "high"], required=True)
    parser.add_argument("--seed", type=int, choices=[1, 2], required=True)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Load configuration and run or resume the selected experiment."""
    args = parse_args()
    base = load_yaml(args.config)
    alpha = 100.0 if args.stats == "low" else 0.2
    config = resolved_run_config(
        base, args.strategy, args.systems, args.stats, alpha, args.seed
    )
    configure_logging(Path(config["project"]["results_dir"]) / "matrix.log")
    try:
        run_experiment(config, fresh=args.fresh)
    except Exception:
        logging.exception("single run failed", extra={"run_id": "-"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
