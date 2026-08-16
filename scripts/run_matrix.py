#!/usr/bin/env python3
"""Background-runnable CLI for the canonical resumable 24-run matrix."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upmf.experiment_matrix import MatrixRunError, run_matrix
from upmf.utils import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse matrix location and explicit destructive fresh-run intent."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_matrix.yaml")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Execute every matrix entry and return failure status after all attempts."""
    args = parse_args()
    configure_logging(Path("results") / "matrix.log")
    try:
        run_matrix(args.config, fresh=args.fresh)
    except MatrixRunError as exc:
        logging.error("%s", exc, extra={"run_id": "matrix"})
        return 1
    except Exception:
        logging.exception("matrix setup failed", extra={"run_id": "matrix"})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
