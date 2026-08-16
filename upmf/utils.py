"""Cross-cutting determinism, device, and logging utilities."""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def derive_seed(seed: int, namespace: str) -> int:
    """Derive an independent deterministic seed for one simulation subsystem."""
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable local training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def numpy_rng(seed: int, namespace: str) -> np.random.Generator:
    """Create an isolated NumPy generator unaffected by other control flow."""
    return np.random.default_rng(derive_seed(seed, namespace))


def resolve_device(requested: str = "auto") -> torch.device:
    """Select an accelerator predictably while preserving an explicit override."""
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_logging(log_path: str | Path, level: int = logging.INFO) -> None:
    """Configure timestamped file and console logs without duplicate handlers."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(run_id)s] %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    class RunIdFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if not hasattr(record, "run_id"):
                record.run_id = "-"
            return True

    run_filter = RunIdFilter()
    file_handler = logging.FileHandler(path, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        handler.addFilter(run_filter)
        root.addHandler(handler)


def cpu_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Move tensors in a state dictionary to CPU for portable checkpointing."""
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }
