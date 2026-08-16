"""MNIST loading and reproducible label-skewed client partitioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


def dataset_targets(dataset: Dataset[Any]) -> np.ndarray:
    """Extract integer targets from torchvision datasets or lightweight fixtures."""
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        if isinstance(targets, torch.Tensor):
            return targets.detach().cpu().numpy().astype(np.int64)
        return np.asarray(targets, dtype=np.int64)
    return np.asarray([int(dataset[index][1]) for index in range(len(dataset))])


def partition_dirichlet(
    dataset: Dataset[Any], num_clients: int, alpha: float, seed: int
) -> list[Subset[Any]]:
    """Partition examples by class using a seeded Dirichlet allocation.

    A repair step moves one item from the largest partition into any empty
    partition. It prevents invalid clients while retaining the intended skew.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    labels = dataset_targets(dataset)
    if len(labels) < num_clients:
        raise ValueError("dataset must contain at least one item per client")
    rng = np.random.default_rng(seed)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts = rng.multinomial(len(indices), proportions)
        offset = 0
        for client_id, count in enumerate(counts):
            end = offset + int(count)
            client_indices[client_id].extend(indices[offset:end].tolist())
            offset = end

    for empty_id in [i for i, values in enumerate(client_indices) if not values]:
        donor_id = max(range(num_clients), key=lambda i: len(client_indices[i]))
        client_indices[empty_id].append(client_indices[donor_id].pop())
    for indices in client_indices:
        rng.shuffle(indices)
    return [Subset(dataset, indices) for indices in client_indices]


def load_mnist(
    root: str | Path,
    train_subset: int | None = None,
    test_subset: int | None = None,
    seed: int = 0,
) -> tuple[Dataset[Any], Dataset[Any]]:
    """Load normalized MNIST and optionally select deterministic smoke subsets."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    root_path = str(Path(root))
    train: Dataset[Any] = datasets.MNIST(
        root_path, train=True, download=True, transform=transform
    )
    test: Dataset[Any] = datasets.MNIST(
        root_path, train=False, download=True, transform=transform
    )
    rng = np.random.default_rng(seed)
    if train_subset is not None and train_subset < len(train):
        indices = rng.choice(len(train), size=train_subset, replace=False).tolist()
        train = Subset(train, indices)
    if test_subset is not None and test_subset < len(test):
        indices = rng.choice(len(test), size=test_subset, replace=False).tolist()
        test = Subset(test, indices)
    return train, test


def class_counts(
    subsets: Sequence[Subset[Any]], num_classes: int = 10
) -> np.ndarray:
    """Return a client-by-class count matrix for diagnostics and tests."""
    result = np.zeros((len(subsets), num_classes), dtype=np.int64)
    for client_id, subset in enumerate(subsets):
        labels = dataset_targets(subset.dataset)[np.asarray(subset.indices)]
        result[client_id] = np.bincount(labels, minlength=num_classes)
    return result
