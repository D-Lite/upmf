"""Tests for deterministic non-IID partitioning and the shared learning path."""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from upmf.client import train_client
from upmf.data import class_counts, partition_dirichlet
from upmf.model import SmallCNN


def _dataset(items_per_class: int = 200) -> TensorDataset:
    images = torch.randn(items_per_class * 10, 1, 28, 28)
    labels = torch.arange(10).repeat_interleave(items_per_class)
    dataset = TensorDataset(images, labels)
    dataset.targets = labels  # type: ignore[attr-defined]
    return dataset


def test_partition_is_reproducible_and_complete() -> None:
    dataset = _dataset()
    first = partition_dirichlet(dataset, 20, 0.2, 7)
    second = partition_dirichlet(dataset, 20, 0.2, 7)
    assert [subset.indices for subset in first] == [subset.indices for subset in second]
    flattened = [index for subset in first for index in subset.indices]
    assert sorted(flattened) == list(range(len(dataset)))
    assert all(len(subset) > 0 for subset in first)


def test_lower_alpha_increases_client_class_skew() -> None:
    dataset = _dataset()
    iid = class_counts(partition_dirichlet(dataset, 20, 100.0, 11))
    skewed = class_counts(partition_dirichlet(dataset, 20, 0.2, 11))
    iid_proportions = iid / iid.sum(axis=1, keepdims=True)
    skewed_proportions = skewed / skewed.sum(axis=1, keepdims=True)
    assert skewed_proportions.var(axis=1).mean() > iid_proportions.var(axis=1).mean()


def test_model_and_client_training_share_expected_contract() -> None:
    model = SmallCNN()
    dataset = _dataset(items_per_class=2)
    assert model(torch.randn(3, 1, 28, 28)).shape == (3, 10)
    result = train_client(
        model,
        dataset,
        steps=1,
        batch_size=8,
        learning_rate=0.01,
        momentum=0.0,
        device=torch.device("cpu"),
        seed=3,
    )
    assert result.completed_steps == 1
    assert result.delta.keys() == model.state_dict().keys()
    assert any(torch.count_nonzero(value) for value in result.delta.values())
