"""The single local-training implementation used by all strategies."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from upmf.types import TensorMap


@dataclass(frozen=True)
class ClientTrainResult:
    """A local model delta and the amount of useful SGD work performed."""

    delta: TensorMap
    completed_steps: int
    mean_loss: float


def train_client(
    global_model: nn.Module,
    dataset: Dataset[Any],
    steps: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    device: torch.device,
    seed: int,
) -> ClientTrainResult:
    """Train from a global snapshot and return a CPU model delta.

    Coordination strategies choose only how many steps to request; optimizer,
    batching, loss, and delta construction are deliberately centralized here.
    """
    if steps < 0:
        raise ValueError("steps cannot be negative")
    if len(dataset) == 0:
        raise ValueError("client dataset cannot be empty")
    if steps == 0:
        return ClientTrainResult(
            delta={
                name: torch.zeros_like(value, device="cpu")
                for name, value in global_model.state_dict().items()
            },
            completed_steps=0,
            mean_loss=0.0,
        )

    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    initial = {
        name: value.detach().cpu().clone()
        for name, value in global_model.state_dict().items()
    }
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.SGD(
        local_model.parameters(), lr=learning_rate, momentum=momentum
    )
    criterion = nn.CrossEntropyLoss()
    iterator = iter(loader)
    losses: list[float] = []
    for _ in range(steps):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, targets = next(iterator)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(local_model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    delta = {
        name: value.detach().cpu() - initial[name]
        for name, value in local_model.state_dict().items()
    }
    return ClientTrainResult(
        delta=delta,
        completed_steps=steps,
        mean_loss=sum(losses) / len(losses),
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: Dataset[Any],
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return mean cross-entropy and accuracy on the shared held-out dataset."""
    original_device = next(model.parameters()).device
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        total_loss += float(criterion(logits, targets).detach().cpu())
        correct += int((logits.argmax(dim=1) == targets).sum().detach().cpu())
        total += int(targets.numel())
    result = (total_loss / total, correct / total)
    model.to(original_device)
    return result
