"""Small MNIST classifier shared by every coordination strategy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class SmallCNN(nn.Module):
    """A compact CNN that trains quickly enough for a laptop experiment matrix."""

    def __init__(
        self,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        kernel_size: int = 3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv2d(1, conv1_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                conv1_channels, conv2_channels, kernel_size, padding=padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(conv2_channels * 7 * 7, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class scores for a batch of MNIST images."""
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, 1))


def build_model(config: Mapping[str, Any]) -> SmallCNN:
    """Construct the shared model strictly from the resolved YAML configuration."""
    return SmallCNN(
        conv1_channels=int(config["conv1_channels"]),
        conv2_channels=int(config["conv2_channels"]),
        kernel_size=int(config["kernel_size"]),
        num_classes=int(config["num_classes"]),
    )


def model_bytes(model: nn.Module) -> int:
    """Count dense parameter bytes transmitted in a global-model downlink."""
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
