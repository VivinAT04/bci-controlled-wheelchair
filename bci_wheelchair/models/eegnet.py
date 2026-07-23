"""Configurable EEGNet architecture for motor-imagery EEG."""

from __future__ import annotations

import math

import torch
from torch import nn


class EEGNet(nn.Module):
    """
    Compact EEGNet architecture for EEG classification.

    Expected input shape:

        batch × 1 × channels × samples
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        dropout_rate: float = 0.5,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int | None = None,
        temporal_kernel_size: int = 64,
        separable_kernel_size: int = 16,
    ) -> None:
        super().__init__()

        if n_channels <= 0:
            raise ValueError("n_channels must be positive.")

        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        if n_classes <= 1:
            raise ValueError(
                "n_classes must be greater than one."
            )

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                "dropout_rate must be in the range [0, 1)."
            )

        if f1 <= 0:
            raise ValueError("f1 must be positive.")

        if depth_multiplier <= 0:
            raise ValueError(
                "depth_multiplier must be positive."
            )

        if f2 is None:
            f2 = f1 * depth_multiplier

        if f2 <= 0:
            raise ValueError("f2 must be positive.")

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.f1 = f1
        self.depth_multiplier = depth_multiplier
        self.f2 = f2

        self.temporal_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, temporal_kernel_size),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(f1),
        )

        spatial_filters = f1 * depth_multiplier

        self.spatial_block = nn.Sequential(
            nn.Conv2d(
                in_channels=f1,
                out_channels=spatial_filters,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 4),
            ),
            nn.Dropout(dropout_rate),
        )

        self.separable_block = nn.Sequential(
            nn.Conv2d(
                in_channels=spatial_filters,
                out_channels=spatial_filters,
                kernel_size=(1, separable_kernel_size),
                padding="same",
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=spatial_filters,
                out_channels=f2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 8),
            ),
            nn.Dropout(dropout_rate),
        )

        flattened_size = self._determine_flattened_size()

        self.classifier = nn.Linear(
            flattened_size,
            n_classes,
        )

    def _determine_flattened_size(self) -> int:
        """Determine classifier input size using a dummy epoch."""

        with torch.no_grad():
            dummy_input = torch.zeros(
                1,
                1,
                self.n_channels,
                self.n_samples,
            )

            dummy_features = self.extract_features(
                dummy_input
            )

        flattened_size = math.prod(
            dummy_features.shape[1:]
        )

        if flattened_size <= 0:
            raise ValueError(
                "The supplied number of samples is too small "
                "for the EEGNet pooling configuration."
            )

        return int(flattened_size)

    def extract_features(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Extract convolutional EEG features."""

        X = self.temporal_block(X)
        X = self.spatial_block(X)
        X = self.separable_block(X)

        return X

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Return unnormalised class scores."""

        X = self.extract_features(X)
        X = torch.flatten(X, start_dim=1)

        return self.classifier(X)


def make_eegnet(
    n_channels: int,
    n_samples: int,
    n_classes: int,
    dropout_rate: float = 0.5,
    f1: int = 8,
    depth_multiplier: int = 2,
    f2: int | None = None,
    temporal_kernel_size: int = 64,
    separable_kernel_size: int = 16,
    device: torch.device | str | None = None,
) -> EEGNet:
    """Create a configurable EEGNet model."""

    model = EEGNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        dropout_rate=dropout_rate,
        f1=f1,
        depth_multiplier=depth_multiplier,
        f2=f2,
        temporal_kernel_size=temporal_kernel_size,
        separable_kernel_size=separable_kernel_size,
    )

    if device is not None:
        model = model.to(device)

    return model


# Alias matching the terminology used in the experiment scripts.
initialise_eegnet = make_eegnet


__all__ = [
    "EEGNet",
    "initialise_eegnet",
    "make_eegnet",
]
