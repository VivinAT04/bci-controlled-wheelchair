"""
Compact EEGNet-style neural network for motor-imagery classification.

Expected input shape:

    (batch, channels, time)

Internally, the model converts this to:

    (batch, 1, channels, time)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EEGNetConfig:
    """Configuration for the EEGNet-style classifier."""

    n_channels: int
    n_times: int
    n_classes: int = 4

    temporal_filters: int = 8
    depth_multiplier: int = 2
    separable_filters: int = 16

    temporal_kernel_size: int = 64
    separable_kernel_size: int = 16

    first_pool_size: int = 4
    second_pool_size: int = 8

    dropout: float = 0.5


class EEGNet(nn.Module):
    """EEGNet-style model for end-to-end EEG classification."""

    def __init__(self, config: EEGNetConfig) -> None:
        super().__init__()

        self.config = config

        depthwise_filters = (
            config.temporal_filters
            * config.depth_multiplier
        )

        self.temporal_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=config.temporal_filters,
                kernel_size=(1, config.temporal_kernel_size),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(config.temporal_filters),
        )

        self.spatial_block = nn.Sequential(
            nn.Conv2d(
                in_channels=config.temporal_filters,
                out_channels=depthwise_filters,
                kernel_size=(config.n_channels, 1),
                groups=config.temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(depthwise_filters),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, config.first_pool_size)
            ),
            nn.Dropout(config.dropout),
        )

        self.separable_block = nn.Sequential(
            nn.Conv2d(
                in_channels=depthwise_filters,
                out_channels=depthwise_filters,
                kernel_size=(1, config.separable_kernel_size),
                padding="same",
                groups=depthwise_filters,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=depthwise_filters,
                out_channels=config.separable_filters,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(config.separable_filters),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, config.second_pool_size)
            ),
            nn.Dropout(config.dropout),
        )

        flattened_size = self._calculate_flattened_size()

        self.classifier = nn.Linear(
            flattened_size,
            config.n_classes,
        )

    def _calculate_flattened_size(self) -> int:
        """Calculate classifier input size using a dummy EEG batch."""
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                1,
                self.config.n_channels,
                self.config.n_times,
            )

            features = self._forward_features(dummy)

        return int(features.flatten(start_dim=1).shape[1])

    def _forward_features(
        self,
        eeg: torch.Tensor,
    ) -> torch.Tensor:
        eeg = self.temporal_block(eeg)
        eeg = self.spatial_block(eeg)
        eeg = self.separable_block(eeg)

        return eeg

    def extract_features(
        self,
        eeg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the learned EEGNet features before classification.

        Input shape:

            (batch, channels, time)
        """
        if eeg.ndim != 3:
            raise ValueError(
                "EEG must have shape "
                "(batch, channels, time), "
                f"but received {tuple(eeg.shape)}."
            )

        eeg = eeg.unsqueeze(1)
        features = self._forward_features(eeg)

        return features.flatten(start_dim=1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        features = self.extract_features(eeg)

        return self.classifier(features)


def initialise_eegnet(
    n_channels: int,
    n_times: int,
    n_classes: int = 4,
    temporal_filters: int = 8,
    depth_multiplier: int = 2,
    separable_filters: int = 16,
    dropout: float = 0.5,
) -> EEGNet:
    """Create an EEGNet model using the supplied dataset dimensions."""
    config = EEGNetConfig(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        temporal_filters=temporal_filters,
        depth_multiplier=depth_multiplier,
        separable_filters=separable_filters,
        dropout=dropout,
    )

    return EEGNet(config)
