"""
Supervised EEG autoencoder.

The model learns two tasks at the same time:

1. Reconstruct the original EEG trial.
2. Predict the motor-imagery class.

After training, only the encoder is needed for latent feature extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SupervisedAutoencoderConfig:
    """Configuration for the supervised EEG autoencoder."""

    n_channels: int
    n_times: int
    latent_dim: int = 32
    hidden_dim: int = 256
    n_classes: int = 4
    dropout: float = 0.25

    @property
    def input_dim(self) -> int:
        """Return the flattened EEG input size."""
        return self.n_channels * self.n_times


class SupervisedEEGEncoder(nn.Module):
    """Convert an EEG trial into a compact latent representation."""

    def __init__(
        self,
        config: SupervisedAutoencoderConfig,
    ) -> None:
        super().__init__()

        second_hidden_dim = max(config.hidden_dim // 2, config.latent_dim)

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, second_hidden_dim),
            nn.BatchNorm1d(second_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(second_hidden_dim, config.latent_dim),
        )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """Create latent features from EEG."""
        return self.network(eeg)


class SupervisedEEGDecoder(nn.Module):
    """Reconstruct EEG from latent features."""

    def __init__(
        self,
        config: SupervisedAutoencoderConfig,
    ) -> None:
        super().__init__()

        second_hidden_dim = max(config.hidden_dim // 2, config.latent_dim)

        self.n_channels = config.n_channels
        self.n_times = config.n_times

        self.network = nn.Sequential(
            nn.Linear(config.latent_dim, second_hidden_dim),
            nn.ReLU(),
            nn.Linear(second_hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct EEG from latent features."""
        reconstruction = self.network(latent)

        return reconstruction.reshape(
            latent.shape[0],
            self.n_channels,
            self.n_times,
        )


class MotorImageryClassificationHead(nn.Module):
    """Predict one of the four motor-imagery classes."""

    def __init__(
        self,
        config: SupervisedAutoencoderConfig,
    ) -> None:
        super().__init__()

        classification_hidden_dim = max(
            config.latent_dim // 2,
            config.n_classes * 2,
        )

        self.network = nn.Sequential(
            nn.Linear(
                config.latent_dim,
                classification_hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                classification_hidden_dim,
                config.n_classes,
            ),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return unnormalised class scores."""
        return self.network(latent)


class SupervisedEEGAutoencoder(nn.Module):
    """
    EEG autoencoder with an additional classification head.

    The forward method returns:

        latent features
        reconstructed EEG
        classification logits
    """

    def __init__(
        self,
        config: SupervisedAutoencoderConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self.encoder = SupervisedEEGEncoder(config)
        self.decoder = SupervisedEEGDecoder(config)
        self.classification_head = MotorImageryClassificationHead(
            config
        )

    def forward(
        self,
        eeg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode, reconstruct and classify an EEG batch."""
        latent = self.encoder(eeg)
        reconstruction = self.decoder(latent)
        logits = self.classification_head(latent)

        return latent, reconstruction, logits

    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        """Return only the latent EEG features."""
        return self.encoder(eeg)
