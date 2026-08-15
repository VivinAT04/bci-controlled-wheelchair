"""Reusable autoencoder for learning compressed EEG representations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AutoencoderConfig:
    """Configuration used to build an EEG autoencoder."""

    n_channels: int
    n_times: int
    latent_dim: int = 32
    hidden_dim: int = 512

    @property
    def input_dim(self) -> int:
        """Return the flattened EEG input size."""
        return self.n_channels * self.n_times


class EEGEncoder(nn.Module):
    """Compress an EEG epoch into a latent feature vector."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()

        self.config = config

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(config.hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, config.latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode EEG epochs.

        Expected input shape:
            (batch_size, n_channels, n_times)

        Returned shape:
            (batch_size, latent_dim)
        """
        self._validate_input(x)
        return self.network(x)

    def _validate_input(self, x: torch.Tensor) -> None:
        """Check that the EEG tensor has the expected dimensions."""
        if x.ndim != 3:
            raise ValueError(
                "EEG input must have shape "
                "(batch_size, n_channels, n_times). "
                f"Received shape: {tuple(x.shape)}"
            )

        expected_shape = (
            self.config.n_channels,
            self.config.n_times,
        )
        received_shape = tuple(x.shape[1:])

        if received_shape != expected_shape:
            raise ValueError(
                f"Expected EEG shape {expected_shape}, "
                f"but received {received_shape}."
            )


class EEGDecoder(nn.Module):
    """Reconstruct an EEG epoch from its latent feature vector."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()

        self.config = config

        self.network = nn.Sequential(
            nn.Linear(config.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )

    def forward(self, latent_features: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features back into EEG epochs.

        Expected input shape:
            (batch_size, latent_dim)

        Returned shape:
            (batch_size, n_channels, n_times)
        """
        reconstructed = self.network(latent_features)

        return reconstructed.reshape(
            -1,
            self.config.n_channels,
            self.config.n_times,
        )


class EEGAutoencoder(nn.Module):
    """Autoencoder containing reusable encoder and decoder components."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()

        self.config = config
        self.encoder = EEGEncoder(config)
        self.decoder = EEGDecoder(config)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and reconstruct an EEG batch.

        Returns:
            reconstructed_eeg:
                Tensor with the same shape as the input.

            latent_features:
                Compressed representation produced by the encoder.
        """
        latent_features = self.encoder(x)
        reconstructed_eeg = self.decoder(latent_features)

        return reconstructed_eeg, latent_features

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the latent EEG features."""
        return self.encoder(x)
