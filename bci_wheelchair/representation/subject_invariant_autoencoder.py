"""
Subject-invariant supervised EEG autoencoder.

The model learns to:
1. reconstruct the EEG trial;
2. predict the motor-imagery class;
3. reduce subject-specific information using adversarial training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd import Function


@dataclass(frozen=True)
class SubjectInvariantAutoencoderConfig:
    """Model configuration."""

    n_channels: int
    n_times: int
    n_subjects: int
    n_classes: int = 4
    latent_dim: int = 32
    hidden_dim: int = 256
    dropout: float = 0.25

    @property
    def input_dim(self) -> int:
        """Return the flattened EEG input size."""
        return self.n_channels * self.n_times


class _GradientReversalFunction(Function):
    """Reverse gradients during the backward pass."""

    @staticmethod
    def forward(
        ctx: object,
        inputs: torch.Tensor,
        coefficient: float,
    ) -> torch.Tensor:
        ctx.coefficient = coefficient
        return inputs.view_as(inputs)

    @staticmethod
    def backward(
        ctx: object,
        gradients: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * gradients, None


class GradientReversal(nn.Module):
    """Gradient-reversal layer."""

    def forward(
        self,
        inputs: torch.Tensor,
        coefficient: float = 1.0,
    ) -> torch.Tensor:
        return _GradientReversalFunction.apply(
            inputs,
            float(coefficient),
        )


class SubjectInvariantEEGEncoder(nn.Module):
    """Encode an EEG trial into a latent vector."""

    def __init__(
        self,
        config: SubjectInvariantAutoencoderConfig,
    ) -> None:
        super().__init__()

        second_hidden_dim = max(
            config.hidden_dim // 2,
            config.latent_dim,
        )

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
        if eeg.ndim != 3:
            raise ValueError(
                "EEG must have shape "
                "(batch, channels, time), "
                f"but received {tuple(eeg.shape)}."
            )

        return self.network(eeg)


class SubjectInvariantEEGDecoder(nn.Module):
    """Reconstruct EEG from latent features."""

    def __init__(
        self,
        config: SubjectInvariantAutoencoderConfig,
    ) -> None:
        super().__init__()

        second_hidden_dim = max(
            config.hidden_dim // 2,
            config.latent_dim,
        )

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
        reconstruction = self.network(latent)

        return reconstruction.reshape(
            latent.shape[0],
            self.n_channels,
            self.n_times,
        )


class MotorImageryHead(nn.Module):
    """Predict the motor-imagery class."""

    def __init__(
        self,
        config: SubjectInvariantAutoencoderConfig,
    ) -> None:
        super().__init__()

        hidden_dim = max(
            config.latent_dim // 2,
            config.n_classes * 2,
        )

        self.network = nn.Sequential(
            nn.Linear(config.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.n_classes),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class SubjectHead(nn.Module):
    """Predict the training subject."""

    def __init__(
        self,
        config: SubjectInvariantAutoencoderConfig,
    ) -> None:
        super().__init__()

        hidden_dim = max(
            config.latent_dim,
            config.n_subjects * 2,
        )

        self.network = nn.Sequential(
            nn.Linear(config.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.n_subjects),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class SubjectInvariantEEGAutoencoder(nn.Module):
    """Autoencoder with class and adversarial subject heads."""

    def __init__(
        self,
        config: SubjectInvariantAutoencoderConfig,
    ) -> None:
        super().__init__()

        self.config = config
        self.encoder = SubjectInvariantEEGEncoder(config)
        self.decoder = SubjectInvariantEEGDecoder(config)
        self.classification_head = MotorImageryHead(config)
        self.gradient_reversal = GradientReversal()
        self.subject_head = SubjectHead(config)

    def forward(
        self,
        eeg: torch.Tensor,
        reversal_coefficient: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        latent = self.encoder(eeg)
        reconstruction = self.decoder(latent)
        class_logits = self.classification_head(latent)

        reversed_latent = self.gradient_reversal(
            latent,
            reversal_coefficient,
        )

        subject_logits = self.subject_head(reversed_latent)

        return (
            latent,
            reconstruction,
            class_logits,
            subject_logits,
        )

    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        """Return only latent features."""
        return self.encoder(eeg)

    def classify(self, eeg: torch.Tensor) -> torch.Tensor:
        """Return motor-imagery logits."""
        latent = self.encoder(eeg)
        return self.classification_head(latent)
