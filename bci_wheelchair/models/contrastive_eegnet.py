"""
EEGNet encoder with a supervised contrastive projection head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from bci_wheelchair.models.eegnet import EEGNet, EEGNetConfig


@dataclass(frozen=True)
class ContrastiveEEGNetConfig:
    """Configuration for contrastive EEGNet."""

    n_channels: int
    n_times: int
    n_classes: int = 4

    temporal_filters: int = 8
    depth_multiplier: int = 2
    separable_filters: int = 16
    dropout: float = 0.5

    projection_hidden_size: int = 128
    projection_size: int = 64


class ContrastiveEEGNet(nn.Module):
    """
    EEGNet classifier with a projection head.

    The EEGNet features are used for classification.

    The projection head is used only for supervised
    contrastive learning.
    """

    def __init__(
        self,
        config: ContrastiveEEGNetConfig,
    ) -> None:
        super().__init__()

        self.config = config

        eegnet_config = EEGNetConfig(
            n_channels=config.n_channels,
            n_times=config.n_times,
            n_classes=config.n_classes,
            temporal_filters=config.temporal_filters,
            depth_multiplier=config.depth_multiplier,
            separable_filters=config.separable_filters,
            dropout=config.dropout,
        )

        self.eegnet = EEGNet(eegnet_config)

        feature_size = self.eegnet.classifier.in_features

        self.projection_head = nn.Sequential(
            nn.Linear(
                feature_size,
                config.projection_hidden_size,
            ),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(
                config.projection_hidden_size,
                config.projection_size,
            ),
        )

    def extract_features(
        self,
        eeg: torch.Tensor,
    ) -> torch.Tensor:
        """Return EEGNet encoder features."""
        return self.eegnet.extract_features(eeg)

    def project(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalised contrastive embeddings."""
        projection = self.projection_head(features)

        return F.normalize(
            projection,
            p=2,
            dim=1,
        )

    def forward(
        self,
        eeg: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Return class logits, features and projection.

        Returns
        -------
        logits:
            Shape (batch, classes).

        features:
            EEGNet encoder features.

        projection:
            Normalised contrastive embedding.
        """
        features = self.extract_features(eeg)
        logits = self.eegnet.classifier(features)
        projection = self.project(features)

        return logits, features, projection


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss.

    Trials with the same motor-imagery label are treated
    as positive examples.
    """

    def __init__(
        self,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                "Temperature must be greater than zero."
            )

        self.temperature = temperature

    def forward(
        self,
        projections: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate supervised contrastive loss.

        Parameters
        ----------
        projections:
            Normalised embeddings with shape
            (batch, projection_size).

        labels:
            Integer class labels with shape (batch,).
        """
        if projections.ndim != 2:
            raise ValueError(
                "Projections must have shape "
                "(batch, features)."
            )

        labels = labels.reshape(-1)

        if projections.shape[0] != labels.shape[0]:
            raise ValueError(
                "Projection and label batch sizes differ."
            )

        device = projections.device
        batch_size = projections.shape[0]

        similarity = torch.matmul(
            projections,
            projections.T,
        ) / self.temperature

        identity_mask = torch.eye(
            batch_size,
            dtype=torch.bool,
            device=device,
        )

        label_mask = (
            labels.unsqueeze(0)
            == labels.unsqueeze(1)
        )

        positive_mask = label_mask & ~identity_mask
        comparison_mask = ~identity_mask

        maximum = similarity.max(
            dim=1,
            keepdim=True,
        ).values.detach()

        logits = similarity - maximum

        exp_logits = (
            torch.exp(logits)
            * comparison_mask.float()
        )

        log_probability = logits - torch.log(
            exp_logits.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        positive_count = positive_mask.sum(dim=1)

        valid_anchors = positive_count > 0

        if not valid_anchors.any():
            return projections.sum() * 0.0

        mean_log_probability = (
            positive_mask.float()
            * log_probability
        ).sum(dim=1) / positive_count.clamp_min(1)

        return -mean_log_probability[
            valid_anchors
        ].mean()
