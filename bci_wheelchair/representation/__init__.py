"""Reusable EEG representation-learning components."""

from bci_wheelchair.representation.autoencoder import (
    AutoencoderConfig,
    EEGAutoencoder,
)
from bci_wheelchair.representation.classifiers import create_classifier

__all__ = [
    "AutoencoderConfig",
    "EEGAutoencoder",
    "create_classifier",
]
