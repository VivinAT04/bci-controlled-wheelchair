"""Reusable EEG representation-learning components."""

from bci_wheelchair.representation.autoencoder import (
    AutoencoderConfig,
    EEGAutoencoder,
)
from bci_wheelchair.representation.classifiers import create_classifier
from bci_wheelchair.representation.training import (
    EEGStandardizer,
    TrainingConfig,
    TrainingHistory,
    extract_latent_features,
    get_device,
    load_autoencoder_checkpoint,
    save_autoencoder_checkpoint,
    train_autoencoder,
)

__all__ = [
    "AutoencoderConfig",
    "EEGAutoencoder",
    "EEGStandardizer",
    "TrainingConfig",
    "TrainingHistory",
    "create_classifier",
    "extract_latent_features",
    "get_device",
    "load_autoencoder_checkpoint",
    "save_autoencoder_checkpoint",
    "train_autoencoder",
]
