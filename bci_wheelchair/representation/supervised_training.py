"""
Training utilities for the supervised EEG autoencoder.
"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bci_wheelchair.representation.supervised_autoencoder import (
    SupervisedAutoencoderConfig,
    SupervisedEEGAutoencoder,
)


@dataclass(frozen=True)
class SupervisedTrainingConfig:
    """Training settings for the supervised autoencoder."""

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    patience: int = 10
    classification_weight: float = 0.5
    random_state: int = 42


@dataclass
class SupervisedTrainingHistory:
    """Store training and validation measurements."""

    training_total_losses: list[float]
    validation_total_losses: list[float]
    training_reconstruction_losses: list[float]
    validation_reconstruction_losses: list[float]
    training_classification_losses: list[float]
    validation_classification_losses: list[float]
    training_accuracies: list[float]
    validation_accuracies: list[float]
    best_epoch: int
    best_validation_loss: float


class SupervisedEEGStandardizer:
    """
    Standardise EEG using statistics calculated from training data only.

    A separate mean and standard deviation are calculated for every
    EEG channel and time sample.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(
        self,
        eeg: np.ndarray,
    ) -> "SupervisedEEGStandardizer":
        """Learn normalisation statistics from training EEG."""
        eeg = self._validate_eeg(eeg)

        self.mean_ = eeg.mean(
            axis=0,
            keepdims=True,
            dtype=np.float64,
        ).astype(np.float32)

        self.scale_ = eeg.std(
            axis=0,
            keepdims=True,
            dtype=np.float64,
        ).astype(np.float32)

        self.scale_[self.scale_ < 1e-8] = 1.0

        return self

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        """Apply the training-data normalisation."""
        eeg = self._validate_eeg(eeg)

        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "The standardizer must be fitted before transform()."
            )

        return (
            (eeg.astype(np.float32) - self.mean_)
            / self.scale_
        ).astype(np.float32)

    def fit_transform(self, eeg: np.ndarray) -> np.ndarray:
        """Fit and transform training EEG."""
        return self.fit(eeg).transform(eeg)

    @staticmethod
    def _validate_eeg(eeg: np.ndarray) -> np.ndarray:
        eeg = np.asarray(eeg)

        if eeg.ndim != 3:
            raise ValueError(
                "EEG must have shape "
                "(trials, channels, time), "
                f"but received {eeg.shape}."
            )

        return eeg


def choose_device() -> torch.device:
    """Choose Apple MPS, CUDA or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def set_random_seed(random_state: int) -> None:
    """Set reproducible random seeds."""
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def encode_labels(
    labels: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert text or numeric labels into integer class indices.

    The same mapping must later be used when interpreting predictions.
    """
    labels = np.asarray(labels)

    class_names = sorted(
        str(label)
        for label in np.unique(labels)
    )

    label_to_index = {
        label: index
        for index, label in enumerate(class_names)
    }

    encoded = np.asarray(
        [
            label_to_index[str(label)]
            for label in labels
        ],
        dtype=np.int64,
    )

    return encoded, class_names


def apply_label_mapping(
    labels: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    """Encode labels using an existing class-name mapping."""
    label_to_index = {
        label: index
        for index, label in enumerate(class_names)
    }

    unknown_labels = sorted(
        {
            str(label)
            for label in labels
            if str(label) not in label_to_index
        }
    )

    if unknown_labels:
        raise ValueError(
            "Test data contains unknown labels: "
            f"{unknown_labels}"
        )

    return np.asarray(
        [
            label_to_index[str(label)]
            for label in labels
        ],
        dtype=np.int64,
    )


def create_data_loader(
    eeg: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch EEG data loader."""
    dataset = TensorDataset(
        torch.from_numpy(eeg.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def calculate_epoch_metrics(
    model: SupervisedEEGAutoencoder,
    data_loader: DataLoader,
    reconstruction_criterion: nn.Module,
    classification_criterion: nn.Module,
    classification_weight: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float, float, float]:
    """
    Run one training or validation epoch.

    When optimizer is provided, model parameters are updated.
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    classification_loss_sum = 0.0
    correct_predictions = 0
    trial_count = 0

    for eeg_batch, label_batch in data_loader:
        eeg_batch = eeg_batch.to(device)
        label_batch = label_batch.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            _, reconstruction, logits = model(eeg_batch)

            reconstruction_loss = reconstruction_criterion(
                reconstruction,
                eeg_batch,
            )

            classification_loss = classification_criterion(
                logits,
                label_batch,
            )

            total_loss = (
                reconstruction_loss
                + classification_weight * classification_loss
            )

            if is_training:
                total_loss.backward()
                optimizer.step()

        batch_size = eeg_batch.shape[0]

        total_loss_sum += total_loss.item() * batch_size
        reconstruction_loss_sum += (
            reconstruction_loss.item() * batch_size
        )
        classification_loss_sum += (
            classification_loss.item() * batch_size
        )

        predictions = logits.argmax(dim=1)

        correct_predictions += (
            predictions == label_batch
        ).sum().item()

        trial_count += batch_size

    return (
        total_loss_sum / trial_count,
        reconstruction_loss_sum / trial_count,
        classification_loss_sum / trial_count,
        correct_predictions / trial_count,
    )


def train_supervised_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_config: SupervisedAutoencoderConfig,
    training_config: SupervisedTrainingConfig,
) -> tuple[
    SupervisedEEGAutoencoder,
    SupervisedEEGStandardizer,
    SupervisedTrainingHistory,
    list[str],
]:
    """
    Train a supervised autoencoder using training-subject data only.

    The validation set is selected only from the training subjects.
    """
    set_random_seed(training_config.random_state)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_encoded, class_names = encode_labels(y_train)

    if len(class_names) != model_config.n_classes:
        raise ValueError(
            f"Expected {model_config.n_classes} classes, "
            f"but found {len(class_names)}: {class_names}"
        )

    trial_indices = np.arange(len(X_train))

    train_indices, validation_indices = train_test_split(
        trial_indices,
        test_size=training_config.validation_fraction,
        random_state=training_config.random_state,
        stratify=y_encoded,
    )

    X_model_train = X_train[train_indices]
    y_model_train = y_encoded[train_indices]

    X_validation = X_train[validation_indices]
    y_validation = y_encoded[validation_indices]

    standardizer = SupervisedEEGStandardizer()
    X_model_train = standardizer.fit_transform(X_model_train)
    X_validation = standardizer.transform(X_validation)

    train_loader = create_data_loader(
        eeg=X_model_train,
        labels=y_model_train,
        batch_size=training_config.batch_size,
        shuffle=True,
    )

    validation_loader = create_data_loader(
        eeg=X_validation,
        labels=y_validation,
        batch_size=training_config.batch_size,
        shuffle=False,
    )

    device = choose_device()

    print(f"Device: {device}")
    print(f"Training trials: {len(X_model_train)}")
    print(f"Validation trials: {len(X_validation)}")
    print(f"Classes: {class_names}")
    print(
        "Classification loss weight: "
        f"{training_config.classification_weight}"
    )

    model = SupervisedEEGAutoencoder(
        model_config
    ).to(device)

    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    training_total_losses: list[float] = []
    validation_total_losses: list[float] = []

    training_reconstruction_losses: list[float] = []
    validation_reconstruction_losses: list[float] = []

    training_classification_losses: list[float] = []
    validation_classification_losses: list[float] = []

    training_accuracies: list[float] = []
    validation_accuracies: list[float] = []

    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, training_config.epochs + 1):
        (
            training_total_loss,
            training_reconstruction_loss,
            training_classification_loss,
            training_accuracy,
        ) = calculate_epoch_metrics(
            model=model,
            data_loader=train_loader,
            reconstruction_criterion=reconstruction_criterion,
            classification_criterion=classification_criterion,
            classification_weight=(
                training_config.classification_weight
            ),
            device=device,
            optimizer=optimizer,
        )

        with torch.no_grad():
            (
                validation_total_loss,
                validation_reconstruction_loss,
                validation_classification_loss,
                validation_accuracy,
            ) = calculate_epoch_metrics(
                model=model,
                data_loader=validation_loader,
                reconstruction_criterion=reconstruction_criterion,
                classification_criterion=classification_criterion,
                classification_weight=(
                    training_config.classification_weight
                ),
                device=device,
                optimizer=None,
            )

        training_total_losses.append(training_total_loss)
        validation_total_losses.append(validation_total_loss)

        training_reconstruction_losses.append(
            training_reconstruction_loss
        )
        validation_reconstruction_losses.append(
            validation_reconstruction_loss
        )

        training_classification_losses.append(
            training_classification_loss
        )
        validation_classification_losses.append(
            validation_classification_loss
        )

        training_accuracies.append(training_accuracy)
        validation_accuracies.append(validation_accuracy)

        improved = (
            validation_total_loss
            < best_validation_loss - 1e-6
        )

        improvement_marker = ""

        if improved:
            best_validation_loss = validation_total_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            improvement_marker = " *"
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train total={training_total_loss:.4f} | "
            f"val total={validation_total_loss:.4f} | "
            f"val reconstruction="
            f"{validation_reconstruction_loss:.4f} | "
            f"val classification="
            f"{validation_classification_loss:.4f} | "
            f"val accuracy="
            f"{validation_accuracy * 100:.2f}%"
            f"{improvement_marker}"
        )

        if (
            epochs_without_improvement
            >= training_config.patience
        ):
            print(
                "Early stopping: validation loss did not "
                f"improve for {training_config.patience} epochs."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Training failed to produce a valid model."
        )

    model.load_state_dict(best_state)
    model.eval()

    history = SupervisedTrainingHistory(
        training_total_losses=training_total_losses,
        validation_total_losses=validation_total_losses,
        training_reconstruction_losses=(
            training_reconstruction_losses
        ),
        validation_reconstruction_losses=(
            validation_reconstruction_losses
        ),
        training_classification_losses=(
            training_classification_losses
        ),
        validation_classification_losses=(
            validation_classification_losses
        ),
        training_accuracies=training_accuracies,
        validation_accuracies=validation_accuracies,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )

    print(
        f"Best epoch: {best_epoch} | "
        f"best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    return model, standardizer, history, class_names


def extract_supervised_latent_features(
    model: SupervisedEEGAutoencoder,
    standardizer: SupervisedEEGStandardizer,
    X: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Extract frozen encoder features from EEG."""
    device = next(model.parameters()).device

    X_standardized = standardizer.transform(X)

    data_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(
                X_standardized.astype(np.float32)
            )
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    feature_batches: list[np.ndarray] = []

    model.eval()

    with torch.no_grad():
        for (eeg_batch,) in data_loader:
            eeg_batch = eeg_batch.to(device)

            latent = model.encode(eeg_batch)

            feature_batches.append(
                latent.detach().cpu().numpy()
            )

    return np.concatenate(feature_batches, axis=0)


def save_supervised_autoencoder_checkpoint(
    path: Path,
    model: SupervisedEEGAutoencoder,
    standardizer: SupervisedEEGStandardizer,
    training_config: SupervisedTrainingConfig,
    history: SupervisedTrainingHistory,
    class_names: list[str],
) -> None:
    """Save the trained model and its normalisation information."""
    if standardizer.mean_ is None:
        raise RuntimeError("Standardizer mean is unavailable.")

    if standardizer.scale_ is None:
        raise RuntimeError("Standardizer scale is unavailable.")

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "standardizer_mean": standardizer.mean_,
        "standardizer_scale": standardizer.scale_,
        "class_names": class_names,
        "history": {
            "training_total_losses": (
                history.training_total_losses
            ),
            "validation_total_losses": (
                history.validation_total_losses
            ),
            "training_reconstruction_losses": (
                history.training_reconstruction_losses
            ),
            "validation_reconstruction_losses": (
                history.validation_reconstruction_losses
            ),
            "training_classification_losses": (
                history.training_classification_losses
            ),
            "validation_classification_losses": (
                history.validation_classification_losses
            ),
            "training_accuracies": (
                history.training_accuracies
            ),
            "validation_accuracies": (
                history.validation_accuracies
            ),
            "best_epoch": history.best_epoch,
            "best_validation_loss": (
                history.best_validation_loss
            ),
        },
    }

    torch.save(checkpoint, path)
