"""Training and feature-extraction utilities for the EEG autoencoder."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bci_wheelchair.representation.autoencoder import (
    AutoencoderConfig,
    EEGAutoencoder,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for autoencoder training."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    patience: int = 15
    random_state: int = 42


@dataclass
class TrainingHistory:
    """Loss values recorded during training."""

    train_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    best_validation_loss: float


class EEGStandardizer:
    """
    Standardize EEG independently for each channel.

    Statistics are calculated only from the training EEG.

    For an input with shape:
        (trials, channels, time)

    one mean and standard deviation are calculated for each channel.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.channel_mean: np.ndarray | None = None
        self.channel_std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "EEGStandardizer":
        """Calculate channel statistics from training EEG."""
        X = validate_eeg_array(X)

        self.channel_mean = X.mean(
            axis=(0, 2),
            keepdims=True,
            dtype=np.float64,
        )

        self.channel_std = X.std(
            axis=(0, 2),
            keepdims=True,
            dtype=np.float64,
        )

        self.channel_std = np.maximum(
            self.channel_std,
            self.epsilon,
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply previously fitted channel standardization."""
        X = validate_eeg_array(X)

        if self.channel_mean is None or self.channel_std is None:
            raise RuntimeError(
                "The EEGStandardizer must be fitted before transform()."
            )

        if X.shape[1] != self.channel_mean.shape[1]:
            raise ValueError(
                "The number of EEG channels does not match the fitted "
                "standardizer."
            )

        standardized = (
            X.astype(np.float32) - self.channel_mean
        ) / self.channel_std

        return standardized.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the standardizer and transform the same training EEG."""
        return self.fit(X).transform(X)

    def state_dict(self) -> dict[str, Any]:
        """Return serializable standardizer parameters."""
        if self.channel_mean is None or self.channel_std is None:
            raise RuntimeError(
                "Cannot save an unfitted EEGStandardizer."
            )

        return {
            "epsilon": self.epsilon,
            "channel_mean": self.channel_mean.astype(np.float32),
            "channel_std": self.channel_std.astype(np.float32),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
    ) -> "EEGStandardizer":
        """Restore a standardizer from saved parameters."""
        standardizer = cls(epsilon=float(state["epsilon"]))

        standardizer.channel_mean = np.asarray(
            state["channel_mean"],
            dtype=np.float32,
        )

        standardizer.channel_std = np.asarray(
            state["channel_std"],
            dtype=np.float32,
        )

        return standardizer


def set_random_seeds(random_state: int) -> None:
    """Set reproducible random seeds."""
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def get_device() -> torch.device:
    """Select the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def validate_eeg_array(X: np.ndarray) -> np.ndarray:
    """Validate an EEG array with trial, channel, and time dimensions."""
    X = np.asarray(X)

    if X.ndim != 3:
        raise ValueError(
            "EEG data must have shape "
            "(n_trials, n_channels, n_times). "
            f"Received shape: {X.shape}"
        )

    if X.shape[0] < 2:
        raise ValueError(
            "At least two EEG trials are required."
        )

    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("EEG data must contain numeric values.")

    if not np.isfinite(X).all():
        raise ValueError(
            "EEG data contains NaN or infinite values."
        )

    return X


def split_training_validation(
    X: np.ndarray,
    validation_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split training EEG into autoencoder training and validation subsets.

    This function must receive training data only. Test EEG must never be
    passed here.
    """
    X = validate_eeg_array(X)

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))

    n_validation = max(
        1,
        int(round(len(X) * validation_fraction)),
    )

    validation_indices = indices[:n_validation]
    training_indices = indices[n_validation:]

    if len(training_indices) == 0:
        raise ValueError(
            "The validation split left no trials for training."
        )

    return X[training_indices], X[validation_indices]


def create_loader(
    X: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch DataLoader for reconstruction training."""
    tensor = torch.from_numpy(
        np.asarray(X, dtype=np.float32)
    )

    dataset = TensorDataset(tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def calculate_epoch_loss(
    model: EEGAutoencoder,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """
    Run one training or validation epoch.

    When an optimizer is supplied, model parameters are updated.
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_trials = 0

    for (batch,) in loader:
        batch = batch.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            reconstructed, _ = model(batch)
            loss = loss_function(reconstructed, batch)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = batch.shape[0]
        total_loss += loss.item() * batch_size
        total_trials += batch_size

    return total_loss / total_trials


def train_autoencoder(
    X_train: np.ndarray,
    model_config: AutoencoderConfig,
    training_config: TrainingConfig,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[
    EEGAutoencoder,
    EEGStandardizer,
    TrainingHistory,
]:
    """
    Train an EEG autoencoder using training EEG only.

    Workflow:
        1. Split the supplied training EEG into train and validation subsets.
        2. Fit normalization using only the autoencoder training subset.
        3. Transform both subsets using the same training statistics.
        4. Train using reconstruction loss.
        5. Restore the model with the best validation loss.

    Test EEG must not be supplied to this function.
    """
    set_random_seeds(training_config.random_state)

    if device is None:
        device = get_device()

    X_train = validate_eeg_array(X_train)

    expected_shape = (
        model_config.n_channels,
        model_config.n_times,
    )

    if tuple(X_train.shape[1:]) != expected_shape:
        raise ValueError(
            f"Model expects EEG shape {expected_shape}, "
            f"but received {tuple(X_train.shape[1:])}."
        )

    X_fit, X_validation = split_training_validation(
        X=X_train,
        validation_fraction=training_config.validation_fraction,
        random_state=training_config.random_state,
    )

    standardizer = EEGStandardizer()
    X_fit = standardizer.fit_transform(X_fit)
    X_validation = standardizer.transform(X_validation)

    training_loader = create_loader(
        X=X_fit,
        batch_size=training_config.batch_size,
        shuffle=True,
    )

    validation_loader = create_loader(
        X=X_validation,
        batch_size=training_config.batch_size,
        shuffle=False,
    )

    model = EEGAutoencoder(model_config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    loss_function = nn.MSELoss()

    train_losses: list[float] = []
    validation_losses: list[float] = []

    best_model_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    if verbose:
        print(f"Device: {device}")
        print(f"Training trials: {len(X_fit)}")
        print(f"Validation trials: {len(X_validation)}")

    for epoch in range(1, training_config.epochs + 1):
        train_loss = calculate_epoch_loss(
            model=model,
            loader=training_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
        )

        validation_loss = calculate_epoch_loss(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            device=device,
            optimizer=None,
        )

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        improved = validation_loss < best_validation_loss

        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            marker = " *" if improved else ""

            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.6f} | "
                f"validation={validation_loss:.6f}"
                f"{marker}"
            )

        if epochs_without_improvement >= training_config.patience:
            if verbose:
                print(
                    "Early stopping: validation loss did not improve "
                    f"for {training_config.patience} epochs."
                )
            break

    if best_model_state is None:
        raise RuntimeError(
            "Training finished without producing a valid model."
        )

    model.load_state_dict(best_model_state)
    model.eval()

    history = TrainingHistory(
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )

    if verbose:
        print(
            f"Best epoch: {history.best_epoch} | "
            f"best validation loss: "
            f"{history.best_validation_loss:.6f}"
        )

    return model, standardizer, history


def extract_latent_features(
    model: EEGAutoencoder,
    standardizer: EEGStandardizer,
    X: np.ndarray,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Convert EEG epochs into latent feature vectors.

    The same standardizer fitted using training EEG must be used for both
    training and held-out EEG.
    """
    if device is None:
        device = get_device()

    X = validate_eeg_array(X)
    X_standardized = standardizer.transform(X)

    loader = create_loader(
        X=X_standardized,
        batch_size=batch_size,
        shuffle=False,
    )

    model = model.to(device)
    model.eval()

    feature_batches: list[np.ndarray] = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            latent_features = model.encode(batch)

            feature_batches.append(
                latent_features.cpu().numpy()
            )

    return np.concatenate(feature_batches, axis=0)


def save_autoencoder_checkpoint(
    path: str | Path,
    model: EEGAutoencoder,
    standardizer: EEGStandardizer,
    training_config: TrainingConfig,
    history: TrainingHistory,
) -> None:
    """Save model weights, normalization, configuration, and history."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_config": asdict(model.config),
        "training_config": asdict(training_config),
        "model_state_dict": model.state_dict(),
        "standardizer": standardizer.state_dict(),
        "history": {
            "train_losses": history.train_losses,
            "validation_losses": history.validation_losses,
            "best_epoch": history.best_epoch,
            "best_validation_loss": history.best_validation_loss,
        },
    }

    torch.save(checkpoint, path)


def load_autoencoder_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[
    EEGAutoencoder,
    EEGStandardizer,
    TrainingConfig,
    TrainingHistory,
]:
    """Load a trained autoencoder and its training-only normalization."""
    if device is None:
        device = get_device()

    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )

    model_config = AutoencoderConfig(
        **checkpoint["model_config"]
    )

    training_config = TrainingConfig(
        **checkpoint["training_config"]
    )

    model = EEGAutoencoder(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    standardizer = EEGStandardizer.from_state_dict(
        checkpoint["standardizer"]
    )

    history_data = checkpoint["history"]

    history = TrainingHistory(
        train_losses=list(history_data["train_losses"]),
        validation_losses=list(
            history_data["validation_losses"]
        ),
        best_epoch=int(history_data["best_epoch"]),
        best_validation_loss=float(
            history_data["best_validation_loss"]
        ),
    )

    return model, standardizer, training_config, history
