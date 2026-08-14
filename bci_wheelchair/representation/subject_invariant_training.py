"""Training utilities for the subject-invariant EEG autoencoder."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bci_wheelchair.representation.subject_invariant_autoencoder import (
    SubjectInvariantAutoencoderConfig,
    SubjectInvariantEEGAutoencoder,
)


@dataclass(frozen=True)
class SubjectInvariantTrainingConfig:
    """Training settings for the subject-invariant autoencoder."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    validation_fraction: float = 0.2
    patience: int = 15
    reconstruction_weight: float = 1.0
    classification_weight: float = 1.0
    subject_weight: float = 0.1
    max_reversal_coefficient: float = 1.0
    random_state: int = 42


@dataclass
class SubjectInvariantTrainingHistory:
    """Store measurements recorded during training."""

    training_total_losses: list[float]
    validation_total_losses: list[float]

    training_reconstruction_losses: list[float]
    validation_reconstruction_losses: list[float]

    training_classification_losses: list[float]
    validation_classification_losses: list[float]

    training_subject_losses: list[float]
    validation_subject_losses: list[float]

    training_class_accuracies: list[float]
    validation_class_accuracies: list[float]

    training_subject_accuracies: list[float]
    validation_subject_accuracies: list[float]

    best_epoch: int
    best_validation_classification_loss: float


class ChannelStandardizer:
    """Standardise each EEG channel using training data only."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(
        self,
        eeg: np.ndarray,
    ) -> "ChannelStandardizer":
        """Learn channel statistics from training EEG."""
        eeg = self._validate(eeg)

        self.mean_ = eeg.mean(
            axis=(0, 2),
            keepdims=True,
            dtype=np.float64,
        ).astype(np.float32)

        self.scale_ = eeg.std(
            axis=(0, 2),
            keepdims=True,
            dtype=np.float64,
        ).astype(np.float32)

        self.scale_[self.scale_ < 1e-8] = 1.0

        return self

    def transform(
        self,
        eeg: np.ndarray,
    ) -> np.ndarray:
        """Apply training-data standardisation."""
        eeg = self._validate(eeg)

        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "The standardizer must be fitted before transform()."
            )

        return (
            (eeg.astype(np.float32) - self.mean_)
            / self.scale_
        ).astype(np.float32)

    def fit_transform(
        self,
        eeg: np.ndarray,
    ) -> np.ndarray:
        """Fit and transform EEG."""
        return self.fit(eeg).transform(eeg)

    @staticmethod
    def _validate(
        eeg: np.ndarray,
    ) -> np.ndarray:
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


def set_random_seed(
    random_state: int,
) -> None:
    """Set reproducible random seeds."""
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def encode_labels(
    labels: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Convert class labels into integer indices."""
    labels = np.asarray(labels)

    class_names = sorted(
        str(label)
        for label in np.unique(labels)
    )

    label_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    encoded_labels = np.asarray(
        [
            label_to_index[str(label)]
            for label in labels
        ],
        dtype=np.int64,
    )

    return encoded_labels, class_names


def apply_label_mapping(
    labels: np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    """Encode labels using an existing class order."""
    label_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
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
            f"Unknown labels: {unknown_labels}"
        )

    return np.asarray(
        [
            label_to_index[str(label)]
            for label in labels
        ],
        dtype=np.int64,
    )


def create_loader(
    eeg: np.ndarray,
    class_labels: np.ndarray,
    subject_labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch data loader."""
    dataset = TensorDataset(
        torch.from_numpy(
            eeg.astype(np.float32)
        ),
        torch.from_numpy(
            class_labels.astype(np.int64)
        ),
        torch.from_numpy(
            subject_labels.astype(np.int64)
        ),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def reversal_schedule(
    epoch_index: int,
    total_epochs: int,
    maximum: float,
) -> float:
    """Gradually increase gradient-reversal strength."""
    progress = (
        epoch_index + 1
    ) / max(total_epochs, 1)

    coefficient = (
        2.0
        / (
            1.0
            + math.exp(-10.0 * progress)
        )
        - 1.0
    )

    return float(
        maximum * coefficient
    )


def run_epoch(
    model: SubjectInvariantEEGAutoencoder,
    loader: DataLoader,
    reconstruction_criterion: nn.Module,
    classification_criterion: nn.Module,
    subject_criterion: nn.Module,
    config: SubjectInvariantTrainingConfig,
    reversal_coefficient: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """Run one training or validation epoch."""
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    reconstruction_loss_sum = 0.0
    classification_loss_sum = 0.0
    subject_loss_sum = 0.0

    correct_classes = 0
    correct_subjects = 0
    trial_count = 0

    for (
        eeg_batch,
        class_batch,
        subject_batch,
    ) in loader:
        eeg_batch = eeg_batch.to(device)
        class_batch = class_batch.to(device)
        subject_batch = subject_batch.to(device)

        if is_training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            is_training
        ):
            (
                _,
                reconstruction,
                class_logits,
                subject_logits,
            ) = model(
                eeg_batch,
                reversal_coefficient=(
                    reversal_coefficient
                ),
            )

            reconstruction_loss = (
                reconstruction_criterion(
                    reconstruction,
                    eeg_batch,
                )
            )

            classification_loss = (
                classification_criterion(
                    class_logits,
                    class_batch,
                )
            )

            subject_loss = subject_criterion(
                subject_logits,
                subject_batch,
            )

            total_loss = (
                config.reconstruction_weight
                * reconstruction_loss
                + config.classification_weight
                * classification_loss
                + config.subject_weight
                * subject_loss
            )

            if is_training:
                total_loss.backward()
                optimizer.step()

        batch_size = eeg_batch.shape[0]

        total_loss_sum += (
            total_loss.item()
            * batch_size
        )

        reconstruction_loss_sum += (
            reconstruction_loss.item()
            * batch_size
        )

        classification_loss_sum += (
            classification_loss.item()
            * batch_size
        )

        subject_loss_sum += (
            subject_loss.item()
            * batch_size
        )

        class_predictions = (
            class_logits.argmax(dim=1)
        )

        subject_predictions = (
            subject_logits.argmax(dim=1)
        )

        correct_classes += (
            class_predictions == class_batch
        ).sum().item()

        correct_subjects += (
            subject_predictions == subject_batch
        ).sum().item()

        trial_count += batch_size

    return (
        total_loss_sum / trial_count,
        reconstruction_loss_sum / trial_count,
        classification_loss_sum / trial_count,
        subject_loss_sum / trial_count,
        correct_classes / trial_count,
        correct_subjects / trial_count,
    )


def train_subject_invariant_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subject_ids: np.ndarray,
    model_config: SubjectInvariantAutoencoderConfig,
    training_config: SubjectInvariantTrainingConfig,
) -> tuple[
    SubjectInvariantEEGAutoencoder,
    ChannelStandardizer,
    SubjectInvariantTrainingHistory,
    list[str],
]:
    """Train the model using training subjects only."""
    set_random_seed(
        training_config.random_state
    )

    X_train = np.asarray(
        X_train,
        dtype=np.float32,
    )

    subject_ids = np.asarray(
        subject_ids,
        dtype=np.int64,
    )

    y_encoded, class_names = encode_labels(
        y_train
    )

    if (
        len(X_train) != len(y_encoded)
        or len(X_train) != len(subject_ids)
    ):
        raise ValueError(
            "EEG, class labels and subject IDs "
            "must have matching lengths."
        )

    if len(class_names) != model_config.n_classes:
        raise ValueError(
            f"Expected {model_config.n_classes} classes, "
            f"but found {class_names}."
        )

    if (
        len(np.unique(subject_ids))
        != model_config.n_subjects
    ):
        raise ValueError(
            "Model n_subjects does not match "
            "the number of training subjects."
        )

    trial_indices = np.arange(
        len(X_train)
    )

    stratification_labels = np.asarray(
        [
            f"{class_label}_{subject_label}"
            for class_label, subject_label
            in zip(
                y_encoded,
                subject_ids,
            )
        ]
    )

    (
        train_indices,
        validation_indices,
    ) = train_test_split(
        trial_indices,
        test_size=(
            training_config.validation_fraction
        ),
        random_state=(
            training_config.random_state
        ),
        stratify=stratification_labels,
    )

    standardizer = ChannelStandardizer()

    X_model_train = (
        standardizer.fit_transform(
            X_train[train_indices]
        )
    )

    X_validation = (
        standardizer.transform(
            X_train[validation_indices]
        )
    )

    train_loader = create_loader(
        eeg=X_model_train,
        class_labels=y_encoded[
            train_indices
        ],
        subject_labels=subject_ids[
            train_indices
        ],
        batch_size=(
            training_config.batch_size
        ),
        shuffle=True,
    )

    validation_loader = create_loader(
        eeg=X_validation,
        class_labels=y_encoded[
            validation_indices
        ],
        subject_labels=subject_ids[
            validation_indices
        ],
        batch_size=(
            training_config.batch_size
        ),
        shuffle=False,
    )

    device = choose_device()

    model = SubjectInvariantEEGAutoencoder(
        model_config
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=(
            training_config.weight_decay
        ),
    )

    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()
    subject_criterion = nn.CrossEntropyLoss()

    history = SubjectInvariantTrainingHistory(
        training_total_losses=[],
        validation_total_losses=[],
        training_reconstruction_losses=[],
        validation_reconstruction_losses=[],
        training_classification_losses=[],
        validation_classification_losses=[],
        training_subject_losses=[],
        validation_subject_losses=[],
        training_class_accuracies=[],
        validation_class_accuracies=[],
        training_subject_accuracies=[],
        validation_subject_accuracies=[],
        best_epoch=0,
        best_validation_classification_loss=(
            float("inf")
        ),
    )

    best_model_state = copy.deepcopy(
        model.state_dict()
    )

    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(
        f"Training trials: "
        f"{len(train_indices)}"
    )
    print(
        f"Validation trials: "
        f"{len(validation_indices)}"
    )
    print(
        f"Training subjects: "
        f"{model_config.n_subjects}"
    )

    for epoch_index in range(
        training_config.epochs
    ):
        reversal_coefficient = (
            reversal_schedule(
                epoch_index=epoch_index,
                total_epochs=(
                    training_config.epochs
                ),
                maximum=(
                    training_config
                    .max_reversal_coefficient
                ),
            )
        )

        training_metrics = run_epoch(
            model=model,
            loader=train_loader,
            reconstruction_criterion=(
                reconstruction_criterion
            ),
            classification_criterion=(
                classification_criterion
            ),
            subject_criterion=(
                subject_criterion
            ),
            config=training_config,
            reversal_coefficient=(
                reversal_coefficient
            ),
            device=device,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            model=model,
            loader=validation_loader,
            reconstruction_criterion=(
                reconstruction_criterion
            ),
            classification_criterion=(
                classification_criterion
            ),
            subject_criterion=(
                subject_criterion
            ),
            config=training_config,
            reversal_coefficient=(
                reversal_coefficient
            ),
            device=device,
            optimizer=None,
        )

        (
            training_total,
            training_reconstruction,
            training_classification,
            training_subject,
            training_class_accuracy,
            training_subject_accuracy,
        ) = training_metrics

        (
            validation_total,
            validation_reconstruction,
            validation_classification,
            validation_subject,
            validation_class_accuracy,
            validation_subject_accuracy,
        ) = validation_metrics

        history.training_total_losses.append(
            training_total
        )
        history.validation_total_losses.append(
            validation_total
        )

        history.training_reconstruction_losses.append(
            training_reconstruction
        )
        history.validation_reconstruction_losses.append(
            validation_reconstruction
        )

        history.training_classification_losses.append(
            training_classification
        )
        history.validation_classification_losses.append(
            validation_classification
        )

        history.training_subject_losses.append(
            training_subject
        )
        history.validation_subject_losses.append(
            validation_subject
        )

        history.training_class_accuracies.append(
            training_class_accuracy
        )
        history.validation_class_accuracies.append(
            validation_class_accuracy
        )

        history.training_subject_accuracies.append(
            training_subject_accuracy
        )
        history.validation_subject_accuracies.append(
            validation_subject_accuracy
        )

        improved = (
            validation_classification
            <
            history.best_validation_classification_loss
        )

        marker = ""

        if improved:
            history.best_validation_classification_loss = (
                validation_classification
            )

            history.best_epoch = (
                epoch_index + 1
            )

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0
            marker = " *"

        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch_index + 1:03d} "
            f"| total "
            f"{training_total:.4f}/"
            f"{validation_total:.4f} "
            f"| reconstruction "
            f"{training_reconstruction:.4f}/"
            f"{validation_reconstruction:.4f} "
            f"| class "
            f"{training_classification:.4f}/"
            f"{validation_classification:.4f} "
            f"| subject "
            f"{training_subject:.4f}/"
            f"{validation_subject:.4f} "
            f"| class acc "
            f"{100 * training_class_accuracy:.2f}%/"
            f"{100 * validation_class_accuracy:.2f}% "
            f"| subject acc "
            f"{100 * training_subject_accuracy:.2f}%/"
            f"{100 * validation_subject_accuracy:.2f}% "
            f"| GRL "
            f"{reversal_coefficient:.3f}"
            f"{marker}"
        )

        if (
            epochs_without_improvement
            >= training_config.patience
        ):
            print(
                "Early stopping: validation "
                "classification loss did not improve."
            )
            break

    model.load_state_dict(
        best_model_state
    )

    print(
        f"Best epoch: "
        f"{history.best_epoch}"
    )

    print(
        "Best validation classification loss: "
        f"{history.best_validation_classification_loss:.6f}"
    )

    return (
        model,
        standardizer,
        history,
        class_names,
    )


def extract_latent_features(
    model: SubjectInvariantEEGAutoencoder,
    standardizer: ChannelStandardizer,
    X: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Extract latent features from EEG."""
    X_standardized = (
        standardizer.transform(X)
    )

    dataset = TensorDataset(
        torch.from_numpy(
            X_standardized.astype(np.float32)
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = next(
        model.parameters()
    ).device

    model.eval()

    feature_batches: list[np.ndarray] = []

    with torch.no_grad():
        for (eeg_batch,) in loader:
            eeg_batch = eeg_batch.to(device)

            latent = model.encode(
                eeg_batch
            )

            feature_batches.append(
                latent.cpu()
                .numpy()
                .astype(np.float32)
            )

    return np.concatenate(
        feature_batches,
        axis=0,
    )


def predict_classes(
    model: SubjectInvariantEEGAutoencoder,
    standardizer: ChannelStandardizer,
    X: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Predict classes using the neural classification head."""
    X_standardized = (
        standardizer.transform(X)
    )

    dataset = TensorDataset(
        torch.from_numpy(
            X_standardized.astype(np.float32)
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = next(
        model.parameters()
    ).device

    model.eval()

    prediction_batches: list[np.ndarray] = []

    with torch.no_grad():
        for (eeg_batch,) in loader:
            eeg_batch = eeg_batch.to(device)

            logits = model.classify(
                eeg_batch
            )

            predictions = logits.argmax(
                dim=1
            )

            prediction_batches.append(
                predictions.cpu().numpy()
            )

    return np.concatenate(
        prediction_batches,
        axis=0,
    )


def save_checkpoint(
    path: Path,
    model: SubjectInvariantEEGAutoencoder,
    standardizer: ChannelStandardizer,
    training_config: SubjectInvariantTrainingConfig,
    history: SubjectInvariantTrainingHistory,
    class_names: list[str],
    training_subjects: list[str],
) -> None:
    """Save the trained model and preprocessing information."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "model_config": asdict(
                model.config
            ),
            "training_config": asdict(
                training_config
            ),
            "history": asdict(
                history
            ),
            "class_names": class_names,
            "training_subjects": (
                training_subjects
            ),
            "standardizer_mean": (
                standardizer.mean_
            ),
            "standardizer_scale": (
                standardizer.scale_
            ),
        },
        path,
    )
