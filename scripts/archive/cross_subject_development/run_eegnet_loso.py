"""
Strict cross-subject EEGNet evaluation using Leave-One-Subject-Out (LOSO).

Research question:
    Can EEGNet improve calibration-free cross-subject motor-imagery
    classification compared with the current Riemannian baseline?

Protocol for every fold:
    Test subject:
        Completely unseen during training, validation, normalisation,
        early stopping and hyperparameter selection.

    Validation subject:
        One subject selected only from the remaining training subjects.

    Stage 1:
        Train on 7 subjects and validate on 1 subject to determine
        the best number of epochs.

    Stage 2:
        Reinitialise EEGNet and train on all 8 available training
        subjects for the selected number of epochs.

    Stage 3:
        Evaluate once on the unseen held-out subject.

Outputs:
    results/cross_subject/eegnet/eegnet_loso/eegnet_loso_subject_results.csv
    results/cross_subject/eegnet/eegnet_loso/eegnet_loso_predictions.csv
    results/cross_subject/eegnet/eegnet_loso/eegnet_loso_confusion_matrix.csv
    results/cross_subject/eegnet/eegnet_loso/eegnet_loso_training_history.csv
    results/cross_subject/eegnet/eegnet_loso/eegnet_loso_summary.txt
    results/cross_subject/eegnet/eegnet_loso/eegnet_loso_accuracy.png
    results/cross_subject/eegnet/eegnet_loso/checkpoints/*.pt

Run:
    python -m scripts.cross_subject.run_eegnet_loso
"""

from __future__ import annotations

import copy
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.data.preprocessing import preprocess_raw


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

mne.set_log_level("ERROR")

RANDOM_SEED = 42

SUBJECTS = [
    "A01T",
    "A02T",
    "A03T",
    "A04T",
    "A05T",
    "A06T",
    "A07T",
    "A08T",
    "A09T",
]

DATA_DIRECTORY = Path("data/raw")
OUTPUT_DIRECTORY = Path("results/cross_subject/eegnet/eegnet_loso")
CHECKPOINT_DIRECTORY = OUTPUT_DIRECTORY / "checkpoints"

FMIN = 8.0
FMAX = 30.0
TMIN = 0.5
TMAX = 2.5

N_CHANNELS = 22
N_CLASSES = 4

BATCH_SIZE = 64
MAX_VALIDATION_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DROPOUT_RATE = 0.5

# Original EEGNet-style parameters.
F1 = 8
DEPTH_MULTIPLIER = 2
F2 = F1 * DEPTH_MULTIPLIER

# Current strict-LOSO Riemannian benchmark.
RIEMANNIAN_BASELINE_ACCURACY = 0.5471
RIEMANNIAN_BASELINE_KAPPA = 0.396

LABEL_TO_INDEX = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "tongue": 3,
}

INDEX_TO_LABEL = {
    value: key
    for key, value in LABEL_TO_INDEX.items()
}

CLASS_NAMES = [
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
]


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def select_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def encode_labels(labels: np.ndarray) -> np.ndarray:
    """Convert string class labels into integer class indices."""

    encoded = []

    for label in labels:
        label_string = str(label)

        if label_string not in LABEL_TO_INDEX:
            raise ValueError(
                f"Unknown label '{label_string}'. "
                f"Expected one of {list(LABEL_TO_INDEX)}."
            )

        encoded.append(LABEL_TO_INDEX[label_string])

    return np.asarray(encoded, dtype=np.int64)


def load_subject(subject: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess one subject.

    Returns:
        X: shape (trials, channels, samples)
        y: integer labels shape (trials,)
    """

    path = DATA_DIRECTORY / f"{subject}.gdf"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {path}"
        )

    print(f"Loading {subject}...")

    raw = load_raw_gdf(str(path))

    X, labels = preprocess_raw(
        raw,
        fmin=FMIN,
        fmax=FMAX,
        tmin=TMIN,
        tmax=TMAX,
    )

    X = np.asarray(X, dtype=np.float32)
    y = encode_labels(labels)

    if X.ndim != 3:
        raise ValueError(
            f"{subject}: expected three-dimensional EEG data, "
            f"but received shape {X.shape}."
        )

    if X.shape[1] != N_CHANNELS:
        raise ValueError(
            f"{subject}: expected {N_CHANNELS} channels, "
            f"but received {X.shape[1]}."
        )

    if len(X) != len(y):
        raise ValueError(
            f"{subject}: trial and label counts do not match."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{subject}: EEG data contain NaN or infinite values."
        )

    class_counts = np.bincount(
        y,
        minlength=N_CLASSES,
    )

    print(
        f"  Shape: {X.shape} | "
        f"Class counts: {class_counts.tolist()}"
    )

    return X, y


def load_all_subjects() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all subjects once and keep them in memory."""

    subject_data = {}

    print("\n" + "=" * 72)
    print("Loading BCI Competition IV-2a subjects")
    print("=" * 72)

    for subject in SUBJECTS:
        subject_data[subject] = load_subject(subject)

    return subject_data


def combine_subjects(
    subject_names: list[str],
    subject_data: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine multiple subjects.

    Returns:
        X
        y
        subject identifier for each trial
    """

    X_parts = []
    y_parts = []
    subject_parts = []

    for subject in subject_names:
        X_subject, y_subject = subject_data[subject]

        X_parts.append(X_subject)
        y_parts.append(y_subject)

        subject_parts.append(
            np.full(
                len(y_subject),
                subject,
                dtype=object,
            )
        )

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(subject_parts, axis=0),
    )


# ---------------------------------------------------------------------
# Training-only normalisation
# ---------------------------------------------------------------------

def calculate_channel_statistics(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate channel-wise mean and standard deviation.

    Statistics are calculated only from the training data.

    Returned arrays have shape:
        (1, channels, 1)
    """

    channel_mean = X_train.mean(
        axis=(0, 2),
        keepdims=True,
    )

    channel_std = X_train.std(
        axis=(0, 2),
        keepdims=True,
    )

    channel_std = np.maximum(
        channel_std,
        1e-8,
    )

    return (
        channel_mean.astype(np.float32),
        channel_std.astype(np.float32),
    )


def standardise_eeg(
    X: np.ndarray,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
) -> np.ndarray:
    """Apply training-derived channel-wise standardisation."""

    standardised = (
        X - channel_mean
    ) / channel_std

    return standardised.astype(np.float32)


# ---------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------

class EEGDataset(Dataset):
    """PyTorch dataset for EEG trials."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        if len(X) != len(y):
            raise ValueError(
                "EEG trials and labels must have equal length."
            )

        # EEGNet input shape:
        # trials × 1 × channels × samples
        self.X = torch.from_numpy(
            X[:, np.newaxis, :, :]
        ).float()

        self.y = torch.from_numpy(
            y
        ).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def create_loader(
    X: np.ndarray,
    y: np.ndarray,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a reproducible DataLoader."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = EEGDataset(X, y)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


# ---------------------------------------------------------------------
# EEGNet model
# ---------------------------------------------------------------------

class EEGNet(nn.Module):
    """
    Compact EEGNet architecture for four-class motor imagery.

    Input:
        batch × 1 × channels × samples
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        dropout_rate: float = 0.5,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
    ) -> None:
        super().__init__()

        self.temporal_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, 64),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(f1),
        )

        self.spatial_block = nn.Sequential(
            nn.Conv2d(
                in_channels=f1,
                out_channels=f1 * depth_multiplier,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(
                f1 * depth_multiplier
            ),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 4)
            ),
            nn.Dropout(dropout_rate),
        )

        self.separable_block = nn.Sequential(
            nn.Conv2d(
                in_channels=f1 * depth_multiplier,
                out_channels=f1 * depth_multiplier,
                kernel_size=(1, 16),
                padding="same",
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=f1 * depth_multiplier,
                out_channels=f2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 8)
            ),
            nn.Dropout(dropout_rate),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(
                1,
                1,
                n_channels,
                n_samples,
            )

            dummy_features = self._extract_features(
                dummy_input
            )

            flattened_size = int(
                np.prod(dummy_features.shape[1:])
            )

        self.classifier = nn.Linear(
            flattened_size,
            n_classes,
        )

    def _extract_features(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        X = self.temporal_block(X)
        X = self.spatial_block(X)
        X = self.separable_block(X)
        return X

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        X = self._extract_features(X)
        X = torch.flatten(X, start_dim=1)
        return self.classifier(X)


def initialise_model(
    n_samples: int,
    device: torch.device,
) -> EEGNet:
    """Create a new EEGNet instance."""

    model = EEGNet(
        n_channels=N_CHANNELS,
        n_samples=n_samples,
        n_classes=N_CLASSES,
        dropout_rate=DROPOUT_RATE,
        f1=F1,
        depth_multiplier=DEPTH_MULTIPLIER,
        f2=F2,
    )

    return model.to(device)


# ---------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------

def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch."""

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimiser.zero_grad(set_to_none=True)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimiser.step()

        total_loss += (
            loss.item() * len(y_batch)
        )

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == y_batch
        ).sum().item()

        total += len(y_batch)

    return (
        total_loss / total,
        correct / total,
    )


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Evaluate a model.

    Returns:
        mean loss
        accuracy
        true labels
        predicted labels
        probabilities
    """

    model.eval()

    total_loss = 0.0
    total = 0

    all_true = []
    all_predictions = []
    all_probabilities = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = probabilities.argmax(
            dim=1
        )

        total_loss += (
            loss.item() * len(y_batch)
        )

        total += len(y_batch)

        all_true.append(
            y_batch.cpu().numpy()
        )

        all_predictions.append(
            predictions.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    true_labels = np.concatenate(all_true)
    predicted_labels = np.concatenate(
        all_predictions
    )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    return (
        total_loss / total,
        accuracy,
        true_labels,
        predicted_labels,
        probabilities,
    )


def select_best_epoch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    n_samples: int,
    device: torch.device,
    fold_number: int,
    test_subject: str,
    validation_subject: str,
) -> tuple[int, list[dict]]:
    """
    Train on seven subjects and select the best epoch using one
    training-domain validation subject.
    """

    model = initialise_model(
        n_samples=n_samples,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )

    train_loader = create_loader(
        X_train,
        y_train,
        shuffle=True,
        seed=RANDOM_SEED + fold_number,
    )

    validation_loader = create_loader(
        X_validation,
        y_validation,
        shuffle=False,
        seed=RANDOM_SEED,
    )

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = 1
    best_state = None

    epochs_without_improvement = 0
    history = []

    for epoch in range(
        1,
        MAX_VALIDATION_EPOCHS + 1,
    ):
        train_loss, train_accuracy = (
            run_training_epoch(
                model,
                train_loader,
                criterion,
                optimiser,
                device,
            )
        )

        (
            validation_loss,
            validation_accuracy,
            _,
            validation_predictions,
            _,
        ) = evaluate_loader(
            model,
            validation_loader,
            criterion,
            device,
        )

        validation_kappa = cohen_kappa_score(
            y_validation,
            validation_predictions,
        )

        scheduler.step(validation_loss)

        current_lr = optimiser.param_groups[0]["lr"]

        history.append(
            {
                "fold": fold_number,
                "test_subject": test_subject,
                "validation_subject": validation_subject,
                "stage": "epoch_selection",
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_kappa": validation_kappa,
                "learning_rate": current_lr,
            }
        )

        improved = (
            validation_loss
            < best_validation_loss - 1e-4
        )

        if improved:
            best_validation_loss = validation_loss
            best_validation_accuracy = (
                validation_accuracy
            )
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"    Epoch {epoch:03d} | "
            f"Train loss {train_loss:.4f} | "
            f"Train acc {train_accuracy * 100:5.1f}% | "
            f"Val loss {validation_loss:.4f} | "
            f"Val acc {validation_accuracy * 100:5.1f}% | "
            f"Val κ {validation_kappa:.3f}"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "    Early stopping: "
                f"no validation improvement for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No valid EEGNet checkpoint was produced."
        )

    checkpoint_path = (
        CHECKPOINT_DIRECTORY
        / f"{test_subject}_validation_checkpoint.pt"
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_loss": (
                best_validation_loss
            ),
            "best_validation_accuracy": (
                best_validation_accuracy
            ),
            "test_subject": test_subject,
            "validation_subject": (
                validation_subject
            ),
        },
        checkpoint_path,
    )

    print(
        f"\n    Best epoch: {best_epoch} | "
        f"Best validation loss: "
        f"{best_validation_loss:.4f} | "
        f"Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    return best_epoch, history


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_samples: int,
    n_epochs: int,
    device: torch.device,
    fold_number: int,
    test_subject: str,
) -> tuple[nn.Module, list[dict]]:
    """
    Reinitialise EEGNet and train on all eight training subjects
    for the selected number of epochs.
    """

    set_random_seed(
        RANDOM_SEED + fold_number
    )

    model = initialise_model(
        n_samples=n_samples,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = create_loader(
        X_train,
        y_train,
        shuffle=True,
        seed=RANDOM_SEED + fold_number,
    )

    history = []

    print(
        f"\n    Retraining on all eight training "
        f"subjects for {n_epochs} epochs..."
    )

    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = (
            run_training_epoch(
                model,
                train_loader,
                criterion,
                optimiser,
                device,
            )
        )

        history.append(
            {
                "fold": fold_number,
                "test_subject": test_subject,
                "validation_subject": "",
                "stage": "final_training",
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": np.nan,
                "validation_accuracy": np.nan,
                "validation_kappa": np.nan,
                "learning_rate": (
                    optimiser.param_groups[0]["lr"]
                ),
            }
        )

        print(
            f"    Final epoch {epoch:03d}/{n_epochs:03d} | "
            f"Loss {train_loss:.4f} | "
            f"Accuracy {train_accuracy * 100:5.1f}%"
        )

    final_checkpoint_path = (
        CHECKPOINT_DIRECTORY
        / f"{test_subject}_final_model.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_epochs": n_epochs,
            "test_subject": test_subject,
        },
        final_checkpoint_path,
    )

    return model, history


# ---------------------------------------------------------------------
# Fold evaluation
# ---------------------------------------------------------------------

def choose_validation_subject(
    training_subjects: list[str],
    test_subject_index: int,
) -> str:
    """
    Choose a validation subject deterministically.

    Selection rotates across folds and never uses the test subject.
    """

    validation_index = (
        test_subject_index
        % len(training_subjects)
    )

    return training_subjects[
        validation_index
    ]


def run_loso_fold(
    fold_number: int,
    test_subject: str,
    subject_data: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> tuple[
    dict,
    list[dict],
    list[dict],
    np.ndarray,
    np.ndarray,
]:
    """Run one complete strict LOSO fold."""

    fold_start_time = time.time()

    training_subjects = [
        subject
        for subject in SUBJECTS
        if subject != test_subject
    ]

    test_subject_index = SUBJECTS.index(
        test_subject
    )

    validation_subject = (
        choose_validation_subject(
            training_subjects,
            test_subject_index,
        )
    )

    epoch_selection_subjects = [
        subject
        for subject in training_subjects
        if subject != validation_subject
    ]

    print("\n" + "=" * 72)
    print(
        f"Fold {fold_number}/9 — "
        f"Unseen test subject: {test_subject}"
    )
    print("=" * 72)

    print(
        "Epoch-selection training subjects: "
        + ", ".join(epoch_selection_subjects)
    )

    print(
        f"Validation subject: {validation_subject}"
    )

    print(
        "Final training subjects: "
        + ", ".join(training_subjects)
    )

    print(
        f"Unseen test subject: {test_subject}"
    )

    X_epoch_train, y_epoch_train, _ = (
        combine_subjects(
            epoch_selection_subjects,
            subject_data,
        )
    )

    X_validation, y_validation, _ = (
        combine_subjects(
            [validation_subject],
            subject_data,
        )
    )

    X_final_train, y_final_train, _ = (
        combine_subjects(
            training_subjects,
            subject_data,
        )
    )

    X_test, y_test, test_subject_labels = (
        combine_subjects(
            [test_subject],
            subject_data,
        )
    )

    # Stage 1 normalisation:
    # calculated using only the seven epoch-selection
    # training subjects.
    selection_mean, selection_std = (
        calculate_channel_statistics(
            X_epoch_train
        )
    )

    X_epoch_train_normalised = standardise_eeg(
        X_epoch_train,
        selection_mean,
        selection_std,
    )

    X_validation_normalised = standardise_eeg(
        X_validation,
        selection_mean,
        selection_std,
    )

    n_samples = X_epoch_train.shape[2]

    print(
        f"\n    Epoch-selection data: "
        f"{X_epoch_train.shape}"
    )

    print(
        f"    Validation data:      "
        f"{X_validation.shape}"
    )

    best_epoch, selection_history = (
        select_best_epoch(
            X_train=X_epoch_train_normalised,
            y_train=y_epoch_train,
            X_validation=(
                X_validation_normalised
            ),
            y_validation=y_validation,
            n_samples=n_samples,
            device=device,
            fold_number=fold_number,
            test_subject=test_subject,
            validation_subject=(
                validation_subject
            ),
        )
    )

    # Stage 2 normalisation:
    # recalculate using all eight final training subjects.
    final_mean, final_std = (
        calculate_channel_statistics(
            X_final_train
        )
    )

    X_final_train_normalised = (
        standardise_eeg(
            X_final_train,
            final_mean,
            final_std,
        )
    )

    # Test data are transformed using only the
    # eight-subject training statistics.
    X_test_normalised = standardise_eeg(
        X_test,
        final_mean,
        final_std,
    )

    final_model, final_history = (
        train_final_model(
            X_train=(
                X_final_train_normalised
            ),
            y_train=y_final_train,
            n_samples=n_samples,
            n_epochs=best_epoch,
            device=device,
            fold_number=fold_number,
            test_subject=test_subject,
        )
    )

    criterion = nn.CrossEntropyLoss()

    test_loader = create_loader(
        X_test_normalised,
        y_test,
        shuffle=False,
        seed=RANDOM_SEED,
    )

    (
        test_loss,
        test_accuracy,
        true_labels,
        predicted_labels,
        probabilities,
    ) = evaluate_loader(
        final_model,
        test_loader,
        criterion,
        device,
    )

    test_kappa = cohen_kappa_score(
        true_labels,
        predicted_labels,
    )

    test_confusion_matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(N_CLASSES),
    )

    class_recalls = recall_score(
        true_labels,
        predicted_labels,
        labels=np.arange(N_CLASSES),
        average=None,
        zero_division=0,
    )

    fold_duration = (
        time.time() - fold_start_time
    )

    print("\n    Test result")
    print("    " + "-" * 60)
    print(
        f"    Accuracy: {test_accuracy * 100:.2f}%"
    )
    print(
        f"    Cohen's κ: {test_kappa:.3f}"
    )
    print(
        f"    Test loss: {test_loss:.4f}"
    )
    print(
        f"    Selected epochs: {best_epoch}"
    )
    print(
        f"    Duration: {fold_duration / 60:.1f} minutes"
    )
    print(
        "    Confusion matrix:"
    )
    print(test_confusion_matrix)

    subject_result = {
        "fold": fold_number,
        "test_subject": test_subject,
        "validation_subject": validation_subject,
        "training_subject_count": len(
            training_subjects
        ),
        "test_trials": len(y_test),
        "selected_epochs": best_epoch,
        "accuracy": test_accuracy,
        "accuracy_percent": (
            test_accuracy * 100
        ),
        "kappa": test_kappa,
        "test_loss": test_loss,
        "left_hand_recall": class_recalls[0],
        "right_hand_recall": class_recalls[1],
        "feet_recall": class_recalls[2],
        "tongue_recall": class_recalls[3],
        "duration_seconds": fold_duration,
    }

    prediction_rows = []

    for trial_index, (
        true_index,
        predicted_index,
        probability_vector,
        subject_label,
    ) in enumerate(
        zip(
            true_labels,
            predicted_labels,
            probabilities,
            test_subject_labels,
        ),
        start=1,
    ):
        prediction_rows.append(
            {
                "fold": fold_number,
                "subject": subject_label,
                "trial": trial_index,
                "true_index": int(
                    true_index
                ),
                "true_label": INDEX_TO_LABEL[
                    int(true_index)
                ],
                "predicted_index": int(
                    predicted_index
                ),
                "predicted_label": (
                    INDEX_TO_LABEL[
                        int(predicted_index)
                    ]
                ),
                "correct": int(
                    true_index
                    == predicted_index
                ),
                "confidence": float(
                    probability_vector.max()
                ),
                "probability_left_hand": float(
                    probability_vector[0]
                ),
                "probability_right_hand": float(
                    probability_vector[1]
                ),
                "probability_feet": float(
                    probability_vector[2]
                ),
                "probability_tongue": float(
                    probability_vector[3]
                ),
            }
        )

    history_rows = (
        selection_history
        + final_history
    )

    return (
        subject_result,
        prediction_rows,
        history_rows,
        true_labels,
        predicted_labels,
    )


# ---------------------------------------------------------------------
# Result export
# ---------------------------------------------------------------------

def save_confusion_matrix(
    matrix: np.ndarray,
    path: Path,
) -> None:
    """Save a labelled confusion matrix."""

    dataframe = pd.DataFrame(
        matrix,
        index=[
            f"true_{name}"
            for name in CLASS_NAMES
        ],
        columns=[
            f"predicted_{name}"
            for name in CLASS_NAMES
        ],
    )

    dataframe.to_csv(path)


def create_accuracy_plot(
    results_dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a subject-wise EEGNet accuracy plot."""

    plt.figure(figsize=(11, 6))

    plt.bar(
        results_dataframe["test_subject"],
        results_dataframe[
            "accuracy_percent"
        ],
    )

    plt.axhline(
        y=RIEMANNIAN_BASELINE_ACCURACY * 100,
        linestyle="--",
        label=(
            "Riemannian LOSO baseline "
            f"({RIEMANNIAN_BASELINE_ACCURACY * 100:.2f}%)"
        ),
    )

    plt.axhline(
        y=25.0,
        linestyle=":",
        label="Four-class chance level (25%)",
    )

    plt.xlabel("Unseen test subject")
    plt.ylabel("Accuracy (%)")
    plt.title(
        "EEGNet Strict LOSO Accuracy by Test Subject"
    )
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_summary(
    subject_results: pd.DataFrame,
    pooled_true: np.ndarray,
    pooled_predictions: np.ndarray,
    total_duration: float,
    output_path: Path,
) -> None:
    """Save the complete experiment summary."""

    mean_accuracy = subject_results[
        "accuracy"
    ].mean()

    standard_deviation_accuracy = (
        subject_results["accuracy"].std(
            ddof=1
        )
    )

    mean_kappa = subject_results[
        "kappa"
    ].mean()

    pooled_accuracy = accuracy_score(
        pooled_true,
        pooled_predictions,
    )

    pooled_kappa = cohen_kappa_score(
        pooled_true,
        pooled_predictions,
    )

    pooled_confusion_matrix = (
        confusion_matrix(
            pooled_true,
            pooled_predictions,
            labels=np.arange(N_CLASSES),
        )
    )

    report = classification_report(
        pooled_true,
        pooled_predictions,
        labels=np.arange(N_CLASSES),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    accuracy_difference = (
        mean_accuracy
        - RIEMANNIAN_BASELINE_ACCURACY
    )

    kappa_difference = (
        mean_kappa
        - RIEMANNIAN_BASELINE_KAPPA
    )

    lines = [
        "=" * 72,
        "EEGNet Strict Leave-One-Subject-Out Evaluation",
        "=" * 72,
        "",
        "Protocol:",
        "  Train on eight subjects.",
        "  Test on one completely unseen subject.",
        "  No test-subject labels or trials are used for training,",
        "  validation, normalisation or early stopping.",
        "",
        "Configuration:",
        f"  Frequency band: {FMIN}-{FMAX} Hz",
        f"  Epoch window: {TMIN}-{TMAX} seconds",
        f"  Channels: {N_CHANNELS}",
        f"  Classes: {N_CLASSES}",
        f"  Batch size: {BATCH_SIZE}",
        f"  Learning rate: {LEARNING_RATE}",
        f"  Weight decay: {WEIGHT_DECAY}",
        f"  Dropout: {DROPOUT_RATE}",
        "",
        "Subject-wise results:",
        subject_results.to_string(index=False),
        "",
        "Overall EEGNet results:",
        (
            f"  Mean subject accuracy: "
            f"{mean_accuracy * 100:.2f}%"
        ),
        (
            f"  Subject accuracy SD: "
            f"{standard_deviation_accuracy * 100:.2f}%"
        ),
        (
            f"  Mean subject kappa: "
            f"{mean_kappa:.3f}"
        ),
        (
            f"  Pooled accuracy: "
            f"{pooled_accuracy * 100:.2f}%"
        ),
        (
            f"  Pooled kappa: "
            f"{pooled_kappa:.3f}"
        ),
        "",
        "Existing Riemannian LOSO benchmark:",
        (
            f"  Accuracy: "
            f"{RIEMANNIAN_BASELINE_ACCURACY * 100:.2f}%"
        ),
        (
            f"  Kappa: "
            f"{RIEMANNIAN_BASELINE_KAPPA:.3f}"
        ),
        "",
        "EEGNet difference from Riemannian benchmark:",
        (
            f"  Accuracy difference: "
            f"{accuracy_difference * 100:+.2f} percentage points"
        ),
        (
            f"  Kappa difference: "
            f"{kappa_difference:+.3f}"
        ),
        "",
        "Pooled confusion matrix:",
        str(pooled_confusion_matrix),
        "",
        "Pooled classification report:",
        report,
        "",
        (
            f"Total experiment duration: "
            f"{total_duration / 60:.1f} minutes"
        ),
        "",
    ]

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:
    """Run all nine strict LOSO EEGNet folds."""

    experiment_start_time = time.time()

    set_random_seed(RANDOM_SEED)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = select_device()

    print("\n" + "=" * 72)
    print("EEGNet Strict Cross-Subject LOSO Experiment")
    print("=" * 72)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print(
        f"Riemannian benchmark: "
        f"{RIEMANNIAN_BASELINE_ACCURACY * 100:.2f}% "
        f"(κ={RIEMANNIAN_BASELINE_KAPPA:.3f})"
    )

    configuration = {
        "subjects": SUBJECTS,
        "random_seed": RANDOM_SEED,
        "frequency_band": [FMIN, FMAX],
        "epoch_window": [TMIN, TMAX],
        "n_channels": N_CHANNELS,
        "n_classes": N_CLASSES,
        "batch_size": BATCH_SIZE,
        "max_validation_epochs": (
            MAX_VALIDATION_EPOCHS
        ),
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout_rate": DROPOUT_RATE,
        "f1": F1,
        "depth_multiplier": DEPTH_MULTIPLIER,
        "f2": F2,
        "device": str(device),
        "riemannian_baseline_accuracy": (
            RIEMANNIAN_BASELINE_ACCURACY
        ),
        "riemannian_baseline_kappa": (
            RIEMANNIAN_BASELINE_KAPPA
        ),
    }

    with (
        OUTPUT_DIRECTORY
        / "eegnet_loso_configuration.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
        )

    subject_data = load_all_subjects()

    subject_results = []
    all_prediction_rows = []
    all_history_rows = []

    pooled_true_parts = []
    pooled_prediction_parts = []

    for fold_number, test_subject in enumerate(
        SUBJECTS,
        start=1,
    ):
        set_random_seed(
            RANDOM_SEED + fold_number
        )

        (
            subject_result,
            prediction_rows,
            history_rows,
            true_labels,
            predicted_labels,
        ) = run_loso_fold(
            fold_number=fold_number,
            test_subject=test_subject,
            subject_data=subject_data,
            device=device,
        )

        subject_results.append(
            subject_result
        )

        all_prediction_rows.extend(
            prediction_rows
        )

        all_history_rows.extend(
            history_rows
        )

        pooled_true_parts.append(
            true_labels
        )

        pooled_prediction_parts.append(
            predicted_labels
        )

        # Save after every fold so progress is preserved.
        pd.DataFrame(
            subject_results
        ).to_csv(
            OUTPUT_DIRECTORY
            / "eegnet_loso_subject_results.csv",
            index=False,
        )

        pd.DataFrame(
            all_prediction_rows
        ).to_csv(
            OUTPUT_DIRECTORY
            / "eegnet_loso_predictions.csv",
            index=False,
        )

        pd.DataFrame(
            all_history_rows
        ).to_csv(
            OUTPUT_DIRECTORY
            / "eegnet_loso_training_history.csv",
            index=False,
        )

    pooled_true = np.concatenate(
        pooled_true_parts
    )

    pooled_predictions = np.concatenate(
        pooled_prediction_parts
    )

    results_dataframe = pd.DataFrame(
        subject_results
    )

    prediction_dataframe = pd.DataFrame(
        all_prediction_rows
    )

    history_dataframe = pd.DataFrame(
        all_history_rows
    )

    overall_confusion_matrix = confusion_matrix(
        pooled_true,
        pooled_predictions,
        labels=np.arange(N_CLASSES),
    )

    results_dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "eegnet_loso_subject_results.csv",
        index=False,
    )

    prediction_dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "eegnet_loso_predictions.csv",
        index=False,
    )

    history_dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "eegnet_loso_training_history.csv",
        index=False,
    )

    save_confusion_matrix(
        overall_confusion_matrix,
        OUTPUT_DIRECTORY
        / "eegnet_loso_confusion_matrix.csv",
    )

    create_accuracy_plot(
        results_dataframe,
        OUTPUT_DIRECTORY
        / "eegnet_loso_accuracy.png",
    )

    total_duration = (
        time.time() - experiment_start_time
    )

    save_summary(
        subject_results=results_dataframe,
        pooled_true=pooled_true,
        pooled_predictions=pooled_predictions,
        total_duration=total_duration,
        output_path=(
            OUTPUT_DIRECTORY
            / "eegnet_loso_summary.txt"
        ),
    )

    mean_accuracy = results_dataframe[
        "accuracy"
    ].mean()

    mean_kappa = results_dataframe[
        "kappa"
    ].mean()

    pooled_accuracy = accuracy_score(
        pooled_true,
        pooled_predictions,
    )

    pooled_kappa = cohen_kappa_score(
        pooled_true,
        pooled_predictions,
    )

    print("\n" + "=" * 72)
    print("Final EEGNet Strict LOSO Results")
    print("=" * 72)

    print(
        results_dataframe[
            [
                "test_subject",
                "validation_subject",
                "selected_epochs",
                "accuracy_percent",
                "kappa",
            ]
        ].to_string(
            index=False,
            formatters={
                "accuracy_percent": (
                    lambda value: f"{value:.2f}"
                ),
                "kappa": (
                    lambda value: f"{value:.3f}"
                ),
            },
        )
    )

    print("\nOverall:")
    print(
        f"  Mean subject accuracy: "
        f"{mean_accuracy * 100:.2f}%"
    )
    print(
        f"  Mean subject kappa: "
        f"{mean_kappa:.3f}"
    )
    print(
        f"  Pooled accuracy: "
        f"{pooled_accuracy * 100:.2f}%"
    )
    print(
        f"  Pooled kappa: "
        f"{pooled_kappa:.3f}"
    )

    print("\nComparison:")
    print(
        f"  EEGNet: "
        f"{mean_accuracy * 100:.2f}%"
    )
    print(
        f"  Riemannian baseline: "
        f"{RIEMANNIAN_BASELINE_ACCURACY * 100:.2f}%"
    )
    print(
        f"  Difference: "
        f"{(mean_accuracy - RIEMANNIAN_BASELINE_ACCURACY) * 100:+.2f} "
        "percentage points"
    )

    print("\nOverall confusion matrix:")
    print(overall_confusion_matrix)

    print("\nSaved outputs:")
    print(
        f"  {OUTPUT_DIRECTORY}/"
        "eegnet_loso_subject_results.csv"
    )
    print(
        f"  {OUTPUT_DIRECTORY}/"
        "eegnet_loso_predictions.csv"
    )
    print(
        f"  {OUTPUT_DIRECTORY}/"
        "eegnet_loso_training_history.csv"
    )
    print(
        f"  {OUTPUT_DIRECTORY}/"
        "eegnet_loso_confusion_matrix.csv"
    )
    print(
        f"  {OUTPUT_DIRECTORY}/"
        "eegnet_loso_accuracy.png"
    )
    print(
        f"  {OUTPUT_DIRECTORY}/"
        "eegnet_loso_summary.txt"
    )

    print(
        f"\nTotal duration: "
        f"{total_duration / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
