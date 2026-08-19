"""
EEGNet cross-subject T-to-E LOSO experiment.

For each held-out subject:

1. Use T sessions from the other eight subjects.
2. Keep the held-out subject's T session out of training.
3. Use the held-out subject's E session only for testing.
3. Take a stratified 15% validation split from every training subject.
4. Select the best epoch using the combined validation data.
5. Reinitialise EEGNet.
6. Train on all trials from the eight training subjects.
7. Test once on the unseen subject.

The complete nine-fold experiment is repeated with three random seeds.

Run:
    python -m scripts.cross_subject.eegnet
"""

from __future__ import annotations

import copy
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
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bci_wheelchair.data.processed_loading import load_processed_subject


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

mne.set_log_level("ERROR")

SUBJECT_IDS = [
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "A06",
    "A07",
    "A08",
    "A09",
]

SEEDS = [42, 123, 2026]

OUTPUT_DIRECTORY = Path("results/cross_subject/eegnet/eegnet_loso_improved")
CHECKPOINT_DIRECTORY = OUTPUT_DIRECTORY / "checkpoints"

FMIN = 4.0
FMAX = 40.0
TMIN = 0.5
TMAX = 2.5

N_CHANNELS = 22
N_CLASSES = 4

VALIDATION_SIZE = 0.15

BATCH_SIZE = 64
MINIMUM_EPOCHS = 20
MAXIMUM_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5

F1 = 8
DEPTH_MULTIPLIER = 2
F2 = F1 * DEPTH_MULTIPLIER

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def encode_labels(labels: np.ndarray) -> np.ndarray:
    encoded_labels = []

    for label in labels:
        label_string = str(label)

        if label_string not in LABEL_TO_INDEX:
            raise ValueError(
                f"Unknown label: {label_string}. "
                f"Expected one of {list(LABEL_TO_INDEX)}."
            )

        encoded_labels.append(
            LABEL_TO_INDEX[label_string]
        )

    return np.asarray(
        encoded_labels,
        dtype=np.int64,
    )


def load_subject(
    subject: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one subject from the cached 4-40 Hz processed dataset.

    Returns
    -------
    X:
        EEG trials with shape
        (trials, channels, samples).

    y:
        Integer class labels with shape
        (trials,).
    """
    print(f"Loading processed data for {subject}...")

    X, labels = load_processed_subject(
        subject,
        config="4-40",
    )

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    labels = np.asarray(labels)
    y = encode_labels(labels)

    if X.ndim != 3:
        raise ValueError(
            f"{subject}: expected EEG shape "
            f"(trials, channels, samples), "
            f"received {X.shape}."
        )

    if X.shape[1] != N_CHANNELS:
        raise ValueError(
            f"{subject}: expected {N_CHANNELS} channels, "
            f"received {X.shape[1]}."
        )

    if len(X) != len(y):
        raise ValueError(
            f"{subject}: trial and label counts differ."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{subject}: EEG data contain "
            "NaN or infinite values."
        )

    class_counts = np.bincount(
        y,
        minlength=N_CLASSES,
    )

    print(
        f"  Shape: {X.shape} | "
        f"Classes: {class_counts.tolist()}"
    )

    return X, y



def load_training_sessions(
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load A01T-A09T training sessions.
    """
    training_data = {}

    print("\n" + "=" * 72)
    print("Loading training sessions A01T-A09T")
    print("=" * 72)

    for subject_id in SUBJECT_IDS:
        session = f"{subject_id}T"
        training_data[subject_id] = load_subject(
            session
        )

    return training_data


def load_evaluation_sessions(
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load A01E-A09E evaluation sessions.
    """
    evaluation_data = {}

    print("\n" + "=" * 72)
    print("Loading evaluation sessions A01E-A09E")
    print("=" * 72)

    for subject_id in SUBJECT_IDS:
        session = f"{subject_id}E"
        evaluation_data[subject_id] = load_subject(
            session
        )

    return evaluation_data


def combine_subjects(
    subjects: list[str],
    subject_data: dict[
        str,
        tuple[np.ndarray, np.ndarray],
    ],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    X_parts = []
    y_parts = []
    subject_parts = []

    for subject_id in subjects:
        X_subject, y_subject = (
            subject_data[subject_id]
        )

        X_parts.append(X_subject)
        y_parts.append(y_subject)

        subject_parts.append(
            np.full(
                len(y_subject),
                subject_id,
                dtype=object,
            )
        )

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(subject_parts, axis=0),
    )


# ---------------------------------------------------------------------
# Subject-balanced validation split
# ---------------------------------------------------------------------

def create_subject_balanced_split(
    training_subjects: list[str],
    subject_data: dict[str, tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Split every training subject separately.

    Each subject contributes approximately 85% training trials and
    15% validation trials. Splitting is stratified by motor-imagery
    class.
    """

    X_train_parts = []
    y_train_parts = []
    train_subject_parts = []

    X_validation_parts = []
    y_validation_parts = []
    validation_subject_parts = []

    for subject_index, subject in enumerate(
        training_subjects
    ):
        X_subject, y_subject = subject_data[subject]

        subject_seed = seed + subject_index

        all_indices = np.arange(
            len(y_subject)
        )

        train_indices, validation_indices = (
            train_test_split(
                all_indices,
                test_size=VALIDATION_SIZE,
                random_state=subject_seed,
                stratify=y_subject,
            )
        )

        X_train_parts.append(
            X_subject[train_indices]
        )

        y_train_parts.append(
            y_subject[train_indices]
        )

        train_subject_parts.append(
            np.full(
                len(train_indices),
                subject,
                dtype=object,
            )
        )

        X_validation_parts.append(
            X_subject[validation_indices]
        )

        y_validation_parts.append(
            y_subject[validation_indices]
        )

        validation_subject_parts.append(
            np.full(
                len(validation_indices),
                subject,
                dtype=object,
            )
        )

    return (
        np.concatenate(X_train_parts, axis=0),
        np.concatenate(y_train_parts, axis=0),
        np.concatenate(train_subject_parts, axis=0),
        np.concatenate(X_validation_parts, axis=0),
        np.concatenate(y_validation_parts, axis=0),
        np.concatenate(validation_subject_parts, axis=0),
    )


# ---------------------------------------------------------------------
# Training-only standardisation
# ---------------------------------------------------------------------

def calculate_channel_statistics(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    return (
        (X - channel_mean) / channel_std
    ).astype(np.float32)


# ---------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------

class EEGDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
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
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        EEGDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


# ---------------------------------------------------------------------
# EEGNet
# ---------------------------------------------------------------------

class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
    ) -> None:
        super().__init__()

        self.temporal_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 64),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(F1),
        )

        self.spatial_block = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1 * DEPTH_MULTIPLIER,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(
                F1 * DEPTH_MULTIPLIER
            ),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 4)
            ),
            nn.Dropout(DROPOUT_RATE),
        )

        self.separable_block = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * DEPTH_MULTIPLIER,
                out_channels=F1 * DEPTH_MULTIPLIER,
                kernel_size=(1, 16),
                padding="same",
                groups=F1 * DEPTH_MULTIPLIER,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=F1 * DEPTH_MULTIPLIER,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(
                kernel_size=(1, 8)
            ),
            nn.Dropout(DROPOUT_RATE),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(
                1,
                1,
                n_channels,
                n_samples,
            )

            dummy_output = self.extract_features(
                dummy_input
            )

            flattened_size = int(
                np.prod(dummy_output.shape[1:])
            )

        self.classifier = nn.Linear(
            flattened_size,
            n_classes,
        )

    def extract_features(
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
        X = self.extract_features(X)
        X = torch.flatten(X, start_dim=1)

        return self.classifier(X)


def initialise_model(
    n_samples: int,
    device: torch.device,
) -> EEGNet:
    model = EEGNet(
        n_channels=N_CHANNELS,
        n_samples=n_samples,
        n_classes=N_CLASSES,
    )

    return model.to(device)


# ---------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

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

        predictions = logits.argmax(dim=1)

        total_loss += (
            loss.item() * len(y_batch)
        )

        total_correct += (
            predictions == y_batch
        ).sum().item()

        total_examples += len(y_batch)

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


@torch.no_grad()
def evaluate_model(
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
    model.eval()

    total_loss = 0.0
    total_examples = 0

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

        total_examples += len(y_batch)

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
        total_loss / total_examples,
        accuracy,
        true_labels,
        predicted_labels,
        probabilities,
    )


def select_training_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    n_samples: int,
    device: torch.device,
    seed: int,
    test_subject: str,
) -> tuple[int, list[dict]]:
    """
    Select the training duration using validation data collected
    from all eight training subjects.
    """

    set_random_seed(seed)

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
        patience=7,
        min_lr=1e-5,
    )

    train_loader = create_loader(
        X_train,
        y_train,
        shuffle=True,
        seed=seed,
    )

    validation_loader = create_loader(
        X_validation,
        y_validation,
        shuffle=False,
        seed=seed,
    )

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = MINIMUM_EPOCHS
    best_state = None

    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(
        1,
        MAXIMUM_EPOCHS + 1,
    ):
        train_loss, train_accuracy = (
            train_one_epoch(
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
            validation_true,
            validation_predictions,
            _,
        ) = evaluate_model(
            model,
            validation_loader,
            criterion,
            device,
        )

        validation_kappa = cohen_kappa_score(
            validation_true,
            validation_predictions,
        )

        scheduler.step(validation_loss)

        current_learning_rate = (
            optimiser.param_groups[0]["lr"]
        )

        history_rows.append(
            {
                "seed": seed,
                "test_subject": test_subject,
                "stage": "epoch_selection",
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": (
                    validation_accuracy
                ),
                "validation_kappa": (
                    validation_kappa
                ),
                "learning_rate": (
                    current_learning_rate
                ),
            }
        )

        print(
            f"    Epoch {epoch:03d} | "
            f"Train loss {train_loss:.4f} | "
            f"Train acc {train_accuracy * 100:5.1f}% | "
            f"Val loss {validation_loss:.4f} | "
            f"Val acc {validation_accuracy * 100:5.1f}% | "
            f"Val κ {validation_kappa:.3f}"
        )

        if epoch >= MINIMUM_EPOCHS:
            improved = (
                validation_loss
                < best_validation_loss - 1e-4
            )

            if improved:
                best_validation_loss = (
                    validation_loss
                )

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

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                print(
                    "    Early stopping after "
                    f"{EARLY_STOPPING_PATIENCE} epochs "
                    "without improvement."
                )
                break

    if best_state is None:
        best_state = copy.deepcopy(
            model.state_dict()
        )

        best_epoch = MAXIMUM_EPOCHS

        best_validation_loss = (
            validation_loss
        )

        best_validation_accuracy = (
            validation_accuracy
        )

    checkpoint_path = (
        CHECKPOINT_DIRECTORY
        / f"seed_{seed}_{test_subject}_selection.pt"
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "seed": seed,
            "test_subject": test_subject,
            "best_epoch": best_epoch,
            "best_validation_loss": (
                best_validation_loss
            ),
            "best_validation_accuracy": (
                best_validation_accuracy
            ),
        },
        checkpoint_path,
    )

    print(
        f"\n    Selected epoch: {best_epoch}"
    )

    print(
        f"    Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    return best_epoch, history_rows


def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_samples: int,
    selected_epochs: int,
    device: torch.device,
    seed: int,
    test_subject: str,
) -> tuple[nn.Module, list[dict]]:
    """
    Train a fresh model on all trials from the eight training
    subjects.
    """

    set_random_seed(seed)

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
        seed=seed,
    )

    history_rows = []

    print(
        f"\n    Final training for "
        f"{selected_epochs} epochs..."
    )

    for epoch in range(
        1,
        selected_epochs + 1,
    ):
        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimiser,
                device,
            )
        )

        history_rows.append(
            {
                "seed": seed,
                "test_subject": test_subject,
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
            f"    Final epoch "
            f"{epoch:03d}/{selected_epochs:03d} | "
            f"Loss {train_loss:.4f} | "
            f"Accuracy {train_accuracy * 100:5.1f}%"
        )

    checkpoint_path = (
        CHECKPOINT_DIRECTORY
        / f"seed_{seed}_{test_subject}_final.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "test_subject": test_subject,
            "training_epochs": selected_epochs,
        },
        checkpoint_path,
    )

    return model, history_rows


# ---------------------------------------------------------------------
# LOSO fold
# ---------------------------------------------------------------------

def run_fold(
    training_data: dict[
        str,
        tuple[np.ndarray, np.ndarray],
    ],
    evaluation_data: dict[
        str,
        tuple[np.ndarray, np.ndarray],
    ],
    test_subject: str,
    seed: int,
    device: torch.device,
) -> tuple[
    dict,
    list[dict],
    list[dict],
    np.ndarray,
    np.ndarray,
]:
    fold_start_time = time.time()

    training_subjects = [
        subject
        for subject in SUBJECT_IDS
        if subject != test_subject
    ]

    training_sessions = [
        f"{subject}T"
        for subject in training_subjects
    ]

    test_session = f"{test_subject}E"

    print("\n" + "=" * 72)
    print(
        f"Seed {seed} | Unseen test subject: {test_subject}"
    )
    print("=" * 72)

    print(
        "Training subjects: "
        + ", ".join(training_subjects)
    )

    (
        X_selection_train,
        y_selection_train,
        selection_train_subjects,
        X_validation,
        y_validation,
        validation_subjects,
    ) = create_subject_balanced_split(
        training_subjects=training_subjects,
        subject_data=training_data,
        seed=seed,
    )

    (
        X_final_train,
        y_final_train,
        final_train_subjects,
    ) = combine_subjects(
        training_subjects,
        training_data,
    )

    (
        X_test,
        y_test,
        test_subject_labels,
    ) = combine_subjects(
        [test_subject],
        evaluation_data,
    )

    print(
        f"Selection training data: "
        f"{X_selection_train.shape}"
    )

    print(
        f"Validation data: "
        f"{X_validation.shape}"
    )

    print(
        f"Final training data: "
        f"{X_final_train.shape}"
    )

    print(
        f"Unseen test data: "
        f"{X_test.shape}"
    )

    # Epoch-selection normalisation.
    selection_mean, selection_std = (
        calculate_channel_statistics(
            X_selection_train
        )
    )

    X_selection_train_normalised = (
        standardise_eeg(
            X_selection_train,
            selection_mean,
            selection_std,
        )
    )

    X_validation_normalised = standardise_eeg(
        X_validation,
        selection_mean,
        selection_std,
    )

    n_samples = X_selection_train.shape[2]

    selected_epochs, selection_history = (
        select_training_epochs(
            X_train=(
                X_selection_train_normalised
            ),
            y_train=y_selection_train,
            X_validation=(
                X_validation_normalised
            ),
            y_validation=y_validation,
            n_samples=n_samples,
            device=device,
            seed=seed,
            test_subject=test_subject,
        )
    )

    # Final model normalisation uses all eight training subjects.
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

    X_test_normalised = standardise_eeg(
        X_test,
        final_mean,
        final_std,
    )

    final_model, final_history = train_final_model(
        X_train=X_final_train_normalised,
        y_train=y_final_train,
        n_samples=n_samples,
        selected_epochs=selected_epochs,
        device=device,
        seed=seed,
        test_subject=test_subject,
    )

    test_loader = create_loader(
        X_test_normalised,
        y_test,
        shuffle=False,
        seed=seed,
    )

    criterion = nn.CrossEntropyLoss()

    (
        test_loss,
        test_accuracy,
        true_labels,
        predicted_labels,
        probabilities,
    ) = evaluate_model(
        final_model,
        test_loader,
        criterion,
        device,
    )

    test_kappa = cohen_kappa_score(
        true_labels,
        predicted_labels,
    )

    fold_confusion_matrix = confusion_matrix(
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

    print("\n    Unseen-subject test result")
    print("    " + "-" * 60)

    print(
        f"    Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"    Cohen's κ: "
        f"{test_kappa:.3f}"
    )

    print(
        f"    Test loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"    Selected epochs: "
        f"{selected_epochs}"
    )

    print(
        f"    Duration: "
        f"{fold_duration / 60:.1f} minutes"
    )

    print("    Confusion matrix:")
    print(fold_confusion_matrix)

    subject_result = {
        "seed": seed,
        "test_subject": test_subject,
        "training_sessions": "|".join(
            training_sessions
        ),
        "test_session": test_session,
        "evaluation": (
            "cross_subject_T_to_E_LOSO"
        ),
        "selected_epochs": selected_epochs,
        "training_trials": len(
            y_final_train
        ),
        "validation_trials": len(
            y_validation
        ),
        "test_trials": len(y_test),
        "accuracy": test_accuracy,
        "accuracy_percent": (
            test_accuracy * 100
        ),
        "kappa": test_kappa,
        "test_loss": test_loss,
        "left_hand_recall": (
            class_recalls[0]
        ),
        "right_hand_recall": (
            class_recalls[1]
        ),
        "feet_recall": (
            class_recalls[2]
        ),
        "tongue_recall": (
            class_recalls[3]
        ),
        "duration_seconds": fold_duration,
    }

    prediction_rows = []

    for trial_index, (
        true_index,
        predicted_index,
        probability_vector,
    ) in enumerate(
        zip(
            true_labels,
            predicted_labels,
            probabilities,
        ),
        start=1,
    ):
        prediction_rows.append(
            {
                "seed": seed,
                "subject": test_subject,
                "training_sessions": "|".join(
                    training_sessions
                ),
                "test_session": test_session,
                "evaluation": (
                    "cross_subject_T_to_E_LOSO"
                ),
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

    return (
        subject_result,
        prediction_rows,
        selection_history + final_history,
        true_labels,
        predicted_labels,
    )


# ---------------------------------------------------------------------
# Saving and plots
# ---------------------------------------------------------------------

def save_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
) -> None:
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

    dataframe.to_csv(output_path)


def create_subject_accuracy_plot(
    summary_dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(11, 6))

    plt.bar(
        summary_dataframe["test_subject"],
        summary_dataframe[
            "mean_accuracy_percent"
        ],
        yerr=summary_dataframe[
            "std_accuracy_percent"
        ],
        capsize=4,
    )

    plt.axhline(
        RIEMANNIAN_BASELINE_ACCURACY * 100,
        linestyle="--",
        label=(
            "Riemannian LOSO baseline "
            f"({RIEMANNIAN_BASELINE_ACCURACY * 100:.2f}%)"
        ),
    )

    plt.axhline(
        25.0,
        linestyle=":",
        label="Chance level (25%)",
    )

    plt.xlabel("Unseen test subject")
    plt.ylabel("Accuracy (%)")
    plt.title(
        "Improved EEGNet Strict LOSO Accuracy\n"
        "Mean and Standard Deviation Across Three Seeds"
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


def create_model_comparison_plot(
    final_mean_accuracy: float,
    final_std_accuracy: float,
    output_path: Path,
) -> None:
    model_names = [
        "FBCSP + LDA",
        "EEGNet",
        "Riemannian",
    ]

    accuracies = [
        39.62,
        final_mean_accuracy * 100,
        RIEMANNIAN_BASELINE_ACCURACY * 100,
    ]

    errors = [
        0.0,
        final_std_accuracy * 100,
        0.0,
    ]

    plt.figure(figsize=(9, 6))

    plt.bar(
        model_names,
        accuracies,
        yerr=errors,
        capsize=5,
    )

    plt.axhline(
        25.0,
        linestyle=":",
        label="Chance level (25%)",
    )

    plt.ylabel("Strict LOSO accuracy (%)")
    plt.title(
        "Calibration-Free Cross-Subject Model Comparison"
    )

    plt.ylim(0, 70)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    experiment_start_time = time.time()

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
    print("Improved EEGNet Strict LOSO Experiment")
    print("=" * 72)

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Seeds: {SEEDS}")
    print(
        f"Validation percentage per subject: "
        f"{VALIDATION_SIZE * 100:.0f}%"
    )
    print(
        f"Epoch range: "
        f"{MINIMUM_EPOCHS}-{MAXIMUM_EPOCHS}"
    )

    configuration = {
        "subjects": SUBJECT_IDS,
        "training_sessions": [
            f"{subject}T"
            for subject in SUBJECT_IDS
        ],
        "evaluation_sessions": [
            f"{subject}E"
            for subject in SUBJECT_IDS
        ],
        "evaluation": (
            "cross_subject_T_to_E_LOSO"
        ),
        "seeds": SEEDS,
        "frequency_band": [
            FMIN,
            FMAX,
        ],
        "epoch_window": [
            TMIN,
            TMAX,
        ],
        "validation_size_per_subject": (
            VALIDATION_SIZE
        ),
        "minimum_epochs": MINIMUM_EPOCHS,
        "maximum_epochs": MAXIMUM_EPOCHS,
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout_rate": DROPOUT_RATE,
        "f1": F1,
        "depth_multiplier": (
            DEPTH_MULTIPLIER
        ),
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
        / "configuration.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
        )

    training_data = (
        load_training_sessions()
    )

    evaluation_data = (
        load_evaluation_sessions()
    )

    all_results = []
    all_predictions = []
    all_history = []

    pooled_true_by_seed = {
        seed: []
        for seed in SEEDS
    }

    pooled_predictions_by_seed = {
        seed: []
        for seed in SEEDS
    }

    for seed in SEEDS:
        print("\n" + "#" * 72)
        print(f"Starting complete LOSO run with seed {seed}")
        print("#" * 72)

        for test_subject in SUBJECT_IDS:
            fold_seed = (
                seed
                + SUBJECT_IDS.index(test_subject)
            )

            (
                result,
                prediction_rows,
                history_rows,
                true_labels,
                predicted_labels,
            ) = run_fold(
                training_data=training_data,
                evaluation_data=evaluation_data,
                test_subject=test_subject,
                seed=fold_seed,
                device=device,
            )

            # Store the main experiment seed rather than only
            # the fold-adjusted internal seed.
            result["experiment_seed"] = seed

            for row in prediction_rows:
                row["experiment_seed"] = seed

            for row in history_rows:
                row["experiment_seed"] = seed

            all_results.append(result)
            all_predictions.extend(
                prediction_rows
            )
            all_history.extend(
                history_rows
            )

            pooled_true_by_seed[seed].append(
                true_labels
            )

            pooled_predictions_by_seed[
                seed
            ].append(
                predicted_labels
            )

            pd.DataFrame(
                all_results
            ).to_csv(
                OUTPUT_DIRECTORY
                / "subject_seed_results.csv",
                index=False,
            )

            pd.DataFrame(
                all_predictions
            ).to_csv(
                OUTPUT_DIRECTORY
                / "predictions.csv",
                index=False,
            )

            pd.DataFrame(
                all_history
            ).to_csv(
                OUTPUT_DIRECTORY
                / "training_history.csv",
                index=False,
            )

    results_dataframe = pd.DataFrame(
        all_results
    )

    predictions_dataframe = pd.DataFrame(
        all_predictions
    )

    history_dataframe = pd.DataFrame(
        all_history
    )

    subject_summary = (
        results_dataframe
        .groupby("test_subject")
        .agg(
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            std_accuracy=(
                "accuracy",
                "std",
            ),
            mean_kappa=(
                "kappa",
                "mean",
            ),
            std_kappa=(
                "kappa",
                "std",
            ),
            mean_selected_epochs=(
                "selected_epochs",
                "mean",
            ),
        )
        .reset_index()
    )

    subject_summary[
        "mean_accuracy_percent"
    ] = (
        subject_summary["mean_accuracy"]
        * 100
    )

    subject_summary[
        "std_accuracy_percent"
    ] = (
        subject_summary["std_accuracy"]
        * 100
    )

    seed_summary_rows = []

    for seed in SEEDS:
        seed_results = results_dataframe[
            results_dataframe[
                "experiment_seed"
            ] == seed
        ]

        pooled_true = np.concatenate(
            pooled_true_by_seed[seed]
        )

        pooled_predictions = np.concatenate(
            pooled_predictions_by_seed[
                seed
            ]
        )

        seed_summary_rows.append(
            {
                "seed": seed,
                "mean_subject_accuracy": (
                    seed_results[
                        "accuracy"
                    ].mean()
                ),
                "mean_subject_accuracy_percent": (
                    seed_results[
                        "accuracy"
                    ].mean()
                    * 100
                ),
                "mean_subject_kappa": (
                    seed_results[
                        "kappa"
                    ].mean()
                ),
                "pooled_accuracy": (
                    accuracy_score(
                        pooled_true,
                        pooled_predictions,
                    )
                ),
                "pooled_kappa": (
                    cohen_kappa_score(
                        pooled_true,
                        pooled_predictions,
                    )
                ),
            }
        )

    seed_summary = pd.DataFrame(
        seed_summary_rows
    )

    final_mean_accuracy = seed_summary[
        "mean_subject_accuracy"
    ].mean()

    final_std_accuracy = seed_summary[
        "mean_subject_accuracy"
    ].std(ddof=1)

    final_mean_kappa = seed_summary[
        "mean_subject_kappa"
    ].mean()

    final_std_kappa = seed_summary[
        "mean_subject_kappa"
    ].std(ddof=1)

    all_pooled_true = np.concatenate(
        [
            np.concatenate(
                pooled_true_by_seed[seed]
            )
            for seed in SEEDS
        ]
    )

    all_pooled_predictions = np.concatenate(
        [
            np.concatenate(
                pooled_predictions_by_seed[
                    seed
                ]
            )
            for seed in SEEDS
        ]
    )

    overall_confusion_matrix = confusion_matrix(
        all_pooled_true,
        all_pooled_predictions,
        labels=np.arange(N_CLASSES),
    )

    results_dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "subject_seed_results.csv",
        index=False,
    )

    predictions_dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "predictions.csv",
        index=False,
    )

    history_dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "training_history.csv",
        index=False,
    )

    subject_summary.to_csv(
        OUTPUT_DIRECTORY
        / "subject_summary.csv",
        index=False,
    )

    seed_summary.to_csv(
        OUTPUT_DIRECTORY
        / "seed_summary.csv",
        index=False,
    )

    save_confusion_matrix(
        overall_confusion_matrix,
        OUTPUT_DIRECTORY
        / "overall_confusion_matrix.csv",
    )

    create_subject_accuracy_plot(
        subject_summary,
        OUTPUT_DIRECTORY
        / "subject_accuracy_mean_std.png",
    )

    create_model_comparison_plot(
        final_mean_accuracy,
        final_std_accuracy,
        OUTPUT_DIRECTORY
        / "model_comparison.png",
    )

    classification_report_text = (
        classification_report(
            all_pooled_true,
            all_pooled_predictions,
            labels=np.arange(N_CLASSES),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    total_duration = (
        time.time() - experiment_start_time
    )

    summary_lines = [
        "=" * 72,
        "Improved EEGNet Strict LOSO Results",
        "=" * 72,
        "",
        "Evaluation protocol:",
        "  Test subject remains completely unseen.",
        (
            "  Validation data contain a stratified 15% "
            "sample from each of the eight training subjects."
        ),
        (
            "  The model is retrained on all eight training "
            "subjects after selecting the epoch."
        ),
        f"  Seeds: {SEEDS}",
        "",
        "Seed-level results:",
        seed_summary.to_string(index=False),
        "",
        "Subject-level mean results:",
        subject_summary.to_string(index=False),
        "",
        "Final EEGNet result across seeds:",
        (
            f"  Mean accuracy: "
            f"{final_mean_accuracy * 100:.2f}%"
        ),
        (
            f"  Accuracy SD: "
            f"{final_std_accuracy * 100:.2f}%"
        ),
        (
            f"  Mean kappa: "
            f"{final_mean_kappa:.3f}"
        ),
        (
            f"  Kappa SD: "
            f"{final_std_kappa:.3f}"
        ),
        "",
        "Riemannian baseline:",
        (
            f"  Accuracy: "
            f"{RIEMANNIAN_BASELINE_ACCURACY * 100:.2f}%"
        ),
        (
            f"  Kappa: "
            f"{RIEMANNIAN_BASELINE_KAPPA:.3f}"
        ),
        "",
        "Difference from Riemannian baseline:",
        (
            f"  Accuracy difference: "
            f"{(final_mean_accuracy - RIEMANNIAN_BASELINE_ACCURACY) * 100:+.2f} "
            "percentage points"
        ),
        (
            f"  Kappa difference: "
            f"{final_mean_kappa - RIEMANNIAN_BASELINE_KAPPA:+.3f}"
        ),
        "",
        "Overall confusion matrix across all seeds:",
        str(overall_confusion_matrix),
        "",
        "Classification report:",
        classification_report_text,
        "",
        (
            f"Total duration: "
            f"{total_duration / 60:.1f} minutes"
        ),
    ]

    (
        OUTPUT_DIRECTORY
        / "summary.txt"
    ).write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("Final Improved EEGNet Results")
    print("=" * 72)

    print("\nSeed-level results:")
    print(
        seed_summary.to_string(
            index=False,
            formatters={
                "mean_subject_accuracy_percent": (
                    lambda value: f"{value:.2f}"
                ),
                "mean_subject_kappa": (
                    lambda value: f"{value:.3f}"
                ),
                "pooled_accuracy": (
                    lambda value: f"{value * 100:.2f}%"
                ),
                "pooled_kappa": (
                    lambda value: f"{value:.3f}"
                ),
            },
        )
    )

    print("\nSubject-level results:")
    print(
        subject_summary[
            [
                "test_subject",
                "mean_accuracy_percent",
                "std_accuracy_percent",
                "mean_kappa",
                "mean_selected_epochs",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean_accuracy_percent": (
                    lambda value: f"{value:.2f}"
                ),
                "std_accuracy_percent": (
                    lambda value: f"{value:.2f}"
                ),
                "mean_kappa": (
                    lambda value: f"{value:.3f}"
                ),
                "mean_selected_epochs": (
                    lambda value: f"{value:.1f}"
                ),
            },
        )
    )

    print("\nOverall:")
    print(
        f"  EEGNet accuracy: "
        f"{final_mean_accuracy * 100:.2f}% "
        f"± {final_std_accuracy * 100:.2f}%"
    )

    print(
        f"  EEGNet kappa: "
        f"{final_mean_kappa:.3f} "
        f"± {final_std_kappa:.3f}"
    )

    print(
        f"  Riemannian accuracy: "
        f"{RIEMANNIAN_BASELINE_ACCURACY * 100:.2f}%"
    )

    print(
        f"  Difference: "
        f"{(final_mean_accuracy - RIEMANNIAN_BASELINE_ACCURACY) * 100:+.2f} "
        "percentage points"
    )

    print("\nSaved outputs:")
    print(
        f"  {OUTPUT_DIRECTORY}/subject_seed_results.csv"
    )
    print(
        f"  {OUTPUT_DIRECTORY}/subject_summary.csv"
    )
    print(
        f"  {OUTPUT_DIRECTORY}/seed_summary.csv"
    )
    print(
        f"  {OUTPUT_DIRECTORY}/predictions.csv"
    )
    print(
        f"  {OUTPUT_DIRECTORY}/training_history.csv"
    )
    print(
        f"  {OUTPUT_DIRECTORY}/overall_confusion_matrix.csv"
    )
    print(
        f"  {OUTPUT_DIRECTORY}/subject_accuracy_mean_std.png"
    )
    print(
        f"  {OUTPUT_DIRECTORY}/model_comparison.png"
    )
    print(
        f"  {OUTPUT_DIRECTORY}/summary.txt"
    )

    print(
        f"\nTotal duration: "
        f"{total_duration / 60:.1f} minutes"
    )


if __name__ == "__main__":
    main()
