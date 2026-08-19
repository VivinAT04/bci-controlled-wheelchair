"""
Within-subject EEGNet evaluation.

Protocol
--------
For each BCI Competition IV 2a training subject independently:

    A01T ... A09T

perform stratified 10-fold cross-validation.

For every outer fold:
    1. Hold out the outer test fold.
    2. Split the remaining trials into training and validation sets.
    3. Calculate EEG standardisation statistics using training data only.
    4. Train EEGNet with early stopping on validation loss.
    5. Restore the best validation checkpoint.
    6. Evaluate once on the untouched outer test fold.

This avoids using the outer test fold for:
    - normalisation
    - model selection
    - early stopping

Run:

    python -m scripts.within_subject.run_eegnet
"""

from __future__ import annotations

import csv
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bci_wheelchair.data.processed_loading import (
    load_processed_subject,
)
from bci_wheelchair.models import make_eegnet


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

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

CLASS_ORDER = [
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
]

CLASS_TO_INDEX = {
    label: index
    for index, label in enumerate(CLASS_ORDER)
}

INDEX_TO_CLASS = {
    index: label
    for label, index in CLASS_TO_INDEX.items()
}

PREPROCESSING = "8-30"

N_SPLITS = 10
VALIDATION_FRACTION = 0.15

BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
DROPOUT_RATE = 0.5

F1 = 8
DEPTH_MULTIPLIER = 2
F2 = 16
TEMPORAL_KERNEL_SIZE = 64
SEPARABLE_KERNEL_SIZE = 16

MAXIMUM_EPOCHS = 300
MINIMUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 25

RANDOM_SEED = 42

RESULTS_DIRECTORY = Path(
    "results/within_subject/eegnet"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "eegnet_within_subject_subject_results.csv"
)

FOLD_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "eegnet_within_subject_fold_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "eegnet_within_subject_predictions.csv"
)

TRAINING_HISTORY_PATH = (
    RESULTS_DIRECTORY
    / "eegnet_within_subject_training_history.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "eegnet_within_subject_overall_summary.csv"
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_random_seed(seed: int) -> None:
    """Set random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Select the best available PyTorch device."""

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def save_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write dictionaries to CSV."""

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------

def encode_labels(
    labels: np.ndarray,
) -> np.ndarray:
    """Convert string labels to integer class indices."""

    encoded = []

    for label in labels:

        label_string = str(label)

        if label_string not in CLASS_TO_INDEX:
            raise ValueError(
                f"Unknown class label: {label_string}"
            )

        encoded.append(
            CLASS_TO_INDEX[label_string]
        )

    return np.asarray(
        encoded,
        dtype=np.int64,
    )


def decode_labels(
    labels: np.ndarray,
) -> np.ndarray:
    """Convert integer class indices back to class labels."""

    return np.asarray(
        [
            INDEX_TO_CLASS[int(label)]
            for label in labels
        ]
    )


# ---------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------

def calculate_channel_statistics(
    X_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate channel-wise statistics using training data only.

    Input:
        trials x channels x samples
    """

    channel_mean = X_train.mean(
        axis=(0, 2),
        keepdims=True,
    )

    channel_std = X_train.std(
        axis=(0, 2),
        keepdims=True,
    )

    channel_std = np.where(
        channel_std < 1e-8,
        1.0,
        channel_std,
    )

    return (
        channel_mean,
        channel_std,
    )


def standardise_eeg(
    X: np.ndarray,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
) -> np.ndarray:
    """Apply training-derived channel-wise standardisation."""

    return np.asarray(
        (X - channel_mean) / channel_std,
        dtype=np.float32,
    )


# ---------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------

class EEGDataset(Dataset):
    """EEG dataset for EEGNet."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:

        self.X = torch.from_numpy(
            np.asarray(
                X,
                dtype=np.float32,
            )
        ).unsqueeze(1)

        self.y = torch.from_numpy(
            np.asarray(
                y,
                dtype=np.int64,
            )
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(
        self,
        index: int,
    ):
        return (
            self.X[index],
            self.y[index],
        )


def create_loader(
    X: np.ndarray,
    y: np.ndarray,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create deterministic DataLoader."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        EEGDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def initialise_model(
    n_channels: int,
    n_samples: int,
    device: torch.device,
) -> nn.Module:
    """Initialise reusable EEGNet architecture."""

    return make_eegnet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=len(CLASS_ORDER),
        dropout_rate=DROPOUT_RATE,
        f1=F1,
        depth_multiplier=DEPTH_MULTIPLIER,
        f2=F2,
        temporal_kernel_size=TEMPORAL_KERNEL_SIZE,
        separable_kernel_size=SEPARABLE_KERNEL_SIZE,
        device=device,
    )


# ---------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(
            logits,
            y_batch,
        )

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        batch_size = len(y_batch)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_correct += int(
            (
                predictions
                == y_batch
            ).sum().item()
        )

        total_examples += batch_size

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
    """Evaluate EEGNet."""

    model.eval()

    total_loss = 0.0
    total_examples = 0

    true_labels = []
    predicted_labels = []
    probabilities = []

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)

        loss = criterion(
            logits,
            y_batch,
        )

        batch_probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        batch_size = len(y_batch)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_examples += batch_size

        true_labels.append(
            y_batch.cpu().numpy()
        )

        predicted_labels.append(
            predictions.cpu().numpy()
        )

        probabilities.append(
            batch_probabilities.cpu().numpy()
        )

    y_true = np.concatenate(
        true_labels
    )

    y_pred = np.concatenate(
        predicted_labels
    )

    probabilities_array = np.concatenate(
        probabilities,
        axis=0,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    return (
        total_loss / total_examples,
        float(accuracy),
        y_true,
        y_pred,
        probabilities_array,
    )


# ---------------------------------------------------------------------
# One outer fold
# ---------------------------------------------------------------------

def run_fold(
    subject: str,
    fold_number: int,
    X: np.ndarray,
    y: np.ndarray,
    train_validation_indices: np.ndarray,
    test_indices: np.ndarray,
    device: torch.device,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:

    fold_seed = (
        RANDOM_SEED
        + fold_number
        + SUBJECTS.index(subject) * 100
    )

    set_random_seed(
        fold_seed
    )

    X_train_validation = X[
        train_validation_indices
    ]

    y_train_validation = y[
        train_validation_indices
    ]

    X_test = X[
        test_indices
    ]

    y_test = y[
        test_indices
    ]

    local_indices = np.arange(
        len(y_train_validation)
    )

    (
        training_local_indices,
        validation_local_indices,
    ) = train_test_split(
        local_indices,
        test_size=VALIDATION_FRACTION,
        random_state=fold_seed,
        stratify=y_train_validation,
    )

    X_train = X_train_validation[
        training_local_indices
    ]

    y_train = y_train_validation[
        training_local_indices
    ]

    X_validation = X_train_validation[
        validation_local_indices
    ]

    y_validation = y_train_validation[
        validation_local_indices
    ]

    # -------------------------------------------------------------
    # Training-only normalisation
    # -------------------------------------------------------------

    channel_mean, channel_std = (
        calculate_channel_statistics(
            X_train
        )
    )

    X_train = standardise_eeg(
        X_train,
        channel_mean,
        channel_std,
    )

    X_validation = standardise_eeg(
        X_validation,
        channel_mean,
        channel_std,
    )

    X_test = standardise_eeg(
        X_test,
        channel_mean,
        channel_std,
    )

    # -------------------------------------------------------------
    # Loaders
    # -------------------------------------------------------------

    train_loader = create_loader(
        X_train,
        y_train,
        shuffle=True,
        seed=fold_seed,
    )

    validation_loader = create_loader(
        X_validation,
        y_validation,
        shuffle=False,
        seed=fold_seed,
    )

    test_loader = create_loader(
        X_test,
        y_test,
        shuffle=False,
        seed=fold_seed,
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = initialise_model(
        n_channels=X.shape[1],
        n_samples=X.shape[2],
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
        )
    )

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    history_rows = []

    start_time = time.time()

    # -------------------------------------------------------------
    # Training
    # -------------------------------------------------------------

    for epoch in range(
        1,
        MAXIMUM_EPOCHS + 1,
    ):

        (
            train_loss,
            train_accuracy,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        (
            validation_loss,
            validation_accuracy,
            validation_true,
            validation_pred,
            _,
        ) = evaluate_model(
            model,
            validation_loader,
            criterion,
            device,
        )

        validation_kappa = (
            cohen_kappa_score(
                validation_true,
                validation_pred,
            )
        )

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        history_rows.append(
            {
                "subject": subject,
                "fold": fold_number,
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

            best_validation_loss = (
                validation_loss
            )

            best_validation_accuracy = (
                validation_accuracy
            )

            best_epoch = epoch

            best_state = deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        elif epoch >= MINIMUM_EPOCHS:

            epochs_without_improvement += 1

        if (
            epoch >= MINIMUM_EPOCHS
            and epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(
            f"No valid EEGNet checkpoint for "
            f"{subject}, fold {fold_number}."
        )

    model.load_state_dict(
        best_state
    )

    # -------------------------------------------------------------
    # Outer test evaluation
    # -------------------------------------------------------------

    (
        test_loss,
        test_accuracy,
        test_true,
        test_pred,
        test_probabilities,
    ) = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
    )

    test_kappa = cohen_kappa_score(
        test_true,
        test_pred,
    )

    recalls = recall_score(
        test_true,
        test_pred,
        labels=np.arange(
            len(CLASS_ORDER)
        ),
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(
        test_true,
        test_pred,
        labels=np.arange(
            len(CLASS_ORDER)
        ),
    )

    duration_seconds = (
        time.time()
        - start_time
    )

    fold_result: dict[str, object] = {
        "subject": subject,
        "fold": fold_number,
        "train_trials": len(X_train),
        "validation_trials": len(
            X_validation
        ),
        "test_trials": len(X_test),
        "selected_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
        "test_loss": test_loss,
        "accuracy": test_accuracy,
        "accuracy_percent": (
            test_accuracy * 100.0
        ),
        "kappa": test_kappa,
        "left_hand_recall": recalls[0],
        "right_hand_recall": recalls[1],
        "feet_recall": recalls[2],
        "tongue_recall": recalls[3],
        "duration_seconds": duration_seconds,
        "preprocessing": PREPROCESSING,
        "model": "EEGNet",
        "evaluation": (
            "within_subject_stratified_10_fold"
        ),
    }

    for true_index, true_label in enumerate(
        CLASS_ORDER
    ):
        for pred_index, pred_label in enumerate(
            CLASS_ORDER
        ):

            fold_result[
                f"cm_{true_label}_pred_{pred_label}"
            ] = int(
                matrix[
                    true_index,
                    pred_index,
                ]
            )

    # -------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------

    prediction_rows = []

    decoded_true = decode_labels(
        test_true
    )

    decoded_pred = decode_labels(
        test_pred
    )

    for local_index, (
        original_trial_index,
        true_label,
        predicted_label,
        probability_vector,
    ) in enumerate(
        zip(
            test_indices,
            decoded_true,
            decoded_pred,
            test_probabilities,
        ),
        start=1,
    ):

        prediction_rows.append(
            {
                "subject": subject,
                "fold": fold_number,
                "fold_trial": local_index,
                "original_trial": (
                    int(original_trial_index) + 1
                ),
                "true_label": true_label,
                "predicted_label": (
                    predicted_label
                ),
                "correct": (
                    true_label
                    == predicted_label
                ),
                "left_hand_probability": (
                    float(
                        probability_vector[0]
                    )
                ),
                "right_hand_probability": (
                    float(
                        probability_vector[1]
                    )
                ),
                "feet_probability": (
                    float(
                        probability_vector[2]
                    )
                ),
                "tongue_probability": (
                    float(
                        probability_vector[3]
                    )
                ),
                "model": "EEGNet",
                "evaluation": (
                    "within_subject_stratified_10_fold"
                ),
                "preprocessing": PREPROCESSING,
            }
        )

    return (
        fold_result,
        prediction_rows,
        history_rows,
    )


# ---------------------------------------------------------------------
# Subject summary
# ---------------------------------------------------------------------

def build_subject_result(
    subject: str,
    subject_prediction_rows: list[
        dict[str, object]
    ],
) -> dict[str, object]:
    """Calculate pooled out-of-fold subject metrics."""

    y_true_labels = np.asarray(
        [
            row["true_label"]
            for row in subject_prediction_rows
        ]
    )

    y_pred_labels = np.asarray(
        [
            row["predicted_label"]
            for row in subject_prediction_rows
        ]
    )

    accuracy = accuracy_score(
        y_true_labels,
        y_pred_labels,
    )

    kappa = cohen_kappa_score(
        y_true_labels,
        y_pred_labels,
    )

    recalls = recall_score(
        y_true_labels,
        y_pred_labels,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true_labels,
        y_pred_labels,
        labels=CLASS_ORDER,
    )

    result: dict[str, object] = {
        "subject": subject,
        "trials": len(
            subject_prediction_rows
        ),
        "folds": N_SPLITS,
        "accuracy": accuracy,
        "accuracy_percent": (
            accuracy * 100.0
        ),
        "kappa": kappa,
        "left_hand_recall": recalls[0],
        "right_hand_recall": recalls[1],
        "feet_recall": recalls[2],
        "tongue_recall": recalls[3],
        "preprocessing": PREPROCESSING,
        "model": "EEGNet",
        "evaluation": (
            "within_subject_stratified_10_fold"
        ),
    }

    for true_index, true_label in enumerate(
        CLASS_ORDER
    ):
        for pred_index, pred_label in enumerate(
            CLASS_ORDER
        ):

            result[
                f"cm_{true_label}_pred_{pred_label}"
            ] = int(
                matrix[
                    true_index,
                    pred_index,
                ]
            )

    return result


# ---------------------------------------------------------------------
# Overall summary
# ---------------------------------------------------------------------

def build_overall_summary(
    subject_results: list[
        dict[str, object]
    ],
) -> dict[str, object]:
    """Calculate mean subject performance."""

    accuracies = np.asarray(
        [
            float(result["accuracy"])
            for result in subject_results
        ]
    )

    kappas = np.asarray(
        [
            float(result["kappa"])
            for result in subject_results
        ]
    )

    return {
        "model": "EEGNet",
        "evaluation": (
            "within_subject_stratified_10_fold"
        ),
        "subjects": len(
            subject_results
        ),
        "folds_per_subject": N_SPLITS,
        "preprocessing": PREPROCESSING,
        "mean_accuracy": float(
            np.mean(accuracies)
        ),
        "mean_accuracy_percent": float(
            np.mean(accuracies)
            * 100.0
        ),
        "std_accuracy_percent": float(
            np.std(accuracies)
            * 100.0
        ),
        "mean_kappa": float(
            np.mean(kappas)
        ),
        "std_kappa": float(
            np.std(kappas)
        ),
        "minimum_accuracy_percent": float(
            np.min(accuracies)
            * 100.0
        ),
        "maximum_accuracy_percent": float(
            np.max(accuracies)
            * 100.0
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Run within-subject EEGNet evaluation."""

    set_random_seed(
        RANDOM_SEED
    )

    device = select_device()

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 78)
    print("Within-Subject EEGNet")
    print("=" * 78)

    print(
        "Evaluation: stratified 10-fold "
        "cross-validation independently per subject"
    )

    print(
        "Validation: 15% of each outer "
        "training fold"
    )

    print(
        "Preprocessing: 8-30 Hz, "
        "0.5-2.5 s"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Maximum epochs: "
        f"{MAXIMUM_EPOCHS}"
    )

    print(
        f"Early stopping patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    all_fold_results = []
    all_prediction_rows = []
    all_history_rows = []
    subject_results = []

    experiment_start = time.time()

    for subject in SUBJECTS:

        print()
        print("=" * 78)
        print(f"Subject: {subject}")
        print("=" * 78)

        X, y_labels = load_processed_subject(
            subject=subject,
            config=PREPROCESSING,
        )

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        y_labels = np.asarray(
            y_labels
        )

        y = encode_labels(
            y_labels
        )

        print(
            f"X shape: {X.shape}"
        )

        print(
            f"Trials: {len(y)}"
        )

        print(
            f"Channels: {X.shape[1]}"
        )

        print(
            f"Samples: {X.shape[2]}"
        )

        outer_cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED,
        )

        subject_prediction_rows = []

        for fold_number, (
            train_validation_indices,
            test_indices,
        ) in enumerate(
            outer_cv.split(
                X,
                y,
            ),
            start=1,
        ):

            print()
            print(
                f"  Fold "
                f"{fold_number:02d}/"
                f"{N_SPLITS}"
            )

            (
                fold_result,
                prediction_rows,
                history_rows,
            ) = run_fold(
                subject=subject,
                fold_number=fold_number,
                X=X,
                y=y,
                train_validation_indices=(
                    train_validation_indices
                ),
                test_indices=test_indices,
                device=device,
            )

            all_fold_results.append(
                fold_result
            )

            all_prediction_rows.extend(
                prediction_rows
            )

            all_history_rows.extend(
                history_rows
            )

            subject_prediction_rows.extend(
                prediction_rows
            )

            print(
                f"    Epoch: "
                f"{fold_result['selected_epoch']}"
            )

            print(
                f"    Accuracy: "
                f"{fold_result['accuracy_percent']:.2f}%"
            )

            print(
                f"    Kappa: "
                f"{fold_result['kappa']:.3f}"
            )

            # Save incrementally.
            save_csv(
                FOLD_RESULTS_PATH,
                all_fold_results,
            )

            save_csv(
                PREDICTIONS_PATH,
                all_prediction_rows,
            )

            save_csv(
                TRAINING_HISTORY_PATH,
                all_history_rows,
            )

        subject_result = build_subject_result(
            subject,
            subject_prediction_rows,
        )

        subject_results.append(
            subject_result
        )

        save_csv(
            SUBJECT_RESULTS_PATH,
            subject_results,
        )

        print()
        print(
            f"{subject} pooled "
            f"out-of-fold accuracy: "
            f"{subject_result['accuracy_percent']:.2f}%"
        )

        print(
            f"{subject} pooled "
            f"kappa: "
            f"{subject_result['kappa']:.3f}"
        )

    overall_summary = build_overall_summary(
        subject_results
    )

    save_csv(
        OVERALL_SUMMARY_PATH,
        [overall_summary],
    )

    total_duration = (
        time.time()
        - experiment_start
    )

    print()
    print("=" * 78)
    print("FINAL EEGNET WITHIN-SUBJECT RESULTS")
    print("=" * 78)

    print()

    print(
        f"{'Subject':<10}"
        f"{'Accuracy':>12}"
        f"{'Kappa':>12}"
    )

    print("-" * 34)

    for result in subject_results:

        print(
            f"{result['subject']:<10}"
            f"{result['accuracy_percent']:>11.2f}%"
            f"{result['kappa']:>12.3f}"
        )

    print("-" * 34)

    print(
        f"{'Mean':<10}"
        f"{overall_summary['mean_accuracy_percent']:>11.2f}%"
        f"{overall_summary['mean_kappa']:>12.3f}"
    )

    print()
    print(
        "Accuracy SD: "
        f"{overall_summary['std_accuracy_percent']:.2f}%"
    )

    print(
        "Kappa SD: "
        f"{overall_summary['std_kappa']:.3f}"
    )

    print(
        "Total duration: "
        f"{total_duration / 60.0:.1f} minutes"
    )

    print()
    print("Saved:")
    print(
        f"  {SUBJECT_RESULTS_PATH}"
    )
    print(
        f"  {FOLD_RESULTS_PATH}"
    )
    print(
        f"  {PREDICTIONS_PATH}"
    )
    print(
        f"  {TRAINING_HISTORY_PATH}"
    )
    print(
        f"  {OVERALL_SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
