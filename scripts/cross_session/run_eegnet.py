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
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bci_wheelchair.data.processed_loading import load_processed_subject
from bci_wheelchair.models import make_eegnet


SUBJECTS = [
    "A01", "A02", "A03", "A04", "A05",
    "A06", "A07", "A08", "A09",
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
    "results/cross_session/eegnet"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "eegnet_cross_session_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "eegnet_cross_session_predictions.csv"
)

TRAINING_HISTORY_PATH = (
    RESULTS_DIRECTORY
    / "eegnet_cross_session_training_history.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "eegnet_cross_session_overall_summary.csv"
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def save_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

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
        )

        writer.writeheader()
        writer.writerows(rows)


def encode_labels(
    labels: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            CLASS_TO_INDEX[str(label)]
            for label in labels
        ],
        dtype=np.int64,
    )


def decode_labels(
    labels: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            INDEX_TO_CLASS[int(label)]
            for label in labels
        ]
    )


def calculate_channel_statistics(
    X_train: np.ndarray,
):
    mean = X_train.mean(
        axis=(0, 2),
        keepdims=True,
    )

    std = X_train.std(
        axis=(0, 2),
        keepdims=True,
    )

    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    return mean, std


def standardise(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        (X - mean) / std,
        dtype=np.float32,
    )


class EEGDataset(Dataset):

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

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            self.X[index],
            self.y[index],
        )


def create_loader(
    X,
    y,
    shuffle,
    seed,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        EEGDataset(X, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def initialise_model(
    n_channels,
    n_samples,
    device,
):
    return make_eegnet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=4,
        dropout_rate=DROPOUT_RATE,
        f1=F1,
        depth_multiplier=DEPTH_MULTIPLIER,
        f2=F2,
        temporal_kernel_size=TEMPORAL_KERNEL_SIZE,
        separable_kernel_size=SEPARABLE_KERNEL_SIZE,
        device=device,
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

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

        correct += int(
            (
                predictions
                == y_batch
            ).sum().item()
        )

        total += batch_size

    return (
        total_loss / total,
        correct / total,
    )


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total = 0

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

        probs = torch.softmax(
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

        total += batch_size

        true_labels.append(
            y_batch.cpu().numpy()
        )

        predicted_labels.append(
            predictions.cpu().numpy()
        )

        probabilities.append(
            probs.cpu().numpy()
        )

    y_true = np.concatenate(
        true_labels
    )

    y_pred = np.concatenate(
        predicted_labels
    )

    probabilities = np.concatenate(
        probabilities,
        axis=0,
    )

    return (
        total_loss / total,
        accuracy_score(
            y_true,
            y_pred,
        ),
        y_true,
        y_pred,
        probabilities,
    )


def run_subject(
    subject,
    device,
):
    train_subject = f"{subject}T"
    test_subject = f"{subject}E"

    seed = (
        RANDOM_SEED
        + SUBJECTS.index(subject)
    )

    set_random_seed(seed)

    print()
    print("=" * 78)
    print(
        f"{subject}: "
        f"{train_subject} -> "
        f"{test_subject}"
    )
    print("=" * 78)

    X_train_session, y_train_labels = (
        load_processed_subject(
            subject=train_subject,
            config=PREPROCESSING,
        )
    )

    X_test, y_test_labels = (
        load_processed_subject(
            subject=test_subject,
            config=PREPROCESSING,
        )
    )

    X_train_session = np.asarray(
        X_train_session,
        dtype=np.float32,
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float32,
    )

    y_train_session = encode_labels(
        np.asarray(y_train_labels)
    )

    y_test = encode_labels(
        np.asarray(y_test_labels)
    )

    indices = np.arange(
        len(y_train_session)
    )

    (
        training_indices,
        validation_indices,
    ) = train_test_split(
        indices,
        test_size=VALIDATION_FRACTION,
        random_state=seed,
        stratify=y_train_session,
    )

    X_train = X_train_session[
        training_indices
    ]

    y_train = y_train_session[
        training_indices
    ]

    X_validation = X_train_session[
        validation_indices
    ]

    y_validation = y_train_session[
        validation_indices
    ]

    print(
        f"Training trials:   "
        f"{len(y_train)}"
    )

    print(
        f"Validation trials: "
        f"{len(y_validation)}"
    )

    print(
        f"Testing trials:    "
        f"{len(y_test)}"
    )

    mean, std = (
        calculate_channel_statistics(
            X_train
        )
    )

    X_train = standardise(
        X_train,
        mean,
        std,
    )

    X_validation = standardise(
        X_validation,
        mean,
        std,
    )

    X_test = standardise(
        X_test,
        mean,
        std,
    )

    train_loader = create_loader(
        X_train,
        y_train,
        True,
        seed,
    )

    validation_loader = create_loader(
        X_validation,
        y_validation,
        False,
        seed,
    )

    test_loader = create_loader(
        X_test,
        y_test,
        False,
        seed,
    )

    model = initialise_model(
        X_train.shape[1],
        X_train.shape[2],
        device,
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=8,
        )
    )

    best_validation_loss = (
        float("inf")
    )

    best_validation_accuracy = 0.0
    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0

    history_rows = []

    start_time = time.time()

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

        history_rows.append(
            {
                "subject": subject,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": (
                    validation_loss
                ),
                "validation_accuracy": (
                    validation_accuracy
                ),
                "validation_kappa": (
                    validation_kappa
                ),
                "learning_rate": (
                    optimizer
                    .param_groups[0]["lr"]
                ),
            }
        )

        improved = (
            validation_loss
            < best_validation_loss
            - 1e-4
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
            and
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(
            f"No checkpoint for {subject}"
        )

    model.load_state_dict(
        best_state
    )

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
        labels=np.arange(4),
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(
        test_true,
        test_pred,
        labels=np.arange(4),
    )

    result = {
        "subject": subject,
        "train_session": train_subject,
        "test_session": test_subject,
        "training_trials": len(y_train),
        "validation_trials": len(
            y_validation
        ),
        "testing_trials": len(y_test),
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
        "duration_seconds": (
            time.time() - start_time
        ),
        "preprocessing": PREPROCESSING,
        "model": "EEGNet",
        "evaluation": (
            "cross_session_AxxT_to_AxxE"
        ),
    }

    for true_index, true_label in enumerate(
        CLASS_ORDER
    ):
        for pred_index, pred_label in enumerate(
            CLASS_ORDER
        ):
            result[
                f"cm_{true_label}_pred_"
                f"{pred_label}"
            ] = int(
                matrix[
                    true_index,
                    pred_index,
                ]
            )

    decoded_true = decode_labels(
        test_true
    )

    decoded_pred = decode_labels(
        test_pred
    )

    prediction_rows = []

    for trial_index, (
        true_label,
        predicted_label,
        probabilities,
    ) in enumerate(
        zip(
            decoded_true,
            decoded_pred,
            test_probabilities,
        ),
        start=1,
    ):
        prediction_rows.append(
            {
                "subject": subject,
                "trial": trial_index,
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
                        probabilities[0]
                    )
                ),
                "right_hand_probability": (
                    float(
                        probabilities[1]
                    )
                ),
                "feet_probability": (
                    float(
                        probabilities[2]
                    )
                ),
                "tongue_probability": (
                    float(
                        probabilities[3]
                    )
                ),
            }
        )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"{subject} Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"{subject} Kappa:    "
        f"{test_kappa:.3f}"
    )

    return (
        result,
        prediction_rows,
        history_rows,
    )


def main():
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
    print("Cross-Session EEGNet")
    print("=" * 78)
    print("Protocol: AxxT -> AxxE")
    print("Preprocessing: 8-30 Hz")
    print(f"Device: {device}")

    subject_results = []
    prediction_rows = []
    history_rows = []

    experiment_start = time.time()

    for subject in SUBJECTS:
        (
            result,
            predictions,
            history,
        ) = run_subject(
            subject,
            device,
        )

        subject_results.append(
            result
        )

        prediction_rows.extend(
            predictions
        )

        history_rows.extend(
            history
        )

        save_csv(
            SUBJECT_RESULTS_PATH,
            subject_results,
        )

        save_csv(
            PREDICTIONS_PATH,
            prediction_rows,
        )

        save_csv(
            TRAINING_HISTORY_PATH,
            history_rows,
        )

    accuracies = np.asarray(
        [
            result["accuracy"]
            for result in subject_results
        ],
        dtype=float,
    )

    kappas = np.asarray(
        [
            result["kappa"]
            for result in subject_results
        ],
        dtype=float,
    )

    summary = {
        "model": "EEGNet",
        "evaluation": (
            "cross_session_AxxT_to_AxxE"
        ),
        "subjects": 9,
        "preprocessing": PREPROCESSING,
        "mean_accuracy": float(
            accuracies.mean()
        ),
        "mean_accuracy_percent": float(
            accuracies.mean() * 100
        ),
        "std_accuracy_percent": float(
            accuracies.std() * 100
        ),
        "mean_kappa": float(
            kappas.mean()
        ),
        "std_kappa": float(
            kappas.std()
        ),
        "minimum_accuracy_percent": float(
            accuracies.min() * 100
        ),
        "maximum_accuracy_percent": float(
            accuracies.max() * 100
        ),
        "total_duration_minutes": (
            time.time()
            - experiment_start
        ) / 60.0,
    }

    save_csv(
        OVERALL_SUMMARY_PATH,
        [summary],
    )

    print()
    print("=" * 78)
    print(
        "FINAL EEGNET CROSS-SESSION RESULTS"
    )
    print("=" * 78)

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
        f"{summary['mean_accuracy_percent']:>11.2f}%"
        f"{summary['mean_kappa']:>12.3f}"
    )

    print()
    print(
        "Accuracy SD: "
        f"{summary['std_accuracy_percent']:.2f}%"
    )

    print(
        "Kappa SD: "
        f"{summary['std_kappa']:.3f}"
    )

    print(
        "Total duration: "
        f"{summary['total_duration_minutes']:.1f} minutes"
    )

    print()
    print("Saved:")
    print(SUBJECT_RESULTS_PATH)
    print(PREDICTIONS_PATH)
    print(TRAINING_HISTORY_PATH)
    print(OVERALL_SUMMARY_PATH)


if __name__ == "__main__":
    main()
