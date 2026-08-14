"""
Run EEGNet-style leave-one-subject-out evaluation.

Example fold:

    Train:
        A01T to A08T

    Test:
        A09T

The held-out subject is never used for:

    - normalisation
    - training
    - validation
    - model selection

Run one fold:

    python -m scripts.eegnet.run_eegnet_loso \
        --test-subject A09

Run all nine folds:

    python -m scripts.eegnet.run_eegnet_loso
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bci_wheelchair.models.eegnet import (
    EEGNet,
    EEGNetConfig,
)


SUBJECTS = [
    f"A{subject_number:02d}"
    for subject_number in range(1, 10)
]


@dataclass(frozen=True)
class TrainingConfig:
    """EEGNet training configuration."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    patience: int = 15
    random_state: int = 42


class ChannelStandardizer:
    """
    Standardise each EEG channel using training data only.

    One mean and standard deviation are calculated per channel.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
    ) -> "ChannelStandardizer":
        """Learn channel statistics from training EEG."""
        X = self._validate(X)

        self.mean_ = X.mean(
            axis=(0, 2),
            keepdims=True,
            dtype=np.float64,
        ).astype(np.float32)

        self.scale_ = X.std(
            axis=(0, 2),
            keepdims=True,
            dtype=np.float64,
        ).astype(np.float32)

        self.scale_[self.scale_ < 1e-8] = 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted channel normalisation."""
        X = self._validate(X)

        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "Standardizer must be fitted first."
            )

        return (
            (X.astype(np.float32) - self.mean_)
            / self.scale_
        ).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform EEG."""
        return self.fit(X).transform(X)

    @staticmethod
    def _validate(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        if X.ndim != 3:
            raise ValueError(
                "EEG must have shape "
                "(trials, channels, time), "
                f"but received {X.shape}."
            )

        return X


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run EEGNet LOSO evaluation."
    )

    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path(
            "data/processed/8-30Hz_0.5-2.5s"
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/eegnet_loso"),
    )

    parser.add_argument(
        "--test-subject",
        type=str,
        default=None,
        help=(
            "Run one held-out subject, such as A09. "
            "Omit to run all folds."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--temporal-filters",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--depth-multiplier",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--separable-filters",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
    )

    return parser.parse_args()


def choose_device() -> torch.device:
    """Use Apple MPS, CUDA or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    """Set reproducible random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subject(
    data_directory: Path,
    subject: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one subject's training session."""
    path = data_directory / f"{subject}T.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Subject file was not found: {path}"
        )

    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"])

    if X.ndim != 3:
        raise ValueError(
            f"Invalid EEG shape in {path}: {X.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"Trial and label counts differ in {path}."
        )

    return X, y


def combine_subjects(
    data_directory: Path,
    subjects: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine EEG trials from several subjects."""
    eeg_arrays: list[np.ndarray] = []
    label_arrays: list[np.ndarray] = []

    for subject in subjects:
        X_subject, y_subject = load_subject(
            data_directory,
            subject,
        )

        eeg_arrays.append(X_subject)
        label_arrays.append(y_subject)

    return (
        np.concatenate(eeg_arrays, axis=0),
        np.concatenate(label_arrays, axis=0),
    )


def create_label_mapping(
    labels: np.ndarray,
) -> tuple[dict[str, int], list[str]]:
    """Create a stable string-to-index label mapping."""
    class_names = sorted(
        str(label)
        for label in np.unique(labels)
    )

    mapping = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    return mapping, class_names


def encode_labels(
    labels: np.ndarray,
    mapping: dict[str, int],
) -> np.ndarray:
    """Convert text labels to integer indices."""
    unknown = sorted(
        {
            str(label)
            for label in labels
            if str(label) not in mapping
        }
    )

    if unknown:
        raise ValueError(
            f"Unknown labels found: {unknown}"
        )

    return np.asarray(
        [
            mapping[str(label)]
            for label in labels
        ],
        dtype=np.int64,
    )


def create_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch DataLoader."""
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def run_epoch(
    model: EEGNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Run one training or validation epoch."""
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    trial_count = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = X_batch.shape[0]

        total_loss += loss.item() * batch_size
        correct += (
            logits.argmax(dim=1) == y_batch
        ).sum().item()

        trial_count += batch_size

    return (
        total_loss / trial_count,
        correct / trial_count,
    )


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_config: EEGNetConfig,
    training_config: TrainingConfig,
) -> tuple[
    EEGNet,
    ChannelStandardizer,
    pd.DataFrame,
]:
    """Train EEGNet using training subjects only."""
    set_random_seed(training_config.random_state)

    indices = np.arange(len(X_train))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=training_config.validation_fraction,
        random_state=training_config.random_state,
        stratify=y_train,
    )

    X_model_train = X_train[train_indices]
    y_model_train = y_train[train_indices]

    X_validation = X_train[validation_indices]
    y_validation = y_train[validation_indices]

    standardizer = ChannelStandardizer()

    X_model_train = standardizer.fit_transform(
        X_model_train
    )

    X_validation = standardizer.transform(
        X_validation
    )

    train_loader = create_loader(
        X_model_train,
        y_model_train,
        training_config.batch_size,
        shuffle=True,
    )

    validation_loader = create_loader(
        X_validation,
        y_validation,
        training_config.batch_size,
        shuffle=False,
    )

    device = choose_device()

    print(f"Device: {device}")
    print(f"Training trials: {len(X_model_train)}")
    print(f"Validation trials: {len(X_validation)}")

    model = EEGNet(model_config).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    history_rows: list[dict[str, float | int]] = []

    for epoch in range(
        1,
        training_config.epochs + 1,
    ):
        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        with torch.no_grad():
            validation_loss, validation_accuracy = (
                run_epoch(
                    model=model,
                    loader=validation_loader,
                    criterion=criterion,
                    device=device,
                    optimizer=None,
                )
            )

        improved = (
            validation_loss
            < best_validation_loss - 1e-6
        )

        marker = ""

        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0
            marker = " *"
        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "epoch": epoch,
                "training_loss": train_loss,
                "validation_loss": validation_loss,
                "training_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_loss:.4f} | "
            f"val loss={validation_loss:.4f} | "
            f"train accuracy="
            f"{train_accuracy * 100:.2f}% | "
            f"val accuracy="
            f"{validation_accuracy * 100:.2f}%"
            f"{marker}"
        )

        if (
            epochs_without_improvement
            >= training_config.patience
        ):
            print(
                "Early stopping: validation loss "
                "did not improve for "
                f"{training_config.patience} epochs."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "No valid EEGNet checkpoint was produced."
        )

    model.load_state_dict(best_state)
    model.eval()

    print(
        f"Best epoch: {best_epoch} | "
        f"best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    history_dataframe = pd.DataFrame(
        history_rows
    )

    history_dataframe.attrs["best_epoch"] = best_epoch
    history_dataframe.attrs[
        "best_validation_loss"
    ] = best_validation_loss

    return (
        model,
        standardizer,
        history_dataframe,
    )


def predict(
    model: EEGNet,
    standardizer: ChannelStandardizer,
    X: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """Predict labels for unseen EEG trials."""
    device = next(model.parameters()).device

    X_standardized = standardizer.transform(X)

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(
                X_standardized.astype(np.float32)
            )
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    prediction_batches: list[np.ndarray] = []

    model.eval()

    with torch.no_grad():
        for (X_batch,) in loader:
            logits = model(X_batch.to(device))

            prediction_batches.append(
                logits.argmax(dim=1)
                .cpu()
                .numpy()
            )

    return np.concatenate(
        prediction_batches,
        axis=0,
    )


def run_fold(
    held_out_subject: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Run one EEGNet LOSO fold."""
    training_subjects = [
        subject
        for subject in SUBJECTS
        if subject != held_out_subject
    ]

    print()
    print("=" * 76)
    print(f"Held-out subject: {held_out_subject}")
    print(
        "Training subjects: "
        f"{', '.join(training_subjects)}"
    )
    print("=" * 76)

    X_train, y_train_text = combine_subjects(
        args.data_directory,
        training_subjects,
    )

    X_test, y_test_text = load_subject(
        args.data_directory,
        held_out_subject,
    )

    mapping, class_names = create_label_mapping(
        y_train_text
    )

    y_train = encode_labels(
        y_train_text,
        mapping,
    )

    y_test = encode_labels(
        y_test_text,
        mapping,
    )

    print(f"Training EEG: {X_train.shape}")
    print(f"Held-out EEG: {X_test.shape}")
    print(f"Classes: {class_names}")

    model_config = EEGNetConfig(
        n_channels=X_train.shape[1],
        n_times=X_train.shape[2],
        n_classes=len(class_names),
        temporal_filters=args.temporal_filters,
        depth_multiplier=args.depth_multiplier,
        separable_filters=args.separable_filters,
        dropout=args.dropout,
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=(
            args.validation_fraction
        ),
        patience=args.patience,
        random_state=args.random_state,
    )

    (
        model,
        standardizer,
        history,
    ) = train_model(
        X_train=X_train,
        y_train=y_train,
        model_config=model_config,
        training_config=training_config,
    )

    predictions = predict(
        model=model,
        standardizer=standardizer,
        X=X_test,
        batch_size=args.batch_size,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    kappa = cohen_kappa_score(
        y_test,
        predictions,
    )

    print()
    print(
        f"EEGNet | accuracy={accuracy * 100:.2f}% "
        f"| kappa={kappa:.4f}"
    )

    fold_directory = (
        args.output_directory / held_out_subject
    )

    fold_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_csv(
        fold_directory / "training_history.csv",
        index=False,
    )

    predicted_text = np.asarray(
        [
            class_names[int(prediction)]
            for prediction in predictions
        ]
    )

    prediction_dataframe = pd.DataFrame(
        {
            "true_label": y_test_text,
            "true_label_index": y_test,
            "predicted_label": predicted_text,
            "predicted_label_index": predictions,
        }
    )

    prediction_dataframe.to_csv(
        fold_directory / "predictions.csv",
        index=False,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(class_names)),
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[
            f"true_{class_name}"
            for class_name in class_names
        ],
        columns=[
            f"predicted_{class_name}"
            for class_name in class_names
        ],
    )

    matrix_dataframe.to_csv(
        fold_directory / "confusion_matrix.csv"
    )

    best_epoch = int(
        history.attrs["best_epoch"]
    )

    best_validation_loss = float(
        history.attrs["best_validation_loss"]
    )

    if args.save_checkpoints:
        if standardizer.mean_ is None:
            raise RuntimeError(
                "Standardizer mean is missing."
            )

        if standardizer.scale_ is None:
            raise RuntimeError(
                "Standardizer scale is missing."
            )

        checkpoint_path = (
            fold_directory
            / f"{held_out_subject}_eegnet.pt"
        )

        torch.save(
            {
                "model_state_dict": (
                    model.state_dict()
                ),
                "model_config": (
                    asdict(model_config)
                ),
                "training_config": (
                    asdict(training_config)
                ),
                "standardizer_mean": (
                    standardizer.mean_
                ),
                "standardizer_scale": (
                    standardizer.scale_
                ),
                "class_names": class_names,
                "best_epoch": best_epoch,
                "best_validation_loss": (
                    best_validation_loss
                ),
            },
            checkpoint_path,
        )

        print(
            f"Saved checkpoint: {checkpoint_path}"
        )

    return {
        "held_out_subject": held_out_subject,
        "model": "eegnet",
        "accuracy": float(accuracy),
        "accuracy_percent": float(
            accuracy * 100
        ),
        "kappa": float(kappa),
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
        "n_training_trials": len(X_train),
        "n_test_trials": len(X_test),
    }


def create_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise EEGNet performance."""
    summary = pd.DataFrame(
        [
            {
                "model": "eegnet",
                "mean_accuracy_percent": (
                    results["accuracy_percent"].mean()
                ),
                "std_accuracy_percent": (
                    results["accuracy_percent"].std()
                ),
                "mean_kappa": (
                    results["kappa"].mean()
                ),
                "std_kappa": (
                    results["kappa"].std()
                ),
                "subjects": (
                    results[
                        "held_out_subject"
                    ].nunique()
                ),
            }
        ]
    )

    summary[
        [
            "std_accuracy_percent",
            "std_kappa",
        ]
    ] = summary[
        [
            "std_accuracy_percent",
            "std_kappa",
        ]
    ].fillna(0.0)

    return summary


def main() -> None:
    """Run one or all EEGNet LOSO folds."""
    args = parse_arguments()

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.test_subject is None:
        held_out_subjects = SUBJECTS
    else:
        held_out_subject = (
            args.test_subject.upper()
        )

        if held_out_subject not in SUBJECTS:
            raise ValueError(
                f"Unknown subject: {held_out_subject}"
            )

        held_out_subjects = [
            held_out_subject
        ]

    results = [
        run_fold(subject, args)
        for subject in held_out_subjects
    ]

    results_dataframe = pd.DataFrame(results)

    results_path = (
        args.output_directory
        / "eegnet_loso_results.csv"
    )

    results_dataframe.to_csv(
        results_path,
        index=False,
    )

    summary_dataframe = create_summary(
        results_dataframe
    )

    summary_path = (
        args.output_directory
        / "eegnet_loso_summary.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    configuration = {
        "evaluation": "EEGNet LOSO",
        "data_directory": str(
            args.data_directory
        ),
        "held_out_subjects": (
            held_out_subjects
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": (
            args.learning_rate
        ),
        "weight_decay": (
            args.weight_decay
        ),
        "validation_fraction": (
            args.validation_fraction
        ),
        "patience": args.patience,
        "dropout": args.dropout,
        "temporal_filters": (
            args.temporal_filters
        ),
        "depth_multiplier": (
            args.depth_multiplier
        ),
        "separable_filters": (
            args.separable_filters
        ),
        "random_state": (
            args.random_state
        ),
    }

    configuration_path = (
        args.output_directory
        / "experiment_configuration.json"
    )

    with configuration_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
        )

    print()
    print("=" * 76)
    print("EEGNET LOSO RESULTS")
    print("=" * 76)

    print(
        results_dataframe[
            [
                "held_out_subject",
                "accuracy_percent",
                "kappa",
                "best_epoch",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)

    print(
        summary_dataframe.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(f"Saved results: {results_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
