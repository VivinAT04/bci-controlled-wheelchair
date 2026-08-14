"""
Run supervised-contrastive EEGNet LOSO evaluation.

Training example:

    A01T to A08T -> training
    A09T         -> unseen test subject

Run one fold:

    python -m scripts.eegnet.run_contrastive_eegnet_loso \
        --test-subject A09

Run all folds:

    python -m scripts.eegnet.run_contrastive_eegnet_loso \
        --save-checkpoints
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

from bci_wheelchair.models.contrastive_eegnet import (
    ContrastiveEEGNet,
    ContrastiveEEGNetConfig,
    SupervisedContrastiveLoss,
)


SUBJECTS = [
    f"A{number:02d}"
    for number in range(1, 10)
]


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    patience: int = 15
    random_state: int = 42

    contrastive_weight: float = 0.2
    temperature: float = 0.1
    noise_standard_deviation: float = 0.02
    time_mask_fraction: float = 0.05


class ChannelStandardizer:
    """Standardise each channel using training data only."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
    ) -> "ChannelStandardizer":
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

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        X = self._validate(X)

        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError(
                "Standardizer has not been fitted."
            )

        return (
            (X.astype(np.float32) - self.mean_)
            / self.scale_
        ).astype(np.float32)

    def fit_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        return self.fit(X).transform(X)

    @staticmethod
    def _validate(
        X: np.ndarray,
    ) -> np.ndarray:
        X = np.asarray(X)

        if X.ndim != 3:
            raise ValueError(
                "Expected EEG shape "
                "(trials, channels, time), "
                f"received {X.shape}."
            )

        return X


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run supervised-contrastive EEGNet LOSO."
        )
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
        default=Path(
            "results/contrastive_eegnet_loso"
        ),
    )

    parser.add_argument(
        "--test-subject",
        type=str,
        default=None,
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
        "--projection-hidden-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--projection-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--noise-standard-deviation",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--time-mask-fraction",
        type=float,
        default=0.05,
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
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_subject_file(
    data_directory: Path,
    subject: str,
) -> Path:
    """Find a subject's training-session NPZ file."""
    preferred_path = data_directory / f"{subject}T.npz"

    if preferred_path.exists():
        return preferred_path

    matches = sorted(
        data_directory.glob(f"{subject}T*.npz")
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find {subject}T.npz in "
            f"{data_directory}."
        )

    return matches[0]


def load_subject(
    data_directory: Path,
    subject: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one subject's training session."""
    path = find_subject_file(
        data_directory,
        subject,
    )

    with np.load(path) as data:
        X = np.asarray(
            data["X"],
            dtype=np.float32,
        )
        y = np.asarray(data["y"]).astype(str)

    if X.ndim != 3:
        raise ValueError(
            f"{path} has invalid EEG shape {X.shape}."
        )

    if len(X) != len(y):
        raise ValueError(
            f"{path} has {len(X)} trials but "
            f"{len(y)} labels."
        )

    return X, y


def create_label_mapping(
    labels: np.ndarray,
) -> dict[str, int]:
    """Create stable integer labels."""
    unique_labels = sorted(
        np.unique(labels).tolist()
    )

    return {
        label: index
        for index, label in enumerate(unique_labels)
    }


def encode_labels(
    labels: np.ndarray,
    mapping: dict[str, int],
) -> np.ndarray:
    """Convert string labels to integer labels."""
    unknown = sorted(
        set(labels.tolist()) - set(mapping)
    )

    if unknown:
        raise ValueError(
            f"Unknown test labels: {unknown}"
        )

    return np.asarray(
        [mapping[label] for label in labels],
        dtype=np.int64,
    )


def load_loso_data(
    data_directory: Path,
    test_subject: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int],
]:
    """Load training subjects and one held-out subject."""
    training_X: list[np.ndarray] = []
    training_y: list[np.ndarray] = []

    for subject in SUBJECTS:
        X, y = load_subject(
            data_directory,
            subject,
        )

        if subject == test_subject:
            test_X = X
            test_labels = y
        else:
            training_X.append(X)
            training_y.append(y)

    train_X = np.concatenate(
        training_X,
        axis=0,
    )

    train_labels = np.concatenate(
        training_y,
        axis=0,
    )

    label_mapping = create_label_mapping(
        train_labels
    )

    train_y = encode_labels(
        train_labels,
        label_mapping,
    )

    test_y = encode_labels(
        test_labels,
        label_mapping,
    )

    return (
        train_X,
        train_y,
        test_X,
        test_y,
        label_mapping,
    )


def augment_eeg(
    eeg: torch.Tensor,
    noise_standard_deviation: float,
    time_mask_fraction: float,
) -> torch.Tensor:
    """
    Produce a mild EEG augmentation.

    Augmentation:
    - small Gaussian noise
    - one short random temporal mask
    """
    augmented = eeg.clone()

    if noise_standard_deviation > 0:
        augmented = augmented + (
            torch.randn_like(augmented)
            * noise_standard_deviation
        )

    n_times = augmented.shape[-1]
    mask_length = max(
        1,
        int(n_times * time_mask_fraction),
    )

    if mask_length < n_times:
        maximum_start = n_times - mask_length

        starts = torch.randint(
            low=0,
            high=maximum_start + 1,
            size=(augmented.shape[0],),
            device=augmented.device,
        )

        for trial_index, start in enumerate(starts):
            start_index = int(start.item())
            end_index = start_index + mask_length

            augmented[
                trial_index,
                :,
                start_index:end_index,
            ] = 0.0

    return augmented


def create_data_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch data loader."""
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def evaluate_model(
    model: ContrastiveEEGNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    """Evaluate classification performance."""
    model.eval()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for eeg, labels in loader:
            eeg = eeg.to(device)
            labels = labels.to(device)

            logits, _, _ = model(eeg)
            predicted = logits.argmax(dim=1)

            predictions.append(
                predicted.cpu().numpy()
            )
            targets.append(
                labels.cpu().numpy()
            )

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
    )

    return accuracy, kappa, y_true, y_pred


def train_one_fold(
    test_subject: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    """Train and evaluate one LOSO fold."""
    print()
    print("=" * 70)
    print(f"Held-out subject: {test_subject}")
    print("=" * 70)

    set_random_seed(args.random_state)

    (
        full_train_X,
        full_train_y,
        test_X,
        test_y,
        label_mapping,
    ) = load_loso_data(
        args.data_directory,
        test_subject,
    )

    (
        train_X,
        validation_X,
        train_y,
        validation_y,
    ) = train_test_split(
        full_train_X,
        full_train_y,
        test_size=args.validation_fraction,
        random_state=args.random_state,
        stratify=full_train_y,
    )

    standardizer = ChannelStandardizer()

    train_X = standardizer.fit_transform(train_X)
    validation_X = standardizer.transform(validation_X)
    test_X = standardizer.transform(test_X)

    print(f"Train:      {train_X.shape}")
    print(f"Validation: {validation_X.shape}")
    print(f"Test:       {test_X.shape}")
    print(f"Labels:     {label_mapping}")

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        random_state=args.random_state,
        contrastive_weight=args.contrastive_weight,
        temperature=args.temperature,
        noise_standard_deviation=(
            args.noise_standard_deviation
        ),
        time_mask_fraction=args.time_mask_fraction,
    )

    model_config = ContrastiveEEGNetConfig(
        n_channels=train_X.shape[1],
        n_times=train_X.shape[2],
        n_classes=len(label_mapping),
        temporal_filters=args.temporal_filters,
        depth_multiplier=args.depth_multiplier,
        separable_filters=args.separable_filters,
        dropout=args.dropout,
        projection_hidden_size=(
            args.projection_hidden_size
        ),
        projection_size=args.projection_size,
    )

    model = ContrastiveEEGNet(
        model_config
    ).to(device)

    train_loader = create_data_loader(
        train_X,
        train_y,
        args.batch_size,
        shuffle=True,
    )

    validation_loader = create_data_loader(
        validation_X,
        validation_y,
        args.batch_size,
        shuffle=False,
    )

    test_loader = create_data_loader(
        test_X,
        test_y,
        args.batch_size,
        shuffle=False,
    )

    classification_loss_function = (
        nn.CrossEntropyLoss()
    )

    contrastive_loss_function = (
        SupervisedContrastiveLoss(
            temperature=args.temperature,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_validation_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_classification_loss = 0.0
        total_contrastive_loss = 0.0
        total_examples = 0
        total_correct = 0

        for eeg, labels in train_loader:
            eeg = eeg.to(device)
            labels = labels.to(device)

            first_view = augment_eeg(
                eeg,
                args.noise_standard_deviation,
                args.time_mask_fraction,
            )

            second_view = augment_eeg(
                eeg,
                args.noise_standard_deviation,
                args.time_mask_fraction,
            )

            combined_eeg = torch.cat(
                [first_view, second_view],
                dim=0,
            )

            combined_labels = torch.cat(
                [labels, labels],
                dim=0,
            )

            optimizer.zero_grad()

            (
                combined_logits,
                _,
                combined_projection,
            ) = model(combined_eeg)

            classification_loss = (
                classification_loss_function(
                    combined_logits,
                    combined_labels,
                )
            )

            contrastive_loss = (
                contrastive_loss_function(
                    combined_projection,
                    combined_labels,
                )
            )

            loss = (
                classification_loss
                + args.contrastive_weight
                * contrastive_loss
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            batch_size = combined_labels.shape[0]

            total_loss += loss.item() * batch_size
            total_classification_loss += (
                classification_loss.item()
                * batch_size
            )
            total_contrastive_loss += (
                contrastive_loss.item()
                * batch_size
            )

            total_examples += batch_size

            predicted = combined_logits.argmax(dim=1)

            total_correct += (
                predicted == combined_labels
            ).sum().item()

        training_loss = (
            total_loss / total_examples
        )

        training_classification_loss = (
            total_classification_loss
            / total_examples
        )

        training_contrastive_loss = (
            total_contrastive_loss
            / total_examples
        )

        training_accuracy = (
            total_correct / total_examples
        )

        (
            validation_accuracy,
            validation_kappa,
            _,
            _,
        ) = evaluate_model(
            model,
            validation_loader,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "classification_loss": (
                    training_classification_loss
                ),
                "contrastive_loss": (
                    training_contrastive_loss
                ),
                "training_accuracy": (
                    training_accuracy
                ),
                "validation_accuracy": (
                    validation_accuracy
                ),
                "validation_kappa": (
                    validation_kappa
                ),
            }
        )

        print(
            f"Epoch {epoch:03d} "
            f"| loss {training_loss:.4f} "
            f"| CE {training_classification_loss:.4f} "
            f"| SupCon {training_contrastive_loss:.4f} "
            f"| train {training_accuracy * 100:.2f}% "
            f"| val {validation_accuracy * 100:.2f}%"
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):
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
            >= args.patience
        ):
            print(
                f"Early stopping after epoch {epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a model state."
        )

    model.load_state_dict(best_state)

    (
        test_accuracy,
        test_kappa,
        y_true,
        y_pred,
    ) = evaluate_model(
        model,
        test_loader,
        device,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(label_mapping)),
    )

    print()
    print(f"Best epoch: {best_epoch}")
    print(
        "Best validation accuracy: "
        f"{best_validation_accuracy * 100:.4f}%"
    )
    print(
        f"Test accuracy: {test_accuracy * 100:.4f}%"
    )
    print(f"Test kappa: {test_kappa:.4f}")

    fold_directory = (
        args.output_directory / test_subject
    )

    fold_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(history).to_csv(
        fold_directory / "training_history.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_label": y_pred,
        }
    ).to_csv(
        fold_directory / "predictions.csv",
        index=False,
    )

    pd.DataFrame(matrix).to_csv(
        fold_directory / "confusion_matrix.csv",
        index=False,
    )

    metadata = {
        "held_out_subject": test_subject,
        "training_config": asdict(training_config),
        "model_config": asdict(model_config),
        "label_mapping": label_mapping,
        "best_epoch": best_epoch,
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
        "test_accuracy": test_accuracy,
        "test_kappa": test_kappa,
    }

    with (
        fold_directory / "metadata.json"
    ).open("w") as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    if args.save_checkpoints:
        torch.save(
            {
                "model_state_dict": best_state,
                "model_config": asdict(model_config),
                "label_mapping": label_mapping,
                "standardizer_mean": (
                    standardizer.mean_
                ),
                "standardizer_scale": (
                    standardizer.scale_
                ),
            },
            fold_directory / "best_model.pt",
        )

    return {
        "held_out_subject": test_subject,
        "accuracy": test_accuracy,
        "accuracy_percent": (
            test_accuracy * 100
        ),
        "kappa": test_kappa,
        "best_epoch": best_epoch,
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
    }


def main() -> None:
    """Run one or all LOSO folds."""
    args = parse_arguments()

    if not args.data_directory.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: "
            f"{args.data_directory}"
        )

    if args.test_subject is not None:
        args.test_subject = (
            args.test_subject.upper()
        )

        if args.test_subject not in SUBJECTS:
            raise ValueError(
                "--test-subject must be from "
                "A01 to A09."
            )

        subjects = [args.test_subject]
    else:
        subjects = SUBJECTS

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = choose_device()

    print(f"Device: {device}")
    print(f"Data: {args.data_directory}")
    print(f"Output: {args.output_directory}")
    print(
        "Loss = cross-entropy + "
        f"{args.contrastive_weight} × "
        "supervised contrastive loss"
    )

    results: list[dict[str, object]] = []

    for subject in subjects:
        result = train_one_fold(
            subject,
            args,
            device,
        )

        results.append(result)

        pd.DataFrame(results).to_csv(
            args.output_directory
            / "contrastive_eegnet_loso_results.csv",
            index=False,
        )

    results_frame = pd.DataFrame(results)

    summary = pd.DataFrame(
        [
            {
                "number_of_subjects": len(results_frame),
                "mean_accuracy": (
                    results_frame["accuracy"].mean()
                ),
                "mean_accuracy_percent": (
                    results_frame[
                        "accuracy_percent"
                    ].mean()
                ),
                "standard_deviation_percent": (
                    results_frame[
                        "accuracy_percent"
                    ].std(ddof=0)
                ),
                "mean_kappa": (
                    results_frame["kappa"].mean()
                ),
            }
        ]
    )

    summary.to_csv(
        args.output_directory
        / "contrastive_eegnet_loso_summary.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        results_frame[
            [
                "held_out_subject",
                "accuracy_percent",
                "kappa",
                "best_epoch",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Mean accuracy: "
        f"{summary.iloc[0]['mean_accuracy_percent']:.4f}%"
    )

    print(
        "Standard deviation: "
        f"{summary.iloc[0]['standard_deviation_percent']:.4f}%"
    )

    print(
        "Mean kappa: "
        f"{summary.iloc[0]['mean_kappa']:.4f}"
    )


if __name__ == "__main__":
    main()
