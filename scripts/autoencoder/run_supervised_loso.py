"""
Run LOSO cross-subject classification using supervised autoencoder features.

Example:

    Train autoencoder:
        A01T to A08T

    Held-out subject:
        A09T

The held-out subject is never used for:

    - standardisation
    - autoencoder training
    - autoencoder validation
    - classification-head training
    - external classifier training
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)

from bci_wheelchair.representation.classifiers import (
    create_classifier,
)
from bci_wheelchair.representation.supervised_autoencoder import (
    SupervisedAutoencoderConfig,
)
from bci_wheelchair.representation.supervised_training import (
    SupervisedTrainingConfig,
    apply_label_mapping,
    extract_supervised_latent_features,
    save_supervised_autoencoder_checkpoint,
    train_supervised_autoencoder,
)


SUBJECTS = [
    f"A{subject_number:02d}"
    for subject_number in range(1, 10)
]

CLASSIFIERS = [
    "lda",
    "linear_svm",
    "rbf_svm",
    "logistic_regression",
    "random_forest",
]


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run LOSO evaluation using supervised "
            "autoencoder latent features."
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
            "results/supervised_autoencoder_loso"
        ),
    )

    parser.add_argument(
        "--test-subject",
        type=str,
        default=None,
        help=(
            "Run one fold, such as A09. "
            "Omit to run all nine subjects."
        ),
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
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
        default=1e-5,
    )

    parser.add_argument(
        "--classification-weight",
        type=float,
        default=0.5,
        help=(
            "Weight applied to classification loss. "
            "Total loss = reconstruction loss + "
            "weight × classification loss."
        ),
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
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


def load_subject(
    data_directory: Path,
    subject: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a subject's training-session EEG."""
    file_path = data_directory / f"{subject}T.npz"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Subject file not found: {file_path}"
        )

    with np.load(file_path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"])

    if X.ndim != 3:
        raise ValueError(
            f"Expected 3D EEG, but {file_path} "
            f"contains {X.shape}."
        )

    if len(X) != len(y):
        raise ValueError(
            f"EEG and label counts do not match in "
            f"{file_path}."
        )

    return X, y


def combine_subjects(
    data_directory: Path,
    subjects: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine multiple subjects."""
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


def run_fold(
    held_out_subject: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Run one supervised-autoencoder LOSO fold."""
    training_subjects = [
        subject
        for subject in SUBJECTS
        if subject != held_out_subject
    ]

    print()
    print("=" * 78)
    print(f"Held-out subject: {held_out_subject}")
    print(
        "Training subjects: "
        f"{', '.join(training_subjects)}"
    )
    print("=" * 78)

    X_train, y_train = combine_subjects(
        args.data_directory,
        training_subjects,
    )

    X_test, y_test = load_subject(
        args.data_directory,
        held_out_subject,
    )

    print(f"Training EEG: {X_train.shape}")
    print(f"Held-out EEG: {X_test.shape}")

    model_config = SupervisedAutoencoderConfig(
        n_channels=X_train.shape[1],
        n_times=X_train.shape[2],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_classes=4,
        dropout=args.dropout,
    )

    training_config = SupervisedTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        classification_weight=(
            args.classification_weight
        ),
        random_state=args.random_state,
    )

    (
        model,
        standardizer,
        history,
        class_names,
    ) = train_supervised_autoencoder(
        X_train=X_train,
        y_train=y_train,
        model_config=model_config,
        training_config=training_config,
    )

    print("Extracting training latent features...")

    train_features = extract_supervised_latent_features(
        model=model,
        standardizer=standardizer,
        X=X_train,
        batch_size=args.batch_size,
    )

    print("Extracting held-out latent features...")

    test_features = extract_supervised_latent_features(
        model=model,
        standardizer=standardizer,
        X=X_test,
        batch_size=args.batch_size,
    )

    print(
        f"Training latent features: "
        f"{train_features.shape}"
    )
    print(
        f"Test latent features: "
        f"{test_features.shape}"
    )

    y_train_encoded = apply_label_mapping(
        y_train,
        class_names,
    )

    y_test_encoded = apply_label_mapping(
        y_test,
        class_names,
    )

    fold_directory = (
        args.output_directory / held_out_subject
    )

    fold_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.save_checkpoints:
        checkpoint_path = (
            fold_directory
            / f"{held_out_subject}_supervised_autoencoder.pt"
        )

        save_supervised_autoencoder_checkpoint(
            path=checkpoint_path,
            model=model,
            standardizer=standardizer,
            training_config=training_config,
            history=history,
            class_names=class_names,
        )

        print(f"Saved checkpoint: {checkpoint_path}")

    history_dataframe = pd.DataFrame(
        {
            "epoch": np.arange(
                1,
                len(history.training_total_losses) + 1,
            ),
            "training_total_loss": (
                history.training_total_losses
            ),
            "validation_total_loss": (
                history.validation_total_losses
            ),
            "training_reconstruction_loss": (
                history.training_reconstruction_losses
            ),
            "validation_reconstruction_loss": (
                history.validation_reconstruction_losses
            ),
            "training_classification_loss": (
                history.training_classification_losses
            ),
            "validation_classification_loss": (
                history.validation_classification_losses
            ),
            "training_accuracy": (
                history.training_accuracies
            ),
            "validation_accuracy": (
                history.validation_accuracies
            ),
        }
    )

    history_dataframe.to_csv(
        fold_directory / "training_history.csv",
        index=False,
    )

    prediction_data: dict[str, object] = {
        "true_label": y_test,
        "true_label_index": y_test_encoded,
    }

    fold_results: list[dict[str, object]] = []

    for classifier_name in CLASSIFIERS:
        classifier = create_classifier(
            classifier_name,
            random_state=args.random_state,
        )

        classifier.fit(
            train_features,
            y_train_encoded,
        )

        predictions = classifier.predict(
            test_features
        )

        accuracy = accuracy_score(
            y_test_encoded,
            predictions,
        )

        kappa = cohen_kappa_score(
            y_test_encoded,
            predictions,
        )

        print(
            f"{classifier_name:20s} | "
            f"accuracy={accuracy * 100:.2f}% | "
            f"kappa={kappa:.4f}"
        )

        predicted_labels = np.asarray(
            [
                class_names[int(prediction)]
                for prediction in predictions
            ]
        )

        prediction_data[
            f"{classifier_name}_prediction_index"
        ] = predictions

        prediction_data[
            f"{classifier_name}_prediction"
        ] = predicted_labels

        matrix = confusion_matrix(
            y_test_encoded,
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
            fold_directory
            / f"{classifier_name}_confusion_matrix.csv"
        )

        fold_results.append(
            {
                "held_out_subject": held_out_subject,
                "classifier": classifier_name,
                "accuracy": float(accuracy),
                "accuracy_percent": float(
                    accuracy * 100
                ),
                "kappa": float(kappa),
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "classification_weight": (
                    args.classification_weight
                ),
                "best_epoch": history.best_epoch,
                "best_validation_loss": (
                    history.best_validation_loss
                ),
                "n_training_trials": len(X_train),
                "n_test_trials": len(X_test),
            }
        )

    pd.DataFrame(prediction_data).to_csv(
        fold_directory / "predictions.csv",
        index=False,
    )

    return fold_results


def create_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise performance across subjects."""
    summary = (
        results
        .groupby("classifier", as_index=False)
        .agg(
            mean_accuracy_percent=(
                "accuracy_percent",
                "mean",
            ),
            std_accuracy_percent=(
                "accuracy_percent",
                "std",
            ),
            mean_kappa=("kappa", "mean"),
            std_kappa=("kappa", "std"),
            subjects=(
                "held_out_subject",
                "nunique",
            ),
        )
    )

    standard_deviation_columns = [
        "std_accuracy_percent",
        "std_kappa",
    ]

    summary[standard_deviation_columns] = (
        summary[standard_deviation_columns]
        .fillna(0.0)
    )

    return summary


def main() -> None:
    """Run one or all LOSO folds."""
    args = parse_arguments()

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.test_subject is None:
        held_out_subjects = SUBJECTS
    else:
        requested_subject = (
            args.test_subject.upper()
        )

        if requested_subject not in SUBJECTS:
            raise ValueError(
                f"Unknown subject: {requested_subject}"
            )

        held_out_subjects = [requested_subject]

    all_results: list[dict[str, object]] = []

    for held_out_subject in held_out_subjects:
        all_results.extend(
            run_fold(
                held_out_subject,
                args,
            )
        )

    results_dataframe = pd.DataFrame(
        all_results
    )

    results_path = (
        args.output_directory
        / "supervised_loso_results.csv"
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
        / "supervised_loso_summary.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    configuration = {
        "evaluation": (
            "supervised autoencoder LOSO"
        ),
        "data_directory": str(
            args.data_directory
        ),
        "held_out_subjects": (
            held_out_subjects
        ),
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "classification_weight": (
            args.classification_weight
        ),
        "validation_fraction": (
            args.validation_fraction
        ),
        "patience": args.patience,
        "random_state": args.random_state,
        "classifiers": CLASSIFIERS,
    }

    with (
        args.output_directory
        / "experiment_configuration.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as configuration_file:
        json.dump(
            configuration,
            configuration_file,
            indent=2,
        )

    print()
    print("=" * 78)
    print("SUPERVISED AUTOENCODER LOSO RESULTS")
    print("=" * 78)

    print(
        results_dataframe[
            [
                "held_out_subject",
                "classifier",
                "accuracy_percent",
                "kappa",
                "best_epoch",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

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
