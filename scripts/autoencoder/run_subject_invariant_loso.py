"""
Run LOSO evaluation using subject-invariant autoencoder features.

The held-out subject is never used during:

- standardisation;
- model training;
- model validation;
- model selection;
- external classifier training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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
from bci_wheelchair.representation.subject_invariant_autoencoder import (
    SubjectInvariantAutoencoderConfig,
)
from bci_wheelchair.representation.subject_invariant_training import (
    SubjectInvariantTrainingConfig,
    apply_label_mapping,
    extract_latent_features,
    predict_classes,
    save_checkpoint,
    train_subject_invariant_autoencoder,
)


SUBJECTS = [
    f"A{subject_number:02d}"
    for subject_number in range(1, 10)
]

CLASSIFIERS = [
    "neural_head",
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
            "Run subject-invariant autoencoder "
            "LOSO evaluation."
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
            "results/"
            "subject_invariant_autoencoder_loso"
        ),
    )

    parser.add_argument(
        "--test-subject",
        type=str,
        default=None,
        help=(
            "Run one fold, such as A09. "
            "Omit to run all subjects."
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
        default=1e-5,
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
        "--reconstruction-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--classification-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--subject-weight",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-reversal-coefficient",
        type=float,
        default=1.0,
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
    """Load one subject's training-session EEG."""
    file_path = (
        data_directory
        / f"{subject}T.npz"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Subject file not found: "
            f"{file_path}"
        )

    with np.load(
        file_path,
        allow_pickle=False,
    ) as data:
        X = np.asarray(
            data["X"],
            dtype=np.float32,
        )

        y = np.asarray(
            data["y"]
        )

    if X.ndim != 3:
        raise ValueError(
            f"Expected three-dimensional EEG "
            f"in {file_path}, but found "
            f"{X.shape}."
        )

    if len(X) != len(y):
        raise ValueError(
            f"EEG and label counts differ "
            f"in {file_path}."
        )

    return X, y


def combine_training_subjects(
    data_directory: Path,
    subjects: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Combine EEG, labels and subject IDs."""
    eeg_arrays: list[np.ndarray] = []
    class_arrays: list[np.ndarray] = []
    subject_id_arrays: list[np.ndarray] = []

    for (
        subject_index,
        subject,
    ) in enumerate(subjects):
        X_subject, y_subject = load_subject(
            data_directory,
            subject,
        )

        eeg_arrays.append(
            X_subject
        )

        class_arrays.append(
            y_subject
        )

        subject_id_arrays.append(
            np.full(
                len(X_subject),
                subject_index,
                dtype=np.int64,
            )
        )

    return (
        np.concatenate(
            eeg_arrays,
            axis=0,
        ),
        np.concatenate(
            class_arrays,
            axis=0,
        ),
        np.concatenate(
            subject_id_arrays,
            axis=0,
        ),
    )


def save_confusion_matrix(
    path: Path,
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    class_names: list[str],
) -> None:
    """Save a labelled confusion matrix."""
    matrix = confusion_matrix(
        y_true,
        y_predicted,
        labels=np.arange(
            len(class_names)
        ),
    )

    dataframe = pd.DataFrame(
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

    dataframe.to_csv(
        path
    )


def run_fold(
    held_out_subject: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Run one LOSO fold."""
    training_subjects = [
        subject
        for subject in SUBJECTS
        if subject != held_out_subject
    ]

    print()
    print("=" * 78)
    print(
        f"Held-out subject: "
        f"{held_out_subject}"
    )
    print(
        "Training subjects: "
        f"{', '.join(training_subjects)}"
    )
    print("=" * 78)

    (
        X_train,
        y_train,
        training_subject_ids,
    ) = combine_training_subjects(
        args.data_directory,
        training_subjects,
    )

    X_test, y_test = load_subject(
        args.data_directory,
        held_out_subject,
    )

    print(
        f"Training EEG: "
        f"{X_train.shape}"
    )

    print(
        f"Held-out EEG: "
        f"{X_test.shape}"
    )

    model_config = (
        SubjectInvariantAutoencoderConfig(
            n_channels=X_train.shape[1],
            n_times=X_train.shape[2],
            n_subjects=len(
                training_subjects
            ),
            n_classes=4,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    )

    training_config = (
        SubjectInvariantTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=(
                args.learning_rate
            ),
            weight_decay=(
                args.weight_decay
            ),
            validation_fraction=(
                args.validation_fraction
            ),
            patience=args.patience,
            reconstruction_weight=(
                args.reconstruction_weight
            ),
            classification_weight=(
                args.classification_weight
            ),
            subject_weight=(
                args.subject_weight
            ),
            max_reversal_coefficient=(
                args
                .max_reversal_coefficient
            ),
            random_state=(
                args.random_state
            ),
        )
    )

    (
        model,
        standardizer,
        history,
        class_names,
    ) = train_subject_invariant_autoencoder(
        X_train=X_train,
        y_train=y_train,
        subject_ids=(
            training_subject_ids
        ),
        model_config=model_config,
        training_config=(
            training_config
        ),
    )

    y_train_encoded = apply_label_mapping(
        y_train,
        class_names,
    )

    y_test_encoded = apply_label_mapping(
        y_test,
        class_names,
    )

    print(
        "Extracting training "
        "latent features..."
    )

    train_features = extract_latent_features(
        model=model,
        standardizer=standardizer,
        X=X_train,
        batch_size=args.batch_size,
    )

    print(
        "Extracting held-out "
        "latent features..."
    )

    test_features = extract_latent_features(
        model=model,
        standardizer=standardizer,
        X=X_test,
        batch_size=args.batch_size,
    )

    print(
        f"Training features: "
        f"{train_features.shape}"
    )

    print(
        f"Test features: "
        f"{test_features.shape}"
    )

    fold_directory = (
        args.output_directory
        / held_out_subject
    )

    fold_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_sets: dict[
        str,
        np.ndarray,
    ] = {}

    prediction_sets[
        "neural_head"
    ] = predict_classes(
        model=model,
        standardizer=standardizer,
        X=X_test,
        batch_size=args.batch_size,
    )

    for classifier_name in CLASSIFIERS:
        if classifier_name == "neural_head":
            continue

        classifier = create_classifier(
            classifier_name
        )

        classifier.fit(
            train_features,
            y_train_encoded,
        )

        prediction_sets[
            classifier_name
        ] = classifier.predict(
            test_features
        )

    fold_results: list[
        dict[str, object]
    ] = []

    for (
        classifier_name,
        predictions,
    ) in prediction_sets.items():
        accuracy = accuracy_score(
            y_test_encoded,
            predictions,
        )

        kappa = cohen_kappa_score(
            y_test_encoded,
            predictions,
        )

        print(
            f"{classifier_name}: "
            f"accuracy="
            f"{100 * accuracy:.2f}% "
            f"| kappa={kappa:.4f}"
        )

        prediction_dataframe = (
            pd.DataFrame(
                {
                    "true_index": (
                        y_test_encoded
                    ),
                    "predicted_index": (
                        predictions
                    ),
                    "true_label": [
                        class_names[index]
                        for index
                        in y_test_encoded
                    ],
                    "predicted_label": [
                        class_names[index]
                        for index
                        in predictions
                    ],
                }
            )
        )

        prediction_dataframe.to_csv(
            fold_directory
            / (
                f"{classifier_name}"
                "_predictions.csv"
            ),
            index=False,
        )

        save_confusion_matrix(
            path=(
                fold_directory
                / (
                    f"{classifier_name}"
                    "_confusion_matrix.csv"
                )
            ),
            y_true=y_test_encoded,
            y_predicted=predictions,
            class_names=class_names,
        )

        fold_results.append(
            {
                "held_out_subject": (
                    held_out_subject
                ),
                "classifier": (
                    classifier_name
                ),
                "accuracy_percent": (
                    100 * accuracy
                ),
                "kappa": kappa,
                "best_epoch": (
                    history.best_epoch
                ),
            }
        )

    metadata = {
        "held_out_subject": (
            held_out_subject
        ),
        "training_subjects": (
            training_subjects
        ),
        "class_names": class_names,
        "model_config": asdict(
            model_config
        ),
        "training_config": asdict(
            training_config
        ),
        "best_epoch": (
            history.best_epoch
        ),
        (
            "best_validation_"
            "classification_loss"
        ): (
            history
            .best_validation_classification_loss
        ),
    }

    metadata_path = (
        fold_directory
        / "metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
        )

    epoch_count = len(
        history.training_total_losses
    )

    history_dataframe = pd.DataFrame(
        {
            "epoch": np.arange(
                1,
                epoch_count + 1,
            ),
            "training_total_loss": (
                history
                .training_total_losses
            ),
            "validation_total_loss": (
                history
                .validation_total_losses
            ),
            (
                "training_"
                "reconstruction_loss"
            ): (
                history
                .training_reconstruction_losses
            ),
            (
                "validation_"
                "reconstruction_loss"
            ): (
                history
                .validation_reconstruction_losses
            ),
            (
                "training_"
                "classification_loss"
            ): (
                history
                .training_classification_losses
            ),
            (
                "validation_"
                "classification_loss"
            ): (
                history
                .validation_classification_losses
            ),
            "training_subject_loss": (
                history
                .training_subject_losses
            ),
            "validation_subject_loss": (
                history
                .validation_subject_losses
            ),
            (
                "training_"
                "class_accuracy"
            ): (
                history
                .training_class_accuracies
            ),
            (
                "validation_"
                "class_accuracy"
            ): (
                history
                .validation_class_accuracies
            ),
            (
                "training_"
                "subject_accuracy"
            ): (
                history
                .training_subject_accuracies
            ),
            (
                "validation_"
                "subject_accuracy"
            ): (
                history
                .validation_subject_accuracies
            ),
        }
    )

    history_dataframe.to_csv(
        fold_directory
        / "training_history.csv",
        index=False,
    )

    if args.save_checkpoints:
        checkpoint_path = (
            fold_directory
            / (
                f"{held_out_subject}"
                "_subject_invariant_"
                "autoencoder.pt"
            )
        )

        save_checkpoint(
            path=checkpoint_path,
            model=model,
            standardizer=standardizer,
            training_config=(
                training_config
            ),
            history=history,
            class_names=class_names,
            training_subjects=(
                training_subjects
            ),
        )

        print(
            f"Saved checkpoint: "
            f"{checkpoint_path}"
        )

    return fold_results


def main() -> None:
    """Run one fold or all LOSO folds."""
    args = parse_arguments()

    if args.test_subject is None:
        held_out_subjects = SUBJECTS
    else:
        held_out_subject = (
            args.test_subject.upper()
        )

        if held_out_subject not in SUBJECTS:
            raise ValueError(
                f"Unknown subject "
                f"{args.test_subject}. "
                f"Choose from {SUBJECTS}."
            )

        held_out_subjects = [
            held_out_subject
        ]

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results: list[
        dict[str, object]
    ] = []

    for held_out_subject in (
        held_out_subjects
    ):
        fold_results = run_fold(
            held_out_subject,
            args,
        )

        all_results.extend(
            fold_results
        )

    results_dataframe = pd.DataFrame(
        all_results
    )

    results_path = (
        args.output_directory
        / (
            "subject_invariant_"
            "autoencoder_loso_results.csv"
        )
    )

    results_dataframe.to_csv(
        results_path,
        index=False,
    )

    summary_dataframe = (
        results_dataframe
        .groupby(
            "classifier",
            as_index=False,
        )
        .agg(
            mean_accuracy_percent=(
                "accuracy_percent",
                "mean",
            ),
            standard_deviation_percent=(
                "accuracy_percent",
                "std",
            ),
            mean_kappa=(
                "kappa",
                "mean",
            ),
            standard_deviation_kappa=(
                "kappa",
                "std",
            ),
            subjects=(
                "held_out_subject",
                "nunique",
            ),
        )
        .fillna(0.0)
        .sort_values(
            "mean_accuracy_percent",
            ascending=False,
        )
    )

    summary_path = (
        args.output_directory
        / (
            "subject_invariant_"
            "autoencoder_loso_summary.csv"
        )
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 78)
    print("FINAL RESULTS")
    print("=" * 78)

    print(
        results_dataframe.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(
        summary_dataframe.to_string(
            index=False
        )
    )

    print(
        f"\nSaved results: "
        f"{results_path}"
    )

    print(
        f"Saved summary: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()
