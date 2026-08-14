"""
Run leave-one-subject-out evaluation using autoencoder EEG features.

Example fold:

    Training subjects:
        A01T, A02T, A03T, A04T,
        A05T, A06T, A07T, A08T

    Held-out test subject:
        A09T

The held-out subject is never used for:

    - autoencoder training
    - autoencoder validation
    - EEG standardization
    - classifier training
    - hyperparameter selection

Run from the project root:

    python -m scripts.autoencoder.run_loso
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from bci_wheelchair.representation import (
    AutoencoderConfig,
    TrainingConfig,
    create_classifier,
    extract_latent_features,
    save_autoencoder_checkpoint,
    train_autoencoder,
)


SUBJECTS = [f"A{number:02d}" for number in range(1, 10)]

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
            "Run LOSO classification using autoencoder latent features."
        )
    )

    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("data/processed/8-30Hz_0.5-2.5s"),
        help="Directory containing A01T.npz through A09T.npz.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("results/autoencoder_loso"),
        help="Directory for results and checkpoints.",
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help="Number of learned autoencoder features.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Size of the first autoencoder hidden layer.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of autoencoder training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Autoencoder batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Autoencoder learning rate.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience.",
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of training-subject EEG used for validation.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--test-subject",
        type=str,
        default=None,
        help=(
            "Run only one held-out subject, for example A09. "
            "Omit to run all nine LOSO folds."
        ),
    )

    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save the trained autoencoder for every LOSO fold.",
    )

    return parser.parse_args()


def load_subject(
    data_directory: Path,
    subject: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one subject's training-session EEG and labels."""
    path = data_directory / f"{subject}T.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Processed subject file was not found: {path}"
        )

    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data["X"])
        y = np.asarray(data["y"])

    if X.ndim != 3:
        raise ValueError(
            f"{path} contains invalid EEG shape: {X.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"{path} has {len(X)} EEG trials but {len(y)} labels."
        )

    return X, y


def combine_training_subjects(
    data_directory: Path,
    training_subjects: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine EEG and labels from all training subjects."""
    eeg_arrays: list[np.ndarray] = []
    label_arrays: list[np.ndarray] = []

    for subject in training_subjects:
        X_subject, y_subject = load_subject(
            data_directory=data_directory,
            subject=subject,
        )

        eeg_arrays.append(X_subject)
        label_arrays.append(y_subject)

    X_train = np.concatenate(eeg_arrays, axis=0)
    y_train = np.concatenate(label_arrays, axis=0)

    return X_train, y_train


def evaluate_classifier(
    classifier_name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    random_state: int,
) -> dict[str, object]:
    """Train one classifier and evaluate the held-out subject."""
    classifier = create_classifier(
        classifier_name,
        random_state=random_state,
    )

    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    kappa = cohen_kappa_score(test_labels, predictions)

    return {
        "classifier": classifier_name,
        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100.0),
        "kappa": float(kappa),
        "predictions": predictions,
    }


def run_fold(
    held_out_subject: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Run one leave-one-subject-out fold."""
    training_subjects = [
        subject
        for subject in SUBJECTS
        if subject != held_out_subject
    ]

    print()
    print("=" * 72)
    print(f"Held-out subject: {held_out_subject}")
    print(f"Training subjects: {', '.join(training_subjects)}")
    print("=" * 72)

    X_train, y_train = combine_training_subjects(
        data_directory=args.data_directory,
        training_subjects=training_subjects,
    )

    X_test, y_test = load_subject(
        data_directory=args.data_directory,
        subject=held_out_subject,
    )

    print(f"Training EEG: {X_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Held-out EEG: {X_test.shape}")
    print(f"Held-out labels: {y_test.shape}")

    model_config = AutoencoderConfig(
        n_channels=X_train.shape[1],
        n_times=X_train.shape[2],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        random_state=args.random_state,
    )

    model, standardizer, history = train_autoencoder(
        X_train=X_train,
        model_config=model_config,
        training_config=training_config,
    )

    print("Extracting training-subject latent features...")

    train_features = extract_latent_features(
        model=model,
        standardizer=standardizer,
        X=X_train,
        batch_size=args.batch_size,
    )

    print("Extracting held-out-subject latent features...")

    test_features = extract_latent_features(
        model=model,
        standardizer=standardizer,
        X=X_test,
        batch_size=args.batch_size,
    )

    print(f"Training latent features: {train_features.shape}")
    print(f"Test latent features: {test_features.shape}")

    fold_directory = args.output_directory / held_out_subject
    fold_directory.mkdir(parents=True, exist_ok=True)

    if args.save_checkpoints:
        checkpoint_path = (
            fold_directory
            / f"{held_out_subject}_autoencoder.pt"
        )

        save_autoencoder_checkpoint(
            path=checkpoint_path,
            model=model,
            standardizer=standardizer,
            training_config=training_config,
            history=history,
        )

        print(f"Saved checkpoint: {checkpoint_path}")

    results: list[dict[str, object]] = []

    prediction_data: dict[str, np.ndarray] = {
        "true_label": y_test,
    }

    for classifier_name in CLASSIFIERS:
        classifier_result = evaluate_classifier(
            classifier_name=classifier_name,
            train_features=train_features,
            train_labels=y_train,
            test_features=test_features,
            test_labels=y_test,
            random_state=args.random_state,
        )

        print(
            f"{classifier_name:20s} | "
            f"accuracy={classifier_result['accuracy_percent']:.2f}% | "
            f"kappa={classifier_result['kappa']:.4f}"
        )

        prediction_data[
            f"{classifier_name}_prediction"
        ] = classifier_result.pop("predictions")

        results.append(
            {
                "held_out_subject": held_out_subject,
                "n_training_subjects": len(training_subjects),
                "n_training_trials": len(X_train),
                "n_test_trials": len(X_test),
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "best_epoch": history.best_epoch,
                "best_validation_loss": (
                    history.best_validation_loss
                ),
                **classifier_result,
            }
        )

    predictions_dataframe = pd.DataFrame(prediction_data)

    predictions_path = (
        fold_directory
        / f"{held_out_subject}_predictions.csv"
    )

    predictions_dataframe.to_csv(
        predictions_path,
        index=False,
    )

    history_path = (
        fold_directory
        / f"{held_out_subject}_training_history.csv"
    )

    history_dataframe = pd.DataFrame(
        {
            "epoch": np.arange(
                1,
                len(history.train_losses) + 1,
            ),
            "training_loss": history.train_losses,
            "validation_loss": history.validation_losses,
        }
    )

    history_dataframe.to_csv(
        history_path,
        index=False,
    )

    return results


def create_summary(
    results_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate mean and standard deviation across held-out subjects."""
    summary = (
        results_dataframe
        .groupby("classifier", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_accuracy_percent=("accuracy_percent", "mean"),
            std_accuracy_percent=("accuracy_percent", "std"),
            mean_kappa=("kappa", "mean"),
            std_kappa=("kappa", "std"),
            subjects=("held_out_subject", "nunique"),
        )
    )

    standard_deviation_columns = [
        "std_accuracy",
        "std_accuracy_percent",
        "std_kappa",
    ]

    summary[standard_deviation_columns] = (
        summary[standard_deviation_columns].fillna(0.0)
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
        requested_subject = args.test_subject.upper()

        if requested_subject not in SUBJECTS:
            raise ValueError(
                f"Unknown subject '{args.test_subject}'. "
                f"Expected one of: {', '.join(SUBJECTS)}"
            )

        held_out_subjects = [requested_subject]

    all_results: list[dict[str, object]] = []

    for held_out_subject in held_out_subjects:
        fold_results = run_fold(
            held_out_subject=held_out_subject,
            args=args,
        )

        all_results.extend(fold_results)

    results_dataframe = pd.DataFrame(all_results)

    results_path = args.output_directory / "loso_results.csv"

    results_dataframe.to_csv(
        results_path,
        index=False,
    )

    summary_dataframe = create_summary(results_dataframe)

    summary_path = args.output_directory / "loso_summary.csv"

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    configuration = {
        "data_directory": str(args.data_directory),
        "subjects": held_out_subjects,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "validation_fraction": args.validation_fraction,
        "random_state": args.random_state,
        "classifiers": CLASSIFIERS,
        "session": "T",
        "evaluation": "LOSO cross-subject",
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
    print("=" * 72)
    print("LOSO RESULTS")
    print("=" * 72)

    display_columns = [
        "held_out_subject",
        "classifier",
        "accuracy_percent",
        "kappa",
        "best_epoch",
    ]

    print(
        results_dataframe[display_columns].to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        summary_dataframe.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print(f"Saved detailed results: {results_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
