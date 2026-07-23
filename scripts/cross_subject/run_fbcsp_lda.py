"""
Strict Leave-One-Subject-Out cross-subject evaluation using FBCSP + LDA.

For every fold:

    Eight subjects -> training set
    One subject    -> completely unseen test set

Important:
- FBCSP and LDA are fitted only on the eight training subjects.
- The held-out subject is used only for testing.
- This prevents test-subject data leakage.

Pipeline:

    Broadband EEG (4-40 Hz)
        -> Filter Bank CSP
        -> Concatenated log-variance CSP features
        -> LDA
        -> Prediction on unseen subject

Run from the project root:

    python -m scripts.cross_subject.run_fbcsp_lda
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import mne
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)

from bci_wheelchair.commands import CLASS_TO_COMMAND
from bci_wheelchair.models import DEFAULT_BANDS, make_fbcsp_lda
from bci_wheelchair.data.processed_loading import load_processed_subject


mne.set_log_level("WARNING")


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

N_COMPONENTS = 4

RESULTS_DIRECTORY = Path("results/cross_subject/predictions")

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "cross_subject_fbcsp_loso_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "cross_subject_fbcsp_loso_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "cross_subject_fbcsp_loso_overall_summary.csv"
)


def load_all_subjects() -> dict[str, dict[str, np.ndarray]]:
    """
    Load broadband epochs for all nine subjects.

    Each subject is loaded independently. No CSP or classifier fitting
    happens here.
    """
    subject_data: dict[str, dict[str, np.ndarray]] = {}

    print("=" * 78)
    print("Loading broadband EEG data for all subjects")
    print("=" * 78)

    for subject in SUBJECTS:
        print(f"Loading processed data for {subject}...")

        X, y = load_processed_subject(
            subject,
            config="4-40",
        )

        subject_data[subject] = {
            "X": np.asarray(X),
            "y": np.asarray(y),
        }

        print(
            f"  Trials: {len(y)}, "
            f"channels: {X.shape[1]}, "
            f"samples: {X.shape[2]}"
        )

    return subject_data


def get_training_data(
    subject_data: dict[str, dict[str, np.ndarray]],
    test_subject: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Pool the eight training subjects for one LOSO fold.
    """
    training_subjects = [
        subject
        for subject in SUBJECTS
        if subject != test_subject
    ]

    X_train = np.concatenate(
        [
            subject_data[subject]["X"]
            for subject in training_subjects
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            subject_data[subject]["y"]
            for subject in training_subjects
        ],
        axis=0,
    )

    return X_train, y_train, training_subjects


def export_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """
    Export dictionary rows to a CSV file.
    """
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def create_subject_result(
    test_subject: str,
    training_subjects: list[str],
    training_trials: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_seconds: float,
    prediction_seconds: float,
) -> dict[str, object]:
    """
    Create the subject-level result for one held-out test subject.
    """
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
    )

    recalls = recall_score(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
    )

    result: dict[str, object] = {
        "test_subject": test_subject,
        "training_subjects": ",".join(training_subjects),
        "training_trials": training_trials,
        "testing_trials": len(y_true),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "kappa": kappa,
        "left_hand_recall": recalls[0],
        "right_hand_recall": recalls[1],
        "feet_recall": recalls[2],
        "tongue_recall": recalls[3],
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
        "prediction_seconds_per_trial": (
            prediction_seconds / len(y_true)
        ),
        "fbcsp_components_per_band": N_COMPONENTS,
        "number_of_frequency_bands": len(DEFAULT_BANDS),
        "total_fbcsp_features": (
            N_COMPONENTS * len(DEFAULT_BANDS)
        ),
    }

    for true_index, true_class in enumerate(CLASS_ORDER):
        for predicted_index, predicted_class in enumerate(
            CLASS_ORDER
        ):
            result[
                f"cm_{true_class}_pred_{predicted_class}"
            ] = int(
                matrix[true_index, predicted_index]
            )

    return result


def create_prediction_rows(
    test_subject: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    probability_classes: np.ndarray,
) -> list[dict[str, object]]:
    """
    Create trial-level prediction rows.
    """
    probability_lookup = {
        class_name: index
        for index, class_name in enumerate(
            probability_classes
        )
    }

    rows: list[dict[str, object]] = []

    for trial_index, (
        true_label,
        predicted_label,
    ) in enumerate(
        zip(y_true, y_pred),
        start=1,
    ):
        row: dict[str, object] = {
            "test_subject": test_subject,
            "trial_index": trial_index,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": int(
                true_label == predicted_label
            ),
            "predicted_command": CLASS_TO_COMMAND.get(
                predicted_label,
                predicted_label,
            ),
        }

        for class_name in CLASS_ORDER:
            class_index = probability_lookup.get(
                class_name
            )

            if class_index is None:
                probability = np.nan
            else:
                probability = probabilities[
                    trial_index - 1,
                    class_index,
                ]

            row[f"probability_{class_name}"] = probability

        rows.append(row)

    return rows


def print_fold_result(
    result: dict[str, object],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Print the results for one held-out subject.
    """
    print()
    print("-" * 78)
    print(
        f"Results for unseen subject "
        f"{result['test_subject']}"
    )
    print("-" * 78)

    print(
        f"Accuracy: {result['accuracy']:.3f} "
        f"({result['accuracy_percent']:.1f}%)"
    )

    print(
        f"Kappa:    {result['kappa']:.3f}"
    )

    print(
        f"Training time: "
        f"{result['training_seconds']:.2f} seconds"
    )

    print(
        f"Prediction time: "
        f"{result['prediction_seconds']:.3f} seconds"
    )

    print("\nPer-class recall:")

    print(
        f"  left_hand:  "
        f"{result['left_hand_recall']:.3f}"
    )

    print(
        f"  right_hand: "
        f"{result['right_hand_recall']:.3f}"
    )

    print(
        f"  feet:       "
        f"{result['feet_recall']:.3f}"
    )

    print(
        f"  tongue:     "
        f"{result['tongue_recall']:.3f}"
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
        )
    )


def create_overall_summary(
    subject_results: list[dict[str, object]],
    all_true_labels: np.ndarray,
    all_predictions: np.ndarray,
    total_seconds: float,
) -> dict[str, object]:
    """
    Calculate overall LOSO statistics.
    """
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

    pooled_accuracy = accuracy_score(
        all_true_labels,
        all_predictions,
    )

    pooled_kappa = cohen_kappa_score(
        all_true_labels,
        all_predictions,
    )

    recalls = recall_score(
        all_true_labels,
        all_predictions,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_true_labels,
        all_predictions,
        labels=CLASS_ORDER,
    )

    summary: dict[str, object] = {
        "pipeline": "FBCSP+LDA",
        "evaluation_protocol": "strict_LOSO",
        "number_of_subjects": len(SUBJECTS),
        "total_test_trials": len(all_true_labels),
        "fbcsp_components_per_band": N_COMPONENTS,
        "number_of_frequency_bands": len(DEFAULT_BANDS),
        "total_fbcsp_features": (
            N_COMPONENTS * len(DEFAULT_BANDS)
        ),
        "mean_subject_accuracy": accuracies.mean(),
        "mean_subject_accuracy_percent": (
            accuracies.mean() * 100.0
        ),
        "std_subject_accuracy": accuracies.std(
            ddof=1
        ),
        "std_subject_accuracy_percent": (
            accuracies.std(ddof=1) * 100.0
        ),
        "minimum_subject_accuracy": accuracies.min(),
        "minimum_subject_accuracy_percent": (
            accuracies.min() * 100.0
        ),
        "maximum_subject_accuracy": accuracies.max(),
        "maximum_subject_accuracy_percent": (
            accuracies.max() * 100.0
        ),
        "mean_subject_kappa": kappas.mean(),
        "std_subject_kappa": kappas.std(
            ddof=1
        ),
        "pooled_accuracy": pooled_accuracy,
        "pooled_accuracy_percent": (
            pooled_accuracy * 100.0
        ),
        "pooled_kappa": pooled_kappa,
        "overall_left_hand_recall": recalls[0],
        "overall_right_hand_recall": recalls[1],
        "overall_feet_recall": recalls[2],
        "overall_tongue_recall": recalls[3],
        "total_experiment_seconds": total_seconds,
    }

    for true_index, true_class in enumerate(CLASS_ORDER):
        for predicted_index, predicted_class in enumerate(
            CLASS_ORDER
        ):
            summary[
                f"cm_{true_class}_pred_{predicted_class}"
            ] = int(
                matrix[true_index, predicted_index]
            )

    return summary


def print_overall_summary(
    summary: dict[str, object],
    subject_results: list[dict[str, object]],
    all_true_labels: np.ndarray,
    all_predictions: np.ndarray,
) -> None:
    """
    Print the complete experiment summary.
    """
    print()
    print("=" * 78)
    print("Final Strict LOSO FBCSP + LDA Results")
    print("=" * 78)

    print()
    print(
        f"{'Subject':<12}"
        f"{'Accuracy':>12}"
        f"{'Kappa':>12}"
    )

    print("-" * 36)

    for result in subject_results:
        print(
            f"{result['test_subject']:<12}"
            f"{result['accuracy_percent']:>11.1f}%"
            f"{result['kappa']:>12.3f}"
        )

    print("-" * 36)

    print(
        f"{'Mean':<12}"
        f"{summary['mean_subject_accuracy_percent']:>11.1f}%"
        f"{summary['mean_subject_kappa']:>12.3f}"
    )

    print()
    print(
        "Mean subject accuracy: "
        f"{summary['mean_subject_accuracy_percent']:.2f}%"
    )

    print(
        "Subject accuracy SD:   "
        f"{summary['std_subject_accuracy_percent']:.2f} "
        "percentage points"
    )

    print(
        "Mean subject kappa:    "
        f"{summary['mean_subject_kappa']:.3f}"
    )

    print(
        "Pooled accuracy:       "
        f"{summary['pooled_accuracy_percent']:.2f}%"
    )

    print(
        "Pooled kappa:          "
        f"{summary['pooled_kappa']:.3f}"
    )

    print("\nOverall confusion matrix:")

    print(
        confusion_matrix(
            all_true_labels,
            all_predictions,
            labels=CLASS_ORDER,
        )
    )

    print("\nOverall per-class recall:")

    print(
        "  left_hand:  "
        f"{summary['overall_left_hand_recall']:.3f}"
    )

    print(
        "  right_hand: "
        f"{summary['overall_right_hand_recall']:.3f}"
    )

    print(
        "  feet:       "
        f"{summary['overall_feet_recall']:.3f}"
    )

    print(
        "  tongue:     "
        f"{summary['overall_tongue_recall']:.3f}"
    )

    print(
        "\nTotal experiment time: "
        f"{summary['total_experiment_seconds']:.2f} seconds"
    )

    print("\nSaved files:")

    print(
        f"  {SUBJECT_RESULTS_PATH}"
    )

    print(
        f"  {PREDICTIONS_PATH}"
    )

    print(
        f"  {OVERALL_SUMMARY_PATH}"
    )


def main() -> None:
    """
    Run all nine strict LOSO folds.
    """
    experiment_start = time.perf_counter()

    print("=" * 78)
    print("Strict Cross-Subject FBCSP + LDA Evaluation")
    print("=" * 78)

    print(
        "\nEach fold trains on eight subjects and tests on "
        "one completely unseen subject."
    )

    print(
        "\nPipeline:"
        "\n- Broadband EEG: 4-40 Hz"
        "\n- Eight frequency sub-bands"
        f"\n- CSP components per band: {N_COMPONENTS}"
        f"\n- Total FBCSP features: "
        f"{N_COMPONENTS * len(DEFAULT_BANDS)}"
        "\n- Linear Discriminant Analysis"
    )

    print("\nFrequency bands:")

    for band_index, (low, high) in enumerate(
        DEFAULT_BANDS,
        start=1,
    ):
        print(
            f"  Band {band_index}: {low}-{high} Hz"
        )

    subject_data = load_all_subjects()

    subject_results: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    all_true_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for fold_number, test_subject in enumerate(
        SUBJECTS,
        start=1,
    ):
        print()
        print("=" * 78)
        print(
            f"Fold {fold_number}/{len(SUBJECTS)}: "
            f"held-out test subject {test_subject}"
        )
        print("=" * 78)

        X_train, y_train, training_subjects = (
            get_training_data(
                subject_data,
                test_subject,
            )
        )

        X_test = subject_data[test_subject]["X"]
        y_test = subject_data[test_subject]["y"]

        print(
            "\nTraining subjects: "
            + ", ".join(training_subjects)
        )

        print(
            f"Training data shape: {X_train.shape}"
        )

        print(
            f"Testing data shape:  {X_test.shape}"
        )

        classifier = make_fbcsp_lda(
            n_components=N_COMPONENTS,
            bands=DEFAULT_BANDS,
        )

        print(
            "\nFitting FBCSP and LDA using training "
            "subjects only..."
        )

        training_start = time.perf_counter()

        classifier.fit(
            X_train,
            y_train,
        )

        training_seconds = (
            time.perf_counter()
            - training_start
        )

        print(
            f"Training completed in "
            f"{training_seconds:.2f} seconds."
        )

        print(
            f"Predicting unseen subject {test_subject}..."
        )

        prediction_start = time.perf_counter()

        y_pred = classifier.predict(
            X_test
        )

        probabilities = classifier.predict_proba(
            X_test
        )

        prediction_seconds = (
            time.perf_counter()
            - prediction_start
        )

        result = create_subject_result(
            test_subject=test_subject,
            training_subjects=training_subjects,
            training_trials=len(y_train),
            y_true=y_test,
            y_pred=y_pred,
            training_seconds=training_seconds,
            prediction_seconds=prediction_seconds,
        )

        subject_results.append(result)

        prediction_rows.extend(
            create_prediction_rows(
                test_subject=test_subject,
                y_true=y_test,
                y_pred=y_pred,
                probabilities=probabilities,
                probability_classes=classifier.classes_,
            )
        )

        all_true_labels.append(
            np.asarray(y_test)
        )

        all_predictions.append(
            np.asarray(y_pred)
        )

        print_fold_result(
            result,
            y_test,
            y_pred,
        )

        # Save progress after every fold.
        export_csv(
            SUBJECT_RESULTS_PATH,
            subject_results,
        )

        export_csv(
            PREDICTIONS_PATH,
            prediction_rows,
        )

    combined_true_labels = np.concatenate(
        all_true_labels
    )

    combined_predictions = np.concatenate(
        all_predictions
    )

    total_seconds = (
        time.perf_counter()
        - experiment_start
    )

    overall_summary = create_overall_summary(
        subject_results=subject_results,
        all_true_labels=combined_true_labels,
        all_predictions=combined_predictions,
        total_seconds=total_seconds,
    )

    export_csv(
        OVERALL_SUMMARY_PATH,
        [overall_summary],
    )

    print_overall_summary(
        summary=overall_summary,
        subject_results=subject_results,
        all_true_labels=combined_true_labels,
        all_predictions=combined_predictions,
    )


if __name__ == "__main__":
    main()
