"""
Leave-One-Subject-Out cross-subject motor-imagery evaluation.

For every fold:

    8 subjects -> training data
    1 subject  -> completely unseen testing data

Pipeline:

    Subject-wise Euclidean Alignment
        -> Regularized FBCSP
        -> PCA retaining 90% variance
        -> Shrinkage LDA

Run from the project root:

    python -m scripts.cross_subject.run_cross_subject_loso
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)

from bci_wheelchair.commands import CLASS_TO_COMMAND
from scripts.cross_subject.run_cross_subject_evaluation import (
    CLASS_ORDER,
    load_and_align_subject,
    make_ea_regularized_classifier,
)


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

RESULTS_DIRECTORY = Path("results/cross_subject/predictions")

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "cross_subject_loso_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "cross_subject_loso_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "cross_subject_loso_overall_summary.csv"
)


def load_all_subjects() -> dict[str, dict[str, object]]:
    """
    Load, preprocess and independently align every subject once.
    """
    subject_data: dict[str, dict[str, object]] = {}

    print("=" * 76)
    print("Loading and independently aligning all subjects")
    print("=" * 76)

    for subject in SUBJECTS:
        X, y, alignment_error = load_and_align_subject(
            subject
        )

        subject_data[subject] = {
            "X": X,
            "y": y,
            "alignment_error": alignment_error,
        }

    return subject_data


def get_training_data(
    subject_data: dict[str, dict[str, object]],
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


def create_subject_result(
    test_subject: str,
    training_subjects: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_seconds: float,
    prediction_seconds: float,
    alignment_error: float,
    retained_pca_components: int,
) -> dict[str, object]:
    """
    Create one result row for one held-out subject.
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
        "training_trials": len(y_true) * len(training_subjects),
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
        "alignment_identity_error": alignment_error,
        "pca_components_retained": retained_pca_components,
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
    Create trial-level prediction rows for one test subject.
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

            row[f"probability_{class_name}"] = (
                probabilities[
                    trial_index - 1,
                    class_index,
                ]
                if class_index is not None
                else np.nan
            )

        rows.append(row)

    return rows


def export_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """
    Export a list of dictionaries to CSV.
    """
    if not rows:
        raise ValueError(
            f"No rows were provided for {path}."
        )

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

    overall_recalls = recall_score(
        all_true_labels,
        all_predictions,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    return {
        "number_of_subjects": len(SUBJECTS),
        "total_test_trials": len(all_true_labels),
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
        "overall_left_hand_recall": overall_recalls[0],
        "overall_right_hand_recall": overall_recalls[1],
        "overall_feet_recall": overall_recalls[2],
        "overall_tongue_recall": overall_recalls[3],
        "total_experiment_seconds": total_seconds,
    }


def print_fold_result(
    result: dict[str, object],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Print one fold result.
    """
    print()
    print("-" * 76)
    print(
        f"Results for unseen subject "
        f"{result['test_subject']}"
    )
    print("-" * 76)

    print(
        f"Accuracy: "
        f"{result['accuracy']:.3f} "
        f"({result['accuracy_percent']:.1f}%)"
    )

    print(
        f"Kappa:    "
        f"{result['kappa']:.3f}"
    )

    print(
        f"PCA components retained: "
        f"{result['pca_components_retained']}"
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


def print_overall_summary(
    summary: dict[str, object],
    subject_results: list[dict[str, object]],
    all_true_labels: np.ndarray,
    all_predictions: np.ndarray,
) -> None:
    """
    Print the final LOSO summary.
    """
    print()
    print("=" * 76)
    print("Final LOSO Cross-Subject Results")
    print("=" * 76)

    print("\nSubject-wise results:")

    print(
        f"{'Subject':<10}"
        f"{'Accuracy':>12}"
        f"{'Kappa':>12}"
    )

    print("-" * 34)

    for result in subject_results:
        print(
            f"{result['test_subject']:<10}"
            f"{result['accuracy_percent']:>11.1f}%"
            f"{result['kappa']:>12.3f}"
        )

    print("-" * 34)

    print(
        f"{'Mean':<10}"
        f"{summary['mean_subject_accuracy_percent']:>11.1f}%"
        f"{summary['mean_subject_kappa']:>12.3f}"
    )

    print(
        "\nMean subject accuracy: "
        f"{summary['mean_subject_accuracy_percent']:.1f}%"
    )

    print(
        "Standard deviation:    "
        f"{summary['std_subject_accuracy_percent']:.1f} "
        "percentage points"
    )

    print(
        "Mean subject kappa:     "
        f"{summary['mean_subject_kappa']:.3f}"
    )

    print(
        "Pooled LOSO accuracy:   "
        f"{summary['pooled_accuracy_percent']:.1f}%"
    )

    print(
        "Pooled LOSO kappa:      "
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
        f"{summary['total_experiment_seconds']:.2f} "
        "seconds"
    )

    print("\nExports:")

    print(
        f"  Subject results: {SUBJECT_RESULTS_PATH}"
    )

    print(
        f"  Predictions:     {PREDICTIONS_PATH}"
    )

    print(
        f"  Overall summary: {OVERALL_SUMMARY_PATH}"
    )


def main() -> None:
    """
    Run all nine LOSO folds.
    """
    experiment_start = time.perf_counter()

    print("=" * 76)
    print("Leave-One-Subject-Out Cross-Subject Evaluation")
    print("=" * 76)

    print(
        "\nFor each fold, eight subjects are used for "
        "training and one subject is completely unseen."
    )

    print(
        "\nPipeline:"
        "\n- Subject-wise Euclidean Alignment"
        "\n- Regularized FBCSP"
        "\n- PCA retaining 90% variance"
        "\n- Shrinkage LDA"
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
        print("=" * 76)
        print(
            f"Fold {fold_number}/{len(SUBJECTS)}: "
            f"unseen test subject {test_subject}"
        )
        print("=" * 76)

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
            f"Training trials: {len(y_train)}"
        )

        print(
            f"Testing trials:  {len(y_test)}"
        )

        classifier = (
            make_ea_regularized_classifier()
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

        pca = classifier.named_steps["pca"]

        retained_pca_components = int(
            pca.n_components_
        )

        result = create_subject_result(
            test_subject=test_subject,
            training_subjects=training_subjects,
            y_true=y_test,
            y_pred=y_pred,
            training_seconds=training_seconds,
            prediction_seconds=prediction_seconds,
            alignment_error=float(
                subject_data[test_subject][
                    "alignment_error"
                ]
            ),
            retained_pca_components=(
                retained_pca_components
            ),
        )

        subject_results.append(result)

        prediction_rows.extend(
            create_prediction_rows(
                test_subject=test_subject,
                y_true=y_test,
                y_pred=y_pred,
                probabilities=probabilities,
                probability_classes=(
                    classifier.classes_
                ),
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
