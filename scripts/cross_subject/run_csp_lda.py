"""
Strict Leave-One-Subject-Out cross-subject evaluation using CSP + LDA.

For every LOSO fold:

    Eight subjects -> training set
    One subject    -> completely unseen test set

Pipeline:

    EEG 8-30 Hz
        -> CSP
        -> LDA
        -> prediction on unseen subject

The held-out subject is never used during model fitting.

Run from the project root:

    python -m scripts.cross_subject.run_csp_lda
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

from bci_wheelchair.data.processed_loading import load_processed_subject
from bci_wheelchair.models import make_csp_lda


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

PREPROCESSING_CONFIG = "8-30"
N_COMPONENTS = 6

RESULTS_DIRECTORY = Path(
    "results/cross_subject/csp_fbcsp"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "cross_subject_csp_lda_loso_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "cross_subject_csp_lda_loso_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "cross_subject_csp_lda_loso_overall_summary.csv"
)


def load_all_subjects() -> dict[str, dict[str, np.ndarray]]:
    """
    Load processed EEG for all subjects.

    No CSP fitting or classifier fitting happens here.
    """
    subject_data: dict[str, dict[str, np.ndarray]] = {}

    print("=" * 78)
    print("Loading 8-30 Hz processed EEG data")
    print("=" * 78)

    for subject in SUBJECTS:
        print(f"Loading {subject}...")

        X, y = load_processed_subject(
            subject=subject,
            config=PREPROCESSING_CONFIG,
        )

        X = np.asarray(X)
        y = np.asarray(y)

        subject_data[subject] = {
            "X": X,
            "y": y,
        }

        print(
            f"  trials={X.shape[0]}, "
            f"channels={X.shape[1]}, "
            f"samples={X.shape[2]}"
        )

    return subject_data


def get_training_data(
    subject_data: dict[str, dict[str, np.ndarray]],
    test_subject: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Pool all subjects except the held-out LOSO test subject.
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


def save_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """
    Save dictionary rows to CSV.
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
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def create_subject_result(
    test_subject: str,
    training_subjects: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_trials: int,
    training_seconds: float,
    prediction_seconds: float,
) -> dict[str, object]:
    """
    Calculate metrics for one held-out subject.
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

    result = {
        "test_subject": test_subject,
        "training_subjects": "|".join(training_subjects),
        "training_trials": training_trials,
        "testing_trials": len(y_true),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "kappa": kappa,
        "left_hand_recall": recalls[0],
        "right_hand_recall": recalls[1],
        "feet_recall": recalls[2],
        "tongue_recall": recalls[3],
        "csp_components": N_COMPONENTS,
        "preprocessing": PREPROCESSING_CONFIG,
        "training_seconds": training_seconds,
        "prediction_seconds": prediction_seconds,
    }

    for true_index, true_label in enumerate(CLASS_ORDER):
        for pred_index, pred_label in enumerate(CLASS_ORDER):
            result[
                f"cm_{true_label}_pred_{pred_label}"
            ] = int(
                matrix[true_index, pred_index]
            )

    return result


def create_prediction_rows(
    test_subject: str,
    training_subjects: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, object]]:
    """
    Export trial-level predictions for one LOSO fold.
    """
    rows: list[dict[str, object]] = []

    for trial_index, (
        true_label,
        predicted_label,
    ) in enumerate(
        zip(y_true, y_pred),
        start=1,
    ):
        rows.append(
            {
                "test_subject": test_subject,
                "trial": trial_index,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": (
                    true_label == predicted_label
                ),
                "model": "CSP_LDA",
                "csp_components": N_COMPONENTS,
                "preprocessing": PREPROCESSING_CONFIG,
                "training_subjects": "|".join(
                    training_subjects
                ),
            }
        )

    return rows


def print_subject_result(
    subject: str,
    result: dict[str, object],
) -> None:
    """
    Print one fold result.
    """
    print(
        f"{subject} Accuracy: "
        f"{result['accuracy_percent']:.2f}%"
    )

    print(
        f"{subject} Kappa:    "
        f"{result['kappa']:.3f}"
    )


def create_overall_summary(
    subject_results: list[dict[str, object]],
) -> dict[str, object]:
    """
    Create overall nine-subject LOSO summary.
    """
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
        "model": "CSP_LDA",
        "evaluation": "strict_LOSO_cross_subject",
        "subjects": len(subject_results),
        "csp_components": N_COMPONENTS,
        "preprocessing": PREPROCESSING_CONFIG,
        "mean_accuracy": float(
            np.mean(accuracies)
        ),
        "mean_accuracy_percent": float(
            np.mean(accuracies) * 100.0
        ),
        "std_accuracy": float(
            np.std(accuracies)
        ),
        "std_accuracy_percent": float(
            np.std(accuracies) * 100.0
        ),
        "mean_kappa": float(
            np.mean(kappas)
        ),
        "std_kappa": float(
            np.std(kappas)
        ),
        "minimum_accuracy_percent": float(
            np.min(accuracies) * 100.0
        ),
        "maximum_accuracy_percent": float(
            np.max(accuracies) * 100.0
        ),
    }


def main() -> None:
    """
    Run strict LOSO CSP + LDA across all nine subjects.
    """
    print()
    print("=" * 78)
    print("Strict Cross-Subject CSP + LDA")
    print("=" * 78)
    print(
        "Protocol: train on 8 subjects, "
        "test on 1 completely unseen subject"
    )
    print(
        f"Preprocessing: {PREPROCESSING_CONFIG}"
    )
    print(
        f"CSP components: {N_COMPONENTS}"
    )

    subject_data = load_all_subjects()

    subject_results: list[
        dict[str, object]
    ] = []

    prediction_rows: list[
        dict[str, object]
    ] = []

    for test_subject in SUBJECTS:
        print()
        print("=" * 78)
        print(
            f"LOSO fold: held out {test_subject}"
        )
        print("=" * 78)

        X_train, y_train, training_subjects = (
            get_training_data(
                subject_data,
                test_subject,
            )
        )

        X_test = subject_data[
            test_subject
        ]["X"]

        y_test = subject_data[
            test_subject
        ]["y"]

        print(
            "Training subjects: "
            + ", ".join(training_subjects)
        )

        print(
            f"Training trials: {len(y_train)}"
        )

        print(
            f"Testing trials:  {len(y_test)}"
        )

        classifier = make_csp_lda(
            n_components=N_COMPONENTS,
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

        prediction_seconds = (
            time.perf_counter()
            - prediction_start
        )

        result = create_subject_result(
            test_subject=test_subject,
            training_subjects=training_subjects,
            y_true=y_test,
            y_pred=y_pred,
            training_trials=len(y_train),
            training_seconds=training_seconds,
            prediction_seconds=prediction_seconds,
        )

        subject_results.append(
            result
        )

        prediction_rows.extend(
            create_prediction_rows(
                test_subject=test_subject,
                training_subjects=training_subjects,
                y_true=y_test,
                y_pred=y_pred,
            )
        )

        print_subject_result(
            test_subject,
            result,
        )

    overall_summary = (
        create_overall_summary(
            subject_results
        )
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
        OVERALL_SUMMARY_PATH,
        [overall_summary],
    )

    print()
    print("=" * 78)
    print("FINAL CSP + LDA LOSO RESULTS")
    print("=" * 78)

    print(
        f"{'Subject':<10}"
        f"{'Accuracy':<15}"
        f"{'Kappa':<12}"
    )

    print("-" * 40)

    for result in subject_results:
        print(
            f"{result['test_subject']:<10}"
            f"{result['accuracy_percent']:<15.2f}"
            f"{result['kappa']:<12.3f}"
        )

    print("-" * 40)

    print(
        f"{'Mean':<10}"
        f"{overall_summary['mean_accuracy_percent']:<15.2f}"
        f"{overall_summary['mean_kappa']:<12.3f}"
    )

    print()
    print(
        "Accuracy standard deviation: "
        f"{overall_summary['std_accuracy_percent']:.2f}%"
    )

    print(
        "Kappa standard deviation: "
        f"{overall_summary['std_kappa']:.3f}"
    )

    print()
    print("Saved:")
    print(
        SUBJECT_RESULTS_PATH
    )
    print(
        PREDICTIONS_PATH
    )
    print(
        OVERALL_SUMMARY_PATH
    )


if __name__ == "__main__":
    main()
