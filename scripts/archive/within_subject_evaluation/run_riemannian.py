"""
Within-subject Riemannian MDM evaluation.

Pipeline:

    EEG epochs
    -> OAS covariance matrices
    -> Riemannian Minimum Distance to Mean (MDM)

Evaluation:

    Leave-One-Out Cross Validation independently for each subject.

Run:

    python -m scripts.within_subject.run_riemannian
"""

from __future__ import annotations

import csv
from pathlib import Path

import mne
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from bci_wheelchair.data.processed_loading import load_processed_subject
from bci_wheelchair.models import make_riemannian_mdm


mne.set_log_level("ERROR")

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

PREPROCESSING = "8-30"

RESULTS_DIRECTORY = Path(
    "results/within_subject/riemannian/mdm"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_mdm_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_mdm_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_mdm_overall_summary.csv"
)


def save_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Save dictionaries to CSV."""

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


def build_subject_result(
    subject: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    """Calculate subject-level metrics."""

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
        "subject": subject,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "kappa": kappa,
        "left_hand_recall": recalls[0],
        "right_hand_recall": recalls[1],
        "feet_recall": recalls[2],
        "tongue_recall": recalls[3],
        "preprocessing": PREPROCESSING,
        "covariance_estimator": "oas",
        "classifier": "riemannian_mdm",
        "evaluation": "within_subject_loocv",
    }

    for true_index, true_label in enumerate(CLASS_ORDER):
        for pred_index, pred_label in enumerate(CLASS_ORDER):
            result[
                f"cm_{true_label}_pred_{pred_label}"
            ] = int(
                matrix[
                    true_index,
                    pred_index,
                ]
            )

    return result


def build_prediction_rows(
    subject: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, object]]:
    """Create trial-level prediction rows."""

    rows = []

    for trial_index, (
        true_label,
        predicted_label,
    ) in enumerate(
        zip(y_true, y_pred),
        start=1,
    ):
        rows.append(
            {
                "subject": subject,
                "trial": trial_index,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": (
                    true_label
                    == predicted_label
                ),
                "model": "Riemannian_MDM",
                "evaluation": "within_subject_loocv",
                "preprocessing": PREPROCESSING,
            }
        )

    return rows


def build_overall_summary(
    subject_results: list[dict[str, object]],
) -> dict[str, object]:
    """Build overall mean/std summary."""

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
        "model": "Riemannian_MDM",
        "evaluation": "within_subject_loocv",
        "subjects": len(subject_results),
        "preprocessing": PREPROCESSING,
        "covariance_estimator": "oas",
        "metric": "riemann",
        "mean_accuracy": float(
            np.mean(accuracies)
        ),
        "mean_accuracy_percent": float(
            np.mean(accuracies) * 100.0
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
    """Run within-subject Riemannian MDM evaluation."""

    print()
    print("=" * 78)
    print("Within-Subject Riemannian MDM")
    print("=" * 78)

    print("Evaluation: LOOCV independently per subject")
    print("Preprocessing: 8-30 Hz")
    print("Covariance estimator: OAS")
    print("Classifier: Riemannian MDM")

    subject_results = []
    prediction_rows = []

    for subject in SUBJECTS:

        print()
        print("=" * 78)
        print(f"Running {subject}")
        print("=" * 78)

        X, y = load_processed_subject(
            subject=subject,
            config=PREPROCESSING,
        )

        print(
            f"Trials: {X.shape[0]}, "
            f"channels: {X.shape[1]}, "
            f"samples: {X.shape[2]}"
        )

        classifier = make_riemannian_mdm(
            covariance_estimator="oas",
            metric="riemann",
        )

        cross_validation = LeaveOneOut()

        y_pred = cross_val_predict(
            classifier,
            X,
            y,
            cv=cross_validation,
        )

        result = build_subject_result(
            subject,
            y,
            y_pred,
        )

        subject_results.append(
            result
        )

        prediction_rows.extend(
            build_prediction_rows(
                subject,
                y,
                y_pred,
            )
        )

        print(
            f"{subject} Accuracy: "
            f"{result['accuracy_percent']:.2f}%"
        )

        print(
            f"{subject} Kappa:    "
            f"{result['kappa']:.3f}"
        )

    overall_summary = build_overall_summary(
        subject_results
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
    print("FINAL WITHIN-SUBJECT RIEMANNIAN MDM RESULTS")
    print("=" * 78)

    print(
        f"{'Subject':<10}"
        f"{'Accuracy':<15}"
        f"{'Kappa':<12}"
    )

    print("-" * 40)

    for result in subject_results:
        print(
            f"{result['subject']:<10}"
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
    print(SUBJECT_RESULTS_PATH)
    print(PREDICTIONS_PATH)
    print(OVERALL_SUMMARY_PATH)


if __name__ == "__main__":
    main()
