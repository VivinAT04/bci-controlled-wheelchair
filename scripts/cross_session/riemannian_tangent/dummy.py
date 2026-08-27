"""
Cross-session Riemannian Tangent-Space + Shrinkage LDA evaluation.

Protocol
--------
Train one model using pooled training-session data:

    A01T + A02T + ... + A09T

Test the same trained model independently on:

    A01E
    A02E
    ...
    A09E

Pipeline:
    EEG epochs
    -> OAS covariance matrices
    -> Riemannian tangent-space projection
    -> Shrinkage LDA

No evaluation-session data are used during model training.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)

from bci_wheelchair.data.processed_loading import (
    load_processed_subject,
)
from bci_wheelchair.models import make_tangent_dummy


SUBJECTS = [
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "A06",
    "A07",
    "A08",
    "A09",
]

CLASS_ORDER = [
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
]

PREPROCESSING = "8-30"


RESULTS_DIRECTORY = Path(
    "results/cross_session/riemannian/tangent_dummy"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "tangent_dummy_cross_session_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "tangent_dummy_cross_session_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "tangent_dummy_cross_session_overall_summary.csv"
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


def load_pooled_training_data() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Load and concatenate A01T-A09T processed EEG."""

    X_blocks = []
    y_blocks = []

    print()
    print("Loading pooled training sessions")
    print("=" * 60)

    for subject in SUBJECTS:

        train_subject = f"{subject}T"

        X_subject, y_subject = (
            load_processed_subject(
                subject=train_subject,
                config=PREPROCESSING,
            )
        )

        X_subject = np.asarray(
            X_subject
        )

        y_subject = np.asarray(
            y_subject
        )

        print(
            f"{train_subject}: "
            f"{len(y_subject)} trials"
        )

        X_blocks.append(
            X_subject
        )

        y_blocks.append(
            y_subject
        )

    X_train = np.concatenate(
        X_blocks,
        axis=0,
    )

    y_train = np.concatenate(
        y_blocks,
        axis=0,
    )

    print("-" * 60)

    print(
        "Total pooled training trials: "
        f"{len(y_train)}"
    )

    print(
        "Pooled training shape: "
        f"{X_train.shape}"
    )

    return X_train, y_train


def build_subject_result(
    subject: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_training_trials: int,
) -> dict[str, object]:
    """Calculate metrics for one evaluation session."""

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

    balanced_accuracy = recall_score(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
    )

    result: dict[str, object] = {
        "subject": subject,
        "training_sessions": "A01T-A09T",
        "test_session": f"{subject}E",
        "accuracy": float(
            accuracy
        ),
        "accuracy_percent": float(
            accuracy * 100.0
        ),
        "kappa": float(
            kappa
        ),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "balanced_accuracy_percent": float(
            balanced_accuracy * 100.0
        ),
        "left_hand_recall": float(
            recalls[0]
        ),
        "right_hand_recall": float(
            recalls[1]
        ),
        "feet_recall": float(
            recalls[2]
        ),
        "tongue_recall": float(
            recalls[3]
        ),
        "preprocessing": PREPROCESSING,
        "covariance_estimator": "oas",
        "tangent_metric": "riemann",
        "lda_solver": "lsqr",
        "lda_shrinkage": "auto",
        "evaluation": (
            "cross_session_pooled_training_all_evaluation_sessions"
        ),
        "n_training_trials": (
            n_training_trials
        ),
        "n_test_trials": len(
            y_true
        ),
    }

    for true_index, true_label in enumerate(
        CLASS_ORDER
    ):
        for pred_index, pred_label in enumerate(
            CLASS_ORDER
        ):
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
        zip(
            y_true,
            y_pred,
        ),
        start=1,
    ):

        rows.append(
            {
                "subject": subject,
                "trial": trial_index,
                "training_sessions": (
                    "A01T-A09T"
                ),
                "test_session": (
                    f"{subject}E"
                ),
                "true_label": true_label,
                "predicted_label": (
                    predicted_label
                ),
                "correct": int(
                    true_label
                    == predicted_label
                ),
                "model": (
                    "Riemannian_Tangent_DUMMY"
                ),
                "evaluation": (
                    "cross_session_pooled_training_all_evaluation_sessions"
                ),
                "preprocessing": (
                    PREPROCESSING
                ),
            }
        )

    return rows


def build_overall_summary(
    subject_results: list[
        dict[str, object]
    ],
) -> dict[str, object]:
    """Calculate summary statistics across A01E-A09E."""

    accuracies = np.asarray(
        [
            float(
                result["accuracy"]
            )
            for result in subject_results
        ]
    )

    kappas = np.asarray(
        [
            float(
                result["kappa"]
            )
            for result in subject_results
        ]
    )

    balanced_accuracies = np.asarray(
        [
            float(
                result[
                    "balanced_accuracy"
                ]
            )
            for result in subject_results
        ]
    )

    return {
        "model": "Riemannian_Tangent_DUMMY",
        "evaluation": (
            "cross_session_pooled_training_all_evaluation_sessions"
        ),
        "training_sessions": (
            "A01T-A09T"
        ),
        "test_sessions": (
            "A01E-A09E"
        ),
        "subjects": len(
            subject_results
        ),
        "preprocessing": PREPROCESSING,
        "covariance_estimator": "oas",
        "tangent_metric": "riemann",
        "lda_solver": "lsqr",
        "lda_shrinkage": "auto",
        "mean_accuracy": float(
            np.mean(accuracies)
        ),
        "mean_accuracy_percent": float(
            np.mean(accuracies)
            * 100.0
        ),
        "std_accuracy_percent": float(
            np.std(accuracies)
            * 100.0
        ),
        "mean_kappa": float(
            np.mean(kappas)
        ),
        "std_kappa": float(
            np.std(kappas)
        ),
        "mean_balanced_accuracy": float(
            np.mean(
                balanced_accuracies
            )
        ),
        "mean_balanced_accuracy_percent": float(
            np.mean(
                balanced_accuracies
            )
            * 100.0
        ),
        "minimum_accuracy_percent": float(
            np.min(accuracies)
            * 100.0
        ),
        "maximum_accuracy_percent": float(
            np.max(accuracies)
            * 100.0
        ),
    }


def main() -> None:
    """Run pooled cross-session Tangent-Space + Dummy."""

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 78)

    print(
        "CROSS-SESSION: "
        "RIEMANNIAN TANGENT SPACE + SHRINKAGE LDA"
    )

    print("=" * 78)

    print(
        "Train once: "
        "A01T + A02T + ... + A09T"
    )

    print(
        "Test same model: "
        "A01E + A02E + ... + A09E"
    )

    print(
        f"Preprocessing: "
        f"{PREPROCESSING} Hz"
    )

    print(
        "Covariance estimator: OAS"
    )

    print(
        "Tangent metric: Riemannian"
    )

    print(
        "Classifier: Shrinkage LDA"
    )


    # ---------------------------------------------------------
    # 1. LOAD AND POOL A01T-A09T
    # ---------------------------------------------------------

    X_train, y_train = (
        load_pooled_training_data()
    )


    # ---------------------------------------------------------
    # 2. TRAIN ONE MODEL
    # ---------------------------------------------------------

    print()

    print(
        "Training one pooled "
        "Tangent-Space + Dummy model..."
    )

    classifier = make_tangent_dummy(
        covariance_estimator="oas",
        tangent_metric="riemann",
    )

    classifier.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )


    # ---------------------------------------------------------
    # 3. TEST SAME MODEL ON A01E-A09E
    # ---------------------------------------------------------

    subject_results = []
    prediction_rows = []

    for subject in SUBJECTS:

        test_subject = (
            f"{subject}E"
        )

        print()
        print("=" * 60)

        print(
            f"Testing pooled model "
            f"on {test_subject}"
        )

        print("=" * 60)

        X_test, y_test = (
            load_processed_subject(
                subject=test_subject,
                config=PREPROCESSING,
            )
        )

        X_test = np.asarray(
            X_test
        )

        y_test = np.asarray(
            y_test
        )

        if (
            X_train.shape[1:]
            != X_test.shape[1:]
        ):
            raise ValueError(
                f"{subject}: "
                "training/test EEG shape "
                "mismatch: "
                f"{X_train.shape[1:]} "
                "vs "
                f"{X_test.shape[1:]}"
            )

        print(
            f"Testing trials: "
            f"{len(y_test)}"
        )

        y_pred = (
            classifier.predict(
                X_test
            )
        )

        result = (
            build_subject_result(
                subject=subject,
                y_true=y_test,
                y_pred=y_pred,
                n_training_trials=(
                    len(y_train)
                ),
            )
        )

        subject_results.append(
            result
        )

        prediction_rows.extend(
            build_prediction_rows(
                subject=subject,
                y_true=y_test,
                y_pred=y_pred,
            )
        )

        print(
            f"{test_subject} Accuracy: "
            f"{result['accuracy_percent']:.2f}%"
        )

        print(
            f"{test_subject} Kappa:    "
            f"{result['kappa']:.3f}"
        )


    # ---------------------------------------------------------
    # 4. OVERALL SUMMARY
    # ---------------------------------------------------------

    overall_summary = (
        build_overall_summary(
            subject_results
        )
    )


    # ---------------------------------------------------------
    # 5. SAVE RESULTS
    # ---------------------------------------------------------

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


    # ---------------------------------------------------------
    # 6. FINAL OUTPUT
    # ---------------------------------------------------------

    print()
    print("=" * 78)

    print(
        "FINAL CROSS-SESSION "
        "TANGENT + LDA RESULTS"
    )

    print("=" * 78)

    print(
        f"{'Test':<10}"
        f"{'Accuracy':<15}"
        f"{'Kappa':<12}"
    )

    print("-" * 40)

    for result in subject_results:

        print(
            f"{result['test_session']:<10}"
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
        "Accuracy SD: "
        f"{overall_summary['std_accuracy_percent']:.2f}%"
    )

    print(
        "Kappa SD: "
        f"{overall_summary['std_kappa']:.3f}"
    )

    print()

    print("Saved:")
    print(SUBJECT_RESULTS_PATH)
    print(PREDICTIONS_PATH)
    print(OVERALL_SUMMARY_PATH)


if __name__ == "__main__":
    main()
