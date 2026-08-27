"""
Cross-subject Riemannian Tangent-Space + Shrinkage LDA evaluation.

Protocol
--------
For each held-out subject:

    Train:
        T sessions from the other eight subjects

    Test:
        E session from the held-out subject

Examples
--------
Held out A01:
    Train: A02T-A09T
    Test:  A01E

Held out A02:
    Train: A01T, A03T-A09T
    Test:  A02E

...

Held out A09:
    Train: A01T-A08T
    Test:  A09E

Pipeline
--------
EEG 8-30 Hz
    -> OAS covariance matrices
    -> Riemannian tangent-space projection
    -> Shrinkage LDA
    -> held-out E-session prediction

The held-out subject contributes no T-session data to training.
No E-session data are used during model fitting.

Run:
    python -m scripts.cross_subject.riemannian_tangent.dummy
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

COVARIANCE_ESTIMATOR = "oas"
TANGENT_METRIC = "riemann"
LDA_SOLVER = "lsqr"
LDA_SHRINKAGE = "auto"


RESULTS_DIRECTORY = Path(
    "results/cross_subject/riemannian_tangent_dummy"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_tangent_dummy_cross_subject_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_tangent_dummy_cross_subject_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_tangent_dummy_cross_subject_overall_summary.csv"
)


def save_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Save dictionary rows to CSV."""

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


def load_training_sessions() -> dict[
    str,
    dict[str, np.ndarray],
]:
    """Load A01T-A09T."""

    training_data = {}

    print()
    print("=" * 78)
    print("Loading training sessions A01T-A09T")
    print("=" * 78)

    for subject in SUBJECTS:
        session = f"{subject}T"

        print(
            f"Loading {session}..."
        )

        X, y = load_processed_subject(
            subject=session,
            config=PREPROCESSING,
        )

        X = np.asarray(X)
        y = np.asarray(y)

        training_data[subject] = {
            "X": X,
            "y": y,
        }

        print(
            f"  trials={len(y)}, "
            f"shape={X.shape}"
        )

    return training_data


def load_evaluation_sessions() -> dict[
    str,
    dict[str, np.ndarray],
]:
    """Load A01E-A09E for held-out testing."""

    evaluation_data = {}

    print()
    print("=" * 78)
    print("Loading evaluation sessions A01E-A09E")
    print("=" * 78)

    for subject in SUBJECTS:
        session = f"{subject}E"

        print(
            f"Loading {session}..."
        )

        X, y = load_processed_subject(
            subject=session,
            config=PREPROCESSING,
        )

        X = np.asarray(X)
        y = np.asarray(y)

        evaluation_data[subject] = {
            "X": X,
            "y": y,
        }

        print(
            f"  trials={len(y)}, "
            f"shape={X.shape}"
        )

    return evaluation_data


def create_cross_subject_fold(
    training_data: dict[
        str,
        dict[str, np.ndarray],
    ],
    evaluation_data: dict[
        str,
        dict[str, np.ndarray],
    ],
    held_out_subject: str,
):
    """
    Create one cross-subject T -> E fold.

    Train:
        T sessions from the other eight subjects.

    Test:
        E session from the held-out subject.
    """

    training_subjects = [
        subject
        for subject in SUBJECTS
        if subject != held_out_subject
    ]

    training_sessions = [
        f"{subject}T"
        for subject in training_subjects
    ]

    X_train = np.concatenate(
        [
            training_data[
                subject
            ]["X"]
            for subject in training_subjects
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            training_data[
                subject
            ]["y"]
            for subject in training_subjects
        ],
        axis=0,
    )

    X_test = evaluation_data[
        held_out_subject
    ]["X"]

    y_test = evaluation_data[
        held_out_subject
    ]["y"]

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        training_subjects,
        training_sessions,
    )


def build_subject_result(
    held_out_subject: str,
    training_sessions: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_trials: int,
    training_seconds: float,
    prediction_seconds: float,
) -> dict[str, object]:
    """Calculate metrics for one held-out subject."""

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
        "subject": held_out_subject,
        "training_sessions": "|".join(
            training_sessions
        ),
        "test_session": (
            f"{held_out_subject}E"
        ),
        "training_trials": training_trials,
        "testing_trials": len(y_true),
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
        "covariance_estimator": (
            COVARIANCE_ESTIMATOR
        ),
        "tangent_metric": (
            TANGENT_METRIC
        ),
        "lda_solver": LDA_SOLVER,
        "lda_shrinkage": (
            LDA_SHRINKAGE
        ),
        "training_seconds": float(
            training_seconds
        ),
        "prediction_seconds": float(
            prediction_seconds
        ),
        "evaluation": (
            "cross_subject_T_to_E_LOSO"
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
    held_out_subject: str,
    training_sessions: list[str],
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
                "subject": (
                    held_out_subject
                ),
                "trial": trial_index,
                "training_sessions": "|".join(
                    training_sessions
                ),
                "test_session": (
                    f"{held_out_subject}E"
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
                "preprocessing": (
                    PREPROCESSING
                ),
                "covariance_estimator": (
                    COVARIANCE_ESTIMATOR
                ),
                "tangent_metric": (
                    TANGENT_METRIC
                ),
                "lda_solver": (
                    LDA_SOLVER
                ),
                "lda_shrinkage": (
                    LDA_SHRINKAGE
                ),
                "evaluation": (
                    "cross_subject_T_to_E_LOSO"
                ),
            }
        )

    return rows


def build_overall_summary(
    subject_results: list[
        dict[str, object]
    ],
) -> dict[str, object]:
    """Calculate statistics across all held-out subjects."""

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
        "model": (
            "Riemannian_Tangent_DUMMY"
        ),
        "evaluation": (
            "cross_subject_T_to_E_LOSO"
        ),
        "protocol": (
            "train_other_8_T_test_held_out_E"
        ),
        "subjects": len(
            subject_results
        ),
        "preprocessing": (
            PREPROCESSING
        ),
        "covariance_estimator": (
            COVARIANCE_ESTIMATOR
        ),
        "tangent_metric": (
            TANGENT_METRIC
        ),
        "lda_solver": (
            LDA_SOLVER
        ),
        "lda_shrinkage": (
            LDA_SHRINKAGE
        ),
        "mean_accuracy": float(
            np.mean(accuracies)
        ),
        "mean_accuracy_percent": float(
            np.mean(accuracies)
            * 100.0
        ),
        "std_accuracy": float(
            np.std(accuracies)
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
    """
    Run nine-fold cross-subject Tangent-Space + Dummy.
    """

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 78)

    print(
        "CROSS-SUBJECT: "
        "RIEMANNIAN TANGENT SPACE + SHRINKAGE LDA"
    )

    print("=" * 78)

    print(
        "Protocol: train on other 8 T sessions, "
        "test on held-out subject E session"
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
        "Classifier: Dummy"
    )

    print(
        "Dummy strategy: stratified"
    )


    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------

    training_data = (
        load_training_sessions()
    )

    evaluation_data = (
        load_evaluation_sessions()
    )


    # ---------------------------------------------------------
    # 2. RUN NINE HELD-OUT SUBJECT FOLDS
    # ---------------------------------------------------------

    subject_results = []
    prediction_rows = []

    for fold_number, held_out_subject in enumerate(
        SUBJECTS,
        start=1,
    ):

        print()
        print("=" * 78)

        print(
            f"Fold {fold_number}/{len(SUBJECTS)}: "
            f"held-out subject {held_out_subject}"
        )

        print("=" * 78)

        (
            X_train,
            y_train,
            X_test,
            y_test,
            training_subjects,
            training_sessions,
        ) = create_cross_subject_fold(
            training_data=training_data,
            evaluation_data=evaluation_data,
            held_out_subject=(
                held_out_subject
            ),
        )

        test_session = (
            f"{held_out_subject}E"
        )

        print(
            "Training sessions: "
            + ", ".join(
                training_sessions
            )
        )

        print(
            f"Test session: "
            f"{test_session}"
        )

        print(
            f"Training trials: "
            f"{len(y_train)}"
        )

        print(
            f"Testing trials:  "
            f"{len(y_test)}"
        )

        print(
            f"Training shape: "
            f"{X_train.shape}"
        )

        print(
            f"Testing shape:  "
            f"{X_test.shape}"
        )

        if (
            X_train.shape[1:]
            != X_test.shape[1:]
        ):
            raise ValueError(
                f"{held_out_subject}: "
                "training/test EEG shape mismatch: "
                f"{X_train.shape[1:]} vs "
                f"{X_test.shape[1:]}"
            )


        # -----------------------------------------------------
        # 3. CREATE FRESH MODEL FOR THIS FOLD
        # -----------------------------------------------------

        classifier = make_tangent_dummy(
            covariance_estimator=(
                COVARIANCE_ESTIMATOR
            ),
            tangent_metric=(
                TANGENT_METRIC
            ),
        )


        # -----------------------------------------------------
        # 4. TRAIN USING OTHER EIGHT T SESSIONS ONLY
        # -----------------------------------------------------

        training_start = (
            time.perf_counter()
        )

        classifier.fit(
            X_train,
            y_train,
        )

        training_seconds = (
            time.perf_counter()
            - training_start
        )


        # -----------------------------------------------------
        # 5. TEST ON HELD-OUT E SESSION
        # -----------------------------------------------------

        prediction_start = (
            time.perf_counter()
        )

        y_pred = classifier.predict(
            X_test
        )

        prediction_seconds = (
            time.perf_counter()
            - prediction_start
        )


        # -----------------------------------------------------
        # 6. METRICS
        # -----------------------------------------------------

        result = build_subject_result(
            held_out_subject=(
                held_out_subject
            ),
            training_sessions=(
                training_sessions
            ),
            y_true=y_test,
            y_pred=y_pred,
            training_trials=(
                len(y_train)
            ),
            training_seconds=(
                training_seconds
            ),
            prediction_seconds=(
                prediction_seconds
            ),
        )

        subject_results.append(
            result
        )

        prediction_rows.extend(
            build_prediction_rows(
                held_out_subject=(
                    held_out_subject
                ),
                training_sessions=(
                    training_sessions
                ),
                y_true=y_test,
                y_pred=y_pred,
            )
        )

        print()

        print(
            f"{test_session} Accuracy: "
            f"{result['accuracy_percent']:.2f}%"
        )

        print(
            f"{test_session} Kappa:    "
            f"{result['kappa']:.3f}"
        )

        print(
            f"Training time: "
            f"{training_seconds:.2f}s"
        )

        print(
            f"Prediction time: "
            f"{prediction_seconds:.3f}s"
        )


    # ---------------------------------------------------------
    # 7. OVERALL SUMMARY
    # ---------------------------------------------------------

    overall_summary = (
        build_overall_summary(
            subject_results
        )
    )


    # ---------------------------------------------------------
    # 8. SAVE RESULTS
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
    # 9. FINAL OUTPUT
    # ---------------------------------------------------------

    print()
    print("=" * 78)

    print(
        "FINAL CROSS-SUBJECT "
        "TANGENT SPACE + SHRINKAGE LDA RESULTS"
    )

    print("=" * 78)

    print(
        f"{'Subject':<10}"
        f"{'Test':<10}"
        f"{'Accuracy':<15}"
        f"{'Kappa':<12}"
    )

    print("-" * 47)

    for result in subject_results:

        print(
            f"{result['subject']:<10}"
            f"{result['test_session']:<10}"
            f"{result['accuracy_percent']:<15.2f}"
            f"{result['kappa']:<12.3f}"
        )

    print("-" * 47)

    print(
        f"{'Mean':<20}"
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

    print(
        "Mean balanced accuracy: "
        f"{overall_summary['mean_balanced_accuracy_percent']:.2f}%"
    )

    print()

    print("Saved:")
    print(SUBJECT_RESULTS_PATH)
    print(PREDICTIONS_PATH)
    print(OVERALL_SUMMARY_PATH)


if __name__ == "__main__":
    main()
