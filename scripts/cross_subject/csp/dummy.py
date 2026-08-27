"""
Cross-subject CSP + Dummy evaluation.

Supervisor-defined LOSO protocol:

For each held-out subject:
    Train using the T sessions of the other eight subjects.
    Test using the E session of the held-out subject.

Example:
    Fold A01:
        Train: A02T-A09T
        Test:  A01E

    Fold A02:
        Train: A01T, A03T-A09T
        Test:  A02E

    ...

    Fold A09:
        Train: A01T-A08T
        Test:  A09E

The held-out subject contributes no training data.

Pipeline:
    EEG 8-30 Hz
        -> CSP
        -> LDA
        -> prediction on held-out subject evaluation session

Run:
    python -m scripts.cross_subject.csp.dummy
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

from bci_wheelchair.data.processed_loading import (
    load_processed_subject,
)
from bci_wheelchair.models import make_csp_dummy


mne.set_log_level("WARNING")


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

PREPROCESSING_CONFIG = "8-30"
N_COMPONENTS = 6

RESULTS_DIRECTORY = Path(
    "results/cross_subject/csp_dummy"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "csp_dummy_cross_subject_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "csp_dummy_cross_subject_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "csp_dummy_cross_subject_overall_summary.csv"
)


def save_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
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


def load_training_session(
    subject: str,
) -> tuple[np.ndarray, np.ndarray]:
    session = f"{subject}T"

    X, y = load_processed_subject(
        subject=session,
        config=PREPROCESSING_CONFIG,
    )

    return (
        np.asarray(X),
        np.asarray(y),
    )


def load_evaluation_session(
    subject: str,
) -> tuple[np.ndarray, np.ndarray]:
    session = f"{subject}E"

    X, y = load_processed_subject(
        subject=session,
        config=PREPROCESSING_CONFIG,
    )

    return (
        np.asarray(X),
        np.asarray(y),
    )


def load_all_data():
    """
    Load all T and E sessions.

    T sessions are candidates for training.
    E sessions are used only as held-out test data.
    """
    training_data = {}
    evaluation_data = {}

    print()
    print("=" * 78)
    print("LOADING CROSS-SUBJECT DATA")
    print("=" * 78)

    for subject in SUBJECTS:
        train_session = f"{subject}T"
        test_session = f"{subject}E"

        print(
            f"Loading {train_session}..."
        )

        X_train, y_train = (
            load_training_session(subject)
        )

        training_data[subject] = {
            "X": X_train,
            "y": y_train,
        }

        print(
            f"  {train_session}: "
            f"{len(y_train)} trials, "
            f"shape={X_train.shape}"
        )

        print(
            f"Loading {test_session}..."
        )

        X_test, y_test = (
            load_evaluation_session(subject)
        )

        evaluation_data[subject] = {
            "X": X_test,
            "y": y_test,
        }

        print(
            f"  {test_session}: "
            f"{len(y_test)} trials, "
            f"shape={X_test.shape}"
        )

    return training_data, evaluation_data


def get_fold_training_data(
    training_data,
    held_out_subject: str,
):
    """
    Pool T sessions from all subjects except the held-out subject.
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
            training_data[subject]["X"]
            for subject in training_subjects
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            training_data[subject]["y"]
            for subject in training_subjects
        ],
        axis=0,
    )

    return (
        X_train,
        y_train,
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
        "subject": held_out_subject,
        "training_sessions": "|".join(
            training_sessions
        ),
        "test_session": (
            f"{held_out_subject}E"
        ),
        "training_trials": training_trials,
        "testing_trials": len(y_true),
        "accuracy": accuracy,
        "accuracy_percent": (
            accuracy * 100.0
        ),
        "kappa": kappa,
        "left_hand_recall": recalls[0],
        "right_hand_recall": recalls[1],
        "feet_recall": recalls[2],
        "tongue_recall": recalls[3],
        "csp_components": N_COMPONENTS,
        "preprocessing": (
            PREPROCESSING_CONFIG
        ),
        "training_seconds": (
            training_seconds
        ),
        "prediction_seconds": (
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
                "subject": held_out_subject,
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
                "correct": (
                    true_label
                    == predicted_label
                ),
                "model": "CSP_DUMMY",
                "csp_components": (
                    N_COMPONENTS
                ),
                "preprocessing": (
                    PREPROCESSING_CONFIG
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
        "model": "CSP_DUMMY",
        "evaluation": (
            "cross_subject_T_to_E_LOSO"
        ),
        "protocol": (
            "train_other_8_T_test_held_out_E"
        ),
        "subjects": len(
            subject_results
        ),
        "csp_components": N_COMPONENTS,
        "preprocessing": (
            PREPROCESSING_CONFIG
        ),
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
    print()
    print("=" * 78)
    print(
        "Cross-Subject CSP + Dummy"
    )
    print("=" * 78)

    print(
        "Protocol:"
    )
    print(
        "  Train: T sessions from "
        "the other 8 subjects"
    )
    print(
        "  Test:  E session from "
        "the held-out subject"
    )

    print(
        f"Preprocessing: "
        f"{PREPROCESSING_CONFIG}"
    )

    print(
        f"CSP components: "
        f"{N_COMPONENTS}"
    )

    (
        training_data,
        evaluation_data,
    ) = load_all_data()

    subject_results = []
    prediction_rows = []

    for held_out_subject in SUBJECTS:

        print()
        print("=" * 78)

        print(
            f"HELD-OUT SUBJECT: "
            f"{held_out_subject}"
        )

        print("=" * 78)

        (
            X_train,
            y_train,
            training_subjects,
            training_sessions,
        ) = get_fold_training_data(
            training_data,
            held_out_subject,
        )

        X_test = (
            evaluation_data[
                held_out_subject
            ]["X"]
        )

        y_test = (
            evaluation_data[
                held_out_subject
            ]["y"]
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

        classifier = make_csp_dummy(
            n_components=N_COMPONENTS,
        )

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

        result = build_subject_result(
            held_out_subject=(
                held_out_subject
            ),
            training_sessions=(
                training_sessions
            ),
            y_true=y_test,
            y_pred=y_pred,
            training_trials=len(
                y_train
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

        print(
            f"{test_session} Accuracy: "
            f"{result['accuracy_percent']:.2f}%"
        )

        print(
            f"{test_session} Kappa:    "
            f"{result['kappa']:.3f}"
        )

    overall_summary = (
        build_overall_summary(
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
    print(
        "FINAL CROSS-SUBJECT "
        "CSP + Dummy RESULTS"
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
