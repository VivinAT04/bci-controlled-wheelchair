"""
Cross-subject FBCSP + LDA evaluation.

Protocol
--------
For each held-out subject:

    Train:
        T sessions from the other eight subjects

    Test:
        E session from the held-out subject

Examples
--------
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

The held-out subject contributes no training-session data.

Pipeline
--------
Broadband EEG (4-40 Hz)
    -> Filter Bank CSP
    -> Concatenated CSP features
    -> LDA
    -> held-out E-session prediction

Run:
    python -m scripts.cross_subject.fbcsp.lda
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
from bci_wheelchair.data.processed_loading import (
    load_processed_subject,
)
from bci_wheelchair.models import (
    DEFAULT_BANDS,
    make_fbcsp_lda,
)


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

PREPROCESSING = "4-40"
N_COMPONENTS = 4


RESULTS_DIRECTORY = Path(
    "results/cross_subject/fbcsp_lda"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "fbcsp_lda_cross_subject_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "fbcsp_lda_cross_subject_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "fbcsp_lda_cross_subject_overall_summary.csv"
)


def save_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Save rows to CSV."""

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
    """Load one subject's T session."""

    session = f"{subject}T"

    X, y = load_processed_subject(
        subject=session,
        config=PREPROCESSING,
    )

    return (
        np.asarray(X),
        np.asarray(y),
    )


def load_evaluation_session(
    subject: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one subject's E session."""

    session = f"{subject}E"

    X, y = load_processed_subject(
        subject=session,
        config=PREPROCESSING,
    )

    return (
        np.asarray(X),
        np.asarray(y),
    )


def load_all_data():
    """
    Load all T and E sessions separately.

    T sessions are candidates for training only.
    E sessions are used only for held-out testing.
    """

    training_data = {}
    evaluation_data = {}

    print()
    print("=" * 78)
    print(
        "LOADING CROSS-SUBJECT FBCSP DATA"
    )
    print("=" * 78)

    for subject in SUBJECTS:

        train_session = f"{subject}T"
        test_session = f"{subject}E"

        print(
            f"Loading {train_session}..."
        )

        X_train, y_train = (
            load_training_session(
                subject
            )
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
            load_evaluation_session(
                subject
            )
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

    return (
        training_data,
        evaluation_data,
    )


def create_cross_subject_fold(
    training_data,
    evaluation_data,
    held_out_subject: str,
):
    """
    Create one fold.

    Training:
        T sessions from all other subjects.

    Testing:
        E session from held-out subject.
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
    """Build metrics for one held-out E session."""

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
        "training_trials": (
            training_trials
        ),
        "testing_trials": (
            len(y_true)
        ),
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
            balanced_accuracy
            * 100.0
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
        "preprocessing": (
            PREPROCESSING
        ),
        "fbcsp_components_per_band": (
            N_COMPONENTS
        ),
        "number_of_frequency_bands": (
            len(DEFAULT_BANDS)
        ),
        "total_fbcsp_features": (
            N_COMPONENTS
            * len(DEFAULT_BANDS)
        ),
        "classifier": "LDA",
        "training_seconds": float(
            training_seconds
        ),
        "prediction_seconds": float(
            prediction_seconds
        ),
        "prediction_seconds_per_trial": float(
            prediction_seconds
            / len(y_true)
        ),
        "evaluation": (
            "cross_subject_T_to_E_LOSO"
        ),
    }

    for true_index, true_class in enumerate(
        CLASS_ORDER
    ):

        for predicted_index, predicted_class in enumerate(
            CLASS_ORDER
        ):

            result[
                f"cm_{true_class}_pred_{predicted_class}"
            ] = int(
                matrix[
                    true_index,
                    predicted_index,
                ]
            )

    return result


def build_prediction_rows(
    held_out_subject: str,
    training_sessions: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    probability_classes: np.ndarray,
) -> list[dict[str, object]]:
    """Create trial-level prediction rows."""

    probability_lookup = {
        class_name: index
        for index, class_name in enumerate(
            probability_classes
        )
    }

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

        row: dict[str, object] = {
            "subject": held_out_subject,
            "trial": trial_index,
            "training_sessions": "|".join(
                training_sessions
            ),
            "test_session": (
                f"{held_out_subject}E"
            ),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": int(
                true_label
                == predicted_label
            ),
            "predicted_command": (
                CLASS_TO_COMMAND.get(
                    predicted_label,
                    predicted_label,
                )
            ),
            "model": "FBCSP_LDA",
            "preprocessing": (
                PREPROCESSING
            ),
            "fbcsp_components_per_band": (
                N_COMPONENTS
            ),
            "evaluation": (
                "cross_subject_T_to_E_LOSO"
            ),
        }

        for class_name in CLASS_ORDER:

            class_index = (
                probability_lookup.get(
                    class_name
                )
            )

            if class_index is None:
                probability = np.nan
            else:
                probability = probabilities[
                    trial_index - 1,
                    class_index,
                ]

            row[
                f"probability_{class_name}"
            ] = float(
                probability
            )

        rows.append(
            row
        )

    return rows


def build_overall_summary(
    subject_results: list[
        dict[str, object]
    ],
    all_true_labels: np.ndarray,
    all_predictions: np.ndarray,
    total_seconds: float,
) -> dict[str, object]:
    """Calculate overall statistics."""

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
        "model": "FBCSP_LDA",
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
        "fbcsp_components_per_band": (
            N_COMPONENTS
        ),
        "number_of_frequency_bands": (
            len(DEFAULT_BANDS)
        ),
        "total_fbcsp_features": (
            N_COMPONENTS
            * len(DEFAULT_BANDS)
        ),
        "classifier": "LDA",
        "mean_accuracy": float(
            np.mean(
                accuracies
            )
        ),
        "mean_accuracy_percent": float(
            np.mean(
                accuracies
            )
            * 100.0
        ),
        "std_accuracy_percent": float(
            np.std(
                accuracies
            )
            * 100.0
        ),
        "mean_kappa": float(
            np.mean(
                kappas
            )
        ),
        "std_kappa": float(
            np.std(
                kappas
            )
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
        "pooled_accuracy": float(
            pooled_accuracy
        ),
        "pooled_accuracy_percent": float(
            pooled_accuracy
            * 100.0
        ),
        "pooled_kappa": float(
            pooled_kappa
        ),
        "overall_left_hand_recall": float(
            recalls[0]
        ),
        "overall_right_hand_recall": float(
            recalls[1]
        ),
        "overall_feet_recall": float(
            recalls[2]
        ),
        "overall_tongue_recall": float(
            recalls[3]
        ),
        "minimum_accuracy_percent": float(
            np.min(
                accuracies
            )
            * 100.0
        ),
        "maximum_accuracy_percent": float(
            np.max(
                accuracies
            )
            * 100.0
        ),
        "total_experiment_seconds": float(
            total_seconds
        ),
    }

    for true_index, true_class in enumerate(
        CLASS_ORDER
    ):

        for predicted_index, predicted_class in enumerate(
            CLASS_ORDER
        ):

            summary[
                f"cm_{true_class}_pred_{predicted_class}"
            ] = int(
                matrix[
                    true_index,
                    predicted_index,
                ]
            )

    return summary


def main() -> None:
    """Run all nine cross-subject folds."""

    experiment_start = (
        time.perf_counter()
    )

    print()
    print("=" * 78)

    print(
        "Cross-Subject FBCSP + LDA"
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
        f"{PREPROCESSING}"
    )

    print(
        f"CSP components per band: "
        f"{N_COMPONENTS}"
    )

    print(
        f"Frequency bands: "
        f"{len(DEFAULT_BANDS)}"
    )

    print(
        f"Total FBCSP features: "
        f"{N_COMPONENTS * len(DEFAULT_BANDS)}"
    )

    print()
    print("Frequency bands:")

    for band_index, (
        low,
        high,
    ) in enumerate(
        DEFAULT_BANDS,
        start=1,
    ):
        print(
            f"  Band {band_index}: "
            f"{low}-{high} Hz"
        )

    (
        training_data,
        evaluation_data,
    ) = load_all_data()

    subject_results = []
    prediction_rows = []

    all_true_labels = []
    all_predictions = []

    for fold_number, held_out_subject in enumerate(
        SUBJECTS,
        start=1,
    ):

        print()
        print("=" * 78)

        print(
            f"Fold {fold_number}/"
            f"{len(SUBJECTS)}: "
            f"held out {held_out_subject}"
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
            training_data=(
                training_data
            ),
            evaluation_data=(
                evaluation_data
            ),
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
            f"Training data shape: "
            f"{X_train.shape}"
        )

        print(
            f"Testing data shape:  "
            f"{X_test.shape}"
        )

        if (
            X_train.shape[1:]
            != X_test.shape[1:]
        ):
            raise ValueError(
                f"{held_out_subject}: "
                "training/test EEG shape "
                "mismatch: "
                f"{X_train.shape[1:]} "
                "vs "
                f"{X_test.shape[1:]}"
            )

        classifier = (
            make_fbcsp_lda(
                n_components=(
                    N_COMPONENTS
                ),
                bands=DEFAULT_BANDS,
            )
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

        probabilities = (
            classifier.predict_proba(
                X_test
            )
        )

        prediction_seconds = (
            time.perf_counter()
            - prediction_start
        )

        result = (
            build_subject_result(
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
                probabilities=(
                    probabilities
                ),
                probability_classes=(
                    classifier.classes_
                ),
            )
        )

        all_true_labels.append(
            np.asarray(
                y_test
            )
        )

        all_predictions.append(
            np.asarray(
                y_pred
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

        save_csv(
            SUBJECT_RESULTS_PATH,
            subject_results,
        )

        save_csv(
            PREDICTIONS_PATH,
            prediction_rows,
        )


    combined_true_labels = (
        np.concatenate(
            all_true_labels
        )
    )

    combined_predictions = (
        np.concatenate(
            all_predictions
        )
    )

    total_seconds = (
        time.perf_counter()
        - experiment_start
    )

    overall_summary = (
        build_overall_summary(
            subject_results=(
                subject_results
            ),
            all_true_labels=(
                combined_true_labels
            ),
            all_predictions=(
                combined_predictions
            ),
            total_seconds=(
                total_seconds
            ),
        )
    )

    save_csv(
        OVERALL_SUMMARY_PATH,
        [overall_summary],
    )


    print()
    print("=" * 78)

    print(
        "FINAL CROSS-SUBJECT "
        "FBCSP + LDA RESULTS"
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
        "Pooled accuracy: "
        f"{overall_summary['pooled_accuracy_percent']:.2f}%"
    )

    print(
        "Pooled kappa: "
        f"{overall_summary['pooled_kappa']:.3f}"
    )

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
