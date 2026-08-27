"""
Cross-subject EA + FBCSP + LDA evaluation.

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
Each session
    -> subject/session-wise Euclidean Alignment

Other eight T sessions
    -> pooled aligned EEG
    -> Regularized FBCSP
    -> PCA retaining 90% variance
    -> RBF-SVM

Held-out E session
    -> independently aligned EEG
    -> fitted FBCSP/PCA/LDA
    -> prediction

No labels from the held-out E session are used during training.

Run:
    python -m scripts.cross_subject.euclidean_alignment.fbcsp.svm
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
from bci_wheelchair.models.euclidean_alignment import (
    CLASS_ORDER,
    load_and_align_subject,
    make_ea_fbcsp_svm,
)


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


RESULTS_DIRECTORY = Path(
    "results/cross_subject/euclidean_alignment/fbcsp/svm"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "ea_fbcsp_svm_cross_subject_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "ea_fbcsp_svm_cross_subject_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "ea_fbcsp_svm_cross_subject_overall_summary.csv"
)


def export_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Export dictionaries to CSV."""

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
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def load_training_sessions():
    """
    Load and independently align A01T-A09T.
    """

    training_data = {}

    print()
    print("=" * 78)
    print("LOADING AND ALIGNING TRAINING SESSIONS")
    print("=" * 78)

    for subject in SUBJECTS:

        session = f"{subject}T"

        X, y, alignment_error = (
            load_and_align_subject(
                session
            )
        )

        training_data[subject] = {
            "X": np.asarray(X),
            "y": np.asarray(y),
            "alignment_error": float(
                alignment_error
            ),
            "session": session,
        }

        print(
            f"{session}: "
            f"{len(y)} trials | "
            f"alignment error="
            f"{alignment_error:.8f}"
        )

    return training_data


def load_evaluation_sessions():
    """
    Load and independently align A01E-A09E.
    """

    evaluation_data = {}

    print()
    print("=" * 78)
    print("LOADING AND ALIGNING EVALUATION SESSIONS")
    print("=" * 78)

    for subject in SUBJECTS:

        session = f"{subject}E"

        X, y, alignment_error = (
            load_and_align_subject(
                session
            )
        )

        evaluation_data[subject] = {
            "X": np.asarray(X),
            "y": np.asarray(y),
            "alignment_error": float(
                alignment_error
            ),
            "session": session,
        }

        print(
            f"{session}: "
            f"{len(y)} trials | "
            f"alignment error="
            f"{alignment_error:.8f}"
        )

    return evaluation_data


def create_cross_subject_fold(
    training_data,
    evaluation_data,
    held_out_subject: str,
):
    """
    Build one cross-subject T -> E fold.
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


def create_subject_result(
    held_out_subject: str,
    training_subjects: list[str],
    training_sessions: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    training_trials: int,
    training_seconds: float,
    prediction_seconds: float,
    alignment_error: float,
    retained_pca_components: int,
) -> dict[str, object]:
    """Create result for one held-out E session."""

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
        "training_subjects": "|".join(
            training_subjects
        ),
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
        "alignment_identity_error": float(
            alignment_error
        ),
        "pca_components_retained": (
            retained_pca_components
        ),
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
        "model": "EA_FBCSP_RBF_SVM",
    }

    for true_index, true_class in enumerate(
        CLASS_ORDER
    ):
        for pred_index, pred_class in enumerate(
            CLASS_ORDER
        ):
            result[
                f"cm_{true_class}_pred_{pred_class}"
            ] = int(
                matrix[
                    true_index,
                    pred_index,
                ]
            )

    return result


def create_prediction_rows(
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
            "training_sessions": "|".join(
                training_sessions
            ),
            "test_session": (
                f"{held_out_subject}E"
            ),
            "trial": trial_index,
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
            "model": "EA_FBCSP_RBF_SVM",
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
            ] = float(probability)

        rows.append(row)

    return rows


def create_overall_summary(
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

    return {
        "model": "EA_FBCSP_RBF_SVM",
        "evaluation": (
            "cross_subject_T_to_E_LOSO"
        ),
        "protocol": (
            "train_other_8_T_test_held_out_E"
        ),
        "subjects": len(
            subject_results
        ),
        "total_test_trials": len(
            all_true_labels
        ),
        "mean_accuracy": float(
            np.mean(accuracies)
        ),
        "mean_accuracy_percent": float(
            np.mean(accuracies)
            * 100.0
        ),
        "std_accuracy_percent": float(
            np.std(
                accuracies,
                ddof=1,
            )
            * 100.0
        ),
        "mean_kappa": float(
            np.mean(kappas)
        ),
        "std_kappa": float(
            np.std(
                kappas,
                ddof=1,
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
        "minimum_accuracy_percent": float(
            np.min(accuracies)
            * 100.0
        ),
        "maximum_accuracy_percent": float(
            np.max(accuracies)
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
        "total_experiment_seconds": float(
            total_seconds
        ),
    }


def main() -> None:
    """Run all nine cross-subject T -> E folds."""

    experiment_start = (
        time.perf_counter()
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 78)

    print(
        "CROSS-SUBJECT: "
        "EA + FBCSP + RBF-SVM"
    )

    print("=" * 78)

    print(
        "Train: other 8 T sessions"
    )

    print(
        "Test: held-out subject E session"
    )

    print(
        "Pipeline:"
    )

    print(
        "  Euclidean Alignment"
    )

    print(
        "  -> Regularized FBCSP"
    )

    print(
        "  -> PCA 90%"
    )

    print(
        "  -> RBF-SVM"
    )


    # ---------------------------------------------------------
    # LOAD SESSIONS
    # ---------------------------------------------------------

    training_data = (
        load_training_sessions()
    )

    evaluation_data = (
        load_evaluation_sessions()
    )


    subject_results = []
    prediction_rows = []

    all_true_labels = []
    all_predictions = []


    # ---------------------------------------------------------
    # NINE T -> E FOLDS
    # ---------------------------------------------------------

    for fold_number, held_out_subject in enumerate(
        SUBJECTS,
        start=1,
    ):

        print()
        print("=" * 78)

        print(
            f"Fold {fold_number}/"
            f"{len(SUBJECTS)}"
        )

        print(
            f"Held-out subject: "
            f"{held_out_subject}"
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
            f"Training trials: "
            f"{len(y_train)}"
        )

        print(
            f"Testing trials: "
            f"{len(y_test)}"
        )

        print(
            f"Training shape: "
            f"{X_train.shape}"
        )

        print(
            f"Testing shape: "
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
        # TRAIN
        # -----------------------------------------------------

        classifier = (
            make_ea_fbcsp_svm()
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


        # -----------------------------------------------------
        # TEST
        # -----------------------------------------------------

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


        # -----------------------------------------------------
        # PCA INFO
        # -----------------------------------------------------

        pca = classifier.named_steps[
            "pca"
        ]

        retained_pca_components = int(
            pca.n_components_
        )


        # -----------------------------------------------------
        # RESULT
        # -----------------------------------------------------

        result = create_subject_result(
            held_out_subject=(
                held_out_subject
            ),
            training_subjects=(
                training_subjects
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
            alignment_error=float(
                evaluation_data[
                    held_out_subject
                ][
                    "alignment_error"
                ]
            ),
            retained_pca_components=(
                retained_pca_components
            ),
        )

        subject_results.append(
            result
        )

        prediction_rows.extend(
            create_prediction_rows(
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

        print()

        print(
            f"{test_session} Accuracy: "
            f"{result['accuracy_percent']:.2f}%"
        )

        print(
            f"{test_session} Kappa: "
            f"{result['kappa']:.3f}"
        )

        print(
            "PCA components retained: "
            f"{retained_pca_components}"
        )


        # Save progress.
        export_csv(
            SUBJECT_RESULTS_PATH,
            subject_results,
        )

        export_csv(
            PREDICTIONS_PATH,
            prediction_rows,
        )


    # ---------------------------------------------------------
    # OVERALL
    # ---------------------------------------------------------

    combined_true = np.concatenate(
        all_true_labels
    )

    combined_predictions = np.concatenate(
        all_predictions
    )

    total_seconds = (
        time.perf_counter()
        - experiment_start
    )

    summary = create_overall_summary(
        subject_results=(
            subject_results
        ),
        all_true_labels=(
            combined_true
        ),
        all_predictions=(
            combined_predictions
        ),
        total_seconds=(
            total_seconds
        ),
    )

    export_csv(
        OVERALL_SUMMARY_PATH,
        [summary],
    )


    # ---------------------------------------------------------
    # FINAL OUTPUT
    # ---------------------------------------------------------

    print()
    print("=" * 78)

    print(
        "FINAL CROSS-SUBJECT "
        "EA + FBCSP + RBF-SVM RESULTS"
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
        f"{summary['mean_accuracy_percent']:<15.2f}"
        f"{summary['mean_kappa']:<12.3f}"
    )

    print()

    print(
        "Accuracy SD: "
        f"{summary['std_accuracy_percent']:.2f}%"
    )

    print(
        "Kappa SD: "
        f"{summary['std_kappa']:.3f}"
    )

    print(
        "Pooled accuracy: "
        f"{summary['pooled_accuracy_percent']:.2f}%"
    )

    print(
        "Pooled kappa: "
        f"{summary['pooled_kappa']:.3f}"
    )

    print()

    print("Saved:")
    print(SUBJECT_RESULTS_PATH)
    print(PREDICTIONS_PATH)
    print(OVERALL_SUMMARY_PATH)


if __name__ == "__main__":
    main()
