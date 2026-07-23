"""
Cross-subject Riemannian classifier with new-user calibration.

Experiment design
-----------------

For every target subject:

    1. Train using the other eight subjects.
    2. Add a small labelled calibration set from the target subject.
    3. Test on the remaining target-subject trials.

Calibration sizes:

    0%   - strict LOSO
    10%  - small calibration
    20%  - medium calibration
    30%  - larger calibration

For 10%, 20%, and 30%, the calibration split is repeated using five
different random seeds.

Classifier:

    Euclidean Alignment
    -> OAS covariance matrices
    -> Riemannian tangent space
    -> StandardScaler
    -> Shrinkage LDA

Important
---------
This script uses the same subject-wise Euclidean Alignment protocol as
the existing LOSO experiment so that the calibration results are directly
comparable with the established 53.67% baseline.

Run:

    python -m scripts.cross_subject.run_cross_subject_riemannian_calibration
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.cross_subject.run_cross_subject_evaluation import (
    CLASS_ORDER,
    load_and_align_subject,
)


# ---------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------

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

CALIBRATION_FRACTIONS = [
    0.0,
    0.10,
    0.20,
    0.30,
]

RANDOM_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

RESULTS_DIRECTORY = Path(
    "results/cross_subject/riemannian/riemannian_calibration"
)

RUN_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_calibration_run_results.csv"
)

SUBJECT_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_calibration_subject_summary.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_calibration_overall_summary.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_calibration_predictions.csv"
)

BEST_CONFIGURATION_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_calibration_best_configuration.csv"
)

STRICT_LOSO_REFERENCE_ACCURACY = 0.5366512345679012
STRICT_LOSO_REFERENCE_KAPPA = 0.382201646090535


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------

def export_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write a list of dictionaries to a CSV file."""
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
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------

def build_classifier() -> Pipeline:
    """
    Build the best Riemannian classifier from the strict LOSO experiment.
    """
    return Pipeline(
        [
            (
                "covariance",
                Covariances(
                    estimator="oas",
                ),
            ),
            (
                "tangent_space",
                TangentSpace(
                    metric="riemann",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto",
                    priors=np.full(
                        len(CLASS_ORDER),
                        1.0 / len(CLASS_ORDER),
                    ),
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_all_subjects() -> dict[str, dict[str, Any]]:
    """
    Load and Euclidean-align every subject independently.
    """
    subject_data: dict[str, dict[str, Any]] = {}

    print("=" * 80)
    print("Loading and Euclidean-aligning all subjects")
    print("=" * 80)

    for subject in SUBJECTS:
        X, y, alignment_error = load_and_align_subject(
            subject
        )

        subject_data[subject] = {
            "X": np.asarray(
                X,
                dtype=np.float64,
            ),
            "y": np.asarray(y),
            "alignment_error": float(
                alignment_error
            ),
        }

    return subject_data


def create_source_training_data(
    subject_data: dict[str, dict[str, Any]],
    target_subject: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """
    Create source training data using the other eight subjects.
    """
    training_subjects = [
        subject
        for subject in SUBJECTS
        if subject != target_subject
    ]

    X_source = np.concatenate(
        [
            subject_data[subject]["X"]
            for subject in training_subjects
        ],
        axis=0,
    )

    y_source = np.concatenate(
        [
            subject_data[subject]["y"]
            for subject in training_subjects
        ],
        axis=0,
    )

    return (
        X_source,
        y_source,
        training_subjects,
    )


# ---------------------------------------------------------------------
# Calibration splitting
# ---------------------------------------------------------------------

def create_calibration_split(
    X_target: np.ndarray,
    y_target: np.ndarray,
    calibration_fraction: float,
    random_seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Split the target subject into calibration and testing portions.

    The split is stratified so that each motor-imagery class is represented.
    """
    total_trials = len(y_target)

    if calibration_fraction == 0.0:
        calibration_indices = np.empty(
            0,
            dtype=int,
        )

        testing_indices = np.arange(
            total_trials,
            dtype=int,
        )

        X_calibration = X_target[
            calibration_indices
        ]

        y_calibration = y_target[
            calibration_indices
        ]

        X_test = X_target[
            testing_indices
        ]

        y_test = y_target[
            testing_indices
        ]

        return (
            X_calibration,
            y_calibration,
            X_test,
            y_test,
            calibration_indices,
            testing_indices,
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=calibration_fraction,
        random_state=random_seed,
    )

    calibration_indices, testing_indices = next(
        splitter.split(
            X_target,
            y_target,
        )
    )

    X_calibration = X_target[
        calibration_indices
    ]

    y_calibration = y_target[
        calibration_indices
    ]

    X_test = X_target[
        testing_indices
    ]

    y_test = y_target[
        testing_indices
    ]

    return (
        X_calibration,
        y_calibration,
        X_test,
        y_test,
        calibration_indices,
        testing_indices,
    )


# ---------------------------------------------------------------------
# Probability helper
# ---------------------------------------------------------------------

def obtain_probabilities(
    classifier: Pipeline,
    X_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Return predicted class probabilities."""
    probabilities = classifier.predict_proba(
        X_test
    )

    classes = np.asarray(
        classifier.named_steps[
            "classifier"
        ].classes_
    )

    return (
        np.asarray(
            probabilities,
            dtype=float,
        ),
        classes,
    )


# ---------------------------------------------------------------------
# Individual calibration run
# ---------------------------------------------------------------------

def run_single_experiment(
    subject_data: dict[str, dict[str, Any]],
    target_subject: str,
    calibration_fraction: float,
    random_seed: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Run one target-subject calibration experiment.
    """
    (
        X_source,
        y_source,
        training_subjects,
    ) = create_source_training_data(
        subject_data=subject_data,
        target_subject=target_subject,
    )

    X_target = subject_data[
        target_subject
    ]["X"]

    y_target = subject_data[
        target_subject
    ]["y"]

    (
        X_calibration,
        y_calibration,
        X_test,
        y_test,
        calibration_indices,
        testing_indices,
    ) = create_calibration_split(
        X_target=X_target,
        y_target=y_target,
        calibration_fraction=calibration_fraction,
        random_seed=random_seed,
    )

    if len(X_calibration) > 0:
        X_train = np.concatenate(
            [
                X_source,
                X_calibration,
            ],
            axis=0,
        )

        y_train = np.concatenate(
            [
                y_source,
                y_calibration,
            ],
            axis=0,
        )
    else:
        X_train = X_source
        y_train = y_source

    classifier = build_classifier()

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

    y_predicted = classifier.predict(
        X_test
    )

    probabilities, probability_classes = (
        obtain_probabilities(
            classifier=classifier,
            X_test=X_test,
        )
    )

    prediction_seconds = (
        time.perf_counter()
        - prediction_start
    )

    accuracy = accuracy_score(
        y_test,
        y_predicted,
    )

    kappa = cohen_kappa_score(
        y_test,
        y_predicted,
    )

    class_recalls = recall_score(
        y_test,
        y_predicted,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_predicted,
        labels=CLASS_ORDER,
    )

    run_result: dict[str, Any] = {
        "target_subject": target_subject,
        "training_subjects": "|".join(
            training_subjects
        ),
        "calibration_fraction": (
            calibration_fraction
        ),
        "calibration_percent": (
            calibration_fraction * 100.0
        ),
        "random_seed": random_seed,
        "source_training_trials": len(
            y_source
        ),
        "calibration_trials": len(
            y_calibration
        ),
        "total_training_trials": len(
            y_train
        ),
        "testing_trials": len(
            y_test
        ),
        "accuracy": float(accuracy),
        "accuracy_percent": float(
            accuracy * 100.0
        ),
        "kappa": float(kappa),
        "left_hand_recall": float(
            class_recalls[0]
        ),
        "right_hand_recall": float(
            class_recalls[1]
        ),
        "feet_recall": float(
            class_recalls[2]
        ),
        "tongue_recall": float(
            class_recalls[3]
        ),
        "training_seconds": float(
            training_seconds
        ),
        "prediction_seconds": float(
            prediction_seconds
        ),
    }

    for true_index, true_class in enumerate(
        CLASS_ORDER
    ):
        for predicted_index, predicted_class in enumerate(
            CLASS_ORDER
        ):
            run_result[
                f"cm_{true_class}_pred_{predicted_class}"
            ] = int(
                matrix[
                    true_index,
                    predicted_index,
                ]
            )

    probability_index = {
        class_name: index
        for index, class_name in enumerate(
            probability_classes
        )
    }

    prediction_rows: list[
        dict[str, Any]
    ] = []

    for local_index, (
        target_trial_index,
        true_class,
        predicted_class,
    ) in enumerate(
        zip(
            testing_indices,
            y_test,
            y_predicted,
        )
    ):
        probability_row = probabilities[
            local_index
        ]

        predicted_probability_index = (
            probability_index[
                predicted_class
            ]
        )

        prediction_row: dict[str, Any] = {
            "target_subject": target_subject,
            "calibration_fraction": (
                calibration_fraction
            ),
            "calibration_percent": (
                calibration_fraction * 100.0
            ),
            "random_seed": random_seed,
            "target_trial_index": int(
                target_trial_index
            ),
            "true_class": true_class,
            "predicted_class": (
                predicted_class
            ),
            "correct": bool(
                true_class
                == predicted_class
            ),
            "confidence": float(
                probability_row[
                    predicted_probability_index
                ]
            ),
        }

        for class_name in CLASS_ORDER:
            prediction_row[
                f"probability_{class_name}"
            ] = float(
                probability_row[
                    probability_index[
                        class_name
                    ]
                ]
            )

        prediction_rows.append(
            prediction_row
        )

    return (
        run_result,
        prediction_rows,
    )


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def create_subject_summary(
    run_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Create mean results for every target subject and calibration size.
    """
    summary_rows: list[
        dict[str, Any]
    ] = []

    for calibration_fraction in CALIBRATION_FRACTIONS:
        for subject in SUBJECTS:
            matching_rows = [
                row
                for row in run_results
                if (
                    row["target_subject"]
                    == subject
                    and row[
                        "calibration_fraction"
                    ]
                    == calibration_fraction
                )
            ]

            if not matching_rows:
                continue

            accuracies = np.asarray(
                [
                    row["accuracy"]
                    for row in matching_rows
                ],
                dtype=float,
            )

            kappas = np.asarray(
                [
                    row["kappa"]
                    for row in matching_rows
                ],
                dtype=float,
            )

            summary_rows.append(
                {
                    "target_subject": subject,
                    "calibration_fraction": (
                        calibration_fraction
                    ),
                    "calibration_percent": (
                        calibration_fraction
                        * 100.0
                    ),
                    "number_of_repeats": len(
                        matching_rows
                    ),
                    "mean_accuracy": float(
                        accuracies.mean()
                    ),
                    "mean_accuracy_percent": float(
                        accuracies.mean()
                        * 100.0
                    ),
                    "std_accuracy": float(
                        accuracies.std(
                            ddof=1
                        )
                        if len(accuracies) > 1
                        else 0.0
                    ),
                    "std_accuracy_percent": float(
                        accuracies.std(
                            ddof=1
                        )
                        * 100.0
                        if len(accuracies) > 1
                        else 0.0
                    ),
                    "minimum_accuracy_percent": float(
                        accuracies.min()
                        * 100.0
                    ),
                    "maximum_accuracy_percent": float(
                        accuracies.max()
                        * 100.0
                    ),
                    "mean_kappa": float(
                        kappas.mean()
                    ),
                    "std_kappa": float(
                        kappas.std(
                            ddof=1
                        )
                        if len(kappas) > 1
                        else 0.0
                    ),
                }
            )

    return summary_rows


def create_overall_summary(
    subject_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Create the overall cross-subject result for each calibration size.

    Each subject contributes equally to the overall mean.
    """
    overall_rows: list[
        dict[str, Any]
    ] = []

    zero_percent_accuracy: float | None = None
    zero_percent_kappa: float | None = None

    for calibration_fraction in CALIBRATION_FRACTIONS:
        matching_rows = [
            row
            for row in subject_summary
            if row[
                "calibration_fraction"
            ]
            == calibration_fraction
        ]

        accuracies = np.asarray(
            [
                row["mean_accuracy"]
                for row in matching_rows
            ],
            dtype=float,
        )

        kappas = np.asarray(
            [
                row["mean_kappa"]
                for row in matching_rows
            ],
            dtype=float,
        )

        mean_accuracy = float(
            accuracies.mean()
        )

        mean_kappa = float(
            kappas.mean()
        )

        if calibration_fraction == 0.0:
            zero_percent_accuracy = (
                mean_accuracy
            )

            zero_percent_kappa = (
                mean_kappa
            )

        accuracy_improvement = (
            mean_accuracy
            - zero_percent_accuracy
            if zero_percent_accuracy
            is not None
            else 0.0
        )

        kappa_improvement = (
            mean_kappa
            - zero_percent_kappa
            if zero_percent_kappa
            is not None
            else 0.0
        )

        overall_rows.append(
            {
                "calibration_fraction": (
                    calibration_fraction
                ),
                "calibration_percent": (
                    calibration_fraction
                    * 100.0
                ),
                "number_of_subjects": len(
                    matching_rows
                ),
                "mean_accuracy": (
                    mean_accuracy
                ),
                "mean_accuracy_percent": (
                    mean_accuracy * 100.0
                ),
                "std_between_subjects": float(
                    accuracies.std(
                        ddof=1
                    )
                ),
                "std_between_subjects_percent": float(
                    accuracies.std(
                        ddof=1
                    )
                    * 100.0
                ),
                "mean_kappa": mean_kappa,
                "std_between_subjects_kappa": float(
                    kappas.std(
                        ddof=1
                    )
                ),
                "accuracy_improvement": float(
                    accuracy_improvement
                ),
                "accuracy_improvement_percent_points": float(
                    accuracy_improvement
                    * 100.0
                ),
                "kappa_improvement": float(
                    kappa_improvement
                ),
                "reference_strict_loso_accuracy": (
                    STRICT_LOSO_REFERENCE_ACCURACY
                ),
                "reference_strict_loso_accuracy_percent": (
                    STRICT_LOSO_REFERENCE_ACCURACY
                    * 100.0
                ),
                "reference_strict_loso_kappa": (
                    STRICT_LOSO_REFERENCE_KAPPA
                ),
            }
        )

    return overall_rows


# ---------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------

def print_overall_results(
    overall_summary: list[dict[str, Any]],
) -> None:
    """Print the final calibration comparison."""
    print()
    print("=" * 80)
    print("Riemannian New-User Calibration Results")
    print("=" * 80)

    print(
        f"\n{'Calibration':<14}"
        f"{'Accuracy':>12}"
        f"{'Subject SD':>14}"
        f"{'Kappa':>11}"
        f"{'Improvement':>16}"
    )

    print("-" * 70)

    for row in overall_summary:
        print(
            f"{row['calibration_percent']:>9.0f}%"
            f"{row['mean_accuracy_percent']:>16.2f}%"
            f"{row['std_between_subjects_percent']:>13.2f}"
            f"{row['mean_kappa']:>11.3f}"
            f"{row['accuracy_improvement_percent_points']:>+15.2f}"
        )

    best_row = max(
        overall_summary,
        key=lambda row: (
            row["mean_accuracy"],
            row["mean_kappa"],
        ),
    )

    print()
    print("=" * 80)
    print("Best Calibration Setting")
    print("=" * 80)

    print(
        f"Calibration size: "
        f"{best_row['calibration_percent']:.0f}%"
    )

    print(
        f"Mean accuracy:    "
        f"{best_row['mean_accuracy_percent']:.2f}%"
    )

    print(
        f"Mean kappa:       "
        f"{best_row['mean_kappa']:.3f}"
    )

    print(
        f"Improvement over 0% calibration: "
        f"{best_row['accuracy_improvement_percent_points']:+.2f} "
        "percentage points"
    )

    print()
    print("Saved files:")
    print(
        f"  {RUN_RESULTS_PATH}"
    )
    print(
        f"  {SUBJECT_SUMMARY_PATH}"
    )
    print(
        f"  {OVERALL_SUMMARY_PATH}"
    )
    print(
        f"  {PREDICTIONS_PATH}"
    )
    print(
        f"  {BEST_CONFIGURATION_PATH}"
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete calibration experiment."""
    total_start = time.perf_counter()

    print("=" * 80)
    print(
        "Cross-Subject Riemannian Calibration Experiment"
    )
    print("=" * 80)

    print(
        "\nClassifier:"
    )

    print(
        "  Euclidean Alignment"
    )

    print(
        "  -> OAS covariance matrices"
    )

    print(
        "  -> Riemannian tangent space"
    )

    print(
        "  -> StandardScaler"
    )

    print(
        "  -> Shrinkage LDA"
    )

    print(
        "\nCalibration sizes:"
    )

    for calibration_fraction in CALIBRATION_FRACTIONS:
        print(
            f"  {calibration_fraction * 100:.0f}%"
        )

    print(
        "\nRepeats for calibrated experiments: "
        f"{len(RANDOM_SEEDS)}"
    )

    subject_data = load_all_subjects()

    run_results: list[
        dict[str, Any]
    ] = []

    prediction_results: list[
        dict[str, Any]
    ] = []

    total_runs = (
        len(SUBJECTS)
        + (
            len(SUBJECTS)
            * (
                len(
                    CALIBRATION_FRACTIONS
                )
                - 1
            )
            * len(RANDOM_SEEDS)
        )
    )

    completed_runs = 0

    for calibration_fraction in CALIBRATION_FRACTIONS:
        seeds = (
            [RANDOM_SEEDS[0]]
            if calibration_fraction == 0.0
            else RANDOM_SEEDS
        )

        print()
        print("#" * 80)
        print(
            f"Calibration size: "
            f"{calibration_fraction * 100:.0f}%"
        )
        print("#" * 80)

        for target_subject in SUBJECTS:
            for random_seed in seeds:
                completed_runs += 1

                print()
                print(
                    f"Run {completed_runs}/{total_runs} | "
                    f"Target: {target_subject} | "
                    f"Calibration: "
                    f"{calibration_fraction * 100:.0f}% | "
                    f"Seed: {random_seed}"
                )

                (
                    run_result,
                    prediction_rows,
                ) = run_single_experiment(
                    subject_data=subject_data,
                    target_subject=target_subject,
                    calibration_fraction=(
                        calibration_fraction
                    ),
                    random_seed=random_seed,
                )

                run_results.append(
                    run_result
                )

                prediction_results.extend(
                    prediction_rows
                )

                print(
                    f"Calibration trials: "
                    f"{run_result['calibration_trials']}"
                )

                print(
                    f"Testing trials:     "
                    f"{run_result['testing_trials']}"
                )

                print(
                    f"Accuracy:           "
                    f"{run_result['accuracy_percent']:.2f}%"
                )

                print(
                    f"Kappa:              "
                    f"{run_result['kappa']:.3f}"
                )

                # Save progress after every run.
                export_csv(
                    RUN_RESULTS_PATH,
                    run_results,
                )

                export_csv(
                    PREDICTIONS_PATH,
                    prediction_results,
                )

    subject_summary = create_subject_summary(
        run_results
    )

    overall_summary = create_overall_summary(
        subject_summary
    )

    export_csv(
        SUBJECT_SUMMARY_PATH,
        subject_summary,
    )

    export_csv(
        OVERALL_SUMMARY_PATH,
        overall_summary,
    )

    best_configuration = max(
        overall_summary,
        key=lambda row: (
            row["mean_accuracy"],
            row["mean_kappa"],
        ),
    ).copy()

    best_configuration[
        "total_experiment_seconds"
    ] = float(
        time.perf_counter()
        - total_start
    )

    export_csv(
        BEST_CONFIGURATION_PATH,
        [best_configuration],
    )

    print_overall_results(
        overall_summary
    )

    print(
        f"\nTotal experiment time: "
        f"{best_configuration['total_experiment_seconds']:.2f} "
        "seconds"
    )


if __name__ == "__main__":
    main()
