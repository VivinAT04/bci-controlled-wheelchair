"""
Strict cross-subject Leave-One-Subject-Out evaluation using
Riemannian geometry classifiers.

For every fold:

    Train on eight subjects
    Test on one completely unseen subject

Pipelines tested:

1. Euclidean Alignment
   -> OAS covariance matrices
   -> Minimum Distance to Mean (MDM)

2. Euclidean Alignment
   -> OAS covariance matrices
   -> Tangent-space projection
   -> StandardScaler
   -> Shrinkage LDA

Current comparison baseline:

    EA + regularized FBCSP + PCA 90% + shrinkage LDA
    Mean LOSO accuracy = 50.96%
    Mean LOSO kappa    = 0.346

Run:

    python -m scripts.cross_subject.run_ea_fbcsp_lda_riemannian
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bci_wheelchair.cross_subject import (
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

RESULTS_DIRECTORY = Path(
    "results/cross_subject/riemannian/loso_riemannian"
)

CONFIGURATION_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "loso_riemannian_configuration_results.csv"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "loso_riemannian_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "loso_riemannian_predictions.csv"
)

BEST_CONFIGURATION_PATH = (
    RESULTS_DIRECTORY
    / "loso_riemannian_best_configuration.csv"
)

BASELINE_NAME = "EA_FBCSP_PCA90_shrinkage_LDA"
BASELINE_ACCURACY = 0.5096450617283951
BASELINE_KAPPA = 0.3461934156378601


# ---------------------------------------------------------------------
# CSV utilities
# ---------------------------------------------------------------------

def export_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write dictionaries to a CSV file."""
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
# Data loading
# ---------------------------------------------------------------------

def load_all_subjects() -> dict[str, dict[str, Any]]:
    """
    Load all subjects and apply Euclidean Alignment independently.

    The alignment reference is calculated separately for each subject.
    Subjects are pooled only after alignment.
    """
    subject_data: dict[str, dict[str, Any]] = {}

    print("=" * 78)
    print("Loading and Euclidean-aligning all subjects")
    print("=" * 78)

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


def create_loso_fold(
    subject_data: dict[str, dict[str, Any]],
    test_subject: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """Create one strict LOSO fold."""
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

    X_test = subject_data[test_subject]["X"]
    y_test = subject_data[test_subject]["y"]

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        training_subjects,
    )


# ---------------------------------------------------------------------
# Riemannian classifiers
# ---------------------------------------------------------------------

def build_mdm_classifier() -> Pipeline:
    """
    Build covariance + Riemannian MDM classifier.
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
                "classifier",
                MDM(
                    metric="riemann",
                ),
            ),
        ]
    )


def build_tangent_lda_classifier() -> Pipeline:
    """
    Build covariance + tangent-space + shrinkage LDA classifier.
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


def build_classifiers() -> dict[str, Pipeline]:
    """Return all configurations to evaluate."""
    return {
        "riemannian_mdm_oas": (
            build_mdm_classifier()
        ),
        "riemannian_tangent_lda_oas": (
            build_tangent_lda_classifier()
        ),
    }


# ---------------------------------------------------------------------
# Probability and confidence handling
# ---------------------------------------------------------------------

def obtain_probabilities(
    classifier: Pipeline,
    X_test: np.ndarray,
    y_predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Obtain class probabilities when available.

    MDM supports predict_proba in pyRiemann. If the installed
    version does not expose it, confidence values are recorded
    as NaN without affecting accuracy evaluation.
    """
    final_classifier = classifier.named_steps[
        "classifier"
    ]

    classes = np.asarray(
        final_classifier.classes_
    )

    if hasattr(classifier, "predict_proba"):
        try:
            probabilities = (
                classifier.predict_proba(
                    X_test
                )
            )

            return (
                np.asarray(
                    probabilities,
                    dtype=float,
                ),
                classes,
            )
        except (
            AttributeError,
            NotImplementedError,
        ):
            pass

    probabilities = np.full(
        (
            len(y_predicted),
            len(classes),
        ),
        np.nan,
        dtype=float,
    )

    return probabilities, classes


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_configuration(
    configuration_name: str,
    classifier_builder,
    subject_data: dict[str, dict[str, Any]],
    configuration_number: int,
    total_configurations: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate one classifier across all LOSO folds."""
    print()
    print("#" * 78)
    print(
        f"Configuration "
        f"{configuration_number}/"
        f"{total_configurations}: "
        f"{configuration_name}"
    )
    print("#" * 78)

    configuration_start = (
        time.perf_counter()
    )

    subject_results: list[
        dict[str, Any]
    ] = []

    prediction_results: list[
        dict[str, Any]
    ] = []

    all_true: list[np.ndarray] = []
    all_predicted: list[np.ndarray] = []

    for fold_number, test_subject in enumerate(
        SUBJECTS,
        start=1,
    ):
        print()
        print("=" * 78)
        print(
            f"{configuration_name} | "
            f"Fold {fold_number}/9 | "
            f"Unseen subject: {test_subject}"
        )
        print("=" * 78)

        (
            X_train,
            y_train,
            X_test,
            y_test,
            training_subjects,
        ) = create_loso_fold(
            subject_data=subject_data,
            test_subject=test_subject,
        )

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
        print(
            f"EEG shape:       "
            f"{X_train.shape[1]} channels × "
            f"{X_train.shape[2]} samples"
        )

        classifier = classifier_builder()

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

        y_predicted = classifier.predict(
            X_test
        )

        prediction_seconds = (
            time.perf_counter()
            - prediction_start
        )

        probabilities, probability_classes = (
            obtain_probabilities(
                classifier=classifier,
                X_test=X_test,
                y_predicted=y_predicted,
            )
        )

        accuracy = accuracy_score(
            y_test,
            y_predicted,
        )

        kappa = cohen_kappa_score(
            y_test,
            y_predicted,
        )

        recalls = recall_score(
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

        subject_result: dict[str, Any] = {
            "configuration": configuration_name,
            "test_subject": test_subject,
            "training_subjects": "|".join(
                training_subjects
            ),
            "training_trials": len(y_train),
            "testing_trials": len(y_test),
            "accuracy": float(accuracy),
            "accuracy_percent": float(
                accuracy * 100.0
            ),
            "kappa": float(kappa),
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
            for (
                predicted_index,
                predicted_class,
            ) in enumerate(CLASS_ORDER):
                subject_result[
                    f"cm_{true_class}_"
                    f"pred_{predicted_class}"
                ] = int(
                    matrix[
                        true_index,
                        predicted_index,
                    ]
                )

        subject_results.append(
            subject_result
        )

        class_to_probability_index = {
            class_name: index
            for index, class_name
            in enumerate(
                probability_classes
            )
        }

        for trial_index, (
            true_class,
            predicted_class,
        ) in enumerate(
            zip(
                y_test,
                y_predicted,
            ),
            start=1,
        ):
            probability_row = (
                probabilities[
                    trial_index - 1
                ]
            )

            predicted_index = (
                class_to_probability_index[
                    predicted_class
                ]
            )

            confidence = float(
                probability_row[
                    predicted_index
                ]
            )

            prediction_row: dict[
                str,
                Any,
            ] = {
                "configuration": (
                    configuration_name
                ),
                "test_subject": test_subject,
                "trial": trial_index,
                "true_class": true_class,
                "predicted_class": (
                    predicted_class
                ),
                "correct": (
                    true_class
                    == predicted_class
                ),
                "confidence": confidence,
            }

            for class_name in CLASS_ORDER:
                class_index = (
                    class_to_probability_index[
                        class_name
                    ]
                )

                prediction_row[
                    f"probability_{class_name}"
                ] = float(
                    probability_row[
                        class_index
                    ]
                )

            prediction_results.append(
                prediction_row
            )

        all_true.append(
            np.asarray(y_test)
        )
        all_predicted.append(
            np.asarray(y_predicted)
        )

        print()
        print(
            f"Accuracy: "
            f"{accuracy * 100.0:.2f}%"
        )
        print(
            f"Kappa:    {kappa:.3f}"
        )
        print(
            "Class recall:"
        )
        print(
            f"  Left hand:  "
            f"{recalls[0]:.3f}"
        )
        print(
            f"  Right hand: "
            f"{recalls[1]:.3f}"
        )
        print(
            f"  Feet:       "
            f"{recalls[2]:.3f}"
        )
        print(
            f"  Tongue:     "
            f"{recalls[3]:.3f}"
        )
        print(
            f"Training time: "
            f"{training_seconds:.2f} seconds"
        )
        print(
            f"Prediction time: "
            f"{prediction_seconds:.2f} seconds"
        )

    combined_true = np.concatenate(
        all_true
    )
    combined_predicted = np.concatenate(
        all_predicted
    )

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

    overall_recalls = recall_score(
        combined_true,
        combined_predicted,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    pooled_accuracy = accuracy_score(
        combined_true,
        combined_predicted,
    )

    pooled_kappa = cohen_kappa_score(
        combined_true,
        combined_predicted,
    )

    configuration_seconds = (
        time.perf_counter()
        - configuration_start
    )

    result: dict[str, Any] = {
        "configuration": configuration_name,
        "mean_accuracy": float(
            accuracies.mean()
        ),
        "mean_accuracy_percent": float(
            accuracies.mean() * 100.0
        ),
        "std_accuracy": float(
            accuracies.std(ddof=1)
        ),
        "std_accuracy_percent": float(
            accuracies.std(ddof=1)
            * 100.0
        ),
        "minimum_accuracy": float(
            accuracies.min()
        ),
        "minimum_accuracy_percent": float(
            accuracies.min() * 100.0
        ),
        "maximum_accuracy": float(
            accuracies.max()
        ),
        "maximum_accuracy_percent": float(
            accuracies.max() * 100.0
        ),
        "mean_kappa": float(
            kappas.mean()
        ),
        "std_kappa": float(
            kappas.std(ddof=1)
        ),
        "pooled_accuracy": float(
            pooled_accuracy
        ),
        "pooled_accuracy_percent": float(
            pooled_accuracy * 100.0
        ),
        "pooled_kappa": float(
            pooled_kappa
        ),
        "overall_left_hand_recall": float(
            overall_recalls[0]
        ),
        "overall_right_hand_recall": float(
            overall_recalls[1]
        ),
        "overall_feet_recall": float(
            overall_recalls[2]
        ),
        "overall_tongue_recall": float(
            overall_recalls[3]
        ),
        "baseline_name": BASELINE_NAME,
        "baseline_accuracy": (
            BASELINE_ACCURACY
        ),
        "baseline_accuracy_percent": (
            BASELINE_ACCURACY * 100.0
        ),
        "accuracy_change": float(
            accuracies.mean()
            - BASELINE_ACCURACY
        ),
        "accuracy_change_percent_points": (
            float(
                (
                    accuracies.mean()
                    - BASELINE_ACCURACY
                )
                * 100.0
            )
        ),
        "baseline_kappa": (
            BASELINE_KAPPA
        ),
        "kappa_change": float(
            kappas.mean()
            - BASELINE_KAPPA
        ),
        "configuration_seconds": float(
            configuration_seconds
        ),
    }

    print()
    print("-" * 78)
    print(
        f"Completed {configuration_name}"
    )
    print("-" * 78)
    print(
        f"Mean LOSO accuracy: "
        f"{result['mean_accuracy_percent']:.2f}%"
    )
    print(
        f"Accuracy SD:        "
        f"{result['std_accuracy_percent']:.2f}"
    )
    print(
        f"Mean LOSO kappa:    "
        f"{result['mean_kappa']:.3f}"
    )
    print(
        f"Change from FBCSP-LDA: "
        f"{result['accuracy_change_percent_points']:+.2f} "
        "percentage points"
    )

    return (
        result,
        subject_results,
        prediction_results,
    )


# ---------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------

def print_final_results(
    configuration_results: list[
        dict[str, Any]
    ],
) -> None:
    """Print ranked configuration results."""
    ranked_results = sorted(
        configuration_results,
        key=lambda result: (
            result["mean_accuracy"],
            result["mean_kappa"],
        ),
        reverse=True,
    )

    print()
    print("=" * 78)
    print("Final Cross-Subject Riemannian LOSO Results")
    print("=" * 78)

    print(
        f"\n{'Rank':<6}"
        f"{'Configuration':<34}"
        f"{'Accuracy':>11}"
        f"{'SD':>9}"
        f"{'Kappa':>9}"
        f"{'vs baseline':>14}"
    )

    print("-" * 83)

    for rank, result in enumerate(
        ranked_results,
        start=1,
    ):
        print(
            f"{rank:<6}"
            f"{result['configuration']:<34}"
            f"{result['mean_accuracy_percent']:>10.2f}%"
            f"{result['std_accuracy_percent']:>8.2f}"
            f"{result['mean_kappa']:>9.3f}"
            f"{result['accuracy_change_percent_points']:>+13.2f}"
        )

    best = ranked_results[0]

    print()
    print("=" * 78)
    print("Best Riemannian Configuration")
    print("=" * 78)
    print(
        f"Configuration: {best['configuration']}"
    )
    print(
        f"Mean accuracy: "
        f"{best['mean_accuracy_percent']:.2f}%"
    )
    print(
        f"Mean kappa:    "
        f"{best['mean_kappa']:.3f}"
    )
    print(
        f"Change from baseline: "
        f"{best['accuracy_change_percent_points']:+.2f} "
        "percentage points"
    )

    if (
        best["mean_accuracy"]
        > BASELINE_ACCURACY
    ):
        print(
            "\nResult: The Riemannian method "
            "improved upon the FBCSP-LDA baseline."
        )
    else:
        print(
            "\nResult: The Riemannian method "
            "did not improve upon the "
            "FBCSP-LDA baseline."
        )

    print()
    print("Saved files:")
    print(
        f"  {CONFIGURATION_RESULTS_PATH}"
    )
    print(
        f"  {SUBJECT_RESULTS_PATH}"
    )
    print(
        f"  {PREDICTIONS_PATH}"
    )
    print(
        f"  {BEST_CONFIGURATION_PATH}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete LOSO Riemannian evaluation."""
    total_start = time.perf_counter()

    print("=" * 78)
    print(
        "Cross-Subject LOSO Riemannian Evaluation"
    )
    print("=" * 78)
    print(
        "\nEvaluation design:"
    )
    print(
        "  Train on eight subjects"
    )
    print(
        "  Test on one completely unseen subject"
    )
    print(
        "  Repeat for all nine subjects"
    )
    print(
        f"\nCurrent FBCSP-LDA baseline: "
        f"{BASELINE_ACCURACY * 100.0:.2f}%"
    )
    print(
        f"Current baseline kappa:     "
        f"{BASELINE_KAPPA:.3f}"
    )

    subject_data = load_all_subjects()

    classifier_templates = (
        build_classifiers()
    )

    configuration_results: list[
        dict[str, Any]
    ] = []

    all_subject_results: list[
        dict[str, Any]
    ] = []

    all_prediction_results: list[
        dict[str, Any]
    ] = []

    total_configurations = len(
        classifier_templates
    )

    for configuration_number, (
        configuration_name,
        classifier_template,
    ) in enumerate(
        classifier_templates.items(),
        start=1,
    ):
        classifier_builder = (
            lambda template=classifier_template:
            build_mdm_classifier()
            if template.named_steps[
                "classifier"
            ].__class__.__name__ == "MDM"
            else build_tangent_lda_classifier()
        )

        (
            configuration_result,
            subject_results,
            prediction_results,
        ) = evaluate_configuration(
            configuration_name=(
                configuration_name
            ),
            classifier_builder=(
                classifier_builder
            ),
            subject_data=subject_data,
            configuration_number=(
                configuration_number
            ),
            total_configurations=(
                total_configurations
            ),
        )

        configuration_results.append(
            configuration_result
        )

        all_subject_results.extend(
            subject_results
        )

        all_prediction_results.extend(
            prediction_results
        )

        # Preserve progress after each model.
        export_csv(
            CONFIGURATION_RESULTS_PATH,
            configuration_results,
        )

        export_csv(
            SUBJECT_RESULTS_PATH,
            all_subject_results,
        )

        export_csv(
            PREDICTIONS_PATH,
            all_prediction_results,
        )

    ranked_results = sorted(
        configuration_results,
        key=lambda result: (
            result["mean_accuracy"],
            result["mean_kappa"],
        ),
        reverse=True,
    )

    best_result = ranked_results[0].copy()

    best_result[
        "total_experiment_seconds"
    ] = float(
        time.perf_counter()
        - total_start
    )

    export_csv(
        BEST_CONFIGURATION_PATH,
        [best_result],
    )

    print_final_results(
        configuration_results
    )

    print(
        f"\nTotal experiment time: "
        f"{best_result['total_experiment_seconds']:.2f} "
        "seconds"
    )


if __name__ == "__main__":
    main()
