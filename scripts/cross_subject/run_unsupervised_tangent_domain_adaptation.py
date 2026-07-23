"""
Strict LOSO unsupervised tangent-space domain adaptation.

Pipeline
--------
8-30 Hz EEG
-> 0.5-2.5 second epoch
-> subject-wise Euclidean Alignment
-> SCM covariance matrices
-> Riemannian Tangent Space
-> unsupervised target adaptation
-> StandardScaler
-> shrinkage LDA

Adaptation methods
------------------
none:
    Standard tangent-space baseline.

mean_alignment:
    Source features are centred using the source mean.
    Target features are centred using the target mean.

coral:
    Source tangent features are transformed so their covariance
    resembles the unseen target subject's tangent-feature covariance.

Important
---------
The target subject labels are never used during:

- preprocessing
- alignment
- tangent-space fitting
- domain adaptation
- scaling
- classifier training

Target EEG features are used without labels for transductive,
unsupervised domain adaptation.

Run:

    python -m scripts.cross_subject.run_unsupervised_tangent_domain_adaptation
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from scripts.cross_subject.run_broadband_riemannian_sweep import (
    SUBJECTS,
    load_subjects,
)

from scripts.cross_subject.run_cross_subject_evaluation import (
    CLASS_ORDER,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

COVARIANCE_ESTIMATOR = "scm"

ADAPTATION_METHODS = [
    "none",
    "mean_alignment",
    "coral",
]

CORAL_REGULARISATION = 1e-3

RESULTS_DIRECTORY = Path(
    "results/cross_subject/riemannian/unsupervised_tangent_domain_adaptation"
)

CONFIGURATION_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "adaptation_configuration_results.csv"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "adaptation_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "adaptation_predictions.csv"
)

BEST_CONFIGURATION_PATH = (
    RESULTS_DIRECTORY
    / "adaptation_best_configuration.csv"
)


# ---------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------

def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write dictionaries to CSV."""
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
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Matrix utilities
# ---------------------------------------------------------------------

def symmetrise(
    matrix: np.ndarray,
) -> np.ndarray:
    """Return an exactly symmetric matrix."""
    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    return (
        matrix
        + matrix.T
    ) / 2.0


def stable_matrix_power(
    matrix: np.ndarray,
    power: float,
    regularisation: float,
) -> np.ndarray:
    """
    Compute a stable symmetric matrix power.

    power=-0.5 gives the inverse square root.
    power=0.5 gives the square root.
    """
    matrix = symmetrise(matrix)

    dimension = matrix.shape[0]

    trace_scale = float(
        np.trace(matrix)
        / max(dimension, 1)
    )

    if not np.isfinite(trace_scale) or trace_scale <= 0:
        trace_scale = 1.0

    regularised_matrix = (
        matrix
        + regularisation
        * trace_scale
        * np.eye(
            dimension,
            dtype=np.float64,
        )
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        symmetrise(regularised_matrix)
    )

    eigenvalue_floor = max(
        regularisation * trace_scale,
        np.finfo(np.float64).eps,
    )

    eigenvalues = np.maximum(
        eigenvalues,
        eigenvalue_floor,
    )

    powered = (
        eigenvectors
        @ np.diag(
            np.power(
                eigenvalues,
                power,
            )
        )
        @ eigenvectors.T
    )

    return symmetrise(powered)


def feature_covariance(
    features: np.ndarray,
) -> np.ndarray:
    """Calculate empirical covariance between feature dimensions."""
    features = np.asarray(
        features,
        dtype=np.float64,
    )

    centered = (
        features
        - features.mean(
            axis=0,
            keepdims=True,
        )
    )

    denominator = max(
        len(features) - 1,
        1,
    )

    covariance = (
        centered.T
        @ centered
    ) / denominator

    return symmetrise(covariance)


# ---------------------------------------------------------------------
# Covariance and tangent features
# ---------------------------------------------------------------------

def extract_tangent_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Extract SCM covariance matrices and tangent-space features.

    Covariance and TangentSpace are fitted using source training
    subjects. Target features are transformed using the source-derived
    tangent reference.
    """
    start = time.perf_counter()

    covariance_transformer = Covariances(
        estimator=COVARIANCE_ESTIMATOR,
    )

    training_covariances = (
        covariance_transformer.fit_transform(
            X_train
        )
    )

    testing_covariances = (
        covariance_transformer.transform(
            X_test
        )
    )

    tangent_space = TangentSpace(
        metric="riemann",
    )

    training_features = (
        tangent_space.fit_transform(
            training_covariances
        )
    )

    testing_features = (
        tangent_space.transform(
            testing_covariances
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        np.asarray(
            training_features,
            dtype=np.float64,
        ),
        np.asarray(
            testing_features,
            dtype=np.float64,
        ),
        float(elapsed),
    )


# ---------------------------------------------------------------------
# Unsupervised adaptation
# ---------------------------------------------------------------------

def apply_no_adaptation(
    training_features: np.ndarray,
    testing_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unchanged source and target tangent features."""
    return (
        training_features.copy(),
        testing_features.copy(),
    )


def apply_mean_alignment(
    training_features: np.ndarray,
    testing_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Centre source and target domains independently.

    Target labels are not required.
    """
    source_mean = training_features.mean(
        axis=0,
        keepdims=True,
    )

    target_mean = testing_features.mean(
        axis=0,
        keepdims=True,
    )

    adapted_training = (
        training_features
        - source_mean
    )

    adapted_testing = (
        testing_features
        - target_mean
    )

    return (
        adapted_training,
        adapted_testing,
    )


def apply_coral(
    training_features: np.ndarray,
    testing_features: np.ndarray,
    regularisation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply unsupervised CORAL adaptation.

    The source domain is whitened using its covariance and recoloured
    using the unseen target subject covariance.

    The target subject labels are never used.
    """
    source_mean = training_features.mean(
        axis=0,
        keepdims=True,
    )

    target_mean = testing_features.mean(
        axis=0,
        keepdims=True,
    )

    centered_source = (
        training_features
        - source_mean
    )

    centered_target = (
        testing_features
        - target_mean
    )

    source_covariance = feature_covariance(
        centered_source
    )

    target_covariance = feature_covariance(
        centered_target
    )

    source_inverse_square_root = (
        stable_matrix_power(
            source_covariance,
            power=-0.5,
            regularisation=regularisation,
        )
    )

    target_square_root = stable_matrix_power(
        target_covariance,
        power=0.5,
        regularisation=regularisation,
    )

    transformation = (
        source_inverse_square_root
        @ target_square_root
    )

    adapted_training = (
        centered_source
        @ transformation
    )

    adapted_testing = centered_target

    return (
        np.asarray(
            adapted_training,
            dtype=np.float64,
        ),
        np.asarray(
            adapted_testing,
            dtype=np.float64,
        ),
    )


def adapt_features(
    method: str,
    training_features: np.ndarray,
    testing_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the requested unsupervised adaptation method."""
    if method == "none":
        return apply_no_adaptation(
            training_features,
            testing_features,
        )

    if method == "mean_alignment":
        return apply_mean_alignment(
            training_features,
            testing_features,
        )

    if method == "coral":
        return apply_coral(
            training_features,
            testing_features,
            regularisation=(
                CORAL_REGULARISATION
            ),
        )

    raise ValueError(
        f"Unknown adaptation method: {method}"
    )


# ---------------------------------------------------------------------
# Probability helper
# ---------------------------------------------------------------------

def softmax(
    scores: np.ndarray,
) -> np.ndarray:
    """Convert classifier scores into normalised values."""
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    if scores.ndim == 1:
        scores = np.column_stack(
            [
                -scores,
                scores,
            ]
        )

    shifted = (
        scores
        - np.max(
            scores,
            axis=1,
            keepdims=True,
        )
    )

    exponentials = np.exp(shifted)

    return (
        exponentials
        / np.sum(
            exponentials,
            axis=1,
            keepdims=True,
        )
    )


def get_probabilities(
    classifier: LinearDiscriminantAnalysis,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return class probabilities or score-derived probabilities."""
    classes = np.asarray(
        classifier.classes_
    )

    if hasattr(
        classifier,
        "predict_proba",
    ):
        try:
            probabilities = classifier.predict_proba(
                features
            )

            return (
                np.asarray(
                    probabilities,
                    dtype=np.float64,
                ),
                classes,
            )
        except (
            AttributeError,
            NotImplementedError,
            ValueError,
        ):
            pass

    scores = classifier.decision_function(
        features
    )

    return (
        softmax(scores),
        classes,
    )


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
]:
    """Calculate four-class evaluation metrics."""
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

    return (
        float(accuracy),
        float(kappa),
        recalls,
        matrix,
    )


# ---------------------------------------------------------------------
# One LOSO fold
# ---------------------------------------------------------------------

def evaluate_fold(
    configuration_id: int,
    adaptation_method: str,
    test_subject: str,
    training_subjects: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Evaluate one adaptation method on one unseen subject."""
    feature_start = time.perf_counter()

    (
        training_features,
        testing_features,
        extraction_seconds,
    ) = extract_tangent_features(
        X_train,
        X_test,
    )

    adaptation_start = time.perf_counter()

    (
        adapted_training_features,
        adapted_testing_features,
    ) = adapt_features(
        method=adaptation_method,
        training_features=training_features,
        testing_features=testing_features,
    )

    adaptation_seconds = (
        time.perf_counter()
        - adaptation_start
    )

    scaler = StandardScaler()

    scaled_training_features = (
        scaler.fit_transform(
            adapted_training_features
        )
    )

    scaled_testing_features = (
        scaler.transform(
            adapted_testing_features
        )
    )

    total_feature_seconds = (
        time.perf_counter()
        - feature_start
    )

    classifier = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
        priors=np.full(
            len(CLASS_ORDER),
            1.0 / len(CLASS_ORDER),
        ),
    )

    fit_start = time.perf_counter()

    classifier.fit(
        scaled_training_features,
        y_train,
    )

    fit_seconds = (
        time.perf_counter()
        - fit_start
    )

    predicted_labels = classifier.predict(
        scaled_testing_features
    )

    probabilities, classifier_classes = (
        get_probabilities(
            classifier,
            scaled_testing_features,
        )
    )

    (
        accuracy,
        kappa,
        recalls,
        matrix,
    ) = calculate_metrics(
        y_test,
        predicted_labels,
    )

    subject_result: dict[str, Any] = {
        "configuration_id": configuration_id,
        "adaptation_method": adaptation_method,
        "covariance_estimator": (
            COVARIANCE_ESTIMATOR
        ),
        "test_subject": test_subject,
        "training_subjects": "|".join(
            training_subjects
        ),
        "training_trials": int(
            len(y_train)
        ),
        "testing_trials": int(
            len(y_test)
        ),
        "tangent_features": int(
            training_features.shape[1]
        ),
        "accuracy": accuracy,
        "accuracy_percent": (
            accuracy * 100.0
        ),
        "kappa": kappa,
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
        "extraction_seconds": float(
            extraction_seconds
        ),
        "adaptation_seconds": float(
            adaptation_seconds
        ),
        "total_feature_seconds": float(
            total_feature_seconds
        ),
        "classifier_fit_seconds": float(
            fit_seconds
        ),
    }

    for true_index, true_class in enumerate(
        CLASS_ORDER
    ):
        for predicted_index, predicted_class in enumerate(
            CLASS_ORDER
        ):
            subject_result[
                f"cm_{true_class}_pred_{predicted_class}"
            ] = int(
                matrix[
                    true_index,
                    predicted_index,
                ]
            )

    probability_index = {
        class_name: class_index
        for class_index, class_name in enumerate(
            classifier_classes
        )
    }

    prediction_rows: list[
        dict[str, Any]
    ] = []

    for trial_index, (
        true_class,
        predicted_class,
    ) in enumerate(
        zip(
            y_test,
            predicted_labels,
        )
    ):
        prediction_row: dict[str, Any] = {
            "configuration_id": configuration_id,
            "adaptation_method": adaptation_method,
            "covariance_estimator": (
                COVARIANCE_ESTIMATOR
            ),
            "test_subject": test_subject,
            "trial_index": trial_index,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "correct": bool(
                true_class == predicted_class
            ),
            "confidence": float(
                np.max(
                    probabilities[
                        trial_index
                    ]
                )
            ),
        }

        for class_name in CLASS_ORDER:
            if class_name in probability_index:
                class_index = probability_index[
                    class_name
                ]

                prediction_row[
                    f"probability_{class_name}"
                ] = float(
                    probabilities[
                        trial_index,
                        class_index,
                    ]
                )

        prediction_rows.append(
            prediction_row
        )

    return (
        subject_result,
        prediction_rows,
    )


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def aggregate_results(
    subject_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate the nine unseen-subject folds."""
    dataframe = pd.DataFrame(
        subject_results
    )

    baseline_group = dataframe[
        dataframe["adaptation_method"]
        == "none"
    ]

    baseline_accuracy = float(
        baseline_group["accuracy"].mean()
    )

    baseline_kappa = float(
        baseline_group["kappa"].mean()
    )

    configuration_results: list[
        dict[str, Any]
    ] = []

    grouped = dataframe.groupby(
        [
            "configuration_id",
            "adaptation_method",
        ],
        sort=True,
    )

    for group_keys, group in grouped:
        (
            configuration_id,
            adaptation_method,
        ) = group_keys

        mean_accuracy = float(
            group["accuracy"].mean()
        )

        mean_kappa = float(
            group["kappa"].mean()
        )

        configuration_results.append(
            {
                "configuration_id": int(
                    configuration_id
                ),
                "adaptation_method": (
                    adaptation_method
                ),
                "covariance_estimator": (
                    COVARIANCE_ESTIMATOR
                ),
                "number_of_subjects": int(
                    len(group)
                ),
                "mean_accuracy": mean_accuracy,
                "mean_accuracy_percent": (
                    mean_accuracy * 100.0
                ),
                "accuracy_standard_deviation_percent": float(
                    group["accuracy"].std(
                        ddof=1
                    )
                    * 100.0
                ),
                "minimum_subject_accuracy_percent": float(
                    group["accuracy"].min()
                    * 100.0
                ),
                "maximum_subject_accuracy_percent": float(
                    group["accuracy"].max()
                    * 100.0
                ),
                "mean_kappa": mean_kappa,
                "kappa_standard_deviation": float(
                    group["kappa"].std(
                        ddof=1
                    )
                ),
                "no_adaptation_accuracy_percent": (
                    baseline_accuracy
                    * 100.0
                ),
                "improvement_over_no_adaptation_percent_points": float(
                    (
                        mean_accuracy
                        - baseline_accuracy
                    )
                    * 100.0
                ),
                "no_adaptation_kappa": (
                    baseline_kappa
                ),
                "kappa_improvement": float(
                    mean_kappa
                    - baseline_kappa
                ),
                "mean_adaptation_seconds": float(
                    group[
                        "adaptation_seconds"
                    ].mean()
                ),
            }
        )

    configuration_results.sort(
        key=lambda row: (
            row["mean_accuracy"],
            row["mean_kappa"],
        ),
        reverse=True,
    )

    for rank, row in enumerate(
        configuration_results,
        start=1,
    ):
        row["rank"] = rank

    return configuration_results


# ---------------------------------------------------------------------
# Final display
# ---------------------------------------------------------------------

def print_final_results(
    configuration_results: list[
        dict[str, Any]
    ],
    subject_results: list[
        dict[str, Any]
    ],
    total_seconds: float,
) -> None:
    """Print ranked adaptation results."""
    print()
    print("=" * 100)
    print(
        "Unsupervised Tangent-Space Domain Adaptation Results"
    )
    print("=" * 100)

    print(
        f"\n{'Rank':<7}"
        f"{'Method':<24}"
        f"{'Accuracy':>12}"
        f"{'Kappa':>12}"
        f"{'Change':>12}"
    )

    print("-" * 67)

    for row in configuration_results:
        print(
            f"{row['rank']:<7}"
            f"{row['adaptation_method']:<24}"
            f"{row['mean_accuracy_percent']:>11.2f}%"
            f"{row['mean_kappa']:>12.3f}"
            f"{row['improvement_over_no_adaptation_percent_points']:>+11.2f}"
        )

    best = configuration_results[0]

    print()
    print("=" * 100)
    print("Best adaptation method")
    print("=" * 100)

    print(
        f"Method:               "
        f"{best['adaptation_method']}"
    )

    print(
        f"Covariance estimator: "
        f"{best['covariance_estimator']}"
    )

    print(
        f"Mean accuracy:        "
        f"{best['mean_accuracy_percent']:.2f}%"
    )

    print(
        f"Mean kappa:           "
        f"{best['mean_kappa']:.3f}"
    )

    print(
        "Change from none:     "
        f"{best['improvement_over_no_adaptation_percent_points']:+.2f} "
        "percentage points"
    )

    best_subject_rows = [
        row
        for row in subject_results
        if row["configuration_id"]
        == best["configuration_id"]
    ]

    print()
    print(
        f"{'Subject':<12}"
        f"{'Accuracy':>12}"
        f"{'Kappa':>12}"
        f"{'Left':>10}"
        f"{'Right':>10}"
        f"{'Feet':>10}"
        f"{'Tongue':>10}"
    )

    print("-" * 76)

    for row in sorted(
        best_subject_rows,
        key=lambda item: item[
            "test_subject"
        ],
    ):
        print(
            f"{row['test_subject']:<12}"
            f"{row['accuracy_percent']:>11.2f}%"
            f"{row['kappa']:>12.3f}"
            f"{row['left_hand_recall']:>10.3f}"
            f"{row['right_hand_recall']:>10.3f}"
            f"{row['feet_recall']:>10.3f}"
            f"{row['tongue_recall']:>10.3f}"
        )

    print()
    print(
        f"Total experiment time: "
        f"{total_seconds:.2f} seconds"
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
    """Run strict LOSO domain adaptation evaluation."""
    total_start = time.perf_counter()

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "Strict LOSO: Unsupervised Tangent-Space Domain Adaptation"
    )
    print("=" * 100)

    print(
        "\nPreprocessing and alignment source:"
    )

    print(
        "  scripts.cross_subject.run_broadband_riemannian_sweep.load_subjects"
    )

    print(
        "  scripts.cross_subject.run_cross_subject_evaluation.load_and_align_subject"
    )

    print(
        f"\nSubjects: {len(SUBJECTS)}"
    )

    print(
        f"Adaptation methods: "
        f"{len(ADAPTATION_METHODS)}"
    )

    print(
        "Total model evaluations: "
        f"{len(SUBJECTS) * len(ADAPTATION_METHODS)}"
    )

    print(
        f"\nCovariance estimator: "
        f"{COVARIANCE_ESTIMATOR}"
    )

    print(
        f"CORAL regularisation: "
        f"{CORAL_REGULARISATION}"
    )

    print(
        "\nTarget labels are used only after prediction "
        "for evaluation."
    )

    loaded_subjects = load_subjects()

    print()
    print("=" * 100)
    print("Alignment verification")
    print("=" * 100)

    for subject in SUBJECTS:
        alignment_error = float(
            loaded_subjects[
                subject
            ]["alignment_error"]
        )

        print(
            f"{subject}: "
            f"{alignment_error:.8f}"
        )

        if alignment_error > 1e-5:
            print(
                "WARNING: Alignment error is higher "
                "than expected."
            )

    subject_results: list[
        dict[str, Any]
    ] = []

    prediction_results: list[
        dict[str, Any]
    ] = []

    for configuration_id, adaptation_method in enumerate(
        ADAPTATION_METHODS,
        start=1,
    ):
        print()
        print("#" * 100)

        print(
            f"Configuration "
            f"{configuration_id}/"
            f"{len(ADAPTATION_METHODS)}: "
            f"{adaptation_method}"
        )

        print("#" * 100)

        for fold_number, test_subject in enumerate(
            SUBJECTS,
            start=1,
        ):
            training_subjects = [
                subject
                for subject in SUBJECTS
                if subject != test_subject
            ]

            X_train = np.concatenate(
                [
                    loaded_subjects[
                        subject
                    ]["X"]
                    for subject in training_subjects
                ],
                axis=0,
            )

            y_train = np.concatenate(
                [
                    loaded_subjects[
                        subject
                    ]["y"]
                    for subject in training_subjects
                ],
                axis=0,
            )

            X_test = loaded_subjects[
                test_subject
            ]["X"]

            y_test = loaded_subjects[
                test_subject
            ]["y"]

            print()
            print(
                f"Fold {fold_number}/"
                f"{len(SUBJECTS)} | "
                f"Unseen subject: {test_subject}"
            )

            print(
                f"Training trials: {len(y_train)}"
            )

            print(
                f"Testing trials:  {len(y_test)}"
            )

            (
                subject_result,
                prediction_rows,
            ) = evaluate_fold(
                configuration_id=(
                    configuration_id
                ),
                adaptation_method=(
                    adaptation_method
                ),
                test_subject=test_subject,
                training_subjects=(
                    training_subjects
                ),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

            subject_results.append(
                subject_result
            )

            prediction_results.extend(
                prediction_rows
            )

            print(
                f"Accuracy: "
                f"{subject_result['accuracy_percent']:.2f}%"
            )

            print(
                f"Kappa:    "
                f"{subject_result['kappa']:.3f}"
            )

            print(
                "Recall:    "
                f"left={subject_result['left_hand_recall']:.3f}, "
                f"right={subject_result['right_hand_recall']:.3f}, "
                f"feet={subject_result['feet_recall']:.3f}, "
                f"tongue={subject_result['tongue_recall']:.3f}"
            )

            write_csv(
                SUBJECT_RESULTS_PATH,
                subject_results,
            )

            write_csv(
                PREDICTIONS_PATH,
                prediction_results,
            )

    configuration_results = (
        aggregate_results(
            subject_results
        )
    )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    for row in configuration_results:
        row[
            "total_experiment_seconds"
        ] = float(
            total_seconds
        )

    write_csv(
        CONFIGURATION_RESULTS_PATH,
        configuration_results,
    )

    write_csv(
        BEST_CONFIGURATION_PATH,
        [
            configuration_results[0]
        ],
    )

    print_final_results(
        configuration_results=(
            configuration_results
        ),
        subject_results=(
            subject_results
        ),
        total_seconds=total_seconds,
    )


if __name__ == "__main__":
    main()
