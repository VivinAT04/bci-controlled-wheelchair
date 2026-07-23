"""
Strict LOSO tangent-space mean-alignment and PCA sweep.

Pipeline
--------
8-30 Hz EEG
-> 0.5-2.5 second epoch
-> subject-wise Euclidean Alignment
-> SCM covariance
-> Riemannian Tangent Space
-> unsupervised source/target mean alignment
-> StandardScaler
-> optional PCA
-> shrinkage LDA

PCA configurations
------------------
none
90% explained variance
95% explained variance
99% explained variance

Important
---------
The unseen target subject labels are used only after prediction to
calculate evaluation metrics.

Run:

    python -m scripts.cross_subject.run_tangent_mean_alignment_pca_sweep
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

from sklearn.decomposition import PCA
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

from bci_wheelchair.cross_subject import (
    CLASS_ORDER,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

COVARIANCE_ESTIMATOR = "scm"

PCA_CONFIGURATIONS: list[tuple[str, float | None]] = [
    ("no_pca", None),
    ("pca_90", 0.90),
    ("pca_95", 0.95),
    ("pca_99", 0.99),
]

RESULTS_DIRECTORY = Path(
    "results/cross_subject/riemannian/tangent_mean_alignment_pca_sweep"
)

CONFIGURATION_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "pca_configuration_results.csv"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "pca_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "pca_predictions.csv"
)

BEST_CONFIGURATION_PATH = (
    RESULTS_DIRECTORY
    / "pca_best_configuration.csv"
)


# ---------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------

def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write a list of dictionaries to CSV."""
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
# Feature extraction
# ---------------------------------------------------------------------

def extract_tangent_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Extract SCM covariance matrices and tangent-space features.

    The tangent-space reference is fitted using training subjects only.
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

    elapsed_seconds = (
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
        float(elapsed_seconds),
    )


# ---------------------------------------------------------------------
# Mean alignment
# ---------------------------------------------------------------------

def apply_mean_alignment(
    training_features: np.ndarray,
    testing_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Centre source and target tangent features independently.

    This uses the unseen target subject's feature mean but does not use
    target labels.
    """
    source_mean = training_features.mean(
        axis=0,
        keepdims=True,
    )

    target_mean = testing_features.mean(
        axis=0,
        keepdims=True,
    )

    aligned_training_features = (
        training_features
        - source_mean
    )

    aligned_testing_features = (
        testing_features
        - target_mean
    )

    return (
        np.asarray(
            aligned_training_features,
            dtype=np.float64,
        ),
        np.asarray(
            aligned_testing_features,
            dtype=np.float64,
        ),
    )


# ---------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------

def apply_optional_pca(
    training_features: np.ndarray,
    testing_features: np.ndarray,
    variance_threshold: float | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
    float,
]:
    """
    Fit PCA using training features only.

    A value such as 0.95 retains enough principal components to explain
    at least 95% of training-feature variance.
    """
    if variance_threshold is None:
        return (
            training_features.copy(),
            testing_features.copy(),
            int(training_features.shape[1]),
            1.0,
        )

    pca = PCA(
        n_components=variance_threshold,
        svd_solver="full",
    )

    reduced_training_features = (
        pca.fit_transform(
            training_features
        )
    )

    reduced_testing_features = (
        pca.transform(
            testing_features
        )
    )

    retained_variance = float(
        np.sum(
            pca.explained_variance_ratio_
        )
    )

    return (
        np.asarray(
            reduced_training_features,
            dtype=np.float64,
        ),
        np.asarray(
            reduced_testing_features,
            dtype=np.float64,
        ),
        int(
            reduced_training_features.shape[1]
        ),
        retained_variance,
    )


# ---------------------------------------------------------------------
# Probability helper
# ---------------------------------------------------------------------

def softmax(
    scores: np.ndarray,
) -> np.ndarray:
    """Convert decision scores into normalised confidence values."""
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

    shifted_scores = (
        scores
        - np.max(
            scores,
            axis=1,
            keepdims=True,
        )
    )

    exponentials = np.exp(
        shifted_scores
    )

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
    """Return classifier probabilities or score-derived probabilities."""
    classes = np.asarray(
        classifier.classes_
    )

    if hasattr(
        classifier,
        "predict_proba",
    ):
        try:
            probabilities = (
                classifier.predict_proba(
                    features
                )
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
    """Calculate accuracy, kappa, per-class recall and confusion matrix."""
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
    configuration_name: str,
    pca_variance: float | None,
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
    """Evaluate one PCA configuration on one unseen subject."""
    feature_start = time.perf_counter()

    (
        training_features,
        testing_features,
        extraction_seconds,
    ) = extract_tangent_features(
        X_train=X_train,
        X_test=X_test,
    )

    original_feature_count = int(
        training_features.shape[1]
    )

    (
        aligned_training_features,
        aligned_testing_features,
    ) = apply_mean_alignment(
        training_features=training_features,
        testing_features=testing_features,
    )

    scaler = StandardScaler()

    scaled_training_features = (
        scaler.fit_transform(
            aligned_training_features
        )
    )

    scaled_testing_features = (
        scaler.transform(
            aligned_testing_features
        )
    )

    pca_start = time.perf_counter()

    (
        final_training_features,
        final_testing_features,
        retained_components,
        retained_variance,
    ) = apply_optional_pca(
        training_features=(
            scaled_training_features
        ),
        testing_features=(
            scaled_testing_features
        ),
        variance_threshold=pca_variance,
    )

    pca_seconds = (
        time.perf_counter()
        - pca_start
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
        final_training_features,
        y_train,
    )

    classifier_fit_seconds = (
        time.perf_counter()
        - fit_start
    )

    prediction_start = time.perf_counter()

    predicted_labels = classifier.predict(
        final_testing_features
    )

    probabilities, classifier_classes = (
        get_probabilities(
            classifier,
            final_testing_features,
        )
    )

    prediction_seconds = (
        time.perf_counter()
        - prediction_start
    )

    (
        accuracy,
        kappa,
        recalls,
        matrix,
    ) = calculate_metrics(
        y_true=y_test,
        y_pred=predicted_labels,
    )

    subject_result: dict[str, Any] = {
        "configuration_id": configuration_id,
        "configuration_name": configuration_name,
        "pca_variance_threshold": (
            pca_variance
            if pca_variance is not None
            else "none"
        ),
        "covariance_estimator": (
            COVARIANCE_ESTIMATOR
        ),
        "adaptation_method": (
            "mean_alignment"
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
        "original_tangent_features": (
            original_feature_count
        ),
        "retained_pca_components": int(
            retained_components
        ),
        "retained_variance": float(
            retained_variance
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
        "pca_seconds": float(
            pca_seconds
        ),
        "total_feature_seconds": float(
            total_feature_seconds
        ),
        "classifier_fit_seconds": float(
            classifier_fit_seconds
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
            "configuration_name": configuration_name,
            "pca_variance_threshold": (
                pca_variance
                if pca_variance is not None
                else "none"
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
    """Aggregate all nine LOSO folds for each PCA configuration."""
    dataframe = pd.DataFrame(
        subject_results
    )

    baseline_group = dataframe[
        dataframe["configuration_name"]
        == "no_pca"
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
            "configuration_name",
            "pca_variance_threshold",
        ],
        sort=True,
        dropna=False,
    )

    for group_keys, group in grouped:
        (
            configuration_id,
            configuration_name,
            pca_variance_threshold,
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
                "configuration_name": (
                    configuration_name
                ),
                "pca_variance_threshold": (
                    pca_variance_threshold
                ),
                "covariance_estimator": (
                    COVARIANCE_ESTIMATOR
                ),
                "adaptation_method": (
                    "mean_alignment"
                ),
                "number_of_subjects": int(
                    len(group)
                ),
                "mean_accuracy": (
                    mean_accuracy
                ),
                "mean_accuracy_percent": (
                    mean_accuracy * 100.0
                ),
                "accuracy_standard_deviation_percent": float(
                    group[
                        "accuracy"
                    ].std(
                        ddof=1
                    )
                    * 100.0
                ),
                "minimum_subject_accuracy_percent": float(
                    group[
                        "accuracy"
                    ].min()
                    * 100.0
                ),
                "maximum_subject_accuracy_percent": float(
                    group[
                        "accuracy"
                    ].max()
                    * 100.0
                ),
                "mean_kappa": (
                    mean_kappa
                ),
                "kappa_standard_deviation": float(
                    group[
                        "kappa"
                    ].std(
                        ddof=1
                    )
                ),
                "mean_retained_components": float(
                    group[
                        "retained_pca_components"
                    ].mean()
                ),
                "mean_retained_variance": float(
                    group[
                        "retained_variance"
                    ].mean()
                ),
                "no_pca_accuracy_percent": (
                    baseline_accuracy
                    * 100.0
                ),
                "improvement_over_no_pca_percent_points": float(
                    (
                        mean_accuracy
                        - baseline_accuracy
                    )
                    * 100.0
                ),
                "no_pca_kappa": (
                    baseline_kappa
                ),
                "kappa_improvement": float(
                    mean_kappa
                    - baseline_kappa
                ),
                "mean_pca_seconds": float(
                    group[
                        "pca_seconds"
                    ].mean()
                ),
                "mean_classifier_fit_seconds": float(
                    group[
                        "classifier_fit_seconds"
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
# Display
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
    """Print ranked PCA results."""
    print()
    print("=" * 100)
    print(
        "Mean Alignment and PCA Sweep Results"
    )
    print("=" * 100)

    print(
        f"\n{'Rank':<7}"
        f"{'Configuration':<20}"
        f"{'Accuracy':>12}"
        f"{'Kappa':>12}"
        f"{'Components':>14}"
        f"{'Change':>12}"
    )

    print("-" * 77)

    for row in configuration_results:
        print(
            f"{row['rank']:<7}"
            f"{row['configuration_name']:<20}"
            f"{row['mean_accuracy_percent']:>11.2f}%"
            f"{row['mean_kappa']:>12.3f}"
            f"{row['mean_retained_components']:>14.1f}"
            f"{row['improvement_over_no_pca_percent_points']:>+11.2f}"
        )

    best = configuration_results[0]

    print()
    print("=" * 100)
    print("Best configuration")
    print("=" * 100)

    print(
        f"Configuration:         "
        f"{best['configuration_name']}"
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
        f"Mean PCA components:  "
        f"{best['mean_retained_components']:.1f}"
    )

    print(
        "Change from no PCA:   "
        f"{best['improvement_over_no_pca_percent_points']:+.2f} "
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
        f"{'PCA':>8}"
        f"{'Left':>10}"
        f"{'Right':>10}"
        f"{'Feet':>10}"
        f"{'Tongue':>10}"
    )

    print("-" * 84)

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
            f"{row['retained_pca_components']:>8}"
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
    """Run the strict LOSO PCA sweep."""
    total_start = time.perf_counter()

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "Strict LOSO: Tangent Mean Alignment and PCA Sweep"
    )
    print("=" * 100)

    print(
        "\nPreprocessing and alignment:"
    )

    print(
        "  8-30 Hz filtering"
    )

    print(
        "  0.5-2.5 second epoch"
    )

    print(
        "  subject-wise Euclidean Alignment"
    )

    print(
        f"\nSubjects: "
        f"{len(SUBJECTS)}"
    )

    print(
        f"Configurations: "
        f"{len(PCA_CONFIGURATIONS)}"
    )

    print(
        "Total model evaluations: "
        f"{len(SUBJECTS) * len(PCA_CONFIGURATIONS)}"
    )

    print(
        f"\nCovariance estimator: "
        f"{COVARIANCE_ESTIMATOR}"
    )

    print(
        "Adaptation method: mean_alignment"
    )

    print(
        "\nTarget labels are used only after prediction."
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

    for configuration_id, (
        configuration_name,
        pca_variance,
    ) in enumerate(
        PCA_CONFIGURATIONS,
        start=1,
    ):
        print()
        print("#" * 100)

        print(
            f"Configuration "
            f"{configuration_id}/"
            f"{len(PCA_CONFIGURATIONS)}: "
            f"{configuration_name}"
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
                configuration_name=(
                    configuration_name
                ),
                pca_variance=(
                    pca_variance
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
                "Features:  "
                f"{subject_result['original_tangent_features']} "
                f"-> "
                f"{subject_result['retained_pca_components']}"
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
