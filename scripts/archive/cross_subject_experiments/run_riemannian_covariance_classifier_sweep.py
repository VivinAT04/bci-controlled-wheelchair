"""
Strict LOSO covariance-estimator and Riemannian-classifier comparison.

The experiment reuses the exact subject loading, preprocessing and
Euclidean Alignment from:

    scripts.cross_subject.run_broadband_riemannian_sweep.load_subjects()

That function internally uses:

    scripts.cross_subject.run_cross_subject_evaluation.load_and_align_subject()

Therefore, all configurations use the same:

    8-30 Hz filtering
    0.5-2.5 second epoch
    subject-wise Euclidean Alignment
    strict Leave-One-Subject-Out evaluation

Only the covariance estimator and classifier are changed.

Configurations
--------------
Covariance estimators:
    OAS
    Ledoit-Wolf
    SCM

Classifiers:
    Tangent Space + StandardScaler + shrinkage LDA
    Minimum Distance to Mean
    Fisher Geodesic MDM

Run:

    python -m scripts.cross_subject.run_riemannian_covariance_classifier_sweep
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyriemann.classification import FgMDM, MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)
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

from bci_wheelchair.models.euclidean_alignment import (
    CLASS_ORDER,
)


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

COVARIANCE_ESTIMATORS = [
    "oas",
    "lwf",
    "scm",
]

CLASSIFIERS = [
    "tangent_lda",
    "mdm",
    "fgmdm",
]

BASELINE_ACCURACY = 0.5366512345679012
BASELINE_KAPPA = 0.382201646090535

RESULTS_DIRECTORY = Path(
    "results/cross_subject/riemannian/riemannian_covariance_classifier_sweep"
)

CONFIGURATION_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_covariance_classifier_configuration_results.csv"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_covariance_classifier_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_covariance_classifier_predictions.csv"
)

BEST_CONFIGURATION_PATH = (
    RESULTS_DIRECTORY
    / "riemannian_covariance_classifier_best_configuration.csv"
)


# ---------------------------------------------------------------------
# CSV helper
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
# Feature extraction
# ---------------------------------------------------------------------

def extract_covariances(
    X_train: np.ndarray,
    X_test: np.ndarray,
    estimator: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate covariance matrices.

    Covariance transformation is fitted on training data only.
    """
    transformer = Covariances(
        estimator=estimator,
    )

    training_covariances = (
        transformer.fit_transform(
            X_train
        )
    )

    testing_covariances = (
        transformer.transform(
            X_test
        )
    )

    return (
        np.asarray(
            training_covariances,
            dtype=np.float64,
        ),
        np.asarray(
            testing_covariances,
            dtype=np.float64,
        ),
    )


def extract_tangent_features(
    training_covariances: np.ndarray,
    testing_covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit the Riemannian tangent-space reference using training data only.
    """
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

    return (
        np.asarray(
            training_features,
            dtype=np.float64,
        ),
        np.asarray(
            testing_features,
            dtype=np.float64,
        ),
    )


# ---------------------------------------------------------------------
# Prediction confidence
# ---------------------------------------------------------------------

def softmax(
    scores: np.ndarray,
) -> np.ndarray:
    """Convert decision scores into normalised values."""
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
    classifier,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Obtain probability or probability-like confidence values.
    """
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
                    X_test
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

    if hasattr(
        classifier,
        "decision_function",
    ):
        scores = (
            classifier.decision_function(
                X_test
            )
        )

        return (
            softmax(scores),
            classes,
        )

    number_of_trials = len(X_test)
    number_of_classes = len(classes)

    probabilities = np.full(
        (
            number_of_trials,
            number_of_classes,
        ),
        1.0 / number_of_classes,
        dtype=np.float64,
    )

    return probabilities, classes


# ---------------------------------------------------------------------
# Tangent-space LDA
# ---------------------------------------------------------------------

def evaluate_tangent_lda(
    training_covariances: np.ndarray,
    testing_covariances: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    int,
]:
    """
    Evaluate:

        covariance
        -> tangent space
        -> StandardScaler
        -> shrinkage LDA
    """
    feature_start = time.perf_counter()

    (
        training_features,
        testing_features,
    ) = extract_tangent_features(
        training_covariances,
        testing_covariances,
    )

    scaler = StandardScaler()

    scaled_training_features = (
        scaler.fit_transform(
            training_features
        )
    )

    scaled_testing_features = (
        scaler.transform(
            testing_features
        )
    )

    feature_seconds = (
        time.perf_counter()
        - feature_start
    )

    classifier = (
        LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
            priors=np.full(
                len(CLASS_ORDER),
                1.0 / len(CLASS_ORDER),
            ),
        )
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

    probabilities, classes = (
        get_probabilities(
            classifier,
            scaled_testing_features,
        )
    )

    return (
        predicted_labels,
        probabilities,
        classes,
        feature_seconds,
        fit_seconds,
        int(training_features.shape[1]),
    )


# ---------------------------------------------------------------------
# MDM and FgMDM
# ---------------------------------------------------------------------

def build_native_riemannian_classifier(
    classifier_name: str,
):
    """Construct MDM or FgMDM."""
    if classifier_name == "mdm":
        return MDM(
            metric="riemann",
            n_jobs=-1,
        )

    if classifier_name == "fgmdm":
        return FgMDM(
            metric="riemann",
            tsupdate=False,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unknown native classifier: {classifier_name}"
    )


def evaluate_native_classifier(
    classifier_name: str,
    training_covariances: np.ndarray,
    testing_covariances: np.ndarray,
    y_train: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Fit and evaluate MDM or FgMDM."""
    classifier = (
        build_native_riemannian_classifier(
            classifier_name
        )
    )

    fit_start = time.perf_counter()

    classifier.fit(
        training_covariances,
        y_train,
    )

    fit_seconds = (
        time.perf_counter()
        - fit_start
    )

    predicted_labels = classifier.predict(
        testing_covariances
    )

    probabilities, classes = (
        get_probabilities(
            classifier,
            testing_covariances,
        )
    )

    return (
        predicted_labels,
        probabilities,
        classes,
        fit_seconds,
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
    """Calculate standard four-class metrics."""
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

def evaluate_fold_configuration(
    configuration_id: int,
    covariance_estimator: str,
    classifier_name: str,
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
    """Evaluate one configuration on one unseen subject."""
    configuration_label = (
        f"{covariance_estimator}_{classifier_name}"
    )

    covariance_start = time.perf_counter()

    (
        training_covariances,
        testing_covariances,
    ) = extract_covariances(
        X_train=X_train,
        X_test=X_test,
        estimator=covariance_estimator,
    )

    covariance_seconds = (
        time.perf_counter()
        - covariance_start
    )

    prediction_start = time.perf_counter()

    if classifier_name == "tangent_lda":
        (
            predicted_labels,
            probabilities,
            classifier_classes,
            feature_seconds,
            classifier_fit_seconds,
            number_of_features,
        ) = evaluate_tangent_lda(
            training_covariances=(
                training_covariances
            ),
            testing_covariances=(
                testing_covariances
            ),
            y_train=y_train,
            y_test=y_test,
        )

    else:
        (
            predicted_labels,
            probabilities,
            classifier_classes,
            classifier_fit_seconds,
        ) = evaluate_native_classifier(
            classifier_name=classifier_name,
            training_covariances=(
                training_covariances
            ),
            testing_covariances=(
                testing_covariances
            ),
            y_train=y_train,
        )

        feature_seconds = 0.0
        number_of_features = 0

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
        "configuration_label": configuration_label,
        "covariance_estimator": covariance_estimator,
        "classifier_name": classifier_name,
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
            number_of_features
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
        "covariance_seconds": float(
            covariance_seconds
        ),
        "feature_seconds": float(
            feature_seconds
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
            "configuration_label": configuration_label,
            "covariance_estimator": covariance_estimator,
            "classifier_name": classifier_name,
            "test_subject": test_subject,
            "trial_index": trial_index,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "correct": bool(
                true_class
                == predicted_class
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
                class_index = (
                    probability_index[
                        class_name
                    ]
                )

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
    """Aggregate nine LOSO folds for each configuration."""
    dataframe = pd.DataFrame(
        subject_results
    )

    configuration_results: list[
        dict[str, Any]
    ] = []

    grouped = dataframe.groupby(
        [
            "configuration_id",
            "configuration_label",
            "covariance_estimator",
            "classifier_name",
        ],
        sort=True,
    )

    for group_keys, group in grouped:
        (
            configuration_id,
            configuration_label,
            covariance_estimator,
            classifier_name,
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
                "configuration_label": (
                    configuration_label
                ),
                "covariance_estimator": (
                    covariance_estimator
                ),
                "classifier_name": (
                    classifier_name
                ),
                "number_of_subjects": int(
                    len(group)
                ),
                "mean_accuracy": mean_accuracy,
                "mean_accuracy_percent": (
                    mean_accuracy * 100.0
                ),
                "accuracy_standard_deviation": float(
                    group[
                        "accuracy"
                    ].std(
                        ddof=1
                    )
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
                "mean_kappa": mean_kappa,
                "kappa_standard_deviation": float(
                    group[
                        "kappa"
                    ].std(
                        ddof=1
                    )
                ),
                "baseline_accuracy_percent": (
                    BASELINE_ACCURACY
                    * 100.0
                ),
                "improvement_over_baseline_percent_points": float(
                    (
                        mean_accuracy
                        - BASELINE_ACCURACY
                    )
                    * 100.0
                ),
                "baseline_kappa": (
                    BASELINE_KAPPA
                ),
                "kappa_improvement": float(
                    mean_kappa
                    - BASELINE_KAPPA
                ),
                "mean_covariance_seconds": float(
                    group[
                        "covariance_seconds"
                    ].mean()
                ),
                "mean_feature_seconds": float(
                    group[
                        "feature_seconds"
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
# Result output
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
    """Print ranked and subject-level results."""
    print()
    print("=" * 100)
    print(
        "Riemannian Covariance and Classifier Sweep Results"
    )
    print("=" * 100)

    print(
        f"\n{'Rank':<7}"
        f"{'Covariance':<16}"
        f"{'Classifier':<20}"
        f"{'Accuracy':>12}"
        f"{'Kappa':>10}"
        f"{'Change':>12}"
    )

    print("-" * 79)

    for row in configuration_results:
        print(
            f"{row['rank']:<7}"
            f"{row['covariance_estimator']:<16}"
            f"{row['classifier_name']:<20}"
            f"{row['mean_accuracy_percent']:>11.2f}%"
            f"{row['mean_kappa']:>10.3f}"
            f"{row['improvement_over_baseline_percent_points']:>+11.2f}"
        )

    best = configuration_results[0]

    print()
    print("=" * 100)
    print("Best configuration")
    print("=" * 100)

    print(
        f"Covariance estimator: "
        f"{best['covariance_estimator']}"
    )

    print(
        f"Classifier:           "
        f"{best['classifier_name']}"
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
        "Change from baseline: "
        f"{best['improvement_over_baseline_percent_points']:+.2f} "
        "percentage points"
    )

    best_subject_results = [
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
        best_subject_results,
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

    baseline_rows = [
        row
        for row in configuration_results
        if row["covariance_estimator"] == "oas"
        and row["classifier_name"] == "tangent_lda"
    ]

    if baseline_rows:
        reproduced = baseline_rows[0]

        difference = abs(
            reproduced[
                "mean_accuracy_percent"
            ]
            - BASELINE_ACCURACY
            * 100.0
        )

        print()
        print("=" * 100)
        print("Baseline reproduction check")
        print("=" * 100)

        print(
            "Expected accuracy:   "
            f"{BASELINE_ACCURACY * 100.0:.4f}%"
        )

        print(
            "Reproduced accuracy: "
            f"{reproduced['mean_accuracy_percent']:.4f}%"
        )

        print(
            "Absolute difference: "
            f"{difference:.6f} percentage points"
        )

        if difference <= 0.05:
            print(
                "PASS: Baseline reproduced successfully."
            )
        else:
            print(
                "WARNING: Baseline was not reproduced."
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
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete strict LOSO experiment."""
    total_start = time.perf_counter()

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print(
        "Strict LOSO: Covariance and Riemannian Classifier Sweep"
    )
    print("=" * 100)

    print(
        "\nExact loading and alignment source:"
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
        "Configurations: "
        f"{len(COVARIANCE_ESTIMATORS) * len(CLASSIFIERS)}"
    )

    print(
        "Total model evaluations: "
        f"{len(SUBJECTS) * len(COVARIANCE_ESTIMATORS) * len(CLASSIFIERS)}"
    )

    print(
        "\nReference baseline:"
    )

    print(
        f"  Accuracy: "
        f"{BASELINE_ACCURACY * 100.0:.2f}%"
    )

    print(
        f"  Kappa:    "
        f"{BASELINE_KAPPA:.3f}"
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

    configuration_id = 0

    for covariance_estimator in (
        COVARIANCE_ESTIMATORS
    ):
        for classifier_name in CLASSIFIERS:
            configuration_id += 1

            print()
            print("#" * 100)

            print(
                f"Configuration {configuration_id}/9: "
                f"{covariance_estimator} + "
                f"{classifier_name}"
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
                    f"Fold {fold_number}/9 | "
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
                ) = evaluate_fold_configuration(
                    configuration_id=(
                        configuration_id
                    ),
                    covariance_estimator=(
                        covariance_estimator
                    ),
                    classifier_name=(
                        classifier_name
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
