"""
Broadband Riemannian LOSO hyperparameter sweep.

This experiment keeps the best broadband feature representation:

    Subject-wise Euclidean Alignment
    -> OAS covariance matrices
    -> Riemannian Tangent Space
    -> StandardScaler
    -> PCA
    -> classifier

The experiment tests several PCA retained-variance values and linear
classifiers under strict Leave-One-Subject-Out evaluation.

Run from the repository root:

    python -m scripts.cross_subject.run_broadband_riemannian_sweep

Outputs:

    results/cross_subject/riemannian/broadband_riemannian_sweep/
        broadband_riemannian_configuration_results.csv
        broadband_riemannian_subject_results.csv
        broadband_riemannian_predictions.csv
        broadband_riemannian_best_configuration.csv
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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.cross_subject.run_cross_subject_evaluation import (
    CLASS_ORDER,
    load_and_align_subject,
)


# ---------------------------------------------------------------------
# Experiment configuration
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

PCA_VARIANCES = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
]

LOGISTIC_REGRESSION_C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
]

LINEAR_SVM_C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
]

RIDGE_ALPHA_VALUES = [
    0.1,
    1.0,
    10.0,
    100.0,
]

RESULTS_DIRECTORY = Path(
    "results/cross_subject/riemannian/broadband_riemannian_sweep"
)

CONFIGURATION_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "broadband_riemannian_configuration_results.csv"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "broadband_riemannian_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "broadband_riemannian_predictions.csv"
)

BEST_CONFIGURATION_PATH = (
    RESULTS_DIRECTORY
    / "broadband_riemannian_best_configuration.csv"
)

BASELINE_ACCURACY = 0.5366512345679012
BASELINE_KAPPA = 0.382201646090535

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# CSV helper
# ---------------------------------------------------------------------

def write_csv(
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
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_subjects() -> dict[str, dict[str, Any]]:
    """
    Load every subject and apply Euclidean Alignment independently.

    The target subject is never combined with the training subjects.
    """
    loaded_subjects: dict[str, dict[str, Any]] = {}

    print("=" * 80)
    print("Loading and Euclidean-aligning all subjects")
    print("=" * 80)

    for subject in SUBJECTS:
        X, y, alignment_error = load_and_align_subject(
            subject
        )

        loaded_subjects[subject] = {
            "X": np.asarray(
                X,
                dtype=np.float64,
            ),
            "y": np.asarray(y),
            "alignment_error": float(
                alignment_error
            ),
        }

    return loaded_subjects


# ---------------------------------------------------------------------
# Configuration generation
# ---------------------------------------------------------------------

def build_configurations() -> list[dict[str, Any]]:
    """Create all PCA and classifier configurations."""
    configurations: list[dict[str, Any]] = []

    configuration_id = 1

    for pca_variance in PCA_VARIANCES:
        configurations.append(
            {
                "configuration_id": configuration_id,
                "pca_variance": pca_variance,
                "classifier_name": "shrinkage_lda",
                "classifier_parameter_name": "none",
                "classifier_parameter_value": "none",
            }
        )

        configuration_id += 1

        for c_value in LOGISTIC_REGRESSION_C_VALUES:
            configurations.append(
                {
                    "configuration_id": configuration_id,
                    "pca_variance": pca_variance,
                    "classifier_name": (
                        "logistic_regression"
                    ),
                    "classifier_parameter_name": "C",
                    "classifier_parameter_value": c_value,
                }
            )

            configuration_id += 1

        for c_value in LINEAR_SVM_C_VALUES:
            configurations.append(
                {
                    "configuration_id": configuration_id,
                    "pca_variance": pca_variance,
                    "classifier_name": "linear_svm",
                    "classifier_parameter_name": "C",
                    "classifier_parameter_value": c_value,
                }
            )

            configuration_id += 1

        for alpha_value in RIDGE_ALPHA_VALUES:
            configurations.append(
                {
                    "configuration_id": configuration_id,
                    "pca_variance": pca_variance,
                    "classifier_name": (
                        "ridge_classifier"
                    ),
                    "classifier_parameter_name": (
                        "alpha"
                    ),
                    "classifier_parameter_value": (
                        alpha_value
                    ),
                }
            )

            configuration_id += 1

    return configurations


# ---------------------------------------------------------------------
# Classifier construction
# ---------------------------------------------------------------------

def build_classifier(
    configuration: dict[str, Any],
):
    """Construct a classifier for one configuration."""
    classifier_name = configuration[
        "classifier_name"
    ]

    parameter_value = configuration[
        "classifier_parameter_value"
    ]

    if classifier_name == "shrinkage_lda":
        return LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
            priors=np.full(
                len(CLASS_ORDER),
                1.0 / len(CLASS_ORDER),
            ),
        )

    if classifier_name == "logistic_regression":
        return LogisticRegression(
            C=float(parameter_value),
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if classifier_name == "linear_svm":
        return SVC(
            C=float(parameter_value),
            kernel="linear",
            probability=False,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if classifier_name == "ridge_classifier":
        return RidgeClassifier(
            alpha=float(parameter_value),
            class_weight="balanced",
        )

    raise ValueError(
        f"Unknown classifier: {classifier_name}"
    )


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def extract_fold_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit covariance and tangent-space transforms using training data only.

    The test subject is transformed using the reference point learned
    entirely from the eight training subjects.
    """
    covariance_transformer = Covariances(
        estimator="oas"
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

    tangent_transformer = TangentSpace(
        metric="riemann"
    )

    training_tangent_features = (
        tangent_transformer.fit_transform(
            training_covariances
        )
    )

    testing_tangent_features = (
        tangent_transformer.transform(
            testing_covariances
        )
    )

    return (
        np.asarray(
            training_tangent_features,
            dtype=np.float64,
        ),
        np.asarray(
            testing_tangent_features,
            dtype=np.float64,
        ),
    )


def prepare_pca_features(
    training_tangent_features: np.ndarray,
    testing_tangent_features: np.ndarray,
    pca_variance: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    StandardScaler,
    PCA,
]:
    """
    Fit StandardScaler and PCA on training features only.
    """
    scaler = StandardScaler()

    scaled_training_features = (
        scaler.fit_transform(
            training_tangent_features
        )
    )

    scaled_testing_features = (
        scaler.transform(
            testing_tangent_features
        )
    )

    pca = PCA(
        n_components=pca_variance,
        svd_solver="full",
        random_state=RANDOM_STATE,
    )

    reduced_training_features = (
        pca.fit_transform(
            scaled_training_features
        )
    )

    reduced_testing_features = (
        pca.transform(
            scaled_testing_features
        )
    )

    return (
        reduced_training_features,
        reduced_testing_features,
        scaler,
        pca,
    )


# ---------------------------------------------------------------------
# Probability extraction
# ---------------------------------------------------------------------

def get_probabilities(
    classifier,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return class probabilities or probability-like confidence values.

    SVM and Logistic Regression provide predict_proba.

    RidgeClassifier provides decision_function, so its scores are
    converted to normalised softmax values for consistent result files.
    """
    classifier_classes = np.asarray(
        classifier.classes_
    )

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(
            X_test
        )

        return (
            np.asarray(
                probabilities,
                dtype=float,
            ),
            classifier_classes,
        )

    decision_scores = classifier.decision_function(
        X_test
    )

    decision_scores = np.asarray(
        decision_scores,
        dtype=float,
    )

    if decision_scores.ndim == 1:
        decision_scores = np.column_stack(
            [
                -decision_scores,
                decision_scores,
            ]
        )

    shifted_scores = (
        decision_scores
        - np.max(
            decision_scores,
            axis=1,
            keepdims=True,
        )
    )

    exponentiated_scores = np.exp(
        shifted_scores
    )

    probabilities = (
        exponentiated_scores
        / np.sum(
            exponentiated_scores,
            axis=1,
            keepdims=True,
        )
    )

    return (
        probabilities,
        classifier_classes,
    )


# ---------------------------------------------------------------------
# One configuration and one subject
# ---------------------------------------------------------------------

def evaluate_configuration(
    configuration: dict[str, Any],
    test_subject: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    reduced_training_features: np.ndarray,
    reduced_testing_features: np.ndarray,
    pca_components: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Evaluate one classifier configuration on one LOSO fold."""
    classifier = build_classifier(
        configuration
    )

    fit_start = time.perf_counter()

    classifier.fit(
        reduced_training_features,
        y_train,
    )

    classifier_fit_seconds = (
        time.perf_counter()
        - fit_start
    )

    prediction_start = time.perf_counter()

    predicted_labels = classifier.predict(
        reduced_testing_features
    )

    probabilities, classifier_classes = (
        get_probabilities(
            classifier,
            reduced_testing_features,
        )
    )

    prediction_seconds = (
        time.perf_counter()
        - prediction_start
    )

    accuracy = accuracy_score(
        y_test,
        predicted_labels,
    )

    kappa = cohen_kappa_score(
        y_test,
        predicted_labels,
    )

    recalls = recall_score(
        y_test,
        predicted_labels,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predicted_labels,
        labels=CLASS_ORDER,
    )

    subject_result: dict[str, Any] = {
        **configuration,
        "test_subject": test_subject,
        "training_trials": int(
            len(y_train)
        ),
        "testing_trials": int(
            len(y_test)
        ),
        "original_tangent_features": int(
            253
        ),
        "pca_components_retained": int(
            pca_components
        ),
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
                f"cm_{true_class}_pred_"
                f"{predicted_class}"
            ] = int(
                matrix[
                    true_index,
                    predicted_index,
                ]
            )

    probability_index = {
        class_name: index
        for index, class_name in enumerate(
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
            **configuration,
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

def aggregate_configuration_results(
    subject_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Calculate the mean LOSO result for every configuration."""
    dataframe = pd.DataFrame(
        subject_results
    )

    configuration_results: list[
        dict[str, Any]
    ] = []

    grouped = dataframe.groupby(
        [
            "configuration_id",
            "pca_variance",
            "classifier_name",
            "classifier_parameter_name",
            "classifier_parameter_value",
        ],
        dropna=False,
        sort=True,
    )

    for group_keys, group in grouped:
        (
            configuration_id,
            pca_variance,
            classifier_name,
            parameter_name,
            parameter_value,
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
                "pca_variance": float(
                    pca_variance
                ),
                "classifier_name": (
                    classifier_name
                ),
                "classifier_parameter_name": (
                    parameter_name
                ),
                "classifier_parameter_value": (
                    parameter_value
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
                "mean_kappa": mean_kappa,
                "kappa_standard_deviation": float(
                    group[
                        "kappa"
                    ].std(
                        ddof=1
                    )
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
                "mean_pca_components": float(
                    group[
                        "pca_components_retained"
                    ].mean()
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
# Output formatting
# ---------------------------------------------------------------------

def print_configuration_name(
    configuration: dict[str, Any],
) -> str:
    """Build a readable configuration label."""
    classifier_name = configuration[
        "classifier_name"
    ]

    parameter_name = configuration[
        "classifier_parameter_name"
    ]

    parameter_value = configuration[
        "classifier_parameter_value"
    ]

    if parameter_name == "none":
        classifier_text = classifier_name
    else:
        classifier_text = (
            f"{classifier_name}"
            f"({parameter_name}="
            f"{parameter_value})"
        )

    return (
        f"PCA "
        f"{configuration['pca_variance'] * 100:.0f}%"
        f" + {classifier_text}"
    )


def print_final_results(
    configuration_results: list[
        dict[str, Any]
    ],
    subject_results: list[
        dict[str, Any]
    ],
    total_seconds: float,
) -> None:
    """Print the ranking and best subject-wise results."""
    print()
    print("=" * 100)
    print("Broadband Riemannian LOSO Sweep Results")
    print("=" * 100)

    print(
        f"\n{'Rank':<6}"
        f"{'PCA':<8}"
        f"{'Classifier':<35}"
        f"{'Accuracy':>12}"
        f"{'Kappa':>10}"
        f"{'Change':>12}"
    )

    print("-" * 85)

    for row in configuration_results[
        :20
    ]:
        parameter_name = row[
            "classifier_parameter_name"
        ]

        parameter_value = row[
            "classifier_parameter_value"
        ]

        if parameter_name == "none":
            classifier_label = row[
                "classifier_name"
            ]
        else:
            classifier_label = (
                f"{row['classifier_name']} "
                f"{parameter_name}="
                f"{parameter_value}"
            )

        print(
            f"{row['rank']:<6}"
            f"{row['pca_variance'] * 100:<7.0f}%"
            f"{classifier_label:<35}"
            f"{row['mean_accuracy_percent']:>11.2f}%"
            f"{row['mean_kappa']:>10.3f}"
            f"{row['improvement_over_baseline_percent_points']:>+11.2f}"
        )

    best_configuration = (
        configuration_results[0]
    )

    best_configuration_id = (
        best_configuration[
            "configuration_id"
        ]
    )

    best_subject_results = [
        row
        for row in subject_results
        if row["configuration_id"]
        == best_configuration_id
    ]

    print()
    print("=" * 100)
    print("Best configuration")
    print("=" * 100)

    print(
        print_configuration_name(
            best_configuration
        )
    )

    print(
        f"Mean accuracy: "
        f"{best_configuration['mean_accuracy_percent']:.2f}%"
    )

    print(
        f"Mean kappa:    "
        f"{best_configuration['mean_kappa']:.3f}"
    )

    print(
        f"Change from previous broadband baseline: "
        f"{best_configuration['improvement_over_baseline_percent_points']:+.2f} "
        "percentage points"
    )

    print()
    print(
        f"{'Subject':<12}"
        f"{'Accuracy':>12}"
        f"{'Kappa':>12}"
        f"{'PCA components':>18}"
    )

    print("-" * 58)

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
            f"{row['pca_components_retained']:>18}"
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
    """Run the complete broadband Riemannian LOSO sweep."""
    experiment_start = time.perf_counter()

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    configurations = build_configurations()

    print("=" * 100)
    print("Strict LOSO: Broadband Riemannian PCA and Classifier Sweep")
    print("=" * 100)

    print(
        f"\nNumber of configurations: "
        f"{len(configurations)}"
    )

    print(
        f"Number of LOSO folds: "
        f"{len(SUBJECTS)}"
    )

    print(
        f"Total classifier evaluations: "
        f"{len(configurations) * len(SUBJECTS)}"
    )

    print(
        "\nPCA retained-variance values:"
    )

    for pca_variance in PCA_VARIANCES:
        print(
            f"  {pca_variance * 100:.0f}%"
        )

    print(
        "\nPrevious broadband Riemannian baseline:"
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

    subject_results: list[
        dict[str, Any]
    ] = []

    prediction_results: list[
        dict[str, Any]
    ] = []

    for fold_number, test_subject in enumerate(
        SUBJECTS,
        start=1,
    ):
        fold_start = time.perf_counter()

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
        print("#" * 100)

        print(
            f"LOSO fold {fold_number}/"
            f"{len(SUBJECTS)}: "
            f"unseen subject {test_subject}"
        )

        print("#" * 100)

        print(
            f"Training subjects: "
            f"{', '.join(training_subjects)}"
        )

        print(
            f"Training trials: {len(y_train)}"
        )

        print(
            f"Testing trials:  {len(y_test)}"
        )

        feature_start = time.perf_counter()

        (
            training_tangent_features,
            testing_tangent_features,
        ) = extract_fold_features(
            X_train=X_train,
            X_test=X_test,
        )

        feature_seconds = (
            time.perf_counter()
            - feature_start
        )

        print(
            f"Tangent features: "
            f"{training_tangent_features.shape[1]}"
        )

        print(
            f"Feature extraction time: "
            f"{feature_seconds:.2f} seconds"
        )

        configurations_by_pca: dict[
            float,
            list[dict[str, Any]],
        ] = {}

        for configuration in configurations:
            pca_variance = float(
                configuration[
                    "pca_variance"
                ]
            )

            configurations_by_pca.setdefault(
                pca_variance,
                [],
            ).append(
                configuration
            )

        for pca_variance in PCA_VARIANCES:
            print()
            print(
                f"  Preparing PCA "
                f"{pca_variance * 100:.0f}%..."
            )

            (
                reduced_training_features,
                reduced_testing_features,
                _,
                pca,
            ) = prepare_pca_features(
                training_tangent_features=(
                    training_tangent_features
                ),
                testing_tangent_features=(
                    testing_tangent_features
                ),
                pca_variance=pca_variance,
            )

            pca_components = int(
                pca.n_components_
            )

            print(
                f"  PCA components retained: "
                f"{pca_components}"
            )

            for configuration in (
                configurations_by_pca[
                    pca_variance
                ]
            ):
                label = print_configuration_name(
                    configuration
                )

                (
                    subject_result,
                    prediction_rows,
                ) = evaluate_configuration(
                    configuration=(
                        configuration
                    ),
                    test_subject=test_subject,
                    y_train=y_train,
                    y_test=y_test,
                    reduced_training_features=(
                        reduced_training_features
                    ),
                    reduced_testing_features=(
                        reduced_testing_features
                    ),
                    pca_components=(
                        pca_components
                    ),
                )

                subject_result[
                    "feature_extraction_seconds"
                ] = float(
                    feature_seconds
                )

                subject_result[
                    "training_subjects"
                ] = "|".join(
                    training_subjects
                )

                subject_results.append(
                    subject_result
                )

                prediction_results.extend(
                    prediction_rows
                )

                print(
                    f"    {label:<55}"
                    f"{subject_result['accuracy_percent']:>7.2f}% "
                    f"kappa="
                    f"{subject_result['kappa']:.3f}"
                )

        write_csv(
            SUBJECT_RESULTS_PATH,
            subject_results,
        )

        write_csv(
            PREDICTIONS_PATH,
            prediction_results,
        )

        fold_seconds = (
            time.perf_counter()
            - fold_start
        )

        print(
            f"\nCompleted {test_subject} in "
            f"{fold_seconds:.2f} seconds"
        )

    configuration_results = (
        aggregate_configuration_results(
            subject_results
        )
    )

    total_seconds = (
        time.perf_counter()
        - experiment_start
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

    best_configuration = dict(
        configuration_results[0]
    )

    write_csv(
        BEST_CONFIGURATION_PATH,
        [best_configuration],
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
