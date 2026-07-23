"""
Cross-subject LOSO evaluation using:

    Subject-wise Euclidean Alignment
        -> Regularized FBCSP
        -> PCA
        -> StandardScaler
        -> RBF-SVM

Every subject is tested as a completely unseen subject:

    Train on 8 subjects
    Test on 1 unseen subject

Configurations:

    C = 0.1, 1.0, 10.0
    gamma = scale, auto

The current LDA benchmark is approximately:

    Mean LOSO accuracy: 50.96%
    Mean LOSO kappa:    0.346

Run:

    python -m scripts.cross_subject.run_cross_subject_loso_svm_sweep
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from bci_wheelchair.cross_subject import (
    BANDS,
    CLASS_ORDER,
    RegularizedFilterBankCSP,
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

# Best FBCSP/PCA configuration from the previous LOSO sweep.
CSP_COMPONENTS = 8
PCA_VARIANCE = 0.90

SVM_C_OPTIONS = [
    0.1,
    1.0,
    10.0,
]

SVM_GAMMA_OPTIONS = [
    "scale",
    "auto",
]

RESULTS_DIRECTORY = Path(
    "results/cross_subject/sweeps/loso_svm_sweep"
)

CONFIGURATION_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "loso_svm_configuration_results.csv"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "loso_svm_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "loso_svm_predictions.csv"
)

BEST_CONFIGURATION_PATH = (
    RESULTS_DIRECTORY
    / "loso_svm_best_configuration.csv"
)

LDA_BASELINE_ACCURACY = 0.5096450617283951
LDA_BASELINE_KAPPA = 0.3461934156378601


# ---------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------

def export_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """
    Export a list of dictionaries to CSV.
    """
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

        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_all_subjects() -> dict[
    str,
    dict[str, object],
]:
    """
    Load and independently align every subject once.

    Each subject receives its own Euclidean Alignment reference.
    The subjects are pooled only after alignment.
    """
    subject_data: dict[
        str,
        dict[str, object],
    ] = {}

    print("=" * 76)
    print("Loading and Euclidean-aligning all subjects")
    print("=" * 76)

    for subject in SUBJECTS:
        (
            X,
            y,
            alignment_error,
        ) = load_and_align_subject(
            subject
        )

        subject_data[subject] = {
            "X": np.asarray(X),
            "y": np.asarray(y),
            "alignment_error": (
                float(alignment_error)
            ),
        }

    return subject_data


def create_loso_fold(
    subject_data: dict[
        str,
        dict[str, object],
    ],
    test_subject: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """
    Create one Leave-One-Subject-Out fold.
    """
    training_subjects = [
        subject
        for subject in SUBJECTS
        if subject != test_subject
    ]

    X_train = np.concatenate(
        [
            np.asarray(
                subject_data[subject]["X"]
            )
            for subject in training_subjects
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            np.asarray(
                subject_data[subject]["y"]
            )
            for subject in training_subjects
        ],
        axis=0,
    )

    X_test = np.asarray(
        subject_data[test_subject]["X"]
    )

    y_test = np.asarray(
        subject_data[test_subject]["y"]
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        training_subjects,
    )


# ---------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------

def build_svm_classifier(
    svm_c: float,
    svm_gamma: str,
) -> Pipeline:
    """
    Build the Euclidean Alignment + FBCSP + RBF-SVM pipeline.

    Euclidean Alignment is applied before this pipeline,
    independently for each subject.
    """
    return Pipeline(
        [
            (
                "fbcsp",
                RegularizedFilterBankCSP(
                    bands=BANDS,
                    n_components=(
                        CSP_COMPONENTS
                    ),
                ),
            ),
            (
                "pca",
                PCA(
                    n_components=(
                        PCA_VARIANCE
                    ),
                    svd_solver="full",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=svm_c,
                    gamma=svm_gamma,
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_configuration(
    subject_data: dict[
        str,
        dict[str, object],
    ],
    svm_c: float,
    svm_gamma: str,
    configuration_number: int,
    total_configurations: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """
    Evaluate one SVM configuration across all nine LOSO folds.
    """
    configuration_name = (
        f"rbf_svm_c{svm_c:g}_"
        f"gamma_{svm_gamma}"
    )

    print()
    print("#" * 76)
    print(
        f"Configuration "
        f"{configuration_number}/"
        f"{total_configurations}: "
        f"{configuration_name}"
    )
    print("#" * 76)

    configuration_start = (
        time.perf_counter()
    )

    subject_results: list[
        dict[str, object]
    ] = []

    prediction_results: list[
        dict[str, object]
    ] = []

    all_true: list[np.ndarray] = []
    all_predicted: list[np.ndarray] = []

    for fold_number, test_subject in enumerate(
        SUBJECTS,
        start=1,
    ):
        print()
        print("=" * 76)
        print(
            f"{configuration_name} | "
            f"Fold {fold_number}/9 | "
            f"Unseen subject: {test_subject}"
        )
        print("=" * 76)

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
            f"Training trials: "
            f"{len(y_train)}"
        )

        print(
            f"Testing trials:  "
            f"{len(y_test)}"
        )

        classifier = build_svm_classifier(
            svm_c=svm_c,
            svm_gamma=svm_gamma,
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

        y_predicted = classifier.predict(
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

        retained_pca_components = int(
            classifier.named_steps[
                "pca"
            ].n_components_
        )

        support_vectors = int(
            classifier.named_steps[
                "svm"
            ].support_.shape[0]
        )

        subject_result: dict[
            str,
            object,
        ] = {
            "configuration": (
                configuration_name
            ),
            "model": (
                "EA_regularized_FBCSP_"
                "PCA_RBF_SVM"
            ),
            "csp_components": (
                CSP_COMPONENTS
            ),
            "pca_variance": (
                PCA_VARIANCE
            ),
            "svm_c": svm_c,
            "svm_gamma": svm_gamma,
            "test_subject": (
                test_subject
            ),
            "training_subjects": "|".join(
                training_subjects
            ),
            "training_trials": (
                len(y_train)
            ),
            "testing_trials": (
                len(y_test)
            ),
            "accuracy": accuracy,
            "accuracy_percent": (
                accuracy * 100.0
            ),
            "kappa": kappa,
            "left_hand_recall": (
                recalls[0]
            ),
            "right_hand_recall": (
                recalls[1]
            ),
            "feet_recall": recalls[2],
            "tongue_recall": recalls[3],
            "pca_components_retained": (
                retained_pca_components
            ),
            "support_vectors": (
                support_vectors
            ),
            "training_seconds": (
                training_seconds
            ),
            "prediction_seconds": (
                prediction_seconds
            ),
        }

        for true_index, true_class in enumerate(
            CLASS_ORDER
        ):
            for (
                predicted_index,
                predicted_class,
            ) in enumerate(
                CLASS_ORDER
            ):
                column_name = (
                    f"cm_{true_class}_pred_"
                    f"{predicted_class}"
                )

                subject_result[
                    column_name
                ] = int(
                    matrix[
                        true_index,
                        predicted_index,
                    ]
                )

        subject_results.append(
            subject_result
        )

        class_indices = {
            class_name: index
            for index, class_name
            in enumerate(
                classifier.named_steps[
                    "svm"
                ].classes_
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
            predicted_probability = float(
                probabilities[
                    trial_index - 1,
                    class_indices[
                        predicted_class
                    ],
                ]
            )

            probability_row = {
                class_name: float(
                    probabilities[
                        trial_index - 1,
                        class_indices[
                            class_name
                        ],
                    ]
                )
                for class_name in CLASS_ORDER
            }

            prediction_results.append(
                {
                    "configuration": (
                        configuration_name
                    ),
                    "test_subject": (
                        test_subject
                    ),
                    "trial": trial_index,
                    "true_class": (
                        true_class
                    ),
                    "predicted_class": (
                        predicted_class
                    ),
                    "correct": (
                        true_class
                        == predicted_class
                    ),
                    "confidence": (
                        predicted_probability
                    ),
                    "probability_left_hand": (
                        probability_row[
                            "left_hand"
                        ]
                    ),
                    "probability_right_hand": (
                        probability_row[
                            "right_hand"
                        ]
                    ),
                    "probability_feet": (
                        probability_row[
                            "feet"
                        ]
                    ),
                    "probability_tongue": (
                        probability_row[
                            "tongue"
                        ]
                    ),
                }
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
            f"Kappa:    "
            f"{kappa:.3f}"
        )

        print(
            f"PCA components retained: "
            f"{retained_pca_components}"
        )

        print(
            f"Support vectors: "
            f"{support_vectors}"
        )

        print(
            f"Training time: "
            f"{training_seconds:.2f} seconds"
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

    pooled_accuracy = accuracy_score(
        combined_true,
        combined_predicted,
    )

    pooled_kappa = cohen_kappa_score(
        combined_true,
        combined_predicted,
    )

    overall_recalls = recall_score(
        combined_true,
        combined_predicted,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    configuration_seconds = (
        time.perf_counter()
        - configuration_start
    )

    configuration_result: dict[
        str,
        object,
    ] = {
        "configuration": (
            configuration_name
        ),
        "model": (
            "EA_regularized_FBCSP_"
            "PCA_RBF_SVM"
        ),
        "csp_components": (
            CSP_COMPONENTS
        ),
        "pca_variance": (
            PCA_VARIANCE
        ),
        "svm_c": svm_c,
        "svm_gamma": svm_gamma,
        "mean_accuracy": (
            float(accuracies.mean())
        ),
        "mean_accuracy_percent": (
            float(
                accuracies.mean()
                * 100.0
            )
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
            accuracies.min()
            * 100.0
        ),
        "maximum_accuracy": float(
            accuracies.max()
        ),
        "maximum_accuracy_percent": float(
            accuracies.max()
            * 100.0
        ),
        "mean_kappa": float(
            kappas.mean()
        ),
        "std_kappa": float(
            kappas.std(ddof=1)
        ),
        "pooled_accuracy": (
            pooled_accuracy
        ),
        "pooled_accuracy_percent": (
            pooled_accuracy * 100.0
        ),
        "pooled_kappa": (
            pooled_kappa
        ),
        "overall_left_hand_recall": (
            overall_recalls[0]
        ),
        "overall_right_hand_recall": (
            overall_recalls[1]
        ),
        "overall_feet_recall": (
            overall_recalls[2]
        ),
        "overall_tongue_recall": (
            overall_recalls[3]
        ),
        "lda_baseline_accuracy": (
            LDA_BASELINE_ACCURACY
        ),
        "lda_baseline_accuracy_percent": (
            LDA_BASELINE_ACCURACY
            * 100.0
        ),
        "accuracy_change_from_lda": (
            accuracies.mean()
            - LDA_BASELINE_ACCURACY
        ),
        "accuracy_change_percent_points": (
            (
                accuracies.mean()
                - LDA_BASELINE_ACCURACY
            )
            * 100.0
        ),
        "lda_baseline_kappa": (
            LDA_BASELINE_KAPPA
        ),
        "kappa_change_from_lda": (
            kappas.mean()
            - LDA_BASELINE_KAPPA
        ),
        "configuration_seconds": (
            configuration_seconds
        ),
    }

    print()
    print("-" * 76)
    print(
        f"Completed "
        f"{configuration_name}"
    )
    print("-" * 76)

    print(
        "Mean LOSO accuracy: "
        f"{configuration_result['mean_accuracy_percent']:.2f}%"
    )

    print(
        "Mean LOSO kappa:    "
        f"{configuration_result['mean_kappa']:.3f}"
    )

    print(
        "Change from LDA:    "
        f"{configuration_result['accuracy_change_percent_points']:+.2f} "
        "percentage points"
    )

    return (
        configuration_result,
        subject_results,
        prediction_results,
    )


# ---------------------------------------------------------------------
# Final reporting
# ---------------------------------------------------------------------

def print_final_results(
    configuration_results: list[
        dict[str, object]
    ],
) -> None:
    """
    Rank and print all SVM configurations.
    """
    ranked_results = sorted(
        configuration_results,
        key=lambda result: (
            result["mean_accuracy"],
            result["mean_kappa"],
        ),
        reverse=True,
    )

    print()
    print("=" * 76)
    print("Final RBF-SVM LOSO Sweep")
    print("=" * 76)

    print(
        f"\n{'Rank':<6}"
        f"{'Configuration':<28}"
        f"{'Accuracy':>11}"
        f"{'SD':>9}"
        f"{'Kappa':>9}"
        f"{'vs LDA':>10}"
    )

    print("-" * 73)

    for rank, result in enumerate(
        ranked_results,
        start=1,
    ):
        print(
            f"{rank:<6}"
            f"{result['configuration']:<28}"
            f"{result['mean_accuracy_percent']:>10.2f}%"
            f"{result['std_accuracy_percent']:>8.2f}"
            f"{result['mean_kappa']:>9.3f}"
            f"{result['accuracy_change_percent_points']:>+9.2f}"
        )

    best = ranked_results[0]

    print()
    print("=" * 76)
    print("Best RBF-SVM Configuration")
    print("=" * 76)

    print(
        f"Configuration: "
        f"{best['configuration']}"
    )

    print(
        f"SVM C:        "
        f"{best['svm_c']}"
    )

    print(
        f"SVM gamma:    "
        f"{best['svm_gamma']}"
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
        f"Change from LDA accuracy: "
        f"{best['accuracy_change_percent_points']:+.2f} "
        "percentage points"
    )

    if (
        best["mean_accuracy"]
        > LDA_BASELINE_ACCURACY
    ):
        print(
            "\nResult: RBF-SVM improved upon "
            "the LDA LOSO baseline."
        )
    else:
        print(
            "\nResult: RBF-SVM did not improve "
            "upon the LDA LOSO baseline."
        )

    print("\nSaved files:")

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
    """
    Run the complete cross-subject RBF-SVM LOSO sweep.
    """
    total_start = time.perf_counter()

    print("=" * 76)
    print(
        "Cross-Subject LOSO "
        "EA + FBCSP + RBF-SVM Sweep"
    )
    print("=" * 76)

    print(
        f"\nCSP components: "
        f"{CSP_COMPONENTS}"
    )

    print(
        f"PCA variance: "
        f"{PCA_VARIANCE:.0%}"
    )

    print(
        "SVM C options: "
        + ", ".join(
            str(value)
            for value in SVM_C_OPTIONS
        )
    )

    print(
        "SVM gamma options: "
        + ", ".join(
            SVM_GAMMA_OPTIONS
        )
    )

    total_configurations = (
        len(SVM_C_OPTIONS)
        * len(SVM_GAMMA_OPTIONS)
    )

    print(
        f"Configurations: "
        f"{total_configurations}"
    )

    print(
        f"Total model fits: "
        f"{total_configurations * len(SUBJECTS)}"
    )

    print(
        f"LDA baseline accuracy: "
        f"{LDA_BASELINE_ACCURACY * 100.0:.2f}%"
    )

    subject_data = load_all_subjects()

    configuration_results: list[
        dict[str, object]
    ] = []

    all_subject_results: list[
        dict[str, object]
    ] = []

    all_prediction_results: list[
        dict[str, object]
    ] = []

    configuration_number = 0

    for svm_c in SVM_C_OPTIONS:
        for svm_gamma in SVM_GAMMA_OPTIONS:
            configuration_number += 1

            (
                configuration_result,
                subject_results,
                prediction_results,
            ) = evaluate_configuration(
                subject_data=subject_data,
                svm_c=svm_c,
                svm_gamma=svm_gamma,
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

            # Save progress after every configuration.
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
        "total_sweep_seconds"
    ] = (
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
        "\nTotal sweep time: "
        f"{best_result['total_sweep_seconds']:.2f} "
        "seconds"
    )


if __name__ == "__main__":
    main()
