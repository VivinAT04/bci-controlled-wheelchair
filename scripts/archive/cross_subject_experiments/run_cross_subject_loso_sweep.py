"""
LOSO hyperparameter sweep for cross-subject EEG classification.

Tests:

    CSP components: 6, 8, 10
    PCA variance:   90%, 95%, 99%

For every configuration, all nine LOSO folds are evaluated:

    8 subjects -> training
    1 subject  -> completely unseen testing

The script reuses the existing:

    Subject-wise Euclidean Alignment
    Regularized Filter-Bank CSP
    Shrinkage LDA

Run:

    python -m scripts.cross_subject.run_cross_subject_loso_sweep
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
)
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)
from sklearn.pipeline import Pipeline

from bci_wheelchair.cross_subject import (
    BANDS,
    CLASS_ORDER,
    RegularizedFilterBankCSP,
    load_and_align_subject,
)


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

CSP_COMPONENT_OPTIONS = [
    6,
    8,
    10,
]

PCA_VARIANCE_OPTIONS = [
    0.90,
    0.95,
    0.99,
]

RESULTS_DIRECTORY = Path("results/cross_subject/sweeps/loso_sweep")

CONFIGURATION_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "loso_configuration_results.csv"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "loso_subject_results.csv"
)

BEST_CONFIGURATION_PATH = (
    RESULTS_DIRECTORY
    / "loso_best_configuration.csv"
)


def export_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """
    Export dictionaries to a CSV file.
    """
    if not rows:
        raise ValueError(
            f"No rows available for {path}."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def load_all_subjects() -> dict[str, dict[str, object]]:
    """
    Load and independently Euclidean-align all nine subjects once.
    """
    data: dict[str, dict[str, object]] = {}

    print("=" * 76)
    print("Loading and aligning all subjects")
    print("=" * 76)

    for subject in SUBJECTS:
        X, y, alignment_error = (
            load_and_align_subject(subject)
        )

        data[subject] = {
            "X": X,
            "y": y,
            "alignment_error": alignment_error,
        }

    return data


def build_classifier(
    csp_components: int,
    pca_variance: float,
) -> Pipeline:
    """
    Build one candidate LOSO classifier.
    """
    return Pipeline(
        [
            (
                "fbcsp",
                RegularizedFilterBankCSP(
                    bands=BANDS,
                    n_components=csp_components,
                ),
            ),
            (
                "pca",
                PCA(
                    n_components=pca_variance,
                    svd_solver="full",
                ),
            ),
            (
                "lda",
                LDA(
                    solver="lsqr",
                    shrinkage="auto",
                    priors=[
                        0.25,
                        0.25,
                        0.25,
                        0.25,
                    ],
                ),
            ),
        ]
    )


def get_fold_data(
    subject_data: dict[str, dict[str, object]],
    test_subject: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """
    Construct one LOSO train-test fold.
    """
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


def evaluate_configuration(
    subject_data: dict[str, dict[str, object]],
    csp_components: int,
    pca_variance: float,
    configuration_number: int,
    total_configurations: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    """
    Run all nine LOSO folds for one configuration.
    """
    configuration_name = (
        f"csp{csp_components}_"
        f"pca{int(pca_variance * 100)}"
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

    configuration_start = time.perf_counter()

    fold_results: list[dict[str, object]] = []

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
        ) = get_fold_data(
            subject_data,
            test_subject,
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

        classifier = build_classifier(
            csp_components=csp_components,
            pca_variance=pca_variance,
        )

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

        retained_components = int(
            classifier.named_steps[
                "pca"
            ].n_components_
        )

        fold_result: dict[str, object] = {
            "configuration": configuration_name,
            "csp_components": csp_components,
            "pca_variance": pca_variance,
            "test_subject": test_subject,
            "training_subjects": "|".join(
                training_subjects
            ),
            "training_trials": len(y_train),
            "testing_trials": len(y_test),
            "accuracy": accuracy,
            "accuracy_percent": accuracy * 100.0,
            "kappa": kappa,
            "left_hand_recall": recalls[0],
            "right_hand_recall": recalls[1],
            "feet_recall": recalls[2],
            "tongue_recall": recalls[3],
            "pca_components_retained": (
                retained_components
            ),
            "training_seconds": training_seconds,
            "prediction_seconds": prediction_seconds,
        }

        for true_index, true_class in enumerate(
            CLASS_ORDER
        ):
            for predicted_index, predicted_class in enumerate(
                CLASS_ORDER
            ):
                fold_result[
                    f"cm_{true_class}_pred_"
                    f"{predicted_class}"
                ] = int(
                    matrix[
                        true_index,
                        predicted_index,
                    ]
                )

        fold_results.append(fold_result)

        all_true.append(
            np.asarray(y_test)
        )

        all_predicted.append(
            np.asarray(y_predicted)
        )

        print(
            f"\nAccuracy: "
            f"{accuracy * 100.0:.1f}%"
        )

        print(
            f"Kappa:    {kappa:.3f}"
        )

        print(
            f"PCA components retained: "
            f"{retained_components}"
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
            for result in fold_results
        ],
        dtype=float,
    )

    kappas = np.asarray(
        [
            result["kappa"]
            for result in fold_results
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

    configuration_result: dict[str, object] = {
        "configuration": configuration_name,
        "csp_components": csp_components,
        "pca_variance": pca_variance,
        "mean_accuracy": accuracies.mean(),
        "mean_accuracy_percent": (
            accuracies.mean() * 100.0
        ),
        "std_accuracy": accuracies.std(
            ddof=1
        ),
        "std_accuracy_percent": (
            accuracies.std(ddof=1) * 100.0
        ),
        "minimum_accuracy": accuracies.min(),
        "minimum_accuracy_percent": (
            accuracies.min() * 100.0
        ),
        "maximum_accuracy": accuracies.max(),
        "maximum_accuracy_percent": (
            accuracies.max() * 100.0
        ),
        "mean_kappa": kappas.mean(),
        "std_kappa": kappas.std(
            ddof=1
        ),
        "pooled_accuracy": pooled_accuracy,
        "pooled_accuracy_percent": (
            pooled_accuracy * 100.0
        ),
        "pooled_kappa": pooled_kappa,
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
        "configuration_seconds": (
            configuration_seconds
        ),
    }

    print()
    print("-" * 76)
    print(
        f"Completed {configuration_name}"
    )
    print("-" * 76)

    print(
        "Mean LOSO accuracy: "
        f"{configuration_result['mean_accuracy_percent']:.1f}%"
    )

    print(
        "Mean LOSO kappa:    "
        f"{configuration_result['mean_kappa']:.3f}"
    )

    print(
        "Accuracy SD:        "
        f"{configuration_result['std_accuracy_percent']:.1f}"
    )

    return (
        configuration_result,
        fold_results,
    )


def print_final_results(
    configuration_results: list[dict[str, object]],
) -> None:
    """
    Print the ranked sweep results.
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
    print("Final LOSO Hyperparameter Sweep")
    print("=" * 76)

    print(
        f"\n{'Rank':<6}"
        f"{'Configuration':<18}"
        f"{'Accuracy':>12}"
        f"{'SD':>10}"
        f"{'Kappa':>10}"
    )

    print("-" * 56)

    for rank, result in enumerate(
        ranked_results,
        start=1,
    ):
        print(
            f"{rank:<6}"
            f"{result['configuration']:<18}"
            f"{result['mean_accuracy_percent']:>11.1f}%"
            f"{result['std_accuracy_percent']:>9.1f}"
            f"{result['mean_kappa']:>10.3f}"
        )

    best = ranked_results[0]

    print()
    print("=" * 76)
    print("Best Configuration")
    print("=" * 76)

    print(
        f"Configuration: "
        f"{best['configuration']}"
    )

    print(
        f"CSP components: "
        f"{best['csp_components']}"
    )

    print(
        f"PCA variance: "
        f"{best['pca_variance']}"
    )

    print(
        f"Mean accuracy: "
        f"{best['mean_accuracy_percent']:.1f}%"
    )

    print(
        f"Mean kappa: "
        f"{best['mean_kappa']:.3f}"
    )

    print(
        f"Feet recall: "
        f"{best['overall_feet_recall']:.3f}"
    )

    print("\nSaved files:")

    print(
        f"  {CONFIGURATION_RESULTS_PATH}"
    )

    print(
        f"  {SUBJECT_RESULTS_PATH}"
    )

    print(
        f"  {BEST_CONFIGURATION_PATH}"
    )


def main() -> None:
    """
    Run the complete LOSO hyperparameter sweep.
    """
    total_start = time.perf_counter()

    print("=" * 76)
    print("Cross-Subject LOSO Hyperparameter Sweep")
    print("=" * 76)

    print(
        "\nCSP options: "
        + ", ".join(
            str(value)
            for value in CSP_COMPONENT_OPTIONS
        )
    )

    print(
        "PCA options: "
        + ", ".join(
            f"{value:.0%}"
            for value in PCA_VARIANCE_OPTIONS
        )
    )

    total_configurations = (
        len(CSP_COMPONENT_OPTIONS)
        * len(PCA_VARIANCE_OPTIONS)
    )

    print(
        f"Configurations: "
        f"{total_configurations}"
    )

    print(
        f"Total model fits: "
        f"{total_configurations * len(SUBJECTS)}"
    )

    subject_data = load_all_subjects()

    configuration_results: list[
        dict[str, object]
    ] = []

    all_subject_results: list[
        dict[str, object]
    ] = []

    configuration_number = 0

    for csp_components in CSP_COMPONENT_OPTIONS:
        for pca_variance in PCA_VARIANCE_OPTIONS:
            configuration_number += 1

            (
                configuration_result,
                fold_results,
            ) = evaluate_configuration(
                subject_data=subject_data,
                csp_components=csp_components,
                pca_variance=pca_variance,
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
                fold_results
            )

            export_csv(
                CONFIGURATION_RESULTS_PATH,
                configuration_results,
            )

            export_csv(
                SUBJECT_RESULTS_PATH,
                all_subject_results,
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
