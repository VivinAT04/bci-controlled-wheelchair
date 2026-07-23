"""
Strict LOSO evaluation using Filter-Bank Riemannian Tangent-Space features.

Pipeline
--------
For each LOSO fold:

    Training:
        Eight subjects

    Testing:
        One completely unseen subject

Feature extraction:
    Subject-wise Euclidean Alignment
    -> nine frequency bands
    -> OAS covariance matrices
    -> Riemannian tangent-space features per band
    -> concatenate all frequency-band features
    -> StandardScaler
    -> PCA retaining 90% variance
    -> Shrinkage LDA

Frequency bands:
    4-8 Hz
    8-12 Hz
    12-16 Hz
    16-20 Hz
    20-24 Hz
    24-28 Hz
    28-32 Hz
    32-36 Hz
    36-40 Hz

Run:
    python -m scripts.cross_subject.run_filterbank_riemannian
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
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

SAMPLING_FREQUENCY = 250.0

FREQUENCY_BANDS = [
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
    (32.0, 36.0),
    (36.0, 40.0),
]

FILTER_ORDER = 4
PCA_VARIANCE = 0.90

RESULTS_DIRECTORY = Path(
    "results/cross_subject/riemannian/filterbank_riemannian_loso"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "filterbank_riemannian_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "filterbank_riemannian_predictions.csv"
)

SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "filterbank_riemannian_summary.csv"
)

BASELINE_RIEMANNIAN_ACCURACY = 0.5366512345679012
BASELINE_RIEMANNIAN_KAPPA = 0.382201646090535


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def write_csv(
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
        )

        writer.writeheader()
        writer.writerows(rows)


def bandpass_filter(
    epochs: np.ndarray,
    low_frequency: float,
    high_frequency: float,
    sampling_frequency: float,
    filter_order: int = 4,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    epochs:
        EEG data shaped:
        trials x channels x samples

    low_frequency:
        Lower cut-off frequency.

    high_frequency:
        Upper cut-off frequency.

    sampling_frequency:
        EEG sampling frequency.

    filter_order:
        Butterworth filter order.
    """
    nyquist_frequency = (
        sampling_frequency / 2.0
    )

    low_normalised = (
        low_frequency
        / nyquist_frequency
    )

    high_normalised = (
        high_frequency
        / nyquist_frequency
    )

    sos = butter(
        N=filter_order,
        Wn=[
            low_normalised,
            high_normalised,
        ],
        btype="bandpass",
        output="sos",
    )

    filtered_epochs = sosfiltfilt(
        sos,
        epochs,
        axis=-1,
    )

    return np.asarray(
        filtered_epochs,
        dtype=np.float64,
    )


# ---------------------------------------------------------------------
# Filter-bank Riemannian transformer
# ---------------------------------------------------------------------

class FilterBankRiemannianTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Generate frequency-specific Riemannian tangent-space features.

    A separate covariance estimator and tangent-space projection are
    fitted for every frequency band. Features from all bands are then
    concatenated.
    """

    def __init__(
        self,
        frequency_bands: list[
            tuple[float, float]
        ],
        sampling_frequency: float = 250.0,
        filter_order: int = 4,
        covariance_estimator: str = "oas",
        tangent_metric: str = "riemann",
    ) -> None:
        self.frequency_bands = (
            frequency_bands
        )

        self.sampling_frequency = (
            sampling_frequency
        )

        self.filter_order = (
            filter_order
        )

        self.covariance_estimator = (
            covariance_estimator
        )

        self.tangent_metric = (
            tangent_metric
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "FilterBankRiemannianTransformer":
        """
        Fit one tangent-space reference point per frequency band.
        """
        self.covariance_transformers_: list[
            Covariances
        ] = []

        self.tangent_transformers_: list[
            TangentSpace
        ] = []

        self.feature_counts_: list[int] = []

        print()
        print(
            "Fitting filter-bank Riemannian features"
        )

        for band_index, (
            low_frequency,
            high_frequency,
        ) in enumerate(
            self.frequency_bands,
            start=1,
        ):
            print(
                f"  Band {band_index}/"
                f"{len(self.frequency_bands)}: "
                f"{low_frequency:.0f}-"
                f"{high_frequency:.0f} Hz"
            )

            filtered_X = bandpass_filter(
                epochs=X,
                low_frequency=low_frequency,
                high_frequency=high_frequency,
                sampling_frequency=(
                    self.sampling_frequency
                ),
                filter_order=(
                    self.filter_order
                ),
            )

            covariance_transformer = (
                Covariances(
                    estimator=(
                        self.covariance_estimator
                    )
                )
            )

            covariance_matrices = (
                covariance_transformer.fit_transform(
                    filtered_X,
                    y,
                )
            )

            tangent_transformer = (
                TangentSpace(
                    metric=self.tangent_metric
                )
            )

            tangent_features = (
                tangent_transformer.fit_transform(
                    covariance_matrices,
                    y,
                )
            )

            self.covariance_transformers_.append(
                covariance_transformer
            )

            self.tangent_transformers_.append(
                tangent_transformer
            )

            self.feature_counts_.append(
                tangent_features.shape[1]
            )

        self.total_features_ = int(
            sum(self.feature_counts_)
        )

        print(
            f"  Total concatenated features: "
            f"{self.total_features_}"
        )

        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Transform EEG trials into concatenated tangent-space features.
        """
        if not hasattr(
            self,
            "tangent_transformers_",
        ):
            raise RuntimeError(
                "The transformer must be fitted "
                "before transform() is called."
            )

        band_features: list[
            np.ndarray
        ] = []

        for (
            low_frequency,
            high_frequency,
        ), covariance_transformer, tangent_transformer in zip(
            self.frequency_bands,
            self.covariance_transformers_,
            self.tangent_transformers_,
        ):
            filtered_X = bandpass_filter(
                epochs=X,
                low_frequency=low_frequency,
                high_frequency=high_frequency,
                sampling_frequency=(
                    self.sampling_frequency
                ),
                filter_order=(
                    self.filter_order
                ),
            )

            covariance_matrices = (
                covariance_transformer.transform(
                    filtered_X
                )
            )

            tangent_features = (
                tangent_transformer.transform(
                    covariance_matrices
                )
            )

            band_features.append(
                tangent_features
            )

        concatenated_features = (
            np.concatenate(
                band_features,
                axis=1,
            )
        )

        return np.asarray(
            concatenated_features,
            dtype=np.float64,
        )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_model() -> Pipeline:
    """
    Build the Filter-Bank Riemannian classifier.
    """
    return Pipeline(
        steps=[
            (
                "filterbank_riemannian",
                FilterBankRiemannianTransformer(
                    frequency_bands=(
                        FREQUENCY_BANDS
                    ),
                    sampling_frequency=(
                        SAMPLING_FREQUENCY
                    ),
                    filter_order=FILTER_ORDER,
                    covariance_estimator="oas",
                    tangent_metric="riemann",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    n_components=PCA_VARIANCE,
                    svd_solver="full",
                ),
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

def load_subjects() -> dict[
    str,
    dict[str, Any],
]:
    """
    Load and Euclidean-align all subjects independently.
    """
    loaded_subjects: dict[
        str,
        dict[str, Any],
    ] = {}

    print("=" * 80)
    print(
        "Loading and Euclidean-aligning subjects"
    )
    print("=" * 80)

    for subject in SUBJECTS:
        X, y, alignment_error = (
            load_and_align_subject(
                subject
            )
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
# LOSO experiment
# ---------------------------------------------------------------------

def run_loso_fold(
    loaded_subjects: dict[
        str,
        dict[str, Any],
    ],
    test_subject: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Train on eight subjects and test on one unseen subject.
    """
    training_subjects = [
        subject
        for subject in SUBJECTS
        if subject != test_subject
    ]

    X_train = np.concatenate(
        [
            loaded_subjects[subject]["X"]
            for subject in training_subjects
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            loaded_subjects[subject]["y"]
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
    print("-" * 80)
    print(
        f"Test subject: {test_subject}"
    )
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
    print("-" * 80)

    model = build_model()

    training_start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    pca_transformer = (
        model.named_steps["pca"]
    )

    retained_components = int(
        pca_transformer.n_components_
    )

    prediction_start = time.perf_counter()

    y_predicted = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    prediction_seconds = (
        time.perf_counter()
        - prediction_start
    )

    classifier_classes = np.asarray(
        model.named_steps[
            "classifier"
        ].classes_
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

    print()
    print(
        f"Accuracy: "
        f"{accuracy * 100.0:.2f}%"
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

    subject_result: dict[
        str,
        Any,
    ] = {
        "test_subject": test_subject,
        "training_subjects": "|".join(
            training_subjects
        ),
        "training_trials": len(
            y_train
        ),
        "testing_trials": len(
            y_test
        ),
        "number_of_frequency_bands": len(
            FREQUENCY_BANDS
        ),
        "pca_variance": PCA_VARIANCE,
        "pca_components_retained": (
            retained_components
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
            subject_result[
                f"cm_{true_class}_pred_"
                f"{predicted_class}"
            ] = int(
                matrix[
                    true_index,
                    predicted_index,
                ]
            )

    class_to_probability_index = {
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
            y_predicted,
        )
    ):
        probability_row = probabilities[
            trial_index
        ]

        prediction_row: dict[
            str,
            Any,
        ] = {
            "test_subject": test_subject,
            "trial_index": trial_index,
            "true_class": true_class,
            "predicted_class": (
                predicted_class
            ),
            "correct": bool(
                true_class
                == predicted_class
            ),
            "confidence": float(
                np.max(
                    probability_row
                )
            ),
        }

        for class_name in CLASS_ORDER:
            probability_index = (
                class_to_probability_index[
                    class_name
                ]
            )

            prediction_row[
                f"probability_{class_name}"
            ] = float(
                probability_row[
                    probability_index
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
# Summary
# ---------------------------------------------------------------------

def create_summary(
    subject_results: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    """
    Create overall LOSO summary.
    """
    accuracies = np.asarray(
        [
            row["accuracy"]
            for row in subject_results
        ],
        dtype=float,
    )

    kappas = np.asarray(
        [
            row["kappa"]
            for row in subject_results
        ],
        dtype=float,
    )

    mean_accuracy = float(
        accuracies.mean()
    )

    mean_kappa = float(
        kappas.mean()
    )

    return {
        "model": (
            "filterbank_riemannian_"
            "pca90_shrinkage_lda"
        ),
        "number_of_subjects": len(
            subject_results
        ),
        "number_of_frequency_bands": len(
            FREQUENCY_BANDS
        ),
        "frequency_bands": "|".join(
            f"{low:.0f}-{high:.0f}"
            for low, high
            in FREQUENCY_BANDS
        ),
        "pca_variance": PCA_VARIANCE,
        "mean_accuracy": mean_accuracy,
        "mean_accuracy_percent": (
            mean_accuracy * 100.0
        ),
        "accuracy_standard_deviation": float(
            accuracies.std(
                ddof=1
            )
        ),
        "accuracy_standard_deviation_percent": float(
            accuracies.std(
                ddof=1
            )
            * 100.0
        ),
        "mean_kappa": mean_kappa,
        "kappa_standard_deviation": float(
            kappas.std(
                ddof=1
            )
        ),
        "minimum_subject_accuracy_percent": float(
            accuracies.min()
            * 100.0
        ),
        "maximum_subject_accuracy_percent": float(
            accuracies.max()
            * 100.0
        ),
        "baseline_riemannian_accuracy": (
            BASELINE_RIEMANNIAN_ACCURACY
        ),
        "baseline_riemannian_accuracy_percent": (
            BASELINE_RIEMANNIAN_ACCURACY
            * 100.0
        ),
        "improvement_over_baseline": float(
            mean_accuracy
            - BASELINE_RIEMANNIAN_ACCURACY
        ),
        "improvement_over_baseline_percent_points": float(
            (
                mean_accuracy
                - BASELINE_RIEMANNIAN_ACCURACY
            )
            * 100.0
        ),
        "baseline_riemannian_kappa": (
            BASELINE_RIEMANNIAN_KAPPA
        ),
        "kappa_improvement": float(
            mean_kappa
            - BASELINE_RIEMANNIAN_KAPPA
        ),
    }


def print_final_results(
    subject_results: list[
        dict[str, Any]
    ],
    summary: dict[str, Any],
) -> None:
    """
    Print the final subject and overall results.
    """
    print()
    print("=" * 80)
    print(
        "Filter-Bank Riemannian LOSO Results"
    )
    print("=" * 80)

    print(
        f"\n{'Subject':<10}"
        f"{'Accuracy':>12}"
        f"{'Kappa':>12}"
        f"{'PCA components':>18}"
    )

    print("-" * 55)

    for row in subject_results:
        print(
            f"{row['test_subject']:<10}"
            f"{row['accuracy_percent']:>11.2f}%"
            f"{row['kappa']:>12.3f}"
            f"{row['pca_components_retained']:>18}"
        )

    print("-" * 55)

    print(
        f"{'Mean':<10}"
        f"{summary['mean_accuracy_percent']:>11.2f}%"
        f"{summary['mean_kappa']:>12.3f}"
    )

    print()
    print(
        f"Accuracy SD: "
        f"{summary['accuracy_standard_deviation_percent']:.2f}"
    )

    print(
        f"Previous broadband Riemannian: "
        f"{summary['baseline_riemannian_accuracy_percent']:.2f}%"
    )

    print(
        f"Improvement: "
        f"{summary['improvement_over_baseline_percent_points']:+.2f} "
        "percentage points"
    )

    print()
    print("Saved files:")

    print(
        f"  {SUBJECT_RESULTS_PATH}"
    )

    print(
        f"  {PREDICTIONS_PATH}"
    )

    print(
        f"  {SUMMARY_PATH}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run all strict LOSO folds.
    """
    experiment_start = (
        time.perf_counter()
    )

    print("=" * 80)
    print(
        "Strict LOSO: Filter-Bank "
        "Riemannian Tangent Space"
    )
    print("=" * 80)

    print(
        "\nFrequency bands:"
    )

    for low_frequency, high_frequency in (
        FREQUENCY_BANDS
    ):
        print(
            f"  {low_frequency:.0f}-"
            f"{high_frequency:.0f} Hz"
        )

    print(
        f"\nPCA retained variance: "
        f"{PCA_VARIANCE * 100:.0f}%"
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
        print()
        print("#" * 80)
        print(
            f"LOSO fold {fold_number}/"
            f"{len(SUBJECTS)}"
        )
        print("#" * 80)

        (
            subject_result,
            prediction_rows,
        ) = run_loso_fold(
            loaded_subjects=(
                loaded_subjects
            ),
            test_subject=test_subject,
        )

        subject_results.append(
            subject_result
        )

        prediction_results.extend(
            prediction_rows
        )

        # Save progress after each fold.
        write_csv(
            SUBJECT_RESULTS_PATH,
            subject_results,
        )

        write_csv(
            PREDICTIONS_PATH,
            prediction_results,
        )

    summary = create_summary(
        subject_results
    )

    summary[
        "total_experiment_seconds"
    ] = float(
        time.perf_counter()
        - experiment_start
    )

    write_csv(
        SUMMARY_PATH,
        [summary],
    )

    print_final_results(
        subject_results=(
            subject_results
        ),
        summary=summary,
    )

    print(
        f"\nTotal experiment time: "
        f"{summary['total_experiment_seconds']:.2f} "
        "seconds"
    )


if __name__ == "__main__":
    main()
