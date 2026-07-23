"""
Cross-subject motor-imagery classification using:

    Subject-wise Euclidean Alignment
        -> Regularized FBCSP
        -> Shrinkage LDA

Training subjects:
    A01T-A08T

Unseen test subject:
    A09T

Important:
    A09 labels are not used for alignment or model training.
    Euclidean Alignment uses only the unlabeled EEG trials of each subject.

Run:
    python -m scripts.cross_subject.export_cross_subject_ea_regularized
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)
from sklearn.pipeline import Pipeline

from bci_wheelchair.commands import CLASS_TO_COMMAND
from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.models import DEFAULT_BANDS
from bci_wheelchair.data.preprocessing import (
    SFREQ,
    bandpass,
    preprocess_raw,
)


mne.set_log_level("ERROR")


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

TRAIN_SUBJECTS = [
    "A01T",
    "A02T",
    "A03T",
    "A04T",
    "A05T",
    "A06T",
    "A07T",
    "A08T",
]

TEST_SUBJECT = "A09T"

DATA_DIRECTORY = Path("data/raw")

OUTPUT_PATH = Path(
    "results/cross_subject/csp_fbcsp/cross_subject_a09_ea_corrected_bands_csp6_predictions.csv"
)

SUMMARY_PATH = Path(
    "results/cross_subject/csp_fbcsp/cross_subject_a09_ea_corrected_bands_csp6_summary.csv"
)

FMIN = 8.0
FMAX = 30.0
TMIN = 0.5
TMAX = 2.5

N_COMPONENTS = 6

# Keep the same bands as the regularized 54.5% experiment.
# This allows Euclidean Alignment to be assessed fairly.
BANDS = [
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 30),
]

CLASS_ORDER = [
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
]

EA_EPSILON = 1e-10


# ---------------------------------------------------------------------
# Euclidean Alignment
# ---------------------------------------------------------------------

def compute_trial_covariance(
    trial: np.ndarray,
) -> np.ndarray:
    """
    Compute the spatial covariance matrix of one EEG trial.

    Parameters
    ----------
    trial:
        EEG trial with shape:
            channels x samples

    Returns
    -------
    covariance:
        Covariance matrix with shape:
            channels x channels
    """
    centered_trial = (
        trial
        - trial.mean(
            axis=1,
            keepdims=True,
        )
    )

    number_of_samples = centered_trial.shape[1]

    covariance = (
        centered_trial
        @ centered_trial.T
    ) / max(number_of_samples - 1, 1)

    return covariance


def compute_reference_covariance(
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute a subject-specific Euclidean reference covariance.

    The reference covariance is the mean covariance matrix across all
    unlabeled trials belonging to one subject.
    """
    if X.ndim != 3:
        raise ValueError(
            "Expected X with shape "
            "(trials, channels, samples), "
            f"but received {X.shape}."
        )

    trial_covariances = np.stack(
        [
            compute_trial_covariance(trial)
            for trial in X
        ],
        axis=0,
    )

    reference_covariance = trial_covariances.mean(
        axis=0
    )

    # Force perfect symmetry before eigendecomposition.
    reference_covariance = (
        reference_covariance
        + reference_covariance.T
    ) / 2.0

    return reference_covariance


def inverse_square_root(
    matrix: np.ndarray,
    epsilon: float = EA_EPSILON,
) -> np.ndarray:
    """
    Compute a numerically stable symmetric matrix inverse square root.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(
        matrix
    )

    # Use a scale-relative eigenvalue floor. EEG covariance values
    # are often far below 1e-10, so an absolute threshold would
    # incorrectly clip valid eigenvalues and prevent proper whitening.
    largest_eigenvalue = float(np.max(eigenvalues))

    if largest_eigenvalue <= 0:
        raise ValueError(
            "Reference covariance is not positive definite."
        )

    relative_floor = max(
        largest_eigenvalue * 1e-12,
        np.finfo(float).eps * largest_eigenvalue,
    )

    eigenvalues = np.maximum(
        eigenvalues,
        relative_floor,
    )

    inverse_sqrt_eigenvalues = (
        1.0 / np.sqrt(eigenvalues)
    )

    inverse_sqrt_matrix = (
        eigenvectors
        @ np.diag(inverse_sqrt_eigenvalues)
        @ eigenvectors.T
    )

    return inverse_sqrt_matrix


def euclidean_align_subject(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Euclidean-align all trials belonging to one subject.

    Each subject receives its own reference covariance:

        R = mean(X_i X_i^T)

    Each trial is transformed using:

        X_aligned = R^(-1/2) X

    Labels are not required.
    """
    reference_covariance = (
        compute_reference_covariance(X)
    )

    whitening_matrix = inverse_square_root(
        reference_covariance
    )

    centered_X = (
        X
        - X.mean(
            axis=2,
            keepdims=True,
        )
    )

    aligned_X = np.einsum(
        "ij,tjk->tik",
        whitening_matrix,
        centered_X,
    )

    return aligned_X, reference_covariance


def alignment_identity_error(
    X_aligned: np.ndarray,
) -> float:
    """
    Measure how close the aligned mean covariance is to identity.

    A lower value indicates that Euclidean Alignment worked as expected.
    """
    aligned_reference = (
        compute_reference_covariance(X_aligned)
    )

    identity = np.eye(
        aligned_reference.shape[0]
    )

    error = np.linalg.norm(
        aligned_reference - identity,
        ord="fro",
    ) / np.linalg.norm(
        identity,
        ord="fro",
    )

    return float(error)


# ---------------------------------------------------------------------
# Regularized FBCSP
# ---------------------------------------------------------------------

class RegularizedFilterBankCSP(
    BaseEstimator,
    TransformerMixin,
):
    """
    Filter-bank CSP using Ledoit-Wolf covariance regularization.
    """

    def __init__(
        self,
        bands=None,
        sfreq: float = SFREQ,
        n_components: int = 4,
    ):
        self.bands = (
            bands
            if bands is not None
            else DEFAULT_BANDS
        )
        self.sfreq = sfreq
        self.n_components = n_components

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        self.csps_ = []

        for low_frequency, high_frequency in self.bands:
            print(
                "  Fitting CSP band: "
                f"{low_frequency}-{high_frequency} Hz"
            )

            X_band = bandpass(
                X,
                low_frequency,
                high_frequency,
                self.sfreq,
            )

            csp = CSP(
                n_components=self.n_components,
                reg="ledoit_wolf",
                log=True,
            )

            csp.fit(
                X_band,
                y,
            )

            self.csps_.append(csp)

        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if not hasattr(self, "csps_"):
            raise RuntimeError(
                "RegularizedFilterBankCSP must be "
                "fitted before transform()."
            )

        feature_blocks = []

        for (
            low_frequency,
            high_frequency,
        ), csp in zip(
            self.bands,
            self.csps_,
        ):
            X_band = bandpass(
                X,
                low_frequency,
                high_frequency,
                self.sfreq,
            )

            band_features = csp.transform(
                X_band
            )

            feature_blocks.append(
                band_features
            )

        return np.concatenate(
            feature_blocks,
            axis=1,
        )


def make_ea_regularized_classifier() -> Pipeline:
    """
    Build regularized FBCSP followed by shrinkage LDA.

    Euclidean Alignment is applied subject-by-subject before this
    classifier receives the pooled training data.
    """
    return Pipeline(
        [
            (
                "fbcsp",
                RegularizedFilterBankCSP(
                    bands=BANDS,
                    n_components=N_COMPONENTS,
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


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_and_align_subject(
    subject_name: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Load, preprocess and Euclidean-align one subject independently.
    """
    subject_path = (
        DATA_DIRECTORY
        / f"{subject_name}.gdf"
    )

    if not subject_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {subject_path}"
        )

    print(f"\nLoading {subject_name}...")

    raw = load_raw_gdf(
        str(subject_path)
    )

    X, y = preprocess_raw(
        raw,
        fmin=FMIN,
        fmax=FMAX,
        tmin=TMIN,
        tmax=TMAX,
    )

    print(
        f"  Original trials: {len(X)}"
    )
    print(
        f"  Original shape:  {X.shape}"
    )

    print(
        "  Applying subject-wise "
        "Euclidean Alignment..."
    )

    X_aligned, _ = euclidean_align_subject(
        X
    )

    identity_error = alignment_identity_error(
        X_aligned
    )

    print(
        f"  Aligned shape:   "
        f"{X_aligned.shape}"
    )

    print(
        "  Alignment identity error: "
        f"{identity_error:.6f}"
    )

    if not np.isfinite(X_aligned).all():
        raise ValueError(
            f"{subject_name} alignment produced "
            "NaN or infinite values."
        )

    return (
        X_aligned,
        y,
        identity_error,
    )


# ---------------------------------------------------------------------
# Result export
# ---------------------------------------------------------------------

def export_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """
    Export trial-level A09 predictions.
    """
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    confidence = probabilities.max(
        axis=1
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
    ) as output_file:
        writer = csv.writer(
            output_file
        )

        writer.writerow(
            [
                "trial",
                "true_class",
                "predicted_class",
                "command",
                "confidence",
                "correct",
                "model",
                "alignment",
                "training_dataset",
                "testing_dataset",
                "data_split",
                "target_labels_used_for_alignment",
                "csp_regularization",
                "lda_shrinkage",
            ]
        )

        for trial_number, (
            true_class,
            predicted_class,
            trial_confidence,
        ) in enumerate(
            zip(
                y_true,
                y_pred,
                confidence,
            ),
            start=1,
        ):
            writer.writerow(
                [
                    trial_number,
                    true_class,
                    predicted_class,
                    CLASS_TO_COMMAND[
                        predicted_class
                    ],
                    f"{trial_confidence:.6f}",
                    (
                        true_class
                        == predicted_class
                    ),
                    (
                        "EA_regularized_FBCSP"
                        "_shrinkage_LDA"
                    ),
                    (
                        "subject_wise_"
                        "euclidean_alignment"
                    ),
                    "A01T.gdf-A08T.gdf",
                    "A09T.gdf",
                    (
                        "unseen_subject_"
                        "cross_subject"
                    ),
                    False,
                    "ledoit_wolf",
                    "auto",
                ]
            )


def export_summary(
    accuracy: float,
    kappa: float,
    matrix: np.ndarray,
    class_recalls: np.ndarray,
    training_time: float,
    prediction_time: float,
    alignment_errors: dict[str, float],
) -> None:
    """
    Export one-row experiment summary.
    """
    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_PATH.open(
        "w",
        newline="",
    ) as output_file:
        writer = csv.writer(
            output_file
        )

        writer.writerow(
            [
                "model",
                "train_subjects",
                "test_subject",
                "accuracy",
                "accuracy_percent",
                "kappa",
                "left_hand_recall",
                "right_hand_recall",
                "feet_recall",
                "tongue_recall",
                "training_time_seconds",
                "prediction_time_seconds",
                "confusion_matrix",
                "alignment_errors",
            ]
        )

        matrix_text = ";".join(
            ",".join(
                str(value)
                for value in row
            )
            for row in matrix
        )

        alignment_text = ";".join(
            (
                f"{subject}:"
                f"{error:.8f}"
            )
            for subject, error
            in alignment_errors.items()
        )

        writer.writerow(
            [
                (
                    "EA_regularized_FBCSP"
                    "_shrinkage_LDA"
                ),
                "A01T-A08T",
                "A09T",
                f"{accuracy:.8f}",
                f"{accuracy * 100:.3f}",
                f"{kappa:.8f}",
                f"{class_recalls[0]:.8f}",
                f"{class_recalls[1]:.8f}",
                f"{class_recalls[2]:.8f}",
                f"{class_recalls[3]:.8f}",
                f"{training_time:.6f}",
                f"{prediction_time:.6f}",
                matrix_text,
                alignment_text,
            ]
        )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 76)
    print(
        "Cross-Subject Euclidean Alignment "
        "+ Regularized FBCSP + Shrinkage LDA"
    )
    print("=" * 76)

    print("\nTraining subjects:")
    print(", ".join(TRAIN_SUBJECTS))

    print(
        f"\nUnseen test subject: "
        f"{TEST_SUBJECT}"
    )

    print("\nProtocol:")
    print(
        "- Each subject is aligned independently."
    )
    print(
        "- A01T-A08T are pooled only after alignment."
    )
    print(
        "- A09T labels are not used for alignment."
    )
    print(
        "- The classifier is trained only on A01T-A08T."
    )

    total_start = time.perf_counter()

    X_train_parts = []
    y_train_parts = []

    alignment_errors: dict[str, float] = {}

    print(
        "\nLoading and aligning "
        "training subjects..."
    )

    for subject_name in TRAIN_SUBJECTS:
        (
            X_subject,
            y_subject,
            identity_error,
        ) = load_and_align_subject(
            subject_name
        )

        X_train_parts.append(
            X_subject
        )

        y_train_parts.append(
            y_subject
        )

        alignment_errors[
            subject_name
        ] = identity_error

    X_train = np.concatenate(
        X_train_parts,
        axis=0,
    )

    y_train = np.concatenate(
        y_train_parts,
        axis=0,
    )

    print(
        "\nLoading and aligning "
        "unseen A09T..."
    )

    (
        X_test,
        y_test,
        test_identity_error,
    ) = load_and_align_subject(
        TEST_SUBJECT
    )

    alignment_errors[
        TEST_SUBJECT
    ] = test_identity_error

    loading_alignment_time = (
        time.perf_counter()
        - total_start
    )

    print("\n" + "=" * 76)
    print("Aligned Dataset Summary")
    print("=" * 76)

    print(
        f"Training trials: "
        f"{len(X_train)}"
    )

    print(
        f"Testing trials:  "
        f"{len(X_test)}"
    )

    print(
        f"Training shape:  "
        f"{X_train.shape}"
    )

    print(
        f"Testing shape:   "
        f"{X_test.shape}"
    )

    print(
        "Loading and alignment time: "
        f"{loading_alignment_time:.2f} "
        "seconds"
    )

    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            "Training and testing EEG shapes "
            "do not match: "
            f"{X_train.shape[1:]} versus "
            f"{X_test.shape[1:]}"
        )

    unique_train_classes, train_counts = (
        np.unique(
            y_train,
            return_counts=True,
        )
    )

    print("\nTraining class counts:")

    for class_name, class_count in zip(
        unique_train_classes,
        train_counts,
    ):
        print(
            f"  {class_name}: "
            f"{class_count}"
        )

    print(
        "\nBuilding EA + regularized FBCSP "
        "+ shrinkage LDA..."
    )

    classifier = (
        make_ea_regularized_classifier()
    )

    print(
        "\nTraining on aligned "
        "A01T-A08T..."
    )

    training_start = time.perf_counter()

    classifier.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    print(
        "\nModel training completed in "
        f"{training_time:.2f} seconds"
    )

    print(
        "\nPredicting aligned unseen A09T..."
    )

    prediction_start = time.perf_counter()

    y_pred = classifier.predict(
        X_test
    )

    probabilities = classifier.predict_proba(
        X_test
    )

    prediction_time = (
        time.perf_counter()
        - prediction_start
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    kappa = cohen_kappa_score(
        y_test,
        y_pred,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
    )

    class_recalls = recall_score(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    print("\n" + "=" * 76)
    print(
        "EA Corrected-Band CSP-6 Cross-Subject Results"
    )
    print("=" * 76)

    print(
        f"Accuracy: {accuracy:.3f} "
        f"({accuracy * 100:.1f}%)"
    )

    print(
        f"Kappa:    {kappa:.3f}"
    )

    print(
        "Prediction time for "
        f"{len(X_test)} trials: "
        f"{prediction_time:.3f} seconds"
    )

    print(
        "Average prediction time "
        "per trial: "
        f"{prediction_time / len(X_test):.6f} "
        "seconds"
    )

    print(
        f"Total experiment time: "
        f"{total_time:.2f} seconds"
    )

    print(
        "\nConfusion-matrix class order:"
    )

    print(CLASS_ORDER)

    print("\nConfusion matrix:")
    print(matrix)

    print("\nPer-class recall:")

    for class_name, recall_value in zip(
        CLASS_ORDER,
        class_recalls,
    ):
        print(
            f"  {class_name:<12}: "
            f"{recall_value:.3f} "
            f"({recall_value * 100:.1f}%)"
        )

    print("\nAlignment identity errors:")

    for subject_name, error in (
        alignment_errors.items()
    ):
        print(
            f"  {subject_name}: "
            f"{error:.8f}"
        )

    export_predictions(
        y_true=y_test,
        y_pred=y_pred,
        probabilities=probabilities,
    )

    export_summary(
        accuracy=accuracy,
        kappa=kappa,
        matrix=matrix,
        class_recalls=class_recalls,
        training_time=training_time,
        prediction_time=prediction_time,
        alignment_errors=alignment_errors,
    )

    print("\n" + "=" * 76)
    print("Comparison")
    print("=" * 76)

    print(
        "Original FBCSP + LDA:"
        "                    "
        "53.1%, kappa 0.375"
    )

    print(
        "Corrected frequency bands:"
        "                 "
        "52.8%, kappa 0.370"
    )

    print(
        "Regularized FBCSP + shrinkage LDA:"
        "        "
        "54.5%, kappa 0.394"
    )

    print(
        "EA + regularized FBCSP + shrinkage LDA: "
        f"{accuracy * 100:.1f}%, "
        f"kappa {kappa:.3f}"
    )

    difference = (
        accuracy * 100.0
        - 54.5
    )

    print(
        "\nChange compared with the current "
        "54.5% best result: "
        f"{difference:+.1f} percentage points"
    )

    if difference > 0:
        print(
            "Result: Euclidean Alignment "
            "improved the current best model."
        )
    elif difference < 0:
        print(
            "Result: Euclidean Alignment "
            "did not improve the current best model."
        )
    else:
        print(
            "Result: Euclidean Alignment "
            "matched the current best model."
        )

    print("\nPrediction export:")
    print(OUTPUT_PATH)

    print("\nSummary export:")
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
