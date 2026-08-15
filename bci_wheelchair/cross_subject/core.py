"""
Reusable utilities for cross-subject EEG experiments.

This module contains:

- Euclidean Alignment
- cached subject loading
- shared class and frequency-band definitions
- the regularized EA + FBCSP + LDA classifier

Experiment scripts should import these utilities instead of importing
another runnable experiment script.
"""

from __future__ import annotations

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
)
from sklearn.pipeline import Pipeline

from bci_wheelchair.data.preprocessing import (
    SFREQ,
    bandpass,
)
from bci_wheelchair.data.processed_loading import (
    load_processed_subject,
)
from bci_wheelchair.features.fbcsp import (
    RegularizedFilterBankCSP,
)


mne.set_log_level("ERROR")

N_COMPONENTS = 10


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


def make_ea_regularized_classifier() -> Pipeline:
    """
    Build regularized FBCSP followed by PCA (90% retained variance) and shrinkage LDA.

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
                    verbose=True,
                ),
            ),
            (
                "pca",
                PCA(
                    n_components=0.9,
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


def load_and_align_subject(
    subject_name: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
]:
    """
    Load cached 8-30 Hz EEG and Euclidean-align one subject independently.
    """
    print(f"\nLoading processed data for {subject_name}...")

    X, y = load_processed_subject(
        subject_name,
        config="8-30",
    )

    X = np.asarray(X)
    y = np.asarray(y)

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
