"""Euclidean Alignment for EEG trials."""

from __future__ import annotations

import numpy as np


def _inverse_square_root(
    matrix: np.ndarray,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Calculate the inverse square root of a symmetric matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    eigenvalues = np.maximum(
        eigenvalues,
        epsilon,
    )

    return (
        eigenvectors
        @ np.diag(1.0 / np.sqrt(eigenvalues))
        @ eigenvectors.T
    )


def calculate_reference_covariance(
    X: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Calculate the mean trial covariance for one subject.

    X shape:
        (trials, channels, time)
    """
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    if X.ndim != 3:
        raise ValueError(
            "X must have shape "
            "(trials, channels, time), "
            f"but received {X.shape}."
        )

    if len(X) == 0:
        raise ValueError(
            "X must contain at least one trial."
        )

    covariance_matrices = np.matmul(
        X,
        np.transpose(X, (0, 2, 1)),
    )

    covariance_matrices /= X.shape[2]

    reference_covariance = (
        covariance_matrices.mean(axis=0)
    )

    reference_covariance += (
        epsilon
        * np.eye(
            X.shape[1],
            dtype=np.float64,
        )
    )

    return reference_covariance


def euclidean_align(
    X: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """
    Apply Euclidean Alignment to one subject.

    Each subject must be aligned separately.
    Labels are not required.
    """
    X = np.asarray(
        X,
        dtype=np.float32,
    )

    reference_covariance = (
        calculate_reference_covariance(
            X,
            epsilon=epsilon,
        )
    )

    whitening_matrix = _inverse_square_root(
        reference_covariance,
        epsilon=epsilon,
    )

    aligned_X = np.einsum(
        "ij,tjk->tik",
        whitening_matrix,
        X,
    )

    return aligned_X.astype(np.float32)
