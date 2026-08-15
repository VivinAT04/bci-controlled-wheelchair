"""Reusable utilities for cross-subject EEG experiments."""

from .core import (
    BANDS,
    CLASS_ORDER,
    EA_EPSILON,
    N_COMPONENTS,
    alignment_identity_error,
    compute_reference_covariance,
    compute_trial_covariance,
    euclidean_align_subject,
    inverse_square_root,
    load_and_align_subject,
    make_ea_regularized_classifier,
)

__all__ = [
    "BANDS",
    "CLASS_ORDER",
    "EA_EPSILON",
    "N_COMPONENTS",
    "alignment_identity_error",
    "compute_reference_covariance",
    "compute_trial_covariance",
    "euclidean_align_subject",
    "inverse_square_root",
    "load_and_align_subject",
    "make_ea_regularized_classifier",
]
