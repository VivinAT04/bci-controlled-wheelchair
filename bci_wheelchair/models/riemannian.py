"""Reusable Riemannian EEG classifier pipelines."""

from __future__ import annotations

from typing import Union

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


GammaValue = Union[str, float]


def make_riemannian_mdm(
    covariance_estimator: str = "oas",
    metric: str = "riemann",
) -> Pipeline:
    """
    Build a covariance-estimation and Minimum Distance to Mean model.
    """

    return Pipeline([
        (
            "covariance",
            Covariances(
                estimator=covariance_estimator,
            ),
        ),
        (
            "mdm",
            MDM(
                metric=metric,
            ),
        ),
    ])


def make_tangent_lda(
    covariance_estimator: str = "oas",
    tangent_metric: str = "riemann",
    solver: str = "lsqr",
    shrinkage: str | float | None = "auto",
) -> Pipeline:
    """
    Build covariance, tangent-space projection and LDA.
    """

    return Pipeline([
        (
            "covariance",
            Covariances(
                estimator=covariance_estimator,
            ),
        ),
        (
            "tangent_space",
            TangentSpace(
                metric=tangent_metric,
            ),
        ),
        (
            "lda",
            LinearDiscriminantAnalysis(
                solver=solver,
                shrinkage=shrinkage,
            ),
        ),
    ])


def make_tangent_svm(
    covariance_estimator: str = "oas",
    tangent_metric: str = "riemann",
    C: float = 1.0,
    gamma: GammaValue = "scale",
    probability: bool = True,
    random_state: int = 42,
) -> Pipeline:
    """
    Build covariance, tangent-space projection and RBF-SVM.
    """

    return Pipeline([
        (
            "covariance",
            Covariances(
                estimator=covariance_estimator,
            ),
        ),
        (
            "tangent_space",
            TangentSpace(
                metric=tangent_metric,
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
                C=C,
                gamma=gamma,
                probability=probability,
                random_state=random_state,
            ),
        ),
    ])


def make_tangent_dummy(
    covariance_estimator: str = "oas",
    tangent_metric: str = "riemann",
    strategy: str = "stratified",
    random_state: int = 42,
) -> Pipeline:
    """
    Build covariance, tangent-space projection and a dummy baseline.
    """

    return Pipeline([
        (
            "covariance",
            Covariances(
                estimator=covariance_estimator,
            ),
        ),
        (
            "tangent_space",
            TangentSpace(
                metric=tangent_metric,
            ),
        ),
        (
            "dummy",
            DummyClassifier(
                strategy=strategy,
                random_state=random_state,
            ),
        ),
    ])


# Backward-compatible descriptive aliases.
make_riemannian_mdm_pipeline = make_riemannian_mdm
make_tangent_lda_pipeline = make_tangent_lda
make_tangent_svm_pipeline = make_tangent_svm


__all__ = [
    "make_riemannian_mdm",
    "make_riemannian_mdm_pipeline",
    "make_tangent_lda",
    "make_tangent_dummy",
    "make_tangent_lda_pipeline",
    "make_tangent_svm",
    "make_tangent_svm_pipeline",
]
