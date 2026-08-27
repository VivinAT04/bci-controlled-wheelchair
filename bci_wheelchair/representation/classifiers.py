"""Factory functions for classifiers used with encoded EEG features."""

from __future__ import annotations

from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SUPPORTED_CLASSIFIERS = (
    "dummy",
    "lda",
    "linear_svm",
    "rbf_svm",
    "logistic_regression",
    "random_forest",
)


def create_classifier(
    name: str,
    random_state: int = 42,
    **parameters: Any,
) -> ClassifierMixin:
    """
    Create a classifier for autoencoder latent features.

    Supported classifier names:
        dummy
        lda
        linear_svm
        rbf_svm
        logistic_regression
        random_forest
    """
    classifier_name = name.strip().lower()

    if classifier_name == "dummy":
        return DummyClassifier(
            strategy=parameters.pop(
                "strategy",
                "stratified",
            ),
            random_state=random_state,
            **parameters,
        )

    if classifier_name == "lda":
        return LinearDiscriminantAnalysis(**parameters)

    if classifier_name == "linear_svm":
        classifier = SVC(
            kernel="linear",
            random_state=random_state,
            **parameters,
        )

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    if classifier_name == "rbf_svm":
        classifier = SVC(
            kernel="rbf",
            random_state=random_state,
            **parameters,
        )

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    if classifier_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            **parameters,
        )

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    if classifier_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
            **parameters,
        )

    supported = ", ".join(SUPPORTED_CLASSIFIERS)

    raise ValueError(
        f"Unknown classifier '{name}'. "
        f"Supported classifiers: {supported}."
    )
