"""CSP-based classifier pipeline constructors."""

from __future__ import annotations

from typing import Union

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
)
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import (
    SelectPercentile,
    f_classif,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from mne.decoding import CSP


def make_csp_transformer(
    n_components: int = 6,
    regularization=None,
    log: bool = True,
) -> CSP:
    """Create a configured MNE Common Spatial Pattern transformer."""
    return CSP(
        n_components=n_components,
        reg=regularization,
        log=log,
    )

GammaValue = Union[str, float]


def make_csp_lda(
    n_components: int = 6,
) -> Pipeline:
    """Build a single-band CSP followed by LDA."""

    return Pipeline([
        (
            "csp",
            make_csp_transformer(
                n_components=n_components,
            ),
        ),
        ("lda", LDA()),
    ])



def make_csp_dummy(
    n_components: int = 6,
    strategy: str = "stratified",
    random_state: int = 42,
) -> Pipeline:
    """Build CSP followed by a dummy baseline classifier."""

    return Pipeline([
        (
            "csp",
            make_csp_transformer(
                n_components=n_components,
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


def make_tuned_csp_transformer(
    n_components: int = 10,
) -> CSP:
    """Create the tuned CSP feature extractor used in final experiments."""

    return CSP(
        n_components=n_components,
        reg=None,
        log=True,
        rank={"eeg": 22},
    )


def make_tuned_csp_lda(
    n_components: int = 10,
) -> Pipeline:
    """Build tuned CSP followed by LDA."""

    return Pipeline([
        (
            "csp",
            make_tuned_csp_transformer(
                n_components=n_components,
            ),
        ),
        (
            "lda",
            LDA(),
        ),
    ])


def make_tuned_csp_svm(
    n_components: int = 10,
    C: float = 1.0,
    gamma: GammaValue = "scale",
) -> Pipeline:
    """Build tuned CSP followed by RBF-SVM."""

    return Pipeline([
        (
            "csp",
            make_tuned_csp_transformer(
                n_components=n_components,
            ),
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                probability=True,
                random_state=42,
            ),
        ),
    ])


def make_tuned_csp_dummy(
    n_components: int = 10,
    strategy: str = "stratified",
    random_state: int = 42,
) -> Pipeline:
    """Build tuned CSP followed by a dummy baseline."""

    return Pipeline([
        (
            "csp",
            make_tuned_csp_transformer(
                n_components=n_components,
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

def make_csp_feature_selected_lda(
    n_components: int = 6,
    percentile: int = 80,
) -> Pipeline:
    """Build CSP, feature selection and LDA."""

    return Pipeline([
        (
            "csp",
            make_csp_transformer(
                n_components=n_components,
            ),
        ),
        (
            "select",
            SelectPercentile(
                score_func=f_classif,
                percentile=percentile,
            ),
        ),
        ("lda", LDA()),
    ])


def make_csp_svm(
    n_components: int = 6,
    C: float = 1.0,
    gamma: GammaValue = "scale",
) -> Pipeline:
    """Build a single-band CSP followed by an RBF-SVM."""

    return Pipeline([
        (
            "csp",
            make_csp_transformer(
                n_components=n_components,
            ),
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                probability=True,
                random_state=42,
            ),
        ),
    ])


def make_csp_feature_selected_svm(
    n_components: int = 6,
    percentile: int = 80,
    C: float = 1.0,
    gamma: GammaValue = "scale",
) -> Pipeline:
    """Build CSP, feature selection and an RBF-SVM."""

    return Pipeline([
        (
            "csp",
            make_csp_transformer(
                n_components=n_components,
            ),
        ),
        (
            "select",
            SelectPercentile(
                score_func=f_classif,
                percentile=percentile,
            ),
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                probability=True,
                random_state=42,
            ),
        ),
    ])


__all__ = [
    "make_csp_transformer",
    "make_tuned_csp_dummy",
    "make_tuned_csp_svm",
    "make_tuned_csp_lda",
    "make_tuned_csp_transformer",
    "make_csp_dummy",
    "make_csp_feature_selected_lda",
    "make_csp_feature_selected_svm",
    "make_csp_lda",
    "make_csp_svm",
]
