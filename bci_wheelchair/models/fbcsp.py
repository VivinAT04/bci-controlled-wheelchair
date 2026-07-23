"""Filter-Bank CSP classifier pipeline constructors."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
)
from sklearn.feature_selection import (
    SelectPercentile,
    f_classif,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from bci_wheelchair.features.fbcsp import (
    DEFAULT_BANDS,
    FilterBankCSP,
    RegularizedFilterBankCSP,
)


Band = Tuple[float, float]
GammaValue = Union[str, float]


def make_fbcsp_lda(
    n_components: int = 4,
    bands: Optional[Sequence[Band]] = None,
) -> Pipeline:
    """Build Filter-Bank CSP followed by LDA."""

    return Pipeline([
        (
            "fbcsp",
            FilterBankCSP(
                n_components=n_components,
                bands=bands,
            ),
        ),
        ("lda", LDA()),
    ])


def make_fbcsp_feature_selected_lda(
    n_components: int = 4,
    bands: Optional[Sequence[Band]] = None,
    percentile: int = 80,
) -> Pipeline:
    """Build Filter-Bank CSP, feature selection and LDA."""

    return Pipeline([
        (
            "fbcsp",
            FilterBankCSP(
                n_components=n_components,
                bands=bands,
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


def make_fbcsp_svm(
    n_components: int = 4,
    bands: Optional[Sequence[Band]] = None,
    C: float = 1.0,
    gamma: GammaValue = "scale",
) -> Pipeline:
    """Build Filter-Bank CSP followed by an RBF-SVM."""

    return Pipeline([
        (
            "fbcsp",
            FilterBankCSP(
                n_components=n_components,
                bands=bands,
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


def make_fbcsp_feature_selected_svm(
    n_components: int = 4,
    bands: Optional[Sequence[Band]] = None,
    percentile: int = 80,
    C: float = 1.0,
    gamma: GammaValue = "scale",
) -> Pipeline:
    """Build Filter-Bank CSP, feature selection and an RBF-SVM."""

    return Pipeline([
        (
            "fbcsp",
            FilterBankCSP(
                n_components=n_components,
                bands=bands,
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
    "DEFAULT_BANDS",
    "FilterBankCSP",
    "RegularizedFilterBankCSP",
    "make_fbcsp_feature_selected_lda",
    "make_fbcsp_feature_selected_svm",
    "make_fbcsp_lda",
    "make_fbcsp_svm",
]
