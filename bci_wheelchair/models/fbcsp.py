"""Filter-Bank CSP classifier pipeline constructors."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

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

import numpy as np
from mne.decoding import CSP
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)

from bci_wheelchair.data.preprocessing import (
    SFREQ,
    bandpass,
)


DEFAULT_BANDS = [
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 32),
    (32, 36),
]


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """Apply CSP independently to multiple frequency bands."""

    def __init__(
        self,
        bands=None,
        sfreq: float = SFREQ,
        n_components: int = 4,
    ):
        self.bands = bands if bands is not None else DEFAULT_BANDS
        self.sfreq = sfreq
        self.n_components = n_components

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """Fit one CSP transformer for every frequency band."""
        self.csps_ = []

        for low_frequency, high_frequency in self.bands:
            X_band = bandpass(
                X,
                low_frequency,
                high_frequency,
                self.sfreq,
            )

            csp = CSP(
                n_components=self.n_components,
                reg=None,
                log=True,
            )

            csp.fit(X_band, y)
            self.csps_.append(csp)

        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Transform trials and concatenate all band-specific features."""
        if not hasattr(self, "csps_"):
            raise RuntimeError(
                "FilterBankCSP must be fitted before transform()."
            )

        feature_blocks = []

        for (
            low_frequency,
            high_frequency,
        ), csp in zip(self.bands, self.csps_):
            X_band = bandpass(
                X,
                low_frequency,
                high_frequency,
                self.sfreq,
            )

            feature_blocks.append(
                csp.transform(X_band)
            )

        return np.concatenate(
            feature_blocks,
            axis=1,
        )


class RegularizedFilterBankCSP(BaseEstimator, TransformerMixin):
    """Filter-bank CSP using Ledoit-Wolf covariance regularization."""

    def __init__(
        self,
        bands=None,
        sfreq: float = SFREQ,
        n_components: int = 4,
        verbose: bool = False,
    ):
        self.bands = bands if bands is not None else DEFAULT_BANDS
        self.sfreq = sfreq
        self.n_components = n_components
        self.verbose = verbose

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """Fit regularized CSP independently for every frequency band."""
        self.csps_ = []

        for low_frequency, high_frequency in self.bands:
            if self.verbose:
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

            csp.fit(X_band, y)
            self.csps_.append(csp)

        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Transform trials and concatenate regularized CSP features."""
        if not hasattr(self, "csps_"):
            raise RuntimeError(
                "RegularizedFilterBankCSP must be fitted "
                "before transform()."
            )

        feature_blocks = []

        for (
            low_frequency,
            high_frequency,
        ), csp in zip(self.bands, self.csps_):
            X_band = bandpass(
                X,
                low_frequency,
                high_frequency,
                self.sfreq,
            )

            feature_blocks.append(
                csp.transform(X_band)
            )

        return np.concatenate(
            feature_blocks,
            axis=1,
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



def make_fbcsp_dummy(
    n_components: int = 4,
    bands: Optional[Sequence[Band]] = None,
    strategy: str = "stratified",
    random_state: int = 42,
) -> Pipeline:
    """Build Filter-Bank CSP followed by a dummy baseline classifier."""

    return Pipeline([
        (
            "fbcsp",
            FilterBankCSP(
                n_components=n_components,
                bands=bands,
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
    "make_fbcsp_dummy",
    "make_fbcsp_feature_selected_lda",
    "make_fbcsp_feature_selected_svm",
    "make_fbcsp_lda",
    "make_fbcsp_svm",
]
