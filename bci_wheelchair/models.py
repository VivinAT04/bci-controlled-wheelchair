"""Classifier pipelines: CSP+LDA baseline and Filter-Bank CSP+LDA."""

import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline

from .preprocessing import SFREQ, bandpass

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


def make_csp_lda(n_components: int = 6) -> Pipeline:
    """Baseline pipeline: single-band CSP followed by LDA."""
    return Pipeline([
        ("csp", CSP(n_components=n_components, reg=None, log=True)),
        ("lda", LDA()),
    ])


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """CSP applied within several frequency sub-bands, features concatenated."""

    def __init__(self, bands=None, sfreq: float = SFREQ, n_components: int = 4):
        self.bands = bands or DEFAULT_BANDS
        self.sfreq = sfreq
        self.n_components = n_components

    def fit(self, X, y):
        self.csps_ = []

        for lo, hi in self.bands:
            csp = CSP(n_components=self.n_components, reg=None, log=True)
            X_band = bandpass(X, lo, hi, self.sfreq)
            csp.fit(X_band, y)
            self.csps_.append(csp)

        return self

    def transform(self, X):
        features = []

        for (lo, hi), csp in zip(self.bands, self.csps_):
            X_band = bandpass(X, lo, hi, self.sfreq)
            features.append(csp.transform(X_band))

        return np.concatenate(features, axis=1)


def make_fbcsp_lda(n_components: int = 4, bands=None) -> Pipeline:
    """Filter-Bank CSP followed by LDA."""
    return Pipeline([
        ("fbcsp", FilterBankCSP(n_components=n_components, bands=bands)),
        ("lda", LDA()),
    ])


def make_eegnet():
    """Stub: add an EEGNet model here in Phase 3."""
    raise NotImplementedError("EEGNet not yet implemented — planned for Phase 3.")