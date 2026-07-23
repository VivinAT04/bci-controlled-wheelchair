"""Filter-bank Common Spatial Pattern feature extractors."""

from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin

from bci_wheelchair.data.preprocessing import SFREQ, bandpass


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
