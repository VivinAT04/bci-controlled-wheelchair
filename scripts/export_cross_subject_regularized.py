"""
Cross-subject FBCSP evaluation with regularised CSP and shrinkage LDA.

Train:
    A01T-A08T

Test:
    A09T

Run:
    python -m scripts.export_cross_subject_regularized
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

from bci_wheelchair.commands import CLASS_TO_COMMAND
from bci_wheelchair.data_loading import load_raw_gdf
from bci_wheelchair.models import DEFAULT_BANDS
from bci_wheelchair.preprocessing import SFREQ, bandpass, preprocess_raw


mne.set_log_level("ERROR")

TRAIN_SUBJECTS = [
    "A01T",
    "A02T",
    "A03T",
    "A04T",
    "A05T",
    "A06T",
    "A07T",
    "A08T",
]

TEST_SUBJECT = "A09T"

DATA_DIRECTORY = Path("data/raw")

OUTPUT_PATH = Path(
    "results/cross_subject_a09_regularized_predictions.csv"
)

FMIN = 8.0
FMAX = 30.0
TMIN = 0.5
TMAX = 2.5

N_COMPONENTS = 4

CLASS_ORDER = [
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
]


class RegularizedFilterBankCSP(
    BaseEstimator,
    TransformerMixin,
):
    """FBCSP using Ledoit-Wolf covariance regularisation."""

    def __init__(
        self,
        bands=None,
        sfreq: float = SFREQ,
        n_components: int = 4,
    ):
        self.bands = bands or DEFAULT_BANDS
        self.sfreq = sfreq
        self.n_components = n_components

    def fit(self, X, y):
        self.csps_ = []

        for low_frequency, high_frequency in self.bands:
            csp = CSP(
                n_components=self.n_components,
                reg="ledoit_wolf",
                log=True,
            )

            X_band = bandpass(
                X,
                low_frequency,
                high_frequency,
                self.sfreq,
            )

            csp.fit(X_band, y)
            self.csps_.append(csp)

        return self

    def transform(self, X):
        features = []

        for (
            low_frequency,
            high_frequency,
        ), csp in zip(
            self.bands,
            self.csps_,
        ):
            X_band = bandpass(
                X,
                low_frequency,
                high_frequency,
                self.sfreq,
            )

            features.append(
                csp.transform(X_band)
            )

        return np.concatenate(
            features,
            axis=1,
        )


def make_regularized_fbcsp_lda() -> Pipeline:
    """Build regularised FBCSP with shrinkage LDA."""
    return Pipeline(
        [
            (
                "fbcsp",
                RegularizedFilterBankCSP(
                    n_components=N_COMPONENTS,
                ),
            ),
            (
                "lda",
                LDA(
                    solver="lsqr",
                    shrinkage="auto",
                ),
            ),
        ]
    )


def load_subject(
    subject_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess one subject."""
    subject_path = (
        DATA_DIRECTORY / f"{subject_name}.gdf"
    )

    if not subject_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {subject_path}"
        )

    print(f"Loading {subject_name}...")

    raw = load_raw_gdf(
        str(subject_path)
    )

    X, y = preprocess_raw(
        raw,
        fmin=FMIN,
        fmax=FMAX,
        tmin=TMIN,
        tmax=TMAX,
    )

    print(
        f"  Trials: {len(X)}, "
        f"shape: {X.shape}"
    )

    return X, y


def main() -> None:
    print("=" * 72)
    print("Cross-Subject Regularized FBCSP + Shrinkage LDA")
    print("=" * 72)

    print("\nTraining subjects:")
    print(", ".join(TRAIN_SUBJECTS))

    print(f"\nTest subject: {TEST_SUBJECT}")

    X_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []

    preprocessing_start = time.perf_counter()

    print("\nLoading training subjects...")

    for subject in TRAIN_SUBJECTS:
        X_subject, y_subject = load_subject(
            subject
        )

        X_train_parts.append(X_subject)
        y_train_parts.append(y_subject)

    X_train = np.concatenate(
        X_train_parts,
        axis=0,
    )

    y_train = np.concatenate(
        y_train_parts,
        axis=0,
    )

    print("\nLoading unseen test subject...")

    X_test, y_test = load_subject(
        TEST_SUBJECT
    )

    preprocessing_time = (
        time.perf_counter()
        - preprocessing_start
    )

    print("\n" + "=" * 72)
    print("Dataset Summary")
    print("=" * 72)

    print(f"Training trials: {len(X_train)}")
    print(f"Testing trials:  {len(X_test)}")
    print(f"Training shape:  {X_train.shape}")
    print(f"Testing shape:   {X_test.shape}")
    print(
        f"Preprocessing time: "
        f"{preprocessing_time:.2f} seconds"
    )

    print(
        "\nBuilding regularized FBCSP "
        "+ shrinkage LDA..."
    )

    classifier = make_regularized_fbcsp_lda()

    print("Training on A01T-A08T...")

    training_start = time.perf_counter()

    classifier.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    print(
        f"Training completed in "
        f"{training_time:.2f} seconds"
    )

    print("\nPredicting A09T...")

    prediction_start = time.perf_counter()

    y_pred = classifier.predict(X_test)
    probabilities = classifier.predict_proba(
        X_test
    )

    prediction_time = (
        time.perf_counter()
        - prediction_start
    )

    confidence = probabilities.max(axis=1)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    kappa = cohen_kappa_score(
        y_test,
        y_pred,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
    )

    print("\n" + "=" * 72)
    print("Regularized Cross-Subject Results")
    print("=" * 72)

    print(
        f"Accuracy: {accuracy:.3f} "
        f"({accuracy * 100:.1f}%)"
    )

    print(f"Kappa:    {kappa:.3f}")

    print(
        f"Prediction time: "
        f"{prediction_time:.3f} seconds"
    )

    print("\nConfusion-matrix class order:")
    print(CLASS_ORDER)

    print("\nConfusion matrix:")
    print(matrix)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
    ) as output_file:
        writer = csv.writer(output_file)

        writer.writerow(
            [
                "trial",
                "true_class",
                "predicted_class",
                "command",
                "confidence",
                "correct",
                "model",
                "training_dataset",
                "testing_dataset",
                "data_split",
                "csp_regularization",
                "lda_shrinkage",
            ]
        )

        for trial_number, (
            true_class,
            predicted_class,
            trial_confidence,
        ) in enumerate(
            zip(
                y_test,
                y_pred,
                confidence,
            ),
            start=1,
        ):
            writer.writerow(
                [
                    trial_number,
                    true_class,
                    predicted_class,
                    CLASS_TO_COMMAND[predicted_class],
                    f"{trial_confidence:.3f}",
                    true_class == predicted_class,
                    "regularized_FBCSP_shrinkage_LDA",
                    "A01T.gdf-A08T.gdf",
                    "A09T.gdf",
                    "unseen_subject_cross_subject",
                    "ledoit_wolf",
                    "auto",
                ]
            )

    print("\nExport complete:")
    print(OUTPUT_PATH)

    print("\nComparison:")
    print("- Baseline accuracy:         53.1%")
    print("- Baseline kappa:            0.375")
    print("- Corrected bands accuracy:  52.8%")
    print("- Corrected bands kappa:     0.370")


if __name__ == "__main__":
    main()
