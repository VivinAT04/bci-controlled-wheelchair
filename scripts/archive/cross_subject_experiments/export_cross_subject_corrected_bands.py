"""
Cross-subject FBCSP + LDA evaluation using frequency bands that match
the 8-30 Hz preprocessing range.

Train:
    A01T-A08T

Test:
    A09T

Run:
    python -m scripts.cross_subject.export_cross_subject_corrected_bands
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import mne
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)

from bci_wheelchair.commands import CLASS_TO_COMMAND
from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.models import make_fbcsp_lda
from bci_wheelchair.data.preprocessing import preprocess_raw


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
    "results/cross_subject/csp_fbcsp/cross_subject_a09_corrected_bands_predictions.csv"
)

FMIN = 8.0
FMAX = 30.0
TMIN = 0.5
TMAX = 2.5

CORRECTED_BANDS = [
    (8, 12),
    (12, 16),
    (16, 20),
    (20, 24),
    (24, 28),
    (28, 30),
]

CLASS_ORDER = [
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
]


def load_subject(
    subject_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess one subject."""
    path = DATA_DIRECTORY / f"{subject_name}.gdf"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    print(f"Loading {subject_name}...")

    raw = load_raw_gdf(str(path))

    X, y = preprocess_raw(
        raw,
        fmin=FMIN,
        fmax=FMAX,
        tmin=TMIN,
        tmax=TMAX,
    )

    print(f"  Trials: {len(X)}, shape: {X.shape}")

    return X, y


def main() -> None:
    print("=" * 72)
    print("Cross-Subject FBCSP + LDA: Corrected Frequency Bands")
    print("=" * 72)

    print("\nTraining subjects:")
    print(", ".join(TRAIN_SUBJECTS))

    print(f"\nTest subject: {TEST_SUBJECT}")

    print("\nCorrected FBCSP bands:")
    for low, high in CORRECTED_BANDS:
        print(f"  {low}-{high} Hz")

    X_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []

    preprocessing_start = time.perf_counter()

    print("\nLoading training subjects...")

    for subject in TRAIN_SUBJECTS:
        X_subject, y_subject = load_subject(subject)

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

    X_test, y_test = load_subject(TEST_SUBJECT)

    preprocessing_time = (
        time.perf_counter() - preprocessing_start
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

    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            "Training and testing EEG shapes do not match: "
            f"{X_train.shape[1:]} versus "
            f"{X_test.shape[1:]}"
        )

    print(
        "\nBuilding FBCSP + LDA with corrected bands..."
    )

    classifier = make_fbcsp_lda(
        n_components=4,
        bands=CORRECTED_BANDS,
    )

    print("Training on A01T-A08T...")

    training_start = time.perf_counter()

    classifier.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.perf_counter() - training_start
    )

    print(
        f"Training completed in "
        f"{training_time:.2f} seconds"
    )

    print("\nPredicting A09T...")

    prediction_start = time.perf_counter()

    y_pred = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)

    prediction_time = (
        time.perf_counter() - prediction_start
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
    print("Corrected-Band Cross-Subject Results")
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
                "frequency_bands",
            ]
        )

        band_description = ";".join(
            f"{low}-{high}"
            for low, high in CORRECTED_BANDS
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
                    "FBCSP_LDA_corrected_bands",
                    "A01T.gdf-A08T.gdf",
                    "A09T.gdf",
                    "unseen_subject_cross_subject",
                    band_description,
                ]
            )

    print("\nExport complete:")
    print(OUTPUT_PATH)

    print("\nComparison baseline:")
    print("- Original bands accuracy: 53.1%")
    print("- Original bands kappa:    0.375")
    print("- Compare the new result against these values.")


if __name__ == "__main__":
    main()
