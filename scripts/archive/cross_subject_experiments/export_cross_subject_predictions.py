"""
Train the existing FBCSP + LDA classifier using subjects A01T-A08T
and test it on the completely unseen subject A09T.

Run:
    python -m scripts.cross_subject.export_cross_subject_predictions
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import mne
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

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
OUTPUT_PATH = Path("results/cross_subject/csp_fbcsp/cross_subject_a09_predictions.csv")

FMIN = 8.0
FMAX = 30.0
TMIN = 0.5
TMAX = 2.5

CLASS_ORDER = [
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
]


def load_and_preprocess_subject(
    subject_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess one BCI Competition IV 2a subject."""
    subject_path = DATA_DIRECTORY / f"{subject_name}.gdf"

    if not subject_path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {subject_path}"
        )

    print(f"Loading {subject_name}...")

    raw = load_raw_gdf(str(subject_path))

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
    print("Cross-Subject FBCSP + LDA Evaluation")
    print("=" * 72)

    print("\nTraining subjects:")
    print(", ".join(TRAIN_SUBJECTS))

    print(f"\nUnseen test subject: {TEST_SUBJECT}")

    training_arrays: list[np.ndarray] = []
    training_labels: list[np.ndarray] = []
    training_subject_ids: list[str] = []

    print("\nLoading and preprocessing training subjects...")

    preprocessing_start = time.perf_counter()

    for subject_name in TRAIN_SUBJECTS:
        X_subject, y_subject = load_and_preprocess_subject(
            subject_name
        )

        training_arrays.append(X_subject)
        training_labels.append(y_subject)

        training_subject_ids.extend(
            [subject_name] * len(y_subject)
        )

    X_train = np.concatenate(
        training_arrays,
        axis=0,
    )

    y_train = np.concatenate(
        training_labels,
        axis=0,
    )

    preprocessing_time = time.perf_counter() - preprocessing_start

    print("\nLoading and preprocessing unseen A09T test subject...")

    X_test, y_test = load_and_preprocess_subject(
        TEST_SUBJECT
    )

    print("\n" + "=" * 72)
    print("Dataset Summary")
    print("=" * 72)

    print(f"Training trials: {len(X_train)}")
    print(f"Testing trials:  {len(X_test)}")
    print(f"Training shape:  {X_train.shape}")
    print(f"Testing shape:   {X_test.shape}")
    print(
        "Preprocessing time: "
        f"{preprocessing_time:.2f} seconds"
    )

    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            "Training and test EEG dimensions do not match: "
            f"{X_train.shape[1:]} versus {X_test.shape[1:]}"
        )

    print("\nBuilding the existing FBCSP + LDA classifier...")

    classifier = make_fbcsp_lda(
        n_components=4
    )

    print(
        "Training on pooled data from "
        "A01T-A08T..."
    )

    training_start = time.perf_counter()

    classifier.fit(
        X_train,
        y_train,
    )

    training_time = time.perf_counter() - training_start

    print(
        f"Model training completed in "
        f"{training_time:.2f} seconds"
    )

    print("\nPredicting the unseen A09T subject...")

    prediction_start = time.perf_counter()

    y_pred = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)

    prediction_time = time.perf_counter() - prediction_start

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
    print("Unseen A09T Cross-Subject Results")
    print("=" * 72)

    print(
        f"Accuracy: {accuracy:.3f} "
        f"({accuracy * 100:.1f}%)"
    )

    print(f"Kappa:    {kappa:.3f}")

    print(
        f"Prediction time for {len(X_test)} trials: "
        f"{prediction_time:.3f} seconds"
    )

    print(
        "Average prediction time per trial: "
        f"{prediction_time / len(X_test):.6f} seconds"
    )

    print("\nConfusion-matrix class order:")
    print(CLASS_ORDER)

    print("\nConfusion matrix:")
    print(matrix)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_dataset_description = (
        "A01T.gdf-A08T.gdf"
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
                "training_time_seconds",
                "prediction_time_seconds",
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
                    "FBCSP_LDA",
                    training_dataset_description,
                    f"{TEST_SUBJECT}.gdf",
                    "unseen_subject_cross_subject",
                    f"{training_time:.3f}",
                    f"{prediction_time:.3f}",
                ]
            )

    print("\nExport complete:")
    print(OUTPUT_PATH)

    print("\nProtocol:")
    print("- Training subjects: A01T-A08T")
    print("- Test subject: A09T")
    print("- A09T was not used during model training")
    print("- Classifier: existing FBCSP + LDA")


if __name__ == "__main__":
    main()
