"""
Train the existing FBCSP + LDA classifier on A09T and generate wheelchair
commands from the unseen A09E evaluation session.

The classifier and the existing training preprocessing are not modified.

Run:
    python -m scripts.export_test_predictions
"""

from __future__ import annotations

import csv
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from bci_wheelchair.commands import CLASS_TO_COMMAND
from bci_wheelchair.data_loading import load_raw_gdf
from bci_wheelchair.models import make_fbcsp_lda
from bci_wheelchair.preprocessing import LABEL_MAP, preprocess_raw


mne.set_log_level("ERROR")

TRAIN_PATH = Path("data/raw/A09T.gdf")
TEST_PATH = Path("data/raw/A09E.gdf")
TEST_LABEL_PATH = Path("data/labels/A09E.mat")
OUTPUT_PATH = Path("results/test_predicted_commands.csv")

TEST_EVENT_ID = {"783": 1}

FMIN = 8.0
FMAX = 30.0
TMIN = 0.5
TMAX = 2.5


def preprocess_evaluation_raw(raw: mne.io.BaseRaw) -> np.ndarray:
    """
    Extract EEG epochs from an evaluation GDF file.

    Evaluation files use event 783 because the true class is hidden inside
    the GDF file. The official labels are loaded separately from A09E.mat.
    """
    raw = raw.copy()

    eog_channels = ["EOG-left", "EOG-central", "EOG-right"]
    raw.drop_channels(eog_channels, on_missing="ignore")

    raw.filter(
        FMIN,
        FMAX,
        fir_design="firwin",
        verbose=False,
    )

    events, _ = mne.events_from_annotations(
        raw,
        event_id=TEST_EVENT_ID,
        verbose=False,
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=TEST_EVENT_ID,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs.get_data()


def load_evaluation_labels(path: Path) -> np.ndarray:
    """Load official BCI Competition evaluation labels from a MAT file."""
    mat_data = loadmat(path)

    if "classlabel" not in mat_data:
        available_keys = [
            key for key in mat_data.keys()
            if not key.startswith("__")
        ]
        raise KeyError(
            "Could not find 'classlabel' in "
            f"{path}. Available keys: {available_keys}"
        )

    numeric_labels = np.asarray(
        mat_data["classlabel"]
    ).reshape(-1).astype(int)

    unknown_labels = sorted(
        set(numeric_labels) - set(LABEL_MAP.keys())
    )

    if unknown_labels:
        raise ValueError(
            f"Unexpected labels in {path}: {unknown_labels}"
        )

    return np.array(
        [LABEL_MAP[label] for label in numeric_labels]
    )


def main() -> None:
    print("=" * 65)
    print("Wheelchair Commands from Unseen Evaluation Data")
    print("=" * 65)

    required_paths = [
        TRAIN_PATH,
        TEST_PATH,
        TEST_LABEL_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print(f"Training EEG: {TRAIN_PATH}")
    print(f"Unseen EEG:   {TEST_PATH}")
    print(f"Test labels:  {TEST_LABEL_PATH}")
    print(f"Output CSV:   {OUTPUT_PATH}")

    print("\nLoading A09T training data...")
    raw_train = load_raw_gdf(str(TRAIN_PATH))

    print("Loading A09E unseen evaluation data...")
    raw_test = load_raw_gdf(str(TEST_PATH))

    print("\nPreprocessing A09T using the existing training preprocessing...")
    X_train, y_train = preprocess_raw(
        raw_train,
        fmin=FMIN,
        fmax=FMAX,
        tmin=TMIN,
        tmax=TMAX,
    )

    print("Extracting A09E trials using evaluation event 783...")
    X_test = preprocess_evaluation_raw(raw_test)

    print("Loading official A09E labels for evaluation only...")
    y_test = load_evaluation_labels(TEST_LABEL_PATH)

    print(f"\nTraining trials: {len(X_train)}")
    print(f"Testing trials:  {len(X_test)}")
    print(f"Testing labels:  {len(y_test)}")
    print(f"Training shape:  {X_train.shape}")
    print(f"Testing shape:   {X_test.shape}")

    if len(X_test) != len(y_test):
        raise ValueError(
            "The number of A09E EEG trials does not match the number "
            f"of official labels: {len(X_test)} trials versus "
            f"{len(y_test)} labels."
        )

    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError(
            "Training and testing EEG shapes are incompatible: "
            f"{X_train.shape[1:]} versus {X_test.shape[1:]}"
        )

    print("\nBuilding the existing FBCSP + LDA classifier...")
    classifier = make_fbcsp_lda(n_components=4)

    print("Training only on A09T...")
    classifier.fit(X_train, y_train)

    print("Predicting wheelchair commands from unseen A09E...")
    y_pred = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    confidence = probabilities.max(axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    class_order = [
        "left_hand",
        "right_hand",
        "feet",
        "tongue",
    ]

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=class_order,
    )

    print("\n" + "=" * 65)
    print("A09E Unseen Evaluation Results")
    print("=" * 65)
    print(f"Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")
    print(f"Kappa:    {kappa:.3f}")

    print("\nConfusion-matrix class order:")
    print(class_order)

    print("\nConfusion matrix:")
    print(matrix)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="") as file:
        writer = csv.writer(file)

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
            ]
        )

        for trial_number, (
            true_class,
            predicted_class,
            trial_confidence,
        ) in enumerate(
            zip(y_test, y_pred, confidence),
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
                    TRAIN_PATH.name,
                    TEST_PATH.name,
                    "unseen_evaluation_session",
                ]
            )

    print(f"\nExported {len(y_pred)} commands to:")
    print(OUTPUT_PATH)

    print("\nImportant:")
    print("- Classifier implementation was not changed.")
    print("- Model was fitted only using A09T.")
    print("- Wheelchair commands were predicted only from A09E.")
    print("- A09E labels were used only after prediction for evaluation.")


if __name__ == "__main__":
    main()
