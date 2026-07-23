"""Preprocess EEG recordings once and save reusable NumPy datasets."""

from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat

from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.data.preprocessing import EOG_CHANNELS, LABEL_MAP


RAW_DIR = Path("data/raw")
LABEL_DIR = Path("data/labels")
PROCESSED_DIR = Path("data/processed")

PREPROCESSING_CONFIGS = [
    {
        "name": "8-30Hz_0.5-2.5s",
        "fmin": 8.0,
        "fmax": 30.0,
        "tmin": 0.5,
        "tmax": 2.5,
    },
    {
        "name": "4-40Hz_0.5-2.5s",
        "fmin": 4.0,
        "fmax": 40.0,
        "tmin": 0.5,
        "tmax": 2.5,
    },
]


def load_labels(label_path: Path) -> np.ndarray:
    """Load the class labels stored in a Dataset 2a .mat file."""
    mat_data = loadmat(label_path)

    if "classlabel" not in mat_data:
        raise KeyError(f"'classlabel' was not found in {label_path}")

    numeric_labels = mat_data["classlabel"].reshape(-1).astype(int)

    return np.array(
        [LABEL_MAP[label] for label in numeric_labels],
        dtype=str,
    )


def preprocess_recording(
    gdf_path: Path,
    label_path: Path,
    fmin: float,
    fmax: float,
    tmin: float,
    tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load, clean, filter and epoch one EEG recording."""
    raw = load_raw_gdf(str(gdf_path)).copy()

    raw.drop_channels(EOG_CHANNELS, on_missing="ignore")
    raw.filter(fmin, fmax, fir_design="firwin", verbose=False)

    # Training files use 769–772. Evaluation files use 783.
    if gdf_path.stem.endswith("T"):
        event_id = {
            "769": 1,
            "770": 2,
            "771": 3,
            "772": 4,
        }
    else:
        event_id = {"783": 1}

    events, _ = mne.events_from_annotations(
        raw,
        event_id=event_id,
        verbose=False,
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data()
    y = load_labels(label_path)

    if len(X) != len(y):
        raise ValueError(
            f"{gdf_path.name}: found {len(X)} EEG trials "
            f"but {len(y)} labels."
        )

    return X, y


def main() -> None:
    """Create all reusable processed datasets."""
    gdf_files = sorted(RAW_DIR.glob("*.gdf"))

    if not gdf_files:
        raise FileNotFoundError(f"No .gdf files found inside {RAW_DIR}")

    for config in PREPROCESSING_CONFIGS:
        output_dir = PROCESSED_DIR / config["name"]
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nCreating: {output_dir}")

        for gdf_path in gdf_files:
            label_path = LABEL_DIR / f"{gdf_path.stem}.mat"
            output_path = output_dir / f"{gdf_path.stem}.npz"

            if not label_path.exists():
                raise FileNotFoundError(f"Missing label file: {label_path}")

            X, y = preprocess_recording(
                gdf_path=gdf_path,
                label_path=label_path,
                fmin=config["fmin"],
                fmax=config["fmax"],
                tmin=config["tmin"],
                tmax=config["tmax"],
            )

            np.savez_compressed(
                output_path,
                X=X,
                y=y,
                sfreq=raw_sfreq(),
                fmin=config["fmin"],
                fmax=config["fmax"],
                tmin=config["tmin"],
                tmax=config["tmax"],
            )

            print(
                f"Saved {output_path} "
                f"| X: {X.shape} | y: {y.shape}"
            )


def raw_sfreq() -> float:
    """Dataset 2a sampling frequency."""
    return 250.0


if __name__ == "__main__":
    main()
