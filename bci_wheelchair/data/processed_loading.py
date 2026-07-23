"""Load previously preprocessed EEG datasets."""

from pathlib import Path

import numpy as np


PROCESSED_DATA_DIR = Path("data/processed")

AVAILABLE_CONFIGS = {
    "8-30": "8-30Hz_0.5-2.5s",
    "4-40": "4-40Hz_0.5-2.5s",
    "8-30Hz_0.5-2.5s": "8-30Hz_0.5-2.5s",
    "4-40Hz_0.5-2.5s": "4-40Hz_0.5-2.5s",
}


def load_processed_subject(
    subject: str,
    config: str = "8-30",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed EEG trials and labels for one recording.

    Examples
    --------
    Training recording:

        X, y = load_processed_subject("A01T", config="8-30")

    Evaluation recording:

        X, y = load_processed_subject("A01E", config="4-40")
    """
    subject = subject.upper()

    if config not in AVAILABLE_CONFIGS:
        valid_configs = ", ".join(sorted(AVAILABLE_CONFIGS))
        raise ValueError(
            f"Unknown preprocessing config: {config}. "
            f"Available configs: {valid_configs}"
        )

    config_directory = AVAILABLE_CONFIGS[config]
    file_path = PROCESSED_DATA_DIR / config_directory / f"{subject}.npz"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {file_path}\n"
            "Run:\n"
            "python -m scripts.preprocessing.create_processed_datasets"
        )

    with np.load(file_path, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"{subject}: X contains {X.shape[0]} trials, "
            f"but y contains {y.shape[0]} labels."
        )

    return X, y


def load_processed_metadata(
    subject: str,
    config: str = "8-30",
) -> dict[str, float]:
    """Load the preprocessing settings stored with one recording."""
    subject = subject.upper()

    if config not in AVAILABLE_CONFIGS:
        raise ValueError(f"Unknown preprocessing config: {config}")

    config_directory = AVAILABLE_CONFIGS[config]
    file_path = PROCESSED_DATA_DIR / config_directory / f"{subject}.npz"

    if not file_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {file_path}")

    with np.load(file_path, allow_pickle=False) as data:
        return {
            "sfreq": float(data["sfreq"]),
            "fmin": float(data["fmin"]),
            "fmax": float(data["fmax"]),
            "tmin": float(data["tmin"]),
            "tmax": float(data["tmax"]),
        }


def processed_subject_exists(
    subject: str,
    config: str = "8-30",
) -> bool:
    """Check whether a processed recording exists."""
    subject = subject.upper()

    if config not in AVAILABLE_CONFIGS:
        return False

    config_directory = AVAILABLE_CONFIGS[config]
    file_path = PROCESSED_DATA_DIR / config_directory / f"{subject}.npz"

    return file_path.exists()
