"""
View raw EEG data from a BCI Competition IV 2a GDF file.

Run from the project root:

    python -m scripts.visualisation.view_eeg
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne


INPUT_PATH = Path("data/raw/A01T.gdf")
DURATION_SECONDS = 1
NUMBER_OF_CHANNELS = 22


def load_raw_eeg(input_path: Path) -> mne.io.BaseRaw:
    """Load a raw EEG recording from a GDF file."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"EEG file not found: {input_path}"
        )

    return mne.io.read_raw_gdf(
        input_path,
        preload=True,
    )


def main() -> None:
    """Load and display the raw EEG recording."""

    print(f"Loading EEG file: {INPUT_PATH}")

    raw = load_raw_eeg(INPUT_PATH)

    print("\nLoaded recording:")
    print(raw)

    raw.plot(
        duration=DURATION_SECONDS,
        n_channels=NUMBER_OF_CHANNELS,
        block=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
