"""Signal preprocessing helpers for BCI Competition IV Dataset 2a."""

import mne
import numpy as np
from scipy.signal import butter, filtfilt

SFREQ = 250
EOG_CHANNELS = ["EOG-left", "EOG-central", "EOG-right"]
EVENT_ID = {"769": 1, "770": 2, "771": 3, "772": 4}
LABEL_MAP = {1: "left_hand", 2: "right_hand", 3: "feet", 4: "tongue"}


def bandpass(X: np.ndarray, lo: float, hi: float, sfreq: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter applied along the last axis."""
    nyq = sfreq / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, X, axis=-1)


def preprocess_raw(raw, fmin: float = 8.0, fmax: float = 30.0,
                   tmin: float = 0.5, tmax: float = 2.5):
    """Convert raw EEG into filtered epochs and labels."""

    raw = raw.copy()

    raw.drop_channels(EOG_CHANNELS, on_missing="ignore")
    raw.filter(fmin, fmax, fir_design="firwin", verbose=False)

    events, _ = mne.events_from_annotations(
        raw,
        event_id=EVENT_ID,
        verbose=False
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_ID,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    X = epochs.get_data()
    y = np.array([LABEL_MAP[code] for code in epochs.events[:, 2]])

    return X, y