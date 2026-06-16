"""Signal preprocessing helpers."""
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass(X: np.ndarray, lo: float, hi: float, sfreq: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter applied along the last axis."""
    nyq = sfreq / 2.0
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, X, axis=-1)
