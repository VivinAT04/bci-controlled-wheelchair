"""Data loading utilities for BCI Competition IV Dataset 2a via MOABB.

MOABB downloads and standardises the raw GDF files for you (cached in
~/mne_data after the first run), so there is no need to hand-parse them.
"""
import mne
from moabb.paradigms import MotorImagery

try:
    from moabb.datasets import BNCI2014_001 as Dataset2a  # moabb >= 1.0
except ImportError:                                          # older moabb
    from moabb.datasets import BNCI2014001 as Dataset2a

mne.set_log_level("WARNING")

SFREQ = 250  # Hz, fixed for Dataset 2a


def load_subject(subject: int, fmin: float = 8.0, fmax: float = 30.0):
    """Load one subject's EEG trials, labels, and metadata.

    Parameters
    ----------
    subject : int
        Subject number, 1-9.
    fmin, fmax : float
        Band-pass filter applied by MOABB before returning data.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_samples)
    y : ndarray, shape (n_trials,) of str
        One of "left_hand", "right_hand", "feet", "tongue".
    meta : pandas.DataFrame
        Per-trial metadata, including a "session" column ("T" or "E").
    """
    paradigm = MotorImagery(fmin=fmin, fmax=fmax, n_classes=4)
    dataset = Dataset2a()
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    return X, y, meta
