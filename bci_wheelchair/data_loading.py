"""Load BCI Competition IV Dataset 2a directly from the local .gdf files.

Verified event codes (confirmed against the real dataset files):
    768  = trial start (cue onset)
    769  = class 1, left hand
    770  = class 2, right hand
    771  = class 3, feet
    772  = class 4, tongue
    1023 = artifact-rejected trial (excluded automatically, since 1023
           trials carry no class code)

Note: this loader only reads "T" (training) session files, because only
those sessions have the true class label embedded in the file. "E"
(evaluation) session files ship with labels withheld; the true labels for
E sessions were released separately as .mat files after the original
competition. Extend this module later if those files are obtained, to
support the proper train-on-T / test-on-E benchmark.
"""
import mne
import numpy as np

mne.set_log_level("WARNING")

SFREQ = 250  # Hz, fixed for Dataset 2a
EOG_CHANNELS = ["EOG-left", "EOG-central", "EOG-right"]
EVENT_ID = {"769": 1, "770": 2, "771": 3, "772": 4}
LABEL_MAP = {1: "left_hand", 2: "right_hand", 3: "feet", 4: "tongue"}


def load_subject_local(gdf_path: str, fmin: float = 8.0, fmax: float = 30.0,
                        tmin: float = 0.5, tmax: float = 2.5):
    """Load one subject's training-session trials from a local .gdf file.

    Parameters
    ----------
    gdf_path : str
        Path to an "...T.gdf" file (e.g. "data/raw/A01T.gdf").
    fmin, fmax : float
        Band-pass filter range in Hz — removes drift, line noise, and
        muscle artifact outside the mu/beta motor-imagery band.
    tmin, tmax : float
        Epoch window in seconds relative to cue onset. 0.5-2.5s is the
        standard window for this dataset.

    Returns
    -------
    X : ndarray, shape (n_trials, 22, n_samples)
    y : ndarray, shape (n_trials,) of str
        One of "left_hand", "right_hand", "feet", "tongue".
    """
    raw = mne.io.read_raw_gdf(gdf_path, preload=True)
    raw.drop_channels(EOG_CHANNELS)
    raw.filter(fmin, fmax, fir_design="firwin", verbose=False)

    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=EVENT_ID, tmin=tmin, tmax=tmax,
                         baseline=None, preload=True, verbose=False)

    X = epochs.get_data()
    y = np.array([LABEL_MAP[code] for code in epochs.events[:, 2]])
    return X, y
