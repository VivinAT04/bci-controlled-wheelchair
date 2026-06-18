"""
Run Filter-Bank CSP + LDA on Subject A01T using Leave-One-Out
Cross Validation (LOOCV).

Run from the project root:
    python -m scripts.run_loocv_fbcsp_all_subjects
"""
import numpy as np
import mne
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from bci_wheelchair.models import make_fbcsp_lda
from bci_wheelchair.data_loading import EOG_CHANNELS, EVENT_ID, LABEL_MAP

mne.set_log_level("WARNING")


def load_subject_broadband(gdf_path, fmin=4.0, fmax=40.0, tmin=0.5, tmax=2.5):
    raw = mne.io.read_raw_gdf(gdf_path, preload=True)
    raw.drop_channels(EOG_CHANNELS)
    raw.filter(fmin, fmax, fir_design="firwin", verbose=False)

    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose=False)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_ID,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data()
    y = np.array([LABEL_MAP[code] for code in epochs.events[:, 2]])
    return X, y


def main():
    subject_code = "A01T"
    path = f"data/raw/{subject_code}.gdf"

    print(f"Loading {subject_code} for FBCSP + LOOCV...")
    X, y = load_subject_broadband(path)

    print(f"Loaded {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} samples/trial")

    clf = make_fbcsp_lda(n_components=4)
    cv = LeaveOneOut()

    print("Running LOOCV. This may take a few minutes...")
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=sorted(set(y)))

    print("\n--- Result: A01T FBCSP+LDA with LOOCV ---")
    print(f"Accuracy = {acc:.3f} ({acc*100:.1f}%)")
    print(f"Kappa    = {kappa:.3f}")
    print(f"Classes  = {sorted(set(y))}")
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()