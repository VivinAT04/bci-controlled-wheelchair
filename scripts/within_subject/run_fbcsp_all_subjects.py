"""
Run Filter-Bank CSP + LDA across all 9 subjects, using the local .gdf files
in data/raw/. Compare against scripts/run_all_subjects.py (plain CSP+LDA).

Run from the project root:
    python -m scripts.within_subject.run_fbcsp_all_subjects
"""
import numpy as np
import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, cohen_kappa_score

from bci_wheelchair.models import make_fbcsp_lda
from bci_wheelchair.data.preprocessing import EOG_CHANNELS, EVENT_ID, LABEL_MAP, SFREQ

mne.set_log_level("WARNING")


def load_subject_broadband(gdf_path, fmin=4.0, fmax=40.0, tmin=0.5, tmax=2.5):
    """Like load_subject_local, but with a WIDE band — FBCSP splits this
    into its own sub-bands internally, so the input here must not already
    be narrowed to 8-30Hz or we'd be filtering out useful sub-bands."""
    raw = mne.io.read_raw_gdf(gdf_path, preload=True)
    raw.drop_channels(EOG_CHANNELS)
    raw.filter(fmin, fmax, fir_design="firwin", verbose=False)

    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=EVENT_ID, tmin=tmin, tmax=tmax,
                         baseline=None, preload=True, verbose=False)

    X = epochs.get_data()
    y = np.array([LABEL_MAP[code] for code in epochs.events[:, 2]])
    return X, y


def main():
    results = []

    for i in range(1, 10):
        subject_code = f"A{i:02d}T"
        path = f"data/raw/{subject_code}.gdf"
        print(f"Loading {subject_code} (broadband, for FBCSP)...")

        X, y = load_subject_broadband(path)

        clf = make_fbcsp_lda(n_components=4)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X, y, cv=cv)

        acc = accuracy_score(y, y_pred)
        kappa = cohen_kappa_score(y, y_pred)
        results.append((subject_code, acc, kappa))
        print(f"  {subject_code}: accuracy={acc:.3f} ({acc*100:.1f}%)  kappa={kappa:.3f}")

    print("\n--- Summary (FBCSP+LDA) ---")
    print(f"{'Subject':10s}{'Accuracy':>12s}{'Kappa':>10s}")
    accs, kaps = [], []
    for code, acc, kappa in results:
        print(f"{code:10s}{acc*100:>11.1f}%{kappa:>10.3f}")
        accs.append(acc)
        kaps.append(kappa)

    print(f"{'Average':10s}{np.mean(accs)*100:>11.1f}%{np.mean(kaps):>10.3f}")
    print(f"{'Std dev':10s}{np.std(accs)*100:>11.1f}%{np.std(kaps):>10.3f}")


if __name__ == "__main__":
    main()
