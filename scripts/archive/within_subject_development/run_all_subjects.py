"""
Run the CSP+LDA baseline across all 9 subjects, using the local .gdf files
in data/raw/.

Run from the project root:
    python -m scripts.within_subject.run_all_subjects
"""
from curses import raw

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, cohen_kappa_score
from mne.decoding import CSP

from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.data.preprocessing import preprocess_raw


def main():
    results = []

    for i in range(1, 10):
        subject_code = f"A{i:02d}T"
        path = f"data/raw/{subject_code}.gdf"
        print(f"Loading {subject_code}...")

        raw = load_raw_gdf(path)
        X, y = preprocess_raw(raw)

        clf = Pipeline([
            ("csp", CSP(n_components=6, reg=None, log=True)),
            ("lda", LDA()),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X, y, cv=cv)

        acc = accuracy_score(y, y_pred)
        kappa = cohen_kappa_score(y, y_pred)
        results.append((subject_code, acc, kappa))
        print(f"  {subject_code}: accuracy={acc:.3f} ({acc*100:.1f}%)  kappa={kappa:.3f}")

    print("\n--- Summary ---")
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
