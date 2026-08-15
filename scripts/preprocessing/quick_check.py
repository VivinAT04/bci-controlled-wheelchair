"""
Quick end-to-end check: load Subject 1, preprocess it, run CSP+LDA.

Run from the project root:
    python -m scripts.preprocessing.quick_check
"""

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.data.preprocessing import preprocess_raw


def main():
    print("Step 1: Loading Subject 1 from data/raw/A01T.gdf...")
    raw = load_raw_gdf("data/raw/A01T.gdf")
    X, y = preprocess_raw(raw)

    print(f"  Loaded {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} samples/trial")

    print("\nStep 2: Data cleaning checks...")
    classes, counts = np.unique(y, return_counts=True)

    for c, n in zip(classes, counts):
        print(f"  {c:12s}: {n} trials")

    if len(set(counts)) > 1:
        print("  WARNING: classes are imbalanced.")
    else:
        print("  Classes are balanced. Good.")

    print("\nStep 3: Preliminary classification CSP + LDA, 5-fold CV...")

    clf = Pipeline([
        ("csp", CSP(n_components=6, reg=None, log=True)),
        ("lda", LDA()),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=sorted(set(y)))

    print("\nRESULT — Subject 1, CSP+LDA, 5-fold CV:")
    print(f"  Accuracy = {acc:.3f} ({acc * 100:.1f}%)")
    print(f"  Kappa    = {kappa:.3f}")
    print("  Chance accuracy for 4 classes = 0.250")
    print(f"\n  Confusion matrix, order {sorted(set(y))}:")
    print(cm)


if __name__ == "__main__":
    main()