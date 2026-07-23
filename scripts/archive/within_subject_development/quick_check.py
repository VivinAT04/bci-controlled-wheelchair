"""
Quick end-to-end check using the cached processed EEG dataset.

Run from the project root:

    python -m scripts.within_subject.quick_check
"""

import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from bci_wheelchair.data.processed_loading import load_processed_subject


SUBJECT = "A01T"
PREPROCESSING_CONFIG = "8-30"


def main() -> None:
    """Load cached EEG data and run a quick CSP and LDA check."""
    print(f"Step 1: Loading processed data for {SUBJECT}...")

    X, y = load_processed_subject(
        subject=SUBJECT,
        config=PREPROCESSING_CONFIG,
    )

    print(
        f"  Loaded {X.shape[0]} trials, "
        f"{X.shape[1]} channels, "
        f"{X.shape[2]} samples/trial"
    )

    print("\nStep 2: Data cleaning checks...")

    classes, counts = np.unique(y, return_counts=True)

    for class_label, count in zip(classes, counts):
        print(f"  {class_label}: {count} trials")

    if len(set(counts)) > 1:
        print("  WARNING: classes are imbalanced.")
    else:
        print("  Classes are balanced. Good.")

    print("\nStep 3: Preliminary classification CSP + LDA, 5-fold CV...")

    classifier = Pipeline(
        [
            (
                "csp",
                CSP(
                    n_components=6,
                    reg=None,
                    log=True,
                    rank={"eeg": 22},
                ),
            ),
            ("lda", LDA()),
        ]
    )

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    predictions = cross_val_predict(
        classifier,
        X,
        y,
        cv=cross_validation,
    )

    accuracy = accuracy_score(y, predictions)
    kappa = cohen_kappa_score(y, predictions)

    class_order = sorted(np.unique(y))
    matrix = confusion_matrix(
        y,
        predictions,
        labels=class_order,
    )

    print("\nRESULT — Subject 1, CSP + LDA, 5-fold CV:")
    print(f"  Accuracy = {accuracy:.3f} ({accuracy * 100:.1f}%)")
    print(f"  Kappa    = {kappa:.3f}")
    print("  Chance accuracy for 4 classes = 0.250")
    print(f"\n  Confusion matrix, order {class_order}:")
    print(matrix)


if __name__ == "__main__":
    main()
