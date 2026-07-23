"""
Run CSP + LDA using Leave-One-Out Cross Validation (LOOCV)
for all BCI Competition IV Dataset 2a training subjects.

Run:
    python -m scripts.within_subject.run_loocv_csp_all_subjects
"""

import warnings
import mne
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from mne.decoding import CSP

from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.data.preprocessing import preprocess_raw


warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


SUBJECTS = [
    "A01T",
    "A02T",
    "A03T",
    "A04T",
    "A05T",
    "A06T",
    "A07T",
    "A08T",
    "A09T",
]


def build_csp_lda():
    return Pipeline([
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
    ])


def run_subject(subject_code):
    path = f"data/raw/{subject_code}.gdf"

    print(f"\nLoading {subject_code}...")

    raw = load_raw_gdf(path)
    X, y = preprocess_raw(raw)

    print(
        f"Loaded {X.shape[0]} trials, "
        f"{X.shape[1]} EEG channels, "
        f"{X.shape[2]} samples/trial"
    )

    clf = build_csp_lda()
    cv = LeaveOneOut()

    print(f"Running LOOCV for {subject_code}...")

    y_pred = cross_val_predict(
        clf,
        X,
        y,
        cv=cv,
        n_jobs=-1,
    )

    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    print(f"{subject_code} Accuracy: {acc:.3f} ({acc * 100:.1f}%)")
    print(f"{subject_code} Kappa:    {kappa:.3f}")

    return {
        "subject": subject_code,
        "accuracy": acc,
        "kappa": kappa,
        "confusion_matrix": cm,
    }


def main():
    print("\n========================================")
    print("CSP + LDA LOOCV: All Subjects")
    print("========================================")

    results = []

    for subject in SUBJECTS:
        result = run_subject(subject)
        results.append(result)

    accuracies = [r["accuracy"] for r in results]
    kappas = [r["kappa"] for r in results]

    print("\n========================================")
    print("Final Subject-wise Results")
    print("========================================")
    print(f"{'Subject':<10}{'Accuracy':<15}{'Kappa':<10}")
    print("----------------------------------------")

    for r in results:
        print(
            f"{r['subject']:<10}"
            f"{r['accuracy'] * 100:<15.1f}"
            f"{r['kappa']:<10.3f}"
        )

    print("----------------------------------------")
    print(
        f"{'Mean':<10}"
        f"{np.mean(accuracies) * 100:<15.1f}"
        f"{np.mean(kappas):<10.3f}"
    )

    print("\n========================================")
    print("Confusion Matrices")
    print("========================================")

    for r in results:
        print(f"\n{r['subject']}")
        print(r["confusion_matrix"])


if __name__ == "__main__":
    main()