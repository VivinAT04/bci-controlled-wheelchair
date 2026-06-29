"""
Run Filter-Bank CSP + LDA using LOOCV for all BCI Competition IV Dataset 2a subjects.

Run:
    python -m scripts.run_loocv_fbcsp_all_subjects
"""

import warnings
import mne
import numpy as np

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from bci_wheelchair.data_loading import load_raw_gdf
from bci_wheelchair.preprocessing import preprocess_raw
from bci_wheelchair.models import make_fbcsp_lda


warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


SUBJECTS = [
    "A01T", "A02T", "A03T", "A04T", "A05T",
    "A06T", "A07T", "A08T", "A09T",
]


def run_subject(subject_code):
    print(f"\nLoading {subject_code}...")

    raw = load_raw_gdf(f"data/raw/{subject_code}.gdf")

    X, y = preprocess_raw(
        raw,
        fmin=4.0,
        fmax=40.0,
        tmin=0.5,
        tmax=2.5,
    )

    print(
        f"Loaded {X.shape[0]} trials, "
        f"{X.shape[1]} EEG channels, "
        f"{X.shape[2]} samples/trial"
    )

    clf = make_fbcsp_lda(n_components=4)
    cv = LeaveOneOut()

    print(f"Running FBCSP + LDA LOOCV for {subject_code}...")

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
    print("FBCSP + LDA LOOCV: All Subjects")
    print("========================================")

    results = []

    for subject in SUBJECTS:
        results.append(run_subject(subject))

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