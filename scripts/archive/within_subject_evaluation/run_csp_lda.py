"""
Run CSP + LDA using Leave-One-Out Cross Validation (LOOCV)
for all BCI Competition IV Dataset 2a training subjects.

The script loads reusable preprocessed EEG data instead of
preprocessing the raw GDF files during every run.

Run:
    python -m scripts.within_subject.run_csp_lda
"""

import warnings

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline

from bci_wheelchair.data.processed_loading import load_processed_subject


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

PREPROCESSING_CONFIG = "8-30"


def build_csp_lda() -> Pipeline:
    """Create the CSP and LDA classification pipeline."""
    return Pipeline(
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


def run_subject(subject_code: str) -> dict:
    """Run LOOCV for one subject."""
    print(f"\nLoading processed data for {subject_code}...")

    X, y = load_processed_subject(
        subject=subject_code,
        config=PREPROCESSING_CONFIG,
    )

    print(
        f"Loaded {X.shape[0]} trials, "
        f"{X.shape[1]} EEG channels, "
        f"{X.shape[2]} samples/trial"
    )

    classifier = build_csp_lda()
    cross_validation = LeaveOneOut()

    print(f"Running LOOCV for {subject_code}...")

    y_pred = cross_val_predict(
        classifier,
        X,
        y,
        cv=cross_validation,
        n_jobs=-1,
    )

    accuracy = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)
    matrix = confusion_matrix(y, y_pred)

    print(
        f"{subject_code} Accuracy: "
        f"{accuracy:.3f} ({accuracy * 100:.1f}%)"
    )
    print(f"{subject_code} Kappa:    {kappa:.3f}")

    return {
        "subject": subject_code,
        "accuracy": accuracy,
        "kappa": kappa,
        "confusion_matrix": matrix,
    }


def print_results(results: list[dict]) -> None:
    """Print the final subject results and confusion matrices."""
    accuracies = [result["accuracy"] for result in results]
    kappas = [result["kappa"] for result in results]

    print("\n========================================")
    print("Final Subject-wise Results")
    print("========================================")
    print(f"{'Subject':<10}{'Accuracy':<15}{'Kappa':<10}")
    print("----------------------------------------")

    for result in results:
        print(
            f"{result['subject']:<10}"
            f"{result['accuracy'] * 100:<15.1f}"
            f"{result['kappa']:<10.3f}"
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

    for result in results:
        print(f"\n{result['subject']}")
        print(result["confusion_matrix"])


def main() -> None:
    """Run CSP and LDA LOOCV for all training subjects."""
    print("\n========================================")
    print("CSP + LDA LOOCV: All Subjects")
    print("========================================")
    print(f"Processed data: {PREPROCESSING_CONFIG} Hz configuration")

    results = []

    for subject in SUBJECTS:
        result = run_subject(subject)
        results.append(result)

    print_results(results)


if __name__ == "__main__":
    main()
