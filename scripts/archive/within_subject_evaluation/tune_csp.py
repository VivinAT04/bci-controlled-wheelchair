"""
Tune CSP + LDA baseline using LOOCV.

Tests:
- different frequency bands
- different CSP component numbers

Run:
    python -m scripts.within_subject.tune_csp
"""

import warnings
import mne
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, cohen_kappa_score

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

FREQUENCY_BANDS = [
    (4, 40),
    (8, 30),
    (8, 35),
    (7, 30),
    (10, 30),
]

CSP_COMPONENTS = [2, 4, 6, 8, 10]


def build_classifier(n_components):
    return Pipeline([
        (
            "csp",
            CSP(
                n_components=n_components,
                reg=None,
                log=True,
                rank={"eeg": 22},
            ),
        ),
        ("lda", LDA()),
    ])


def evaluate_subject(subject_code, fmin, fmax, n_components):
    raw = load_raw_gdf(f"data/raw/{subject_code}.gdf")
    X, y = preprocess_raw(raw, fmin=fmin, fmax=fmax)

    clf = build_classifier(n_components)
    cv = LeaveOneOut()

    y_pred = cross_val_predict(
        clf,
        X,
        y,
        cv=cv,
        n_jobs=-1,
    )

    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)

    return acc, kappa


def main():
    print("\n========================================")
    print("Tuning CSP + LDA Baseline")
    print("========================================")

    all_results = []

    for fmin, fmax in FREQUENCY_BANDS:
        for n_components in CSP_COMPONENTS:
            print(
                f"\nTesting band {fmin}-{fmax} Hz, "
                f"CSP components={n_components}"
            )

            subject_accs = []
            subject_kappas = []

            for subject in SUBJECTS:
                acc, kappa = evaluate_subject(
                    subject,
                    fmin,
                    fmax,
                    n_components,
                )

                subject_accs.append(acc)
                subject_kappas.append(kappa)

                print(
                    f"{subject}: "
                    f"Accuracy={acc * 100:.1f}%, "
                    f"Kappa={kappa:.3f}"
                )

            mean_acc = np.mean(subject_accs)
            mean_kappa = np.mean(subject_kappas)

            result = {
                "band": f"{fmin}-{fmax}",
                "fmin": fmin,
                "fmax": fmax,
                "components": n_components,
                "mean_accuracy": mean_acc,
                "mean_kappa": mean_kappa,
            }

            all_results.append(result)

            print(
                f"Mean: Accuracy={mean_acc * 100:.1f}%, "
                f"Kappa={mean_kappa:.3f}"
            )

    print("\n========================================")
    print("Final Tuning Summary")
    print("========================================")
    print(f"{'Band':<12}{'Components':<15}{'Mean Acc':<15}{'Mean Kappa'}")
    print("-" * 60)

    all_results = sorted(
        all_results,
        key=lambda x: x["mean_accuracy"],
        reverse=True,
    )

    for r in all_results:
        print(
            f"{r['band']:<12}"
            f"{r['components']:<15}"
            f"{r['mean_accuracy'] * 100:<15.1f}"
            f"{r['mean_kappa']:.3f}"
        )

    best = all_results[0]

    print("\n========================================")
    print("Best Configuration")
    print("========================================")
    print(f"Band: {best['band']} Hz")
    print(f"CSP components: {best['components']}")
    print(f"Mean Accuracy: {best['mean_accuracy'] * 100:.1f}%")
    print(f"Mean Kappa: {best['mean_kappa']:.3f}")


if __name__ == "__main__":
    main()