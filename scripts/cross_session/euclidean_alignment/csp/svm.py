"""
Cross-session EA + CSP + RBF-SVM.

Supervisor Method 1:
    Train once:
        A01T + A02T + ... + A09T

    Test same fitted model:
        A01E, A02E, ..., A09E

Each T/E session is independently Euclidean-aligned.

Pipeline:
    Euclidean Alignment
    -> CSP
    -> RBF-SVM
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
)

from bci_wheelchair.models.euclidean_alignment import (
    load_and_align_subject,
    make_ea_csp_svm,
)


SUBJECTS = [
    "A01",
    "A02",
    "A03",
    "A04",
    "A05",
    "A06",
    "A07",
    "A08",
    "A09",
]

RESULTS_DIRECTORY = Path(
    "results/cross_session/euclidean_alignment/csp/svm"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "ea_csp_svm_cross_session_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "ea_csp_svm_cross_session_predictions.csv"
)

SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "ea_csp_svm_cross_session_overall_summary.csv"
)


def save_csv(path, rows):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    start_time = time.perf_counter()

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 78)
    print(
        "CROSS-SESSION: "
        "EA + CSP + RBF-SVM"
    )
    print("=" * 78)

    print(
        "Train once: "
        "A01T + A02T + ... + A09T"
    )

    print(
        "Test same model: "
        "A01E + A02E + ... + A09E"
    )

    print()
    print(
        "Pipeline: EA -> CSP "
        "-> RBF-SVM"
    )


    # =========================================================
    # LOAD + ALIGN TRAINING SESSIONS
    # =========================================================

    print()
    print("=" * 78)
    print("LOADING POOLED TRAINING DATA")
    print("=" * 78)

    X_train_parts = []
    y_train_parts = []

    for subject in SUBJECTS:

        session = f"{subject}T"

        X, y, alignment_error = (
            load_and_align_subject(
                session
            )
        )

        X = np.asarray(X)
        y = np.asarray(y)

        X_train_parts.append(X)
        y_train_parts.append(y)

        print(
            f"{session}: "
            f"{len(y)} trials | "
            f"EA error="
            f"{alignment_error:.8f}"
        )


    X_train = np.concatenate(
        X_train_parts,
        axis=0,
    )

    y_train = np.concatenate(
        y_train_parts,
        axis=0,
    )

    print("-" * 78)

    print(
        f"Total pooled training trials: "
        f"{len(y_train)}"
    )

    print(
        f"Pooled training shape: "
        f"{X_train.shape}"
    )


    # =========================================================
    # TRAIN ONE MODEL
    # =========================================================

    print()
    print("=" * 78)
    print(
        "TRAINING ONE POOLED "
        "EA + CSP + RBF-SVM MODEL"
    )
    print("=" * 78)

    classifier = (
        make_ea_csp_svm()
    )

    training_start = time.perf_counter()

    classifier.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    print(
        f"Training complete: "
        f"{training_time:.2f}s"
    )

    if "pca" in classifier.named_steps:
        pca_components = int(
            classifier.named_steps[
                "pca"
            ].n_components_
        )
        print(
            f"PCA components retained: "
            f"{pca_components}"
        )
    else:
        pca_components = None


    # =========================================================
    # TEST A01E-A09E
    # =========================================================

    subject_rows = []
    prediction_rows = []

    all_true = []
    all_pred = []

    for subject in SUBJECTS:

        session = f"{subject}E"

        print()
        print("=" * 78)
        print(
            f"Testing pooled model on "
            f"{session}"
        )
        print("=" * 78)

        X_test, y_test, alignment_error = (
            load_and_align_subject(
                session
            )
        )

        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        y_pred = classifier.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        kappa = cohen_kappa_score(
            y_test,
            y_pred,
        )

        print(
            f"{session} Accuracy: "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"{session} Kappa:    "
            f"{kappa:.3f}"
        )

        subject_rows.append(
            {
                "subject": subject,
                "test_session": session,
                "training_sessions": (
                    "A01T-A09T"
                ),
                "accuracy": float(
                    accuracy
                ),
                "accuracy_percent": float(
                    accuracy * 100
                ),
                "kappa": float(kappa),
                "alignment_error": float(
                    alignment_error
                ),
                "n_test_trials": len(
                    y_test
                ),
            }
        )

        for trial_index, (
            true_label,
            predicted_label,
        ) in enumerate(
            zip(
                y_test,
                y_pred,
            ),
            start=1,
        ):

            prediction_rows.append(
                {
                    "subject": subject,
                    "test_session": session,
                    "trial": trial_index,
                    "true_label": (
                        true_label
                    ),
                    "predicted_label": (
                        predicted_label
                    ),
                    "correct": int(
                        true_label
                        == predicted_label
                    ),
                }
            )

        all_true.append(y_test)
        all_pred.append(y_pred)


    # =========================================================
    # FINAL METRICS
    # =========================================================

    accuracies = np.asarray(
        [
            row["accuracy"]
            for row in subject_rows
        ]
    )

    kappas = np.asarray(
        [
            row["kappa"]
            for row in subject_rows
        ]
    )

    all_true = np.concatenate(
        all_true
    )

    all_pred = np.concatenate(
        all_pred
    )

    pooled_accuracy = accuracy_score(
        all_true,
        all_pred,
    )

    pooled_kappa = cohen_kappa_score(
        all_true,
        all_pred,
    )

    mean_accuracy = float(
        np.mean(accuracies)
    )

    mean_kappa = float(
        np.mean(kappas)
    )

    accuracy_sd = float(
        np.std(
            accuracies,
            ddof=1,
        )
    )

    kappa_sd = float(
        np.std(
            kappas,
            ddof=1,
        )
    )

    duration = (
        time.perf_counter()
        - start_time
    )


    # =========================================================
    # SAVE RESULTS
    # =========================================================

    summary = {
        "evaluation": "Cross-Session",
        "method": (
            "EA + CSP + "
            "RBF-SVM"
        ),
        "training_sessions": (
            "A01T-A09T"
        ),
        "test_sessions": (
            "A01E-A09E"
        ),
        "mean_accuracy": (
            mean_accuracy
        ),
        "mean_accuracy_percent": (
            mean_accuracy * 100
        ),
        "mean_kappa": mean_kappa,
        "accuracy_sd": accuracy_sd,
        "accuracy_sd_percent": (
            accuracy_sd * 100
        ),
        "kappa_sd": kappa_sd,
        "pooled_accuracy": (
            pooled_accuracy
        ),
        "pooled_accuracy_percent": (
            pooled_accuracy * 100
        ),
        "pooled_kappa": (
            pooled_kappa
        ),
        "training_trials": len(
            y_train
        ),
        "test_trials": len(
            all_true
        ),
        "pca_components": (
            pca_components
        ),
        "duration_seconds": (
            duration
        ),
    }

    save_csv(
        SUBJECT_RESULTS_PATH,
        subject_rows,
    )

    save_csv(
        PREDICTIONS_PATH,
        prediction_rows,
    )

    save_csv(
        SUMMARY_PATH,
        [summary],
    )


    # =========================================================
    # PRINT FINAL TABLE
    # =========================================================

    print()
    print("=" * 78)
    print(
        "FINAL EA + CSP + RBF-SVM "
        "CROSS-SESSION RESULTS"
    )
    print("=" * 78)

    print(
        f"{'Test':<10}"
        f"{'Accuracy':>12}"
        f"{'Kappa':>12}"
    )

    print("-" * 34)

    for row in subject_rows:

        print(
            f"{row['test_session']:<10}"
            f"{row['accuracy_percent']:>11.2f}%"
            f"{row['kappa']:>12.3f}"
        )

    print("-" * 34)

    print(
        f"{'Mean':<10}"
        f"{mean_accuracy * 100:>11.2f}%"
        f"{mean_kappa:>12.3f}"
    )

    print()
    print(
        f"Accuracy SD: "
        f"{accuracy_sd * 100:.2f}%"
    )

    print(
        f"Kappa SD:    "
        f"{kappa_sd:.3f}"
    )

    print()
    print(
        f"Pooled accuracy: "
        f"{pooled_accuracy * 100:.2f}%"
    )

    print(
        f"Pooled kappa:    "
        f"{pooled_kappa:.3f}"
    )

    print()
    print(
        f"Duration: "
        f"{duration:.2f}s"
    )

    print()
    print("Saved:")
    print(
        SUBJECT_RESULTS_PATH
    )
    print(
        PREDICTIONS_PATH
    )
    print(
        SUMMARY_PATH
    )


if __name__ == "__main__":
    main()
