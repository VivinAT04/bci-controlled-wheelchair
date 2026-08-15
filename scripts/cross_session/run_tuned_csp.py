"""
Cross-session evaluation using Tuned CSP + LDA.

Protocol:

    Train: AxxT
    Test:  AxxE

Tuned configuration:

    Band: 8-30 Hz
    CSP components: 10
    Classifier: LDA

Run from project root:

    python -m scripts.cross_session.run_tuned_csp
"""

from __future__ import annotations

import csv
from pathlib import Path

import mne
import numpy as np
from mne.decoding import CSP
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)
from sklearn.pipeline import Pipeline

from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.data.preprocessing import preprocess_raw


mne.set_log_level("ERROR")


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

CLASS_ORDER = [
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
]

LABEL_MAP = {
    1: "left_hand",
    2: "right_hand",
    3: "feet",
    4: "tongue",
}

FMIN = 8.0
FMAX = 30.0
TMIN = 0.5
TMAX = 2.5
N_COMPONENTS = 10

RESULTS_DIRECTORY = Path(
    "results/cross_session/tuned_csp"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "tuned_csp_cross_session_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "tuned_csp_cross_session_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "tuned_csp_cross_session_overall_summary.csv"
)


def make_tuned_csp_lda() -> Pipeline:
    """Create tuned CSP + LDA pipeline."""

    return Pipeline(
        [
            (
                "csp",
                CSP(
                    n_components=N_COMPONENTS,
                    reg=None,
                    log=True,
                    rank={"eeg": 22},
                ),
            ),
            (
                "lda",
                LDA(),
            ),
        ]
    )


def preprocess_evaluation_raw(raw) -> np.ndarray:
    """
    Preprocess evaluation-session GDF.

    Evaluation files contain generic event code 783.
    True class labels are stored separately in MAT files.
    """

    raw = raw.copy()

    raw.drop_channels(
        [
            "EOG-left",
            "EOG-central",
            "EOG-right",
        ],
        on_missing="ignore",
    )

    raw.filter(
        FMIN,
        FMAX,
        fir_design="firwin",
        verbose=False,
    )

    events, _ = mne.events_from_annotations(
        raw,
        event_id={"783": 1},
        verbose=False,
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"783": 1},
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs.get_data()


def load_evaluation_labels(
    label_file: Path,
) -> np.ndarray:
    """Load true AxxE labels from official MAT file."""

    mat_data = loadmat(
        label_file
    )

    numeric_labels = (
        mat_data["classlabel"]
        .reshape(-1)
    )

    return np.asarray(
        [
            LABEL_MAP[int(label)]
            for label in numeric_labels
        ]
    )


def save_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Save dictionary rows to CSV."""

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def create_subject_result(
    subject: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    """Calculate metrics for one subject."""

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    kappa = cohen_kappa_score(
        y_true,
        y_pred,
    )

    recalls = recall_score(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
    )

    result = {
        "subject": subject,
        "train_session": f"{subject}T",
        "test_session": f"{subject}E",
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "kappa": kappa,
        "left_hand_recall": recalls[0],
        "right_hand_recall": recalls[1],
        "feet_recall": recalls[2],
        "tongue_recall": recalls[3],
        "fmin": FMIN,
        "fmax": FMAX,
        "tmin": TMIN,
        "tmax": TMAX,
        "csp_components": N_COMPONENTS,
    }

    for true_index, true_label in enumerate(
        CLASS_ORDER
    ):
        for pred_index, pred_label in enumerate(
            CLASS_ORDER
        ):
            result[
                f"cm_{true_label}_pred_{pred_label}"
            ] = int(
                matrix[
                    true_index,
                    pred_index,
                ]
            )

    return result


def create_prediction_rows(
    subject: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, object]]:
    """Create trial-level prediction rows."""

    rows = []

    for trial_index, (
        true_label,
        predicted_label,
    ) in enumerate(
        zip(
            y_true,
            y_pred,
        ),
        start=1,
    ):

        rows.append(
            {
                "subject": subject,
                "trial": trial_index,
                "train_session": f"{subject}T",
                "test_session": f"{subject}E",
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": (
                    true_label
                    == predicted_label
                ),
                "model": "Tuned_CSP_LDA",
                "fmin": FMIN,
                "fmax": FMAX,
                "csp_components": N_COMPONENTS,
            }
        )

    return rows


def create_overall_summary(
    subject_results: list[dict[str, object]],
) -> dict[str, object]:
    """Create overall cross-session summary."""

    accuracies = np.asarray(
        [
            float(
                result["accuracy"]
            )
            for result
            in subject_results
        ]
    )

    kappas = np.asarray(
        [
            float(
                result["kappa"]
            )
            for result
            in subject_results
        ]
    )

    return {
        "model": "Tuned_CSP_LDA",
        "evaluation": "cross_session_AxxT_to_AxxE",
        "subjects": len(
            subject_results
        ),
        "fmin": FMIN,
        "fmax": FMAX,
        "tmin": TMIN,
        "tmax": TMAX,
        "csp_components": N_COMPONENTS,
        "mean_accuracy": float(
            np.mean(
                accuracies
            )
        ),
        "mean_accuracy_percent": float(
            np.mean(
                accuracies
            )
            * 100.0
        ),
        "std_accuracy_percent": float(
            np.std(
                accuracies
            )
            * 100.0
        ),
        "mean_kappa": float(
            np.mean(
                kappas
            )
        ),
        "std_kappa": float(
            np.std(
                kappas
            )
        ),
        "minimum_accuracy_percent": float(
            np.min(
                accuracies
            )
            * 100.0
        ),
        "maximum_accuracy_percent": float(
            np.max(
                accuracies
            )
            * 100.0
        ),
    }


def main() -> None:
    """Run tuned CSP cross-session evaluation."""

    print()
    print("=" * 78)
    print("Cross-Session Tuned CSP + LDA")
    print("=" * 78)

    print(
        "Protocol: AxxT -> AxxE"
    )

    print(
        f"Band: {FMIN:.0f}-{FMAX:.0f} Hz"
    )

    print(
        f"CSP components: {N_COMPONENTS}"
    )

    subject_results = []
    prediction_rows = []

    for subject in SUBJECTS:

        print()
        print("=" * 78)
        print(
            f"{subject}: "
            f"{subject}T -> {subject}E"
        )
        print("=" * 78)

        train_file = Path(
            f"data/raw/{subject}T.gdf"
        )

        test_file = Path(
            f"data/raw/{subject}E.gdf"
        )

        label_file = Path(
            f"data/labels/{subject}E.mat"
        )

        raw_train = load_raw_gdf(
            str(train_file)
        )

        raw_test = load_raw_gdf(
            str(test_file)
        )

        X_train, y_train = preprocess_raw(
            raw_train,
            fmin=FMIN,
            fmax=FMAX,
            tmin=TMIN,
            tmax=TMAX,
        )

        X_test = preprocess_evaluation_raw(
            raw_test
        )

        y_test = load_evaluation_labels(
            label_file
        )

        if len(X_test) != len(y_test):
            raise ValueError(
                f"{subject}: "
                f"{len(X_test)} test epochs "
                f"but {len(y_test)} labels."
            )

        print(
            f"Training trials: {len(y_train)}"
        )

        print(
            f"Testing trials:  {len(y_test)}"
        )

        classifier = make_tuned_csp_lda()

        classifier.fit(
            X_train,
            y_train,
        )

        y_pred = classifier.predict(
            X_test
        )

        result = create_subject_result(
            subject=subject,
            y_true=y_test,
            y_pred=y_pred,
        )

        subject_results.append(
            result
        )

        prediction_rows.extend(
            create_prediction_rows(
                subject=subject,
                y_true=y_test,
                y_pred=y_pred,
            )
        )

        print(
            f"{subject} Accuracy: "
            f"{result['accuracy_percent']:.2f}%"
        )

        print(
            f"{subject} Kappa:    "
            f"{result['kappa']:.3f}"
        )

    overall_summary = (
        create_overall_summary(
            subject_results
        )
    )

    save_csv(
        SUBJECT_RESULTS_PATH,
        subject_results,
    )

    save_csv(
        PREDICTIONS_PATH,
        prediction_rows,
    )

    save_csv(
        OVERALL_SUMMARY_PATH,
        [overall_summary],
    )

    print()
    print("=" * 78)
    print(
        "FINAL TUNED CSP + LDA "
        "CROSS-SESSION RESULTS"
    )
    print("=" * 78)

    print(
        f"{'Subject':<10}"
        f"{'Accuracy':<15}"
        f"{'Kappa':<12}"
    )

    print("-" * 40)

    for result in subject_results:
        print(
            f"{result['subject']:<10}"
            f"{result['accuracy_percent']:<15.2f}"
            f"{result['kappa']:<12.3f}"
        )

    print("-" * 40)

    print(
        f"{'Mean':<10}"
        f"{overall_summary['mean_accuracy_percent']:<15.2f}"
        f"{overall_summary['mean_kappa']:<12.3f}"
    )

    print()

    print(
        "Accuracy standard deviation: "
        f"{overall_summary['std_accuracy_percent']:.2f}%"
    )

    print(
        "Kappa standard deviation: "
        f"{overall_summary['std_kappa']:.3f}"
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
        OVERALL_SUMMARY_PATH
    )


if __name__ == "__main__":
    main()
