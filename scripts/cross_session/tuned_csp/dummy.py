from __future__ import annotations

import csv
from pathlib import Path

import mne
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)

from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.data.preprocessing import preprocess_raw
from bci_wheelchair.models import make_tuned_csp_dummy


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
    "results/cross_session/tuned_csp_dummy"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "tuned_csp_dummy_cross_session_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "tuned_csp_dummy_cross_session_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "tuned_csp_dummy_cross_session_overall_summary.csv"
)


def preprocess_evaluation_raw(
    raw: mne.io.BaseRaw,
) -> np.ndarray:
    """Preprocess one AxxE evaluation-session EEG file."""

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
    """Load official AxxE labels."""

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
    """Save rows to CSV."""

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


def load_pooled_training_data() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Load and concatenate A01T-A09T."""

    X_all = []
    y_all = []

    print()
    print("Loading pooled training sessions")
    print("=" * 60)

    for subject in SUBJECTS:

        train_file = Path(
            f"data/raw/{subject}T.gdf"
        )

        if not train_file.exists():
            raise FileNotFoundError(
                train_file
            )

        raw_train = load_raw_gdf(
            str(train_file)
        )

        X_subject, y_subject = preprocess_raw(
            raw_train,
            fmin=FMIN,
            fmax=FMAX,
            tmin=TMIN,
            tmax=TMAX,
        )

        print(
            f"{subject}T: "
            f"{len(y_subject)} trials"
        )

        X_all.append(
            np.asarray(X_subject)
        )

        y_all.append(
            np.asarray(y_subject)
        )

    X_train = np.concatenate(
        X_all,
        axis=0,
    )

    y_train = np.concatenate(
        y_all,
        axis=0,
    )

    print("-" * 60)

    print(
        "Total pooled training trials: "
        f"{len(y_train)}"
    )

    print(
        "Pooled training shape: "
        f"{X_train.shape}"
    )

    return X_train, y_train


def calculate_subject_result(
    subject: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_training_trials: int,
) -> dict[str, object]:
    """Calculate metrics for one evaluation session."""

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

    balanced_accuracy = recall_score(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
    )

    result = {
        "subject": subject,
        "method": "Tuned CSP + Dummy",
        "evaluation": (
            "cross_session_pooled_training_all_evaluation_sessions"
        ),
        "training_sessions": "A01T-A09T",
        "test_session": f"{subject}E",
        "fmin": FMIN,
        "fmax": FMAX,
        "tmin": TMIN,
        "tmax": TMAX,
        "csp_components": N_COMPONENTS,
        "accuracy": float(accuracy),
        "accuracy_percent": float(
            accuracy * 100.0
        ),
        "kappa": float(kappa),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "balanced_accuracy_percent": float(
            balanced_accuracy * 100.0
        ),
        "left_hand_recall": float(
            recalls[0]
        ),
        "right_hand_recall": float(
            recalls[1]
        ),
        "feet_recall": float(
            recalls[2]
        ),
        "tongue_recall": float(
            recalls[3]
        ),
        "n_training_trials": (
            n_training_trials
        ),
        "n_test_trials": len(y_true),
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
                "training_sessions": (
                    "A01T-A09T"
                ),
                "test_session": (
                    f"{subject}E"
                ),
                "true_label": true_label,
                "predicted_label": (
                    predicted_label
                ),
                "correct": int(
                    true_label
                    == predicted_label
                ),
                "model": "Tuned_CSP_DUMMY",
                "fmin": FMIN,
                "fmax": FMAX,
                "csp_components": (
                    N_COMPONENTS
                ),
            }
        )

    return rows


def create_overall_summary(
    subject_results: list[
        dict[str, object]
    ],
) -> dict[str, object]:
    """Calculate overall mean across A01E-A09E."""

    accuracies = np.asarray(
        [
            float(
                result["accuracy"]
            )
            for result in subject_results
        ]
    )

    kappas = np.asarray(
        [
            float(
                result["kappa"]
            )
            for result in subject_results
        ]
    )

    balanced_accuracies = np.asarray(
        [
            float(
                result[
                    "balanced_accuracy"
                ]
            )
            for result in subject_results
        ]
    )

    return {
        "method": "Tuned CSP + Dummy",
        "evaluation": (
            "cross_session_pooled_training_all_evaluation_sessions"
        ),
        "training_sessions": (
            "A01T-A09T"
        ),
        "test_sessions": (
            "A01E-A09E"
        ),
        "subjects": len(
            subject_results
        ),
        "fmin": FMIN,
        "fmax": FMAX,
        "tmin": TMIN,
        "tmax": TMAX,
        "csp_components": N_COMPONENTS,
        "mean_accuracy": float(
            np.mean(accuracies)
        ),
        "mean_accuracy_percent": float(
            np.mean(accuracies)
            * 100.0
        ),
        "std_accuracy_percent": float(
            np.std(accuracies)
            * 100.0
        ),
        "mean_kappa": float(
            np.mean(kappas)
        ),
        "std_kappa": float(
            np.std(kappas)
        ),
        "mean_balanced_accuracy": float(
            np.mean(
                balanced_accuracies
            )
        ),
        "mean_balanced_accuracy_percent": float(
            np.mean(
                balanced_accuracies
            )
            * 100.0
        ),
        "minimum_accuracy_percent": float(
            np.min(accuracies)
            * 100.0
        ),
        "maximum_accuracy_percent": float(
            np.max(accuracies)
            * 100.0
        ),
    }


def main() -> None:
    """Run supervisor Method 1 Tuned CSP + Dummy."""

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 78)

    print(
        "CROSS-SESSION: TUNED CSP + LDA"
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

    print(
        f"Band: "
        f"{FMIN:.0f}-{FMAX:.0f} Hz"
    )

    print(
        f"CSP components: "
        f"{N_COMPONENTS}"
    )


    # ---------------------------------------------------------
    # 1. LOAD AND POOL A01T-A09T
    # ---------------------------------------------------------

    X_train, y_train = (
        load_pooled_training_data()
    )


    # ---------------------------------------------------------
    # 2. TRAIN EXACTLY ONE MODEL
    # ---------------------------------------------------------

    print()

    print(
        "Training one pooled "
        "Tuned CSP + Dummy model..."
    )

    classifier = make_tuned_csp_dummy()

    classifier.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )


    # ---------------------------------------------------------
    # 3. TEST SAME MODEL ON A01E-A09E
    # ---------------------------------------------------------

    subject_results = []
    prediction_rows = []

    for subject in SUBJECTS:

        print()
        print("=" * 60)

        print(
            f"Testing pooled model "
            f"on {subject}E"
        )

        print("=" * 60)

        test_file = Path(
            f"data/raw/{subject}E.gdf"
        )

        label_file = Path(
            f"data/labels/{subject}E.mat"
        )

        if not test_file.exists():
            raise FileNotFoundError(
                test_file
            )

        if not label_file.exists():
            raise FileNotFoundError(
                label_file
            )

        raw_test = load_raw_gdf(
            str(test_file)
        )

        X_test = (
            preprocess_evaluation_raw(
                raw_test
            )
        )

        y_test = (
            load_evaluation_labels(
                label_file
            )
        )

        if len(X_test) != len(
            y_test
        ):
            raise ValueError(
                f"{subject}: "
                f"{len(X_test)} EEG trials "
                "!= "
                f"{len(y_test)} labels"
            )

        if (
            X_train.shape[1:]
            != X_test.shape[1:]
        ):
            raise ValueError(
                f"{subject}: "
                "training/test EEG shape "
                "mismatch: "
                f"{X_train.shape[1:]} "
                "vs "
                f"{X_test.shape[1:]}"
            )

        y_pred = classifier.predict(
            X_test
        )

        result = (
            calculate_subject_result(
                subject=subject,
                y_true=y_test,
                y_pred=y_pred,
                n_training_trials=(
                    len(y_train)
                ),
            )
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
            f"{subject}E Accuracy: "
            f"{result['accuracy_percent']:.2f}%"
        )

        print(
            f"{subject}E Kappa:    "
            f"{result['kappa']:.3f}"
        )


    # ---------------------------------------------------------
    # 4. OVERALL SUMMARY
    # ---------------------------------------------------------

    overall_summary = (
        create_overall_summary(
            subject_results
        )
    )


    # ---------------------------------------------------------
    # 5. SAVE RESULTS
    # ---------------------------------------------------------

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


    # ---------------------------------------------------------
    # 6. FINAL OUTPUT
    # ---------------------------------------------------------

    print()
    print("=" * 78)

    print(
        "FINAL TUNED CSP + LDA "
        "CROSS-SESSION RESULTS"
    )

    print("=" * 78)

    print(
        f"{'Test':<10}"
        f"{'Accuracy':<15}"
        f"{'Kappa':<12}"
    )

    print("-" * 40)

    for result in subject_results:

        print(
            f"{result['test_session']:<10}"
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
        "Accuracy SD: "
        f"{overall_summary['std_accuracy_percent']:.2f}%"
    )

    print(
        "Kappa SD: "
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
