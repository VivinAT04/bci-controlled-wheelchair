"""
Export wheelchair commands from pooled cross-session FBCSP + LDA predictions.

Protocol
--------
Train one FBCSP + LDA classifier using pooled training-session data:

    A01T + A02T + ... + A09T

Use the same trained classifier to generate predictions and wheelchair
commands independently for:

    A01E
    A02E
    ...
    A09E

Evaluation-session labels are used only after prediction to calculate
performance metrics.
"""

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
)

from bci_wheelchair.commands import CLASS_TO_COMMAND
from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.data.preprocessing import (
    LABEL_MAP,
    preprocess_raw,
)
from bci_wheelchair.models import make_fbcsp_lda


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

TEST_EVENT_ID = {
    "783": 1,
}

FMIN = 4.0
FMAX = 40.0
TMIN = 0.5
TMAX = 2.5

N_COMPONENTS = 4


RESULTS_DIRECTORY = Path(
    "results/cross_session/predictions"
)

COMMANDS_PATH = (
    RESULTS_DIRECTORY
    / "fbcsp_lda_cross_session_predicted_commands.csv"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "fbcsp_lda_cross_session_command_subject_results.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "fbcsp_lda_cross_session_command_overall_summary.csv"
)


def preprocess_evaluation_raw(
    raw: mne.io.BaseRaw,
) -> np.ndarray:
    """Extract EEG epochs from one AxxE evaluation file."""

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
        event_id=TEST_EVENT_ID,
        verbose=False,
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=TEST_EVENT_ID,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs.get_data()


def load_evaluation_labels(
    path: Path,
) -> np.ndarray:
    """Load official evaluation labels from MAT."""

    mat_data = loadmat(
        path
    )

    if "classlabel" not in mat_data:

        available_keys = [
            key
            for key in mat_data.keys()
            if not key.startswith("__")
        ]

        raise KeyError(
            "Could not find 'classlabel' in "
            f"{path}. Available keys: "
            f"{available_keys}"
        )

    numeric_labels = np.asarray(
        mat_data["classlabel"]
    ).reshape(-1).astype(int)

    unknown_labels = sorted(
        set(numeric_labels)
        - set(LABEL_MAP.keys())
    )

    if unknown_labels:
        raise ValueError(
            f"Unexpected labels in "
            f"{path}: {unknown_labels}"
        )

    return np.asarray(
        [
            LABEL_MAP[label]
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
                fieldnames.append(
                    key
                )

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
        writer.writerows(
            rows
        )


def load_pooled_training_data() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Load and concatenate A01T-A09T."""

    X_blocks = []
    y_blocks = []

    print()
    print("=" * 72)
    print(
        "LOADING POOLED TRAINING DATA"
    )
    print("=" * 72)

    for subject in SUBJECTS:

        train_path = Path(
            f"data/raw/{subject}T.gdf"
        )

        if not train_path.exists():
            raise FileNotFoundError(
                train_path
            )

        raw_train = load_raw_gdf(
            str(train_path)
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

        X_blocks.append(
            np.asarray(
                X_subject
            )
        )

        y_blocks.append(
            np.asarray(
                y_subject
            )
        )

    X_train = np.concatenate(
        X_blocks,
        axis=0,
    )

    y_train = np.concatenate(
        y_blocks,
        axis=0,
    )

    print("-" * 72)

    print(
        "Total pooled training trials: "
        f"{len(y_train)}"
    )

    print(
        "Pooled training shape: "
        f"{X_train.shape}"
    )

    return (
        X_train,
        y_train,
    )


def main() -> None:
    """Export A01E-A09E wheelchair commands."""

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)

    print(
        "CROSS-SESSION WHEELCHAIR "
        "COMMAND EXPORT"
    )

    print("=" * 72)

    print(
        "Train once: "
        "A01T + A02T + ... + A09T"
    )

    print(
        "Test same model: "
        "A01E + A02E + ... + A09E"
    )

    print(
        "Model: FBCSP + LDA"
    )


    # ---------------------------------------------------------
    # 1. POOLED TRAINING DATA
    # ---------------------------------------------------------

    X_train, y_train = (
        load_pooled_training_data()
    )


    # ---------------------------------------------------------
    # 2. TRAIN ONE FBCSP + LDA MODEL
    # ---------------------------------------------------------

    print()
    print(
        "Training one pooled "
        "FBCSP + LDA classifier..."
    )

    classifier = make_fbcsp_lda(
        n_components=N_COMPONENTS,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )


    # ---------------------------------------------------------
    # 3. TEST A01E-A09E AND EXPORT COMMANDS
    # ---------------------------------------------------------

    command_rows = []
    subject_results = []

    for subject in SUBJECTS:

        test_path = Path(
            f"data/raw/{subject}E.gdf"
        )

        label_path = Path(
            f"data/labels/{subject}E.mat"
        )

        if not test_path.exists():
            raise FileNotFoundError(
                test_path
            )

        if not label_path.exists():
            raise FileNotFoundError(
                label_path
            )

        print()
        print("=" * 72)

        print(
            f"Testing same model on "
            f"{subject}E"
        )

        print("=" * 72)

        raw_test = load_raw_gdf(
            str(test_path)
        )

        X_test = (
            preprocess_evaluation_raw(
                raw_test
            )
        )

        y_test = (
            load_evaluation_labels(
                label_path
            )
        )

        if len(X_test) != len(
            y_test
        ):
            raise ValueError(
                f"{subject}E: "
                f"{len(X_test)} EEG trials "
                "!= "
                f"{len(y_test)} labels"
            )

        if (
            X_train.shape[1:]
            != X_test.shape[1:]
        ):
            raise ValueError(
                f"{subject}E: "
                "training/test EEG shape "
                "mismatch: "
                f"{X_train.shape[1:]} "
                "vs "
                f"{X_test.shape[1:]}"
            )

        y_pred = classifier.predict(
            X_test
        )

        probabilities = (
            classifier.predict_proba(
                X_test
            )
        )

        confidence = (
            probabilities.max(
                axis=1
            )
        )

        accuracy = accuracy_score(
            y_test,
            y_pred,
        )

        kappa = cohen_kappa_score(
            y_test,
            y_pred,
        )

        matrix = confusion_matrix(
            y_test,
            y_pred,
            labels=CLASS_ORDER,
        )

        subject_results.append(
            {
                "subject": subject,
                "training_sessions": (
                    "A01T-A09T"
                ),
                "test_session": (
                    f"{subject}E"
                ),
                "model": (
                    "FBCSP_LDA"
                ),
                "accuracy": float(
                    accuracy
                ),
                "accuracy_percent": float(
                    accuracy * 100.0
                ),
                "kappa": float(
                    kappa
                ),
                "n_training_trials": (
                    len(y_train)
                ),
                "n_test_trials": (
                    len(y_test)
                ),
                "cm_left_hand_pred_left_hand": int(
                    matrix[0, 0]
                ),
                "cm_left_hand_pred_right_hand": int(
                    matrix[0, 1]
                ),
                "cm_left_hand_pred_feet": int(
                    matrix[0, 2]
                ),
                "cm_left_hand_pred_tongue": int(
                    matrix[0, 3]
                ),
                "cm_right_hand_pred_left_hand": int(
                    matrix[1, 0]
                ),
                "cm_right_hand_pred_right_hand": int(
                    matrix[1, 1]
                ),
                "cm_right_hand_pred_feet": int(
                    matrix[1, 2]
                ),
                "cm_right_hand_pred_tongue": int(
                    matrix[1, 3]
                ),
                "cm_feet_pred_left_hand": int(
                    matrix[2, 0]
                ),
                "cm_feet_pred_right_hand": int(
                    matrix[2, 1]
                ),
                "cm_feet_pred_feet": int(
                    matrix[2, 2]
                ),
                "cm_feet_pred_tongue": int(
                    matrix[2, 3]
                ),
                "cm_tongue_pred_left_hand": int(
                    matrix[3, 0]
                ),
                "cm_tongue_pred_right_hand": int(
                    matrix[3, 1]
                ),
                "cm_tongue_pred_feet": int(
                    matrix[3, 2]
                ),
                "cm_tongue_pred_tongue": int(
                    matrix[3, 3]
                ),
            }
        )

        for trial_number, (
            true_class,
            predicted_class,
            trial_confidence,
        ) in enumerate(
            zip(
                y_test,
                y_pred,
                confidence,
            ),
            start=1,
        ):

            command_rows.append(
                {
                    "subject": subject,
                    "trial": trial_number,
                    "true_class": (
                        true_class
                    ),
                    "predicted_class": (
                        predicted_class
                    ),
                    "command": (
                        CLASS_TO_COMMAND[
                            predicted_class
                        ]
                    ),
                    "confidence": float(
                        trial_confidence
                    ),
                    "correct": int(
                        true_class
                        == predicted_class
                    ),
                    "model": (
                        "FBCSP_LDA"
                    ),
                    "training_dataset": (
                        "A01T-A09T"
                    ),
                    "testing_dataset": (
                        f"{subject}E"
                    ),
                    "data_split": (
                        "cross_session_evaluation"
                    ),
                }
            )

        print(
            f"{subject}E Accuracy: "
            f"{accuracy * 100.0:.2f}%"
        )

        print(
            f"{subject}E Kappa: "
            f"{kappa:.3f}"
        )


    # ---------------------------------------------------------
    # 4. OVERALL SUMMARY
    # ---------------------------------------------------------

    accuracies = np.asarray(
        [
            float(
                row["accuracy"]
            )
            for row in subject_results
        ]
    )

    kappas = np.asarray(
        [
            float(
                row["kappa"]
            )
            for row in subject_results
        ]
    )

    overall_summary = {
        "model": "FBCSP_LDA",
        "training_sessions": (
            "A01T-A09T"
        ),
        "test_sessions": (
            "A01E-A09E"
        ),
        "subjects": len(
            SUBJECTS
        ),
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
        "n_training_trials": (
            len(y_train)
        ),
        "n_exported_commands": (
            len(command_rows)
        ),
    }


    # ---------------------------------------------------------
    # 5. SAVE
    # ---------------------------------------------------------

    save_csv(
        COMMANDS_PATH,
        command_rows,
    )

    save_csv(
        SUBJECT_RESULTS_PATH,
        subject_results,
    )

    save_csv(
        OVERALL_SUMMARY_PATH,
        [overall_summary],
    )


    # ---------------------------------------------------------
    # 6. FINAL OUTPUT
    # ---------------------------------------------------------

    print()
    print("=" * 72)

    print(
        "COMMAND EXPORT COMPLETE"
    )

    print("=" * 72)

    print(
        "Mean accuracy: "
        f"{overall_summary['mean_accuracy_percent']:.2f}%"
    )

    print(
        "Mean kappa: "
        f"{overall_summary['mean_kappa']:.3f}"
    )

    print(
        "Commands exported: "
        f"{overall_summary['n_exported_commands']}"
    )

    print()

    print("Saved:")
    print(COMMANDS_PATH)
    print(SUBJECT_RESULTS_PATH)
    print(OVERALL_SUMMARY_PATH)


if __name__ == "__main__":
    main()
