"""
Within-subject Autoencoder + RBF-SVM evaluation.

For each subject:
    EEG trials
    -> train autoencoder using training folds only
    -> extract frozen latent features
    -> train RBF-SVM
    -> predict held-out trials

Evaluation:
    5-fold stratified cross-validation independently per subject.

Important:
    The autoencoder is trained separately inside every CV fold.
    Held-out trials are never used for training or normalisation.

Run:
    python -m scripts.within_subject.run_autoencoder_rbf_svm
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from bci_wheelchair.data.processed_loading import load_processed_subject
from bci_wheelchair.representation import (
    AutoencoderConfig,
    create_classifier,
)
from bci_wheelchair.representation.training import (
    TrainingConfig,
    extract_latent_features,
    train_autoencoder,
)


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

CLASS_ORDER = [
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
]

PREPROCESSING = "8-30"

LATENT_DIM = 32
HIDDEN_DIM = 256

N_SPLITS = 5
RANDOM_STATE = 42

TRAINING_CONFIG = TrainingConfig(
    epochs=50,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=1e-5,
    validation_fraction=0.2,
    patience=10,
    random_state=RANDOM_STATE,
)

RESULTS_DIRECTORY = Path(
    "results/within_subject/autoencoder_rbf_svm"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "autoencoder_rbf_svm_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "autoencoder_rbf_svm_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "autoencoder_rbf_svm_overall_summary.csv"
)


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames: list[str] = []

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
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def evaluate_subject(
    subject: str,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:

    print()
    print("=" * 78)
    print(f"Running {subject}")
    print("=" * 78)

    X, y = load_processed_subject(
        subject=subject,
        config=PREPROCESSING,
    )

    print(
        f"Trials: {X.shape[0]}, "
        f"channels: {X.shape[1]}, "
        f"samples: {X.shape[2]}"
    )

    model_config = AutoencoderConfig(
        n_channels=X.shape[1],
        n_times=X.shape[2],
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    y_pred_all = np.empty(
        len(y),
        dtype=object,
    )

    prediction_rows: list[
        dict[str, object]
    ] = []

    for fold_index, (
        train_indices,
        test_indices,
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print()
        print(
            f"{subject} Fold {fold_index}/{N_SPLITS}"
        )

        X_train = X[train_indices]
        y_train = y[train_indices]

        X_test = X[test_indices]
        y_test = y[test_indices]

        print(
            f"Training trials: {len(y_train)}"
        )
        print(
            f"Testing trials:  {len(y_test)}"
        )

        autoencoder, standardizer, history = (
            train_autoencoder(
                X_train=X_train,
                model_config=model_config,
                training_config=TRAINING_CONFIG,
            )
        )

        train_features = extract_latent_features(
            model=autoencoder,
            standardizer=standardizer,
            X=X_train,
        )

        test_features = extract_latent_features(
            model=autoencoder,
            standardizer=standardizer,
            X=X_test,
        )

        classifier = create_classifier(
            "rbf_svm",
            random_state=RANDOM_STATE,
        )

        classifier.fit(
            train_features,
            y_train,
        )

        fold_predictions = classifier.predict(
            test_features
        )

        y_pred_all[
            test_indices
        ] = fold_predictions

        fold_accuracy = accuracy_score(
            y_test,
            fold_predictions,
        )

        print(
            f"Fold accuracy: "
            f"{fold_accuracy * 100.0:.2f}%"
        )

        for local_index, (
            dataset_index,
            true_label,
            predicted_label,
        ) in enumerate(
            zip(
                test_indices,
                y_test,
                fold_predictions,
            ),
            start=1,
        ):
            prediction_rows.append(
                {
                    "subject": subject,
                    "fold": fold_index,
                    "fold_trial": local_index,
                    "dataset_trial_index": int(
                        dataset_index
                    ),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "correct": (
                        true_label
                        == predicted_label
                    ),
                    "best_autoencoder_epoch": (
                        history.best_epoch
                    ),
                    "best_validation_loss": (
                        history.best_validation_loss
                    ),
                    "model": "Autoencoder_RBF_SVM",
                    "evaluation": (
                        "within_subject_5fold"
                    ),
                }
            )

    accuracy = accuracy_score(
        y,
        y_pred_all,
    )

    kappa = cohen_kappa_score(
        y,
        y_pred_all,
    )

    recalls = recall_score(
        y,
        y_pred_all,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        y_pred_all,
        labels=CLASS_ORDER,
    )

    result: dict[str, object] = {
        "subject": subject,
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "kappa": kappa,
        "left_hand_recall": recalls[0],
        "right_hand_recall": recalls[1],
        "feet_recall": recalls[2],
        "tongue_recall": recalls[3],
        "preprocessing": PREPROCESSING,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "cv_folds": N_SPLITS,
        "classifier": "rbf_svm",
        "evaluation": "within_subject_5fold",
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

    print()
    print(
        f"{subject} Accuracy: "
        f"{accuracy * 100.0:.2f}%"
    )

    print(
        f"{subject} Kappa:    "
        f"{kappa:.3f}"
    )

    return result, prediction_rows


def build_overall_summary(
    subject_results: list[dict[str, object]],
) -> dict[str, object]:

    accuracies = np.asarray(
        [
            float(result["accuracy"])
            for result in subject_results
        ]
    )

    kappas = np.asarray(
        [
            float(result["kappa"])
            for result in subject_results
        ]
    )

    return {
        "model": "Autoencoder_RBF_SVM",
        "evaluation": "within_subject_5fold",
        "subjects": len(subject_results),
        "preprocessing": PREPROCESSING,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "cv_folds": N_SPLITS,
        "classifier": "rbf_svm",
        "mean_accuracy": float(
            np.mean(accuracies)
        ),
        "mean_accuracy_percent": float(
            np.mean(accuracies) * 100.0
        ),
        "std_accuracy_percent": float(
            np.std(accuracies) * 100.0
        ),
        "mean_kappa": float(
            np.mean(kappas)
        ),
        "std_kappa": float(
            np.std(kappas)
        ),
        "minimum_accuracy_percent": float(
            np.min(accuracies) * 100.0
        ),
        "maximum_accuracy_percent": float(
            np.max(accuracies) * 100.0
        ),
    }


def main() -> None:

    print()
    print("=" * 78)
    print(
        "Within-Subject Autoencoder + RBF-SVM"
    )
    print("=" * 78)

    print(
        "Protocol: 5-fold stratified CV "
        "independently per subject"
    )
    print(
        "Autoencoder trained separately "
        "inside each CV fold"
    )
    print(
        f"Preprocessing: {PREPROCESSING} Hz"
    )
    print(
        f"Latent dimension: {LATENT_DIM}"
    )
    print(
        f"Hidden dimension: {HIDDEN_DIM}"
    )
    print(
        "Classifier: RBF-SVM"
    )

    subject_results = []
    prediction_rows = []

    for subject in SUBJECTS:

        result, subject_predictions = (
            evaluate_subject(subject)
        )

        subject_results.append(
            result
        )

        prediction_rows.extend(
            subject_predictions
        )

    overall_summary = build_overall_summary(
        subject_results
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
        "FINAL WITHIN-SUBJECT "
        "AUTOENCODER + RBF-SVM RESULTS"
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
    print(SUBJECT_RESULTS_PATH)
    print(PREDICTIONS_PATH)
    print(OVERALL_SUMMARY_PATH)


if __name__ == "__main__":
    main()
