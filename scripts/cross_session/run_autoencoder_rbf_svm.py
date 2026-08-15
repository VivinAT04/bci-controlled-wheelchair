"""
Cross-session Autoencoder + RBF-SVM evaluation.

Protocol:
    Train autoencoder on AxxT only.
    Extract latent features from AxxT and AxxE.
    Train RBF-SVM on AxxT latent features.
    Test on AxxE latent features.

Important:
    AxxE is never used for:
        - autoencoder training
        - validation
        - EEG standardisation
        - classifier training

Run:
    python -m scripts.cross_session.run_autoencoder_rbf_svm
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

PREPROCESSING = "8-30"

LATENT_DIM = 32
HIDDEN_DIM = 256
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
    "results/cross_session/autoencoder_rbf_svm"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "autoencoder_rbf_svm_cross_session_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "autoencoder_rbf_svm_cross_session_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "autoencoder_rbf_svm_cross_session_overall_summary.csv"
)


def save_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Save dictionaries to CSV."""

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


def build_subject_result(
    subject: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    best_epoch: int,
    best_validation_loss: float,
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

    result: dict[str, object] = {
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
        "preprocessing": PREPROCESSING,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "classifier": "rbf_svm",
        "best_autoencoder_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "evaluation": "cross_session_AxxT_to_AxxE",
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


def build_prediction_rows(
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
        zip(y_true, y_pred),
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
                "model": "Autoencoder_RBF_SVM",
                "evaluation": (
                    "cross_session_AxxT_to_AxxE"
                ),
                "preprocessing": PREPROCESSING,
            }
        )

    return rows


def build_overall_summary(
    subject_results: list[dict[str, object]],
) -> dict[str, object]:
    """Calculate mean and standard deviation."""

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
        "evaluation": "cross_session_AxxT_to_AxxE",
        "subjects": len(subject_results),
        "preprocessing": PREPROCESSING,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
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
    """Run all cross-session autoencoder experiments."""

    print()
    print("=" * 78)
    print(
        "Cross-Session Autoencoder + RBF-SVM"
    )
    print("=" * 78)

    print("Protocol: AxxT -> AxxE")
    print(
        "Autoencoder training: AxxT only"
    )
    print(
        "Normalisation statistics: AxxT only"
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
    print("Classifier: RBF-SVM")

    subject_results = []
    prediction_rows = []

    for subject in SUBJECTS:

        train_subject = f"{subject}T"
        test_subject = f"{subject}E"

        print()
        print("=" * 78)
        print(
            f"{subject}: "
            f"{train_subject} -> {test_subject}"
        )
        print("=" * 78)

        X_train, y_train = load_processed_subject(
            subject=train_subject,
            config=PREPROCESSING,
        )

        X_test, y_test = load_processed_subject(
            subject=test_subject,
            config=PREPROCESSING,
        )

        print(
            f"Training trials: {len(y_train)}"
        )

        print(
            f"Testing trials:  {len(y_test)}"
        )

        model_config = AutoencoderConfig(
            n_channels=X_train.shape[1],
            n_times=X_train.shape[2],
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
        )

        (
            autoencoder,
            standardizer,
            history,
        ) = train_autoencoder(
            X_train=X_train,
            model_config=model_config,
            training_config=TRAINING_CONFIG,
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

        print(
            "Latent training shape: "
            f"{train_features.shape}"
        )

        print(
            "Latent testing shape:  "
            f"{test_features.shape}"
        )

        classifier = create_classifier(
            "rbf_svm",
            random_state=RANDOM_STATE,
        )

        classifier.fit(
            train_features,
            y_train,
        )

        y_pred = classifier.predict(
            test_features
        )

        result = build_subject_result(
            subject=subject,
            y_true=y_test,
            y_pred=y_pred,
            best_epoch=history.best_epoch,
            best_validation_loss=(
                history.best_validation_loss
            ),
        )

        subject_results.append(
            result
        )

        prediction_rows.extend(
            build_prediction_rows(
                subject=subject,
                y_true=y_test,
                y_pred=y_pred,
            )
        )

        print()
        print(
            f"{subject} Accuracy: "
            f"{result['accuracy_percent']:.2f}%"
        )

        print(
            f"{subject} Kappa:    "
            f"{result['kappa']:.3f}"
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
        "FINAL CROSS-SESSION "
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
