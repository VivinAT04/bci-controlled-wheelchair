"""
Strict cross-subject Supervised Autoencoder + RBF-SVM LOSO evaluation.

Protocol:
    Train on 8 subjects.
    Test on 1 completely unseen subject.

For every LOSO fold:
    training subjects
        -> supervised autoencoder
        -> training-only standardisation
        -> latent features
        -> RBF-SVM

    held-out subject
        -> same standardizer
        -> same frozen encoder
        -> RBF-SVM prediction

No held-out subject trials are used for:
    - supervised autoencoder training
    - validation
    - normalisation fitting
    - classifier training

Run:
    python -m scripts.cross_subject.run_supervised_autoencoder_rbf_svm
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

from bci_wheelchair.data.processed_loading import (
    load_processed_subject,
)

from bci_wheelchair.representation.classifiers import (
    create_classifier,
)

from bci_wheelchair.representation.supervised_autoencoder import (
    SupervisedAutoencoderConfig,
)

from bci_wheelchair.representation.supervised_training import (
    SupervisedTrainingConfig,
    extract_supervised_latent_features,
    train_supervised_autoencoder,
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
DROPOUT = 0.25

CLASSIFICATION_WEIGHT = 0.5

RANDOM_STATE = 42


TRAINING_CONFIG = SupervisedTrainingConfig(
    epochs=50,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=1e-5,
    validation_fraction=0.2,
    patience=10,
    classification_weight=CLASSIFICATION_WEIGHT,
    random_state=RANDOM_STATE,
)


RESULTS_DIRECTORY = Path(
    "results/cross_subject/"
    "supervised_autoencoder_rbf_svm"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "supervised_autoencoder_rbf_svm_loso_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "supervised_autoencoder_rbf_svm_loso_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "supervised_autoencoder_rbf_svm_loso_overall_summary.csv"
)


def save_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:

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


def load_all_subjects():
    subject_data = {}

    print("=" * 78)
    print("Loading all subjects")
    print("=" * 78)

    for subject in SUBJECTS:

        print(f"Loading {subject}...")

        X, y = load_processed_subject(
            subject=subject,
            config=PREPROCESSING,
        )

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        y = np.asarray(y)

        subject_data[subject] = {
            "X": X,
            "y": y,
        }

        print(
            f"  trials={X.shape[0]}, "
            f"channels={X.shape[1]}, "
            f"samples={X.shape[2]}"
        )

    return subject_data


def create_loso_fold(
    subject_data,
    test_subject,
):

    training_subjects = [
        subject
        for subject in SUBJECTS
        if subject != test_subject
    ]

    X_train = np.concatenate(
        [
            subject_data[subject]["X"]
            for subject in training_subjects
        ],
        axis=0,
    )

    y_train = np.concatenate(
        [
            subject_data[subject]["y"]
            for subject in training_subjects
        ],
        axis=0,
    )

    X_test = subject_data[test_subject]["X"]
    y_test = subject_data[test_subject]["y"]

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        training_subjects,
    )


def main():

    print()
    print("=" * 78)
    print(
        "Strict Cross-Subject "
        "Supervised Autoencoder + RBF-SVM"
    )
    print("=" * 78)

    print(
        "Protocol: train on 8 subjects, "
        "test on 1 unseen subject"
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
        f"Classification weight: "
        f"{CLASSIFICATION_WEIGHT}"
    )

    print("Classifier: RBF-SVM")

    subject_data = load_all_subjects()

    subject_results = []
    prediction_rows = []

    for test_subject in SUBJECTS:

        (
            X_train,
            y_train,
            X_test,
            y_test,
            training_subjects,
        ) = create_loso_fold(
            subject_data,
            test_subject,
        )

        print()
        print("=" * 78)
        print(
            f"LOSO fold: held out "
            f"{test_subject}"
        )
        print("=" * 78)

        print(
            "Training subjects: "
            + ", ".join(training_subjects)
        )

        print(
            f"Training trials: {len(y_train)}"
        )

        print(
            f"Testing trials:  {len(y_test)}"
        )

        model_config = (
            SupervisedAutoencoderConfig(
                n_channels=X_train.shape[1],
                n_times=X_train.shape[2],
                latent_dim=LATENT_DIM,
                hidden_dim=HIDDEN_DIM,
                n_classes=len(CLASS_ORDER),
                dropout=DROPOUT,
            )
        )

        (
            autoencoder,
            standardizer,
            history,
            learned_class_names,
        ) = train_supervised_autoencoder(
            X_train=X_train,
            y_train=y_train,
            model_config=model_config,
            training_config=TRAINING_CONFIG,
        )

        print(
            "Learned classes: "
            f"{learned_class_names}"
        )

        train_features = (
            extract_supervised_latent_features(
                model=autoencoder,
                standardizer=standardizer,
                X=X_train,
            )
        )

        test_features = (
            extract_supervised_latent_features(
                model=autoencoder,
                standardizer=standardizer,
                X=X_test,
            )
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

        y_predicted = classifier.predict(
            test_features
        )

        accuracy = accuracy_score(
            y_test,
            y_predicted,
        )

        kappa = cohen_kappa_score(
            y_test,
            y_predicted,
        )

        recalls = recall_score(
            y_test,
            y_predicted,
            labels=CLASS_ORDER,
            average=None,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test,
            y_predicted,
            labels=CLASS_ORDER,
        )

        result = {
            "test_subject": test_subject,
            "training_subjects": (
                "|".join(training_subjects)
            ),
            "training_trials": len(y_train),
            "testing_trials": len(y_test),
            "accuracy": accuracy,
            "accuracy_percent": (
                accuracy * 100.0
            ),
            "kappa": kappa,
            "left_hand_recall": recalls[0],
            "right_hand_recall": recalls[1],
            "feet_recall": recalls[2],
            "tongue_recall": recalls[3],
            "preprocessing": PREPROCESSING,
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "classification_weight": (
                CLASSIFICATION_WEIGHT
            ),
            "classifier": "rbf_svm",
            "best_epoch": (
                history.best_epoch
            ),
            "best_validation_loss": (
                history.best_validation_loss
            ),
            "evaluation": (
                "strict_LOSO_cross_subject"
            ),
        }

        for true_index, true_label in enumerate(
            CLASS_ORDER
        ):
            for pred_index, pred_label in enumerate(
                CLASS_ORDER
            ):

                result[
                    f"cm_{true_label}_pred_"
                    f"{pred_label}"
                ] = int(
                    matrix[
                        true_index,
                        pred_index,
                    ]
                )

        subject_results.append(
            result
        )

        for trial_index, (
            true_label,
            predicted_label,
        ) in enumerate(
            zip(
                y_test,
                y_predicted,
            ),
            start=1,
        ):

            prediction_rows.append(
                {
                    "test_subject": (
                        test_subject
                    ),
                    "trial": trial_index,
                    "true_label": true_label,
                    "predicted_label": (
                        predicted_label
                    ),
                    "correct": (
                        true_label
                        == predicted_label
                    ),
                    "model": (
                        "Supervised_Autoencoder_"
                        "RBF_SVM"
                    ),
                    "evaluation": (
                        "strict_LOSO_cross_subject"
                    ),
                }
            )

        print()
        print(
            f"{test_subject} Accuracy: "
            f"{accuracy * 100.0:.2f}%"
        )

        print(
            f"{test_subject} Kappa:    "
            f"{kappa:.3f}"
        )

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

    summary = {
        "model": (
            "Supervised_Autoencoder_RBF_SVM"
        ),
        "evaluation": (
            "strict_LOSO_cross_subject"
        ),
        "subjects": len(SUBJECTS),
        "preprocessing": PREPROCESSING,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "classification_weight": (
            CLASSIFICATION_WEIGHT
        ),
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
        [summary],
    )

    print()
    print("=" * 78)
    print(
        "FINAL CROSS-SUBJECT SUPERVISED "
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
            f"{result['test_subject']:<10}"
            f"{result['accuracy_percent']:<15.2f}"
            f"{result['kappa']:<12.3f}"
        )

    print("-" * 40)

    print(
        f"{'Mean':<10}"
        f"{summary['mean_accuracy_percent']:<15.2f}"
        f"{summary['mean_kappa']:<12.3f}"
    )

    print()

    print(
        "Accuracy standard deviation: "
        f"{summary['std_accuracy_percent']:.2f}%"
    )

    print(
        "Kappa standard deviation: "
        f"{summary['std_kappa']:.3f}"
    )

    print()
    print("Saved:")
    print(SUBJECT_RESULTS_PATH)
    print(PREDICTIONS_PATH)
    print(OVERALL_SUMMARY_PATH)


if __name__ == "__main__":
    main()
