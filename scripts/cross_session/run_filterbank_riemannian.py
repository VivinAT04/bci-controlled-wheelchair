"""
Cross-session Filter-Bank Riemannian evaluation.

Protocol:
    Train on AxxT
    Test on AxxE

Pipeline:
    EEG
    -> multiple frequency bands
    -> OAS covariance matrices per band
    -> Riemannian tangent-space features per band
    -> concatenate features
    -> StandardScaler
    -> PCA retaining 90% variance
    -> Shrinkage LDA

Run:
    python -m scripts.cross_session.run_filterbank_riemannian
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bci_wheelchair.data.processed_loading import load_processed_subject


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

PREPROCESSING = "4-40"
SAMPLING_FREQUENCY = 250.0

FREQUENCY_BANDS = [
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 28.0),
    (28.0, 32.0),
    (32.0, 36.0),
    (36.0, 40.0),
]

FILTER_ORDER = 4
PCA_VARIANCE = 0.90

RESULTS_DIRECTORY = Path(
    "results/cross_session/riemannian/filterbank"
)

SUBJECT_RESULTS_PATH = (
    RESULTS_DIRECTORY
    / "filterbank_riemannian_cross_session_subject_results.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "filterbank_riemannian_cross_session_predictions.csv"
)

OVERALL_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "filterbank_riemannian_cross_session_overall_summary.csv"
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


def bandpass_filter(
    epochs,
    low_frequency,
    high_frequency,
):
    nyquist_frequency = SAMPLING_FREQUENCY / 2.0

    sos = butter(
        N=FILTER_ORDER,
        Wn=[
            low_frequency / nyquist_frequency,
            high_frequency / nyquist_frequency,
        ],
        btype="bandpass",
        output="sos",
    )

    return np.asarray(
        sosfiltfilt(
            sos,
            epochs,
            axis=-1,
        ),
        dtype=np.float64,
    )


class FilterBankRiemannianTransformer(
    BaseEstimator,
    TransformerMixin,
):
    def __init__(
        self,
        frequency_bands=None,
    ):
        self.frequency_bands = (
            FREQUENCY_BANDS
            if frequency_bands is None
            else frequency_bands
        )

    def fit(self, X, y=None):
        self.covariance_transformers_ = []
        self.tangent_transformers_ = []

        for low_frequency, high_frequency in self.frequency_bands:

            X_band = bandpass_filter(
                X,
                low_frequency,
                high_frequency,
            )

            covariance = Covariances(
                estimator="oas",
            )

            covariance_matrices = covariance.fit_transform(
                X_band,
                y,
            )

            tangent = TangentSpace(
                metric="riemann",
            )

            tangent.fit(
                covariance_matrices,
                y,
            )

            self.covariance_transformers_.append(
                covariance
            )

            self.tangent_transformers_.append(
                tangent
            )

        return self

    def transform(self, X):
        features = []

        for (
            low_frequency,
            high_frequency,
        ), covariance, tangent in zip(
            self.frequency_bands,
            self.covariance_transformers_,
            self.tangent_transformers_,
        ):

            X_band = bandpass_filter(
                X,
                low_frequency,
                high_frequency,
            )

            covariance_matrices = covariance.transform(
                X_band
            )

            tangent_features = tangent.transform(
                covariance_matrices
            )

            features.append(
                tangent_features
            )

        return np.concatenate(
            features,
            axis=1,
        )


def build_classifier():
    return Pipeline(
        [
            (
                "filterbank_riemannian",
                FilterBankRiemannianTransformer(),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "pca",
                PCA(
                    n_components=PCA_VARIANCE,
                    svd_solver="full",
                ),
            ),
            (
                "lda",
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto",
                ),
            ),
        ]
    )


def build_subject_result(
    subject,
    y_true,
    y_pred,
):
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
        "preprocessing": PREPROCESSING,
        "frequency_bands": len(FREQUENCY_BANDS),
        "covariance_estimator": "oas",
        "tangent_metric": "riemann",
        "pca_variance": PCA_VARIANCE,
        "classifier": "shrinkage_lda",
        "evaluation": "cross_session_AxxT_to_AxxE",
    }

    for true_index, true_label in enumerate(CLASS_ORDER):
        for pred_index, pred_label in enumerate(CLASS_ORDER):
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
    subject,
    y_true,
    y_pred,
):
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
                "correct": true_label == predicted_label,
                "model": "FilterBank_Riemannian",
                "evaluation": "cross_session_AxxT_to_AxxE",
            }
        )

    return rows


def build_overall_summary(
    subject_results,
):
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
        "model": "FilterBank_Riemannian",
        "evaluation": "cross_session_AxxT_to_AxxE",
        "subjects": len(subject_results),
        "preprocessing": PREPROCESSING,
        "frequency_bands": len(FREQUENCY_BANDS),
        "covariance_estimator": "oas",
        "tangent_metric": "riemann",
        "pca_variance": PCA_VARIANCE,
        "classifier": "shrinkage_lda",
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


def main():
    print()
    print("=" * 78)
    print("Cross-Session Filter-Bank Riemannian")
    print("=" * 78)

    print("Protocol: AxxT -> AxxE")
    print("Input preprocessing: 4-40 Hz")
    print(f"Frequency bands: {len(FREQUENCY_BANDS)}")
    print("Covariance estimator: OAS")
    print("Tangent metric: Riemannian")
    print("PCA: 90% variance")
    print("Classifier: Shrinkage LDA")

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

        classifier = build_classifier()

        classifier.fit(
            X_train,
            y_train,
        )

        y_pred = classifier.predict(
            X_test
        )

        result = build_subject_result(
            subject,
            y_test,
            y_pred,
        )

        subject_results.append(
            result
        )

        prediction_rows.extend(
            build_prediction_rows(
                subject,
                y_test,
                y_pred,
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
        "FINAL CROSS-SESSION FILTER-BANK "
        "RIEMANNIAN RESULTS"
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
