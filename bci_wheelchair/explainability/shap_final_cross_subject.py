"""
Final Cross-Subject SHAP analysis.

Exact model:
    EA + CSP + Shrinkage LDA

Exact evaluation protocol:
    Leave-One-Subject-Out (LOSO)

For held-out subject Axx:
    Train:
        T sessions from the remaining eight subjects

    Test:
        AxxE

Each subject/session is independently Euclidean aligned using the
same project implementation used by the final cross-subject experiment.

The exact classifier factory is:
    make_ea_csp_lda()

SHAP is computed in CSP feature space. The fitted Shrinkage LDA
predict_proba function is explained using Kernel SHAP.

This script validates every LOSO fold against the already-final
cross-subject subject_results.csv before accepting the SHAP output.
"""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import shap

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
)

from bci_wheelchair.models.euclidean_alignment import (
    load_and_align_subject,
    make_ea_csp_lda,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42

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

BACKGROUND_SIZE = 30

SAMPLES_PER_CLASS_PER_SUBJECT = 8

SHAP_NSAMPLES = 250


REFERENCE_RESULTS = Path(
    "results/cross_subject/euclidean_alignment/csp/lda/"
    "ea_csp_lda_cross_subject_subject_results.csv"
)

OUTPUT_DIR = Path(
    "results/explainability/"
    "final_cross_subject_ea_csp_lda"
)

FIGURE_DATA_DIR = Path(
    "results/dissertation_figure_data"
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def balanced_indices(
    labels: np.ndarray,
    n_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Reproducible class-balanced selection.
    """

    labels = np.asarray(labels)

    selected: list[int] = []

    for class_name in CLASS_ORDER:
        indices = np.flatnonzero(
            labels == class_name
        )

        if len(indices) == 0:
            raise RuntimeError(
                f"No samples found for class: {class_name}"
            )

        n_select = min(
            n_per_class,
            len(indices),
        )

        chosen = rng.choice(
            indices,
            size=n_select,
            replace=False,
        )

        selected.extend(
            chosen.tolist()
        )

    return np.asarray(
        selected,
        dtype=int,
    )


def normalize_shap_values(
    values,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """
    Normalize SHAP result into:

        samples x features x classes
    """

    if isinstance(values, list):
        array = np.stack(
            [
                np.asarray(value)
                for value in values
            ],
            axis=-1,
        )
    else:
        array = np.asarray(values)

    expected = (
        n_samples,
        n_features,
        n_classes,
    )

    if array.shape == expected:
        return array

    if array.shape == (
        n_classes,
        n_samples,
        n_features,
    ):
        return np.moveaxis(
            array,
            0,
            -1,
        )

    if array.shape == (
        n_samples,
        n_classes,
        n_features,
    ):
        return np.moveaxis(
            array,
            1,
            -1,
        )

    raise RuntimeError(
        "Unexpected SHAP output shape. "
        f"Received {array.shape}; "
        f"expected equivalent of {expected}."
    )


def find_csp_step(classifier):
    """
    Locate the fitted CSP transformer from the exact project pipeline.
    """

    if not hasattr(
        classifier,
        "named_steps",
    ):
        raise RuntimeError(
            "EA+CSP+LDA factory did not return a sklearn Pipeline."
        )

    print(
        "Pipeline steps:",
        list(
            classifier.named_steps.keys()
        ),
    )

    # Preferred exact name.
    if "csp" in classifier.named_steps:
        return (
            "csp",
            classifier.named_steps["csp"],
        )

    # Defensive fallback.
    for name, step in classifier.named_steps.items():
        if (
            hasattr(step, "transform")
            and hasattr(step, "fit")
        ):
            class_name = (
                step.__class__.__name__.lower()
            )

            if "csp" in class_name:
                return name, step

    raise RuntimeError(
        "Could not locate CSP step in "
        f"{list(classifier.named_steps.keys())}"
    )


def find_final_classifier(classifier):
    """
    Return the final fitted classifier step.
    """

    if not hasattr(
        classifier,
        "steps",
    ):
        raise RuntimeError(
            "Expected sklearn Pipeline."
        )

    name, estimator = (
        classifier.steps[-1]
    )

    if not hasattr(
        estimator,
        "predict_proba",
    ):
        raise RuntimeError(
            f"Final step '{name}' does not support predict_proba."
        )

    return name, estimator


def feature_names(
    n_features: int,
) -> list[str]:
    return [
        f"CSP{i + 1}"
        for i in range(
            n_features
        )
    ]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_training_subject(
    subject: str,
):
    session = f"{subject}T"

    X, y, alignment_error = (
        load_and_align_subject(
            session
        )
    )

    return (
        X,
        y,
        float(alignment_error),
    )


def load_evaluation_subject(
    subject: str,
):
    session = f"{subject}E"

    X, y, alignment_error = (
        load_and_align_subject(
            session
        )
    )

    return (
        X,
        y,
        float(alignment_error),
    )


# ---------------------------------------------------------------------
# One LOSO fold
# ---------------------------------------------------------------------

def run_fold(
    held_out_subject: str,
    reference_results: pd.DataFrame,
    rng: np.random.Generator,
):
    print()
    print("=" * 92)
    print(
        f"HELD-OUT SUBJECT: {held_out_subject}"
    )
    print("=" * 92)

    training_subjects = [
        subject
        for subject in SUBJECTS
        if subject != held_out_subject
    ]

    X_training_parts = []
    y_training_parts = []

    training_alignment_errors = []

    for subject in training_subjects:
        X_subject, y_subject, error = (
            load_training_subject(
                subject
            )
        )

        X_training_parts.append(
            X_subject
        )

        y_training_parts.append(
            y_subject
        )

        training_alignment_errors.append(
            error
        )

    X_train = np.concatenate(
        X_training_parts,
        axis=0,
    )

    y_train = np.concatenate(
        y_training_parts,
        axis=0,
    )

    X_test, y_test, test_alignment_error = (
        load_evaluation_subject(
            held_out_subject
        )
    )

    print(
        "Training subjects:",
        " ".join(
            training_subjects
        ),
    )

    print(
        "Training trials:",
        len(y_train),
    )

    print(
        "Evaluation trials:",
        len(y_test),
    )

    # -------------------------------------------------------------
    # EXACT FINAL PROJECT MODEL
    # -------------------------------------------------------------

    classifier = (
        make_ea_csp_lda()
    )

    classifier.fit(
        X_train,
        y_train,
    )

    predictions = classifier.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    kappa = cohen_kappa_score(
        y_test,
        predictions,
    )

    print(
        f"Reproduced accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Reproduced kappa: "
        f"{kappa:.3f}"
    )

    # -------------------------------------------------------------
    # Verify against FINAL experiment
    # -------------------------------------------------------------

    reference_row = reference_results[
        reference_results["subject"]
        == held_out_subject
    ]

    if len(reference_row) != 1:
        raise RuntimeError(
            "Could not uniquely locate final result for "
            f"{held_out_subject}."
        )

    reference_row = (
        reference_row.iloc[0]
    )

    reference_accuracy = float(
        reference_row[
            "accuracy_percent"
        ]
    )

    reference_kappa = float(
        reference_row[
            "kappa"
        ]
    )

    accuracy_difference = abs(
        accuracy * 100.0
        - reference_accuracy
    )

    kappa_difference = abs(
        kappa
        - reference_kappa
    )

    print(
        f"Reference accuracy: "
        f"{reference_accuracy:.2f}%"
    )

    print(
        f"Reference kappa: "
        f"{reference_kappa:.3f}"
    )

    if accuracy_difference > 0.01:
        raise RuntimeError(
            f"{held_out_subject}: accuracy reproduction failed. "
            f"Computed={accuracy * 100:.4f}%, "
            f"reference={reference_accuracy:.4f}%."
        )

    if kappa_difference > 0.001:
        raise RuntimeError(
            f"{held_out_subject}: kappa reproduction failed. "
            f"Computed={kappa:.6f}, "
            f"reference={reference_kappa:.6f}."
        )

    print(
        "✅ Final fold reproduced exactly."
    )

    # -------------------------------------------------------------
    # Extract fitted CSP feature space
    # -------------------------------------------------------------

    csp_name, csp = find_csp_step(
        classifier
    )

    classifier_name, lda = (
        find_final_classifier(
            classifier
        )
    )

    print(
        "CSP step:",
        csp_name,
    )

    print(
        "Classifier step:",
        classifier_name,
    )

    X_train_csp = csp.transform(
        X_train
    )

    X_test_csp = csp.transform(
        X_test
    )

    print(
        "CSP feature count:",
        X_train_csp.shape[1],
    )

    names = feature_names(
        X_train_csp.shape[1]
    )

    # -------------------------------------------------------------
    # SHAP background
    # -------------------------------------------------------------

    background_size = min(
        BACKGROUND_SIZE,
        len(X_train_csp),
    )

    background_indices = rng.choice(
        len(X_train_csp),
        size=background_size,
        replace=False,
    )

    background = X_train_csp[
        background_indices
    ]

    # -------------------------------------------------------------
    # Balanced held-out E-session sample
    # -------------------------------------------------------------

    explain_indices = balanced_indices(
        y_test,
        SAMPLES_PER_CLASS_PER_SUBJECT,
        rng,
    )

    X_explain = X_test_csp[
        explain_indices
    ]

    y_explain = y_test[
        explain_indices
    ]

    print(
        "SHAP background samples:",
        len(background),
    )

    print(
        "Explained held-out E trials:",
        len(X_explain),
    )

    # -------------------------------------------------------------
    # Explain fitted Shrinkage LDA in CSP feature space
    # -------------------------------------------------------------

    def predict_probability(
        csp_features,
    ):
        return lda.predict_proba(
            np.asarray(
                csp_features
            )
        )

    explainer = shap.KernelExplainer(
        predict_probability,
        background,
    )

    raw_shap_values = (
        explainer.shap_values(
            X_explain,
            nsamples=SHAP_NSAMPLES,
            silent=True,
        )
    )

    shap_values = normalize_shap_values(
        raw_shap_values,
        n_samples=len(
            X_explain
        ),
        n_features=X_explain.shape[1],
        n_classes=len(
            lda.classes_
        ),
    )

    print(
        "SHAP shape:",
        shap_values.shape,
    )

    return {
        "subject": held_out_subject,
        "training_subjects": (
            training_subjects
        ),
        "accuracy": float(
            accuracy
        ),
        "kappa": float(
            kappa
        ),
        "reference_accuracy": (
            reference_accuracy
        ),
        "reference_kappa": (
            reference_kappa
        ),
        "training_alignment_error_mean": float(
            np.mean(
                training_alignment_errors
            )
        ),
        "test_alignment_error": float(
            test_alignment_error
        ),
        "classes": [
            str(value)
            for value in lda.classes_
        ],
        "feature_names": names,
        "X_explain": X_explain,
        "y_explain": y_explain,
        "shap_values": shap_values,
    }


# ---------------------------------------------------------------------
# Full LOSO SHAP
# ---------------------------------------------------------------------

def main():
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    if not REFERENCE_RESULTS.exists():
        raise FileNotFoundError(
            f"Missing final reference results: "
            f"{REFERENCE_RESULTS}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_results = pd.read_csv(
        REFERENCE_RESULTS
    )

    if len(reference_results) != 9:
        raise RuntimeError(
            "Expected nine final LOSO subject rows, "
            f"found {len(reference_results)}."
        )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    fold_results = []

    for held_out_subject in SUBJECTS:
        fold = run_fold(
            held_out_subject=(
                held_out_subject
            ),
            reference_results=(
                reference_results
            ),
            rng=rng,
        )

        fold_results.append(
            fold
        )

    # -------------------------------------------------------------
    # Cross-fold consistency
    # -------------------------------------------------------------

    feature_counts = {
        len(
            fold["feature_names"]
        )
        for fold in fold_results
    }

    if len(feature_counts) != 1:
        raise RuntimeError(
            "CSP feature count differs across folds: "
            f"{feature_counts}"
        )

    class_orders = {
        tuple(
            fold["classes"]
        )
        for fold in fold_results
    }

    if len(class_orders) != 1:
        raise RuntimeError(
            "Classifier class order differs across folds."
        )

    feature_names_final = (
        fold_results[0][
            "feature_names"
        ]
    )

    classes_final = (
        fold_results[0][
            "classes"
        ]
    )

    all_shap = np.concatenate(
        [
            fold["shap_values"]
            for fold in fold_results
        ],
        axis=0,
    )

    all_features = np.concatenate(
        [
            fold["X_explain"]
            for fold in fold_results
        ],
        axis=0,
    )

    all_labels = np.concatenate(
        [
            fold["y_explain"]
            for fold in fold_results
        ],
        axis=0,
    )

    print()
    print("=" * 92)
    print(
        "AGGREGATING FINAL CROSS-SUBJECT SHAP"
    )
    print("=" * 92)

    print(
        "Combined SHAP shape:",
        all_shap.shape,
    )

    # -------------------------------------------------------------
    # Global importance
    # -------------------------------------------------------------

    global_importance = np.mean(
        np.abs(
            all_shap
        ),
        axis=(0, 2),
    )

    global_dataframe = (
        pd.DataFrame(
            {
                "feature": (
                    feature_names_final
                ),
                "mean_abs_shap": (
                    global_importance
                ),
            }
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    global_dataframe.insert(
        0,
        "rank",
        np.arange(
            1,
            len(
                global_dataframe
            )
            + 1,
        ),
    )

    global_output = (
        OUTPUT_DIR
        / (
            "shap_cross_subject_"
            "global_feature_importance.csv"
        )
    )

    global_dataframe.to_csv(
        global_output,
        index=False,
    )

    # -------------------------------------------------------------
    # Class-specific importance
    # -------------------------------------------------------------

    class_rows = []

    for class_index, class_name in enumerate(
        classes_final
    ):
        importance = np.mean(
            np.abs(
                all_shap[
                    :,
                    :,
                    class_index,
                ]
            ),
            axis=0,
        )

        for feature_name, value in zip(
            feature_names_final,
            importance,
        ):
            class_rows.append(
                {
                    "class": class_name,
                    "feature": feature_name,
                    "mean_abs_shap": float(
                        value
                    ),
                }
            )

    class_dataframe = pd.DataFrame(
        class_rows
    )

    class_output = (
        OUTPUT_DIR
        / (
            "shap_cross_subject_"
            "class_feature_importance.csv"
        )
    )

    class_dataframe.to_csv(
        class_output,
        index=False,
    )

    # -------------------------------------------------------------
    # Subject summary
    # -------------------------------------------------------------

    subject_rows = []

    for fold in fold_results:
        subject_rows.append(
            {
                "subject": (
                    fold["subject"]
                ),
                "accuracy_percent": (
                    fold["accuracy"]
                    * 100.0
                ),
                "kappa": (
                    fold["kappa"]
                ),
                "reference_accuracy_percent": (
                    fold[
                        "reference_accuracy"
                    ]
                ),
                "reference_kappa": (
                    fold[
                        "reference_kappa"
                    ]
                ),
                "explained_samples": len(
                    fold[
                        "X_explain"
                    ]
                ),
                "training_alignment_error_mean": (
                    fold[
                        "training_alignment_error_mean"
                    ]
                ),
                "test_alignment_error": (
                    fold[
                        "test_alignment_error"
                    ]
                ),
            }
        )

    subject_dataframe = pd.DataFrame(
        subject_rows
    )

    subject_output = (
        OUTPUT_DIR
        / (
            "shap_cross_subject_"
            "subject_summary.csv"
        )
    )

    subject_dataframe.to_csv(
        subject_output,
        index=False,
    )

    # -------------------------------------------------------------
    # Dissertation figure-data CSVs
    # -------------------------------------------------------------

    global_dataframe.to_csv(
        FIGURE_DATA_DIR
        / (
            "shap_cross_subject_"
            "global_feature_importance.csv"
        ),
        index=False,
    )

    class_heatmap = (
        class_dataframe.pivot(
            index="class",
            columns="feature",
            values="mean_abs_shap",
        )
    )

    class_heatmap.to_csv(
        FIGURE_DATA_DIR
        / (
            "shap_cross_subject_"
            "class_feature_heatmap.csv"
        )
    )

    # -------------------------------------------------------------
    # Raw arrays
    # -------------------------------------------------------------

    np.save(
        OUTPUT_DIR
        / "shap_values.npy",
        all_shap,
    )

    np.save(
        OUTPUT_DIR
        / "explained_features.npy",
        all_features,
    )

    np.save(
        OUTPUT_DIR
        / "explained_labels.npy",
        all_labels,
    )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    mean_accuracy = float(
        subject_dataframe[
            "accuracy_percent"
        ].mean()
    )

    mean_kappa = float(
        subject_dataframe[
            "kappa"
        ].mean()
    )

    metadata = {
        "protocol": (
            "Cross-Subject T-to-E LOSO"
        ),
        "model": (
            "EA + CSP + Shrinkage LDA"
        ),
        "factory": (
            "make_ea_csp_lda"
        ),
        "subjects": SUBJECTS,
        "training_sessions": (
            "T sessions from eight non-held-out subjects"
        ),
        "evaluation_sessions": (
            "held-out subject E session"
        ),
        "background_size": (
            BACKGROUND_SIZE
        ),
        "samples_per_class_per_subject": (
            SAMPLES_PER_CLASS_PER_SUBJECT
        ),
        "shap_nsamples": (
            SHAP_NSAMPLES
        ),
        "random_state": (
            RANDOM_STATE
        ),
        "feature_names": (
            feature_names_final
        ),
        "classifier_classes": (
            classes_final
        ),
        "mean_accuracy_percent": (
            mean_accuracy
        ),
        "mean_kappa": (
            mean_kappa
        ),
        "total_explained_samples": int(
            len(
                all_features
            )
        ),
        "shap_shape": [
            int(value)
            for value in all_shap.shape
        ],
    }

    metadata_output = (
        OUTPUT_DIR
        / "shap_cross_subject_metadata.json"
    )

    with open(
        metadata_output,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print(
        "Top Cross-Subject SHAP features:"
    )

    print(
        global_dataframe.head(
            10
        ).to_string(
            index=False
        )
    )

    print()
    print(
        "LOSO subject summary:"
    )

    print(
        subject_dataframe[
            [
                "subject",
                "accuracy_percent",
                "kappa",
                "explained_samples",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Mean accuracy: "
        f"{mean_accuracy:.2f}%"
    )

    print(
        f"Mean kappa: "
        f"{mean_kappa:.3f}"
    )

    print()
    print(
        "Outputs:",
        OUTPUT_DIR,
    )

    print()
    print("=" * 92)
    print(
        "✅ FINAL CROSS-SUBJECT SHAP COMPLETE"
    )
    print("=" * 92)


if __name__ == "__main__":
    main()
