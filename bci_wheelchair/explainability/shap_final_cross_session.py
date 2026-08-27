"""
Final dissertation SHAP/XAI experiment.

Explains the final cross-session winning classifier:

    A01T-A09T
        -> session-wise Euclidean Alignment
        -> pooled aligned training data
        -> Regularized FBCSP
        -> PCA retaining 90% variance
        -> StandardScaler
        -> RBF-SVM

Evaluation:

    A01E-A09E
        -> independently Euclidean-aligned
        -> fitted FBCSP/PCA/scaler/SVM

Interpretability strategy
-------------------------
SHAP explanations are calculated in the ORIGINAL FBCSP feature space.

The prediction function exposed to SHAP receives FBCSP features and
internally applies:

    PCA -> StandardScaler -> fitted RBF-SVM

Therefore the explanation still describes the actual final classifier,
while retaining interpretable frequency-band CSP feature names rather
than opaque PCA-component names.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
)

from bci_wheelchair.models.euclidean_alignment import (
    load_and_align_subject,
    make_ea_fbcsp_svm,
)


# =====================================================================
# CONFIGURATION
# =====================================================================

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

RANDOM_STATE = 42

# Number of evaluation examples per class, per subject.
#
# 8 x 4 classes x 9 subjects = up to 288 explained trials.
EXAMPLES_PER_CLASS_PER_SUBJECT = 8

# Number of representative training examples used as SHAP background.
BACKGROUND_SIZE = 30

# Kernel SHAP approximation budget.
SHAP_NSAMPLES = 250


RESULTS_DIR = Path(
    "results/explainability/final_ea_fbcsp_svm"
)

SUBJECT_DIR = RESULTS_DIR / "subjects"

MASTER_FEATURE_PATH = (
    RESULTS_DIR
    / "shap_final_feature_importance.csv"
)

BAND_PATH = (
    RESULTS_DIR
    / "shap_final_frequency_band_importance.csv"
)

CLASS_BAND_PATH = (
    RESULTS_DIR
    / "shap_final_class_frequency_importance.csv"
)

SUBJECT_SUMMARY_PATH = (
    RESULTS_DIR
    / "shap_final_subject_summary.csv"
)

METADATA_PATH = (
    RESULTS_DIR
    / "shap_final_metadata.json"
)


DISSERTATION_DATA_DIR = Path(
    "results/dissertation_figure_data"
)

DISSERTATION_FIGURE_DIR = Path(
    "dissertation_figures"
)


GLOBAL_DATA_PATH = (
    DISSERTATION_DATA_DIR
    / "shap_cross_session_global_feature_importance.csv"
)

FREQUENCY_DATA_PATH = (
    DISSERTATION_DATA_DIR
    / "shap_cross_session_frequency_band_importance.csv"
)

CLASS_FREQUENCY_DATA_PATH = (
    DISSERTATION_DATA_DIR
    / "shap_cross_session_class_frequency_heatmap.csv"
)


GLOBAL_FIGURE_PATH = (
    DISSERTATION_FIGURE_DIR
    / "shap_cross_session_global_feature_importance.png"
)

FREQUENCY_FIGURE_PATH = (
    DISSERTATION_FIGURE_DIR
    / "shap_cross_session_frequency_band_importance.png"
)

CLASS_FREQUENCY_FIGURE_PATH = (
    DISSERTATION_FIGURE_DIR
    / "shap_cross_session_class_frequency_heatmap.png"
)


# =====================================================================
# HELPERS
# =====================================================================

def format_band(
    low: float,
    high: float,
) -> str:
    """Return dissertation-friendly frequency-band name."""

    def clean(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return str(value)

    return (
        f"{clean(low)}-{clean(high)} Hz"
    )


def build_feature_names(
    bands,
    n_components: int,
) -> list[str]:
    """
    Build feature names in exactly the same order emitted by FBCSP.
    """

    names = []

    for low, high in bands:
        band_name = (
            f"{int(low) if float(low).is_integer() else low}"
            "-"
            f"{int(high) if float(high).is_integer() else high}"
            "Hz"
        )

        for component in range(
            1,
            n_components + 1,
        ):
            names.append(
                f"{band_name}_CSP{component}"
            )

    return names


def stratified_subject_sample(
    X: np.ndarray,
    y: np.ndarray,
    n_per_class: int,
    rng: np.random.Generator,
):
    """
    Select a balanced explanation subset from one evaluation session.
    """

    indices = []

    for class_name in CLASS_ORDER:

        class_indices = np.flatnonzero(
            y == class_name
        )

        if len(class_indices) == 0:
            continue

        take = min(
            n_per_class,
            len(class_indices),
        )

        chosen = rng.choice(
            class_indices,
            size=take,
            replace=False,
        )

        indices.extend(
            chosen.tolist()
        )

    indices = np.asarray(
        sorted(indices),
        dtype=int,
    )

    return (
        X[indices],
        y[indices],
        indices,
    )


def normalise_shap_output(
    shap_values,
    n_samples: int,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """
    Convert SHAP API variants into:

        samples x features x classes
    """

    if isinstance(
        shap_values,
        list,
    ):
        arrays = [
            np.asarray(value)
            for value in shap_values
        ]

        result = np.stack(
            arrays,
            axis=-1,
        )

    else:
        result = np.asarray(
            shap_values
        )

        # Newer SHAP:
        # samples x features x outputs
        if (
            result.ndim == 3
            and result.shape
            == (
                n_samples,
                n_features,
                n_classes,
            )
        ):
            pass

        # outputs x samples x features
        elif (
            result.ndim == 3
            and result.shape
            == (
                n_classes,
                n_samples,
                n_features,
            )
        ):
            result = np.transpose(
                result,
                (1, 2, 0),
            )

        else:
            raise ValueError(
                "Unexpected SHAP output shape: "
                f"{result.shape}"
            )

    expected = (
        n_samples,
        n_features,
        n_classes,
    )

    if result.shape != expected:
        raise ValueError(
            "Normalised SHAP shape mismatch. "
            f"Expected {expected}, received "
            f"{result.shape}."
        )

    return np.asarray(
        result,
        dtype=np.float64,
    )


# =====================================================================
# LOAD FINAL CROSS-SESSION DATA
# =====================================================================

def load_cross_session_data():
    """
    Load and independently EA-align all T and E sessions.
    """

    train_X_parts = []
    train_y_parts = []

    evaluation = {}

    alignment_errors = {}

    print()
    print("=" * 78)
    print("LOADING + EUCLIDEAN-ALIGNING A01T-A09T")
    print("=" * 78)

    for subject in SUBJECTS:

        session = f"{subject}T"

        X, y, error = (
            load_and_align_subject(
                session
            )
        )

        X = np.asarray(X)
        y = np.asarray(y)

        train_X_parts.append(X)
        train_y_parts.append(y)

        alignment_errors[session] = float(
            error
        )

        print(
            f"{session}: "
            f"{len(y)} trials | "
            f"EA error={error:.8f}"
        )

    X_train = np.concatenate(
        train_X_parts,
        axis=0,
    )

    y_train = np.concatenate(
        train_y_parts,
        axis=0,
    )

    print()
    print(
        "Pooled training shape: "
        f"{X_train.shape}"
    )

    print()
    print("=" * 78)
    print("LOADING + EUCLIDEAN-ALIGNING A01E-A09E")
    print("=" * 78)

    for subject in SUBJECTS:

        session = f"{subject}E"

        X, y, error = (
            load_and_align_subject(
                session
            )
        )

        X = np.asarray(X)
        y = np.asarray(y)

        evaluation[subject] = {
            "X": X,
            "y": y,
        }

        alignment_errors[session] = float(
            error
        )

        print(
            f"{session}: "
            f"{len(y)} trials | "
            f"EA error={error:.8f}"
        )

    return (
        X_train,
        y_train,
        evaluation,
        alignment_errors,
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    start = time.perf_counter()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUBJECT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DISSERTATION_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DISSERTATION_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    print()
    print("=" * 78)
    print("FINAL DISSERTATION SHAP/XAI")
    print("EA + FBCSP + RBF-SVM")
    print("=" * 78)

    (
        X_train,
        y_train,
        evaluation,
        alignment_errors,
    ) = load_cross_session_data()

    # -----------------------------------------------------------------
    # TRAIN EXACT FINAL CLASSIFIER
    # -----------------------------------------------------------------

    print()
    print("=" * 78)
    print("TRAINING EXACT FINAL CROSS-SESSION MODEL")
    print("=" * 78)

    classifier = (
        make_ea_fbcsp_svm()
    )

    classifier.fit(
        X_train,
        y_train,
    )

    required_steps = [
        "fbcsp",
        "pca",
        "scaler",
        "svm",
    ]

    for step in required_steps:
        if step not in classifier.named_steps:
            raise RuntimeError(
                "Final model missing pipeline step: "
                f"{step}"
            )

    fbcsp = classifier.named_steps[
        "fbcsp"
    ]

    pca = classifier.named_steps[
        "pca"
    ]

    scaler = classifier.named_steps[
        "scaler"
    ]

    svm = classifier.named_steps[
        "svm"
    ]

    bands = list(
        fbcsp.bands
    )

    n_components = int(
        fbcsp.n_components
    )

    feature_names = build_feature_names(
        bands,
        n_components,
    )

    expected_features = (
        len(bands)
        * n_components
    )

    print()
    print(
        f"Frequency bands: {bands}"
    )
    print(
        "CSP components per band: "
        f"{n_components}"
    )
    print(
        "Original interpretable FBCSP features: "
        f"{expected_features}"
    )
    print(
        "PCA components retained: "
        f"{pca.n_components_}"
    )
    print(
        f"SVM kernel: {svm.kernel}"
    )
    print(
        f"SVM C: {svm.C}"
    )
    print(
        f"SVM gamma: {svm.gamma}"
    )

    if len(feature_names) != expected_features:
        raise RuntimeError(
            "Feature-name count mismatch."
        )

    # -----------------------------------------------------------------
    # VERIFY FINAL MODEL PERFORMANCE
    # -----------------------------------------------------------------

    all_true = []
    all_pred = []

    subject_metrics = []

    print()
    print("=" * 78)
    print("VERIFYING CROSS-SESSION PERFORMANCE")
    print("=" * 78)

    for subject in SUBJECTS:

        X = evaluation[
            subject
        ]["X"]

        y = evaluation[
            subject
        ]["y"]

        pred = classifier.predict(
            X
        )

        accuracy = accuracy_score(
            y,
            pred,
        )

        kappa = cohen_kappa_score(
            y,
            pred,
        )

        subject_metrics.append(
            {
                "subject": subject,
                "accuracy_percent": (
                    accuracy * 100.0
                ),
                "kappa": kappa,
            }
        )

        all_true.append(y)
        all_pred.append(pred)

        print(
            f"{subject}E: "
            f"{accuracy * 100:.2f}% | "
            f"kappa={kappa:.3f}"
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

    mean_accuracy = np.mean(
        [
            row["accuracy_percent"]
            for row in subject_metrics
        ]
    )

    mean_kappa = np.mean(
        [
            row["kappa"]
            for row in subject_metrics
        ]
    )

    print()
    print(
        "Mean cross-session accuracy: "
        f"{mean_accuracy:.2f}%"
    )

    print(
        "Mean cross-session kappa: "
        f"{mean_kappa:.3f}"
    )

    print(
        "Pooled accuracy: "
        f"{pooled_accuracy * 100:.2f}%"
    )

    print(
        "Pooled kappa: "
        f"{pooled_kappa:.3f}"
    )

    # Expected dissertation result.
    if abs(
        mean_accuracy - 60.49
    ) > 0.10:
        raise RuntimeError(
            "STOP: final model did not reproduce "
            "the expected ~60.49% cross-session "
            f"accuracy. Obtained {mean_accuracy:.2f}%."
        )

    if abs(
        mean_kappa - 0.473
    ) > 0.005:
        raise RuntimeError(
            "STOP: final model did not reproduce "
            "the expected ~0.473 kappa. "
            f"Obtained {mean_kappa:.3f}."
        )

    print()
    print(
        "✅ Final classifier reproduction check passed."
    )

    # -----------------------------------------------------------------
    # FIT FBCSP FEATURE REPRESENTATION
    # -----------------------------------------------------------------

    print()
    print("=" * 78)
    print("EXTRACTING INTERPRETABLE FBCSP FEATURES")
    print("=" * 78)

    X_train_features = (
        fbcsp.transform(
            X_train
        )
    )

    if (
        X_train_features.shape[1]
        != expected_features
    ):
        raise RuntimeError(
            "FBCSP produced an unexpected "
            "number of features: "
            f"{X_train_features.shape}"
        )

    # -----------------------------------------------------------------
    # SHAP PREDICTION WRAPPER
    # -----------------------------------------------------------------

    def predict_from_fbcsp(
        features,
    ):
        """
        Predict class probabilities from raw FBCSP features.

        SHAP therefore attributes the complete downstream:
            PCA -> scaler -> RBF-SVM
        prediction back to the interpretable FBCSP inputs.
        """

        features = np.asarray(
            features,
            dtype=np.float64,
        )

        reduced = pca.transform(
            features
        )

        scaled = scaler.transform(
            reduced
        )

        return svm.predict_proba(
            scaled
        )

    # -----------------------------------------------------------------
    # BACKGROUND DATA
    # -----------------------------------------------------------------

    print()
    print("=" * 78)
    print("BUILDING SHAP BACKGROUND")
    print("=" * 78)

    if len(X_train_features) <= BACKGROUND_SIZE:
        background = (
            X_train_features
        )
    else:
        background_indices = rng.choice(
            len(X_train_features),
            size=BACKGROUND_SIZE,
            replace=False,
        )

        background = (
            X_train_features[
                background_indices
            ]
        )

    print(
        "Background samples: "
        f"{len(background)}"
    )

    print(
        "Background feature dimension: "
        f"{background.shape[1]}"
    )

    # KernelExplainer is appropriate because the final classifier
    # includes a nonlinear RBF-SVM.
    explainer = shap.KernelExplainer(
        predict_from_fbcsp,
        background,
    )

    # -----------------------------------------------------------------
    # EXPLAIN ALL NINE EVALUATION SESSIONS
    # -----------------------------------------------------------------

    all_shap_blocks = []
    all_feature_blocks = []
    all_label_blocks = []
    all_subject_blocks = []

    subject_summary_rows = []

    for subject_index, subject in enumerate(
        SUBJECTS,
        start=1,
    ):

        print()
        print("=" * 78)
        print(
            f"SHAP SUBJECT "
            f"{subject_index}/9: {subject}E"
        )
        print("=" * 78)

        X_eeg = evaluation[
            subject
        ]["X"]

        y = evaluation[
            subject
        ]["y"]

        X_features = (
            fbcsp.transform(
                X_eeg
            )
        )

        (
            X_selected,
            y_selected,
            selected_indices,
        ) = stratified_subject_sample(
            X_features,
            y,
            EXAMPLES_PER_CLASS_PER_SUBJECT,
            rng,
        )

        print(
            f"Explaining {len(X_selected)} "
            f"balanced evaluation trials."
        )

        shap_start = time.perf_counter()

        raw_shap_values = (
            explainer.shap_values(
                X_selected,
                nsamples=SHAP_NSAMPLES,
            )
        )

        shap_values = normalise_shap_output(
            raw_shap_values,
            n_samples=len(X_selected),
            n_features=expected_features,
            n_classes=len(
                svm.classes_
            ),
        )

        shap_seconds = (
            time.perf_counter()
            - shap_start
        )

        print(
            f"SHAP completed in "
            f"{shap_seconds:.1f}s"
        )

        # Save subject-level global importance.
        subject_importance = (
            np.abs(
                shap_values
            ).mean(
                axis=(0, 2)
            )
        )

        subject_df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": (
                    subject_importance
                ),
            }
        ).sort_values(
            "mean_abs_shap",
            ascending=False,
        )

        subject_path = (
            SUBJECT_DIR
            / (
                f"{subject}_"
                "feature_importance.csv"
            )
        )

        subject_df.to_csv(
            subject_path,
            index=False,
        )

        subject_summary_rows.append(
            {
                "subject": subject,
                "n_explained_trials": (
                    len(X_selected)
                ),
                "shap_seconds": (
                    shap_seconds
                ),
                "top_feature": (
                    subject_df.iloc[
                        0
                    ]["feature"]
                ),
                "top_feature_mean_abs_shap": (
                    subject_df.iloc[
                        0
                    ]["mean_abs_shap"]
                ),
            }
        )

        all_shap_blocks.append(
            shap_values
        )

        all_feature_blocks.append(
            X_selected
        )

        all_label_blocks.append(
            y_selected
        )

        all_subject_blocks.append(
            np.repeat(
                subject,
                len(X_selected),
            )
        )

    # -----------------------------------------------------------------
    # GLOBAL AGGREGATION
    # -----------------------------------------------------------------

    all_shap = np.concatenate(
        all_shap_blocks,
        axis=0,
    )

    all_features = np.concatenate(
        all_feature_blocks,
        axis=0,
    )

    all_labels = np.concatenate(
        all_label_blocks,
        axis=0,
    )

    all_subjects = np.concatenate(
        all_subject_blocks,
        axis=0,
    )

    print()
    print("=" * 78)
    print("AGGREGATING FINAL SHAP RESULTS")
    print("=" * 78)

    print(
        "Combined SHAP shape: "
        f"{all_shap.shape}"
    )

    # -----------------------------------------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # -----------------------------------------------------------------

    global_importance = (
        np.abs(
            all_shap
        ).mean(
            axis=(0, 2)
        )
    )

    feature_rows = []

    feature_index = 0

    for low, high in bands:

        band_name = format_band(
            low,
            high,
        )

        for component in range(
            1,
            n_components + 1,
        ):

            feature_rows.append(
                {
                    "feature": (
                        feature_names[
                            feature_index
                        ]
                    ),
                    "frequency_band": (
                        band_name
                    ),
                    "csp_component": (
                        component
                    ),
                    "mean_abs_shap": float(
                        global_importance[
                            feature_index
                        ]
                    ),
                }
            )

            feature_index += 1

    feature_df = pd.DataFrame(
        feature_rows
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    feature_df.to_csv(
        MASTER_FEATURE_PATH,
        index=False,
    )

    feature_df.to_csv(
        GLOBAL_DATA_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # FREQUENCY-BAND IMPORTANCE
    # -----------------------------------------------------------------

    band_df = (
        feature_df.groupby(
            "frequency_band",
            as_index=False,
        )["mean_abs_shap"]
        .mean()
    )

    band_order = [
        format_band(
            low,
            high,
        )
        for low, high in bands
    ]

    band_df[
        "frequency_band"
    ] = pd.Categorical(
        band_df[
            "frequency_band"
        ],
        categories=band_order,
        ordered=True,
    )

    band_df = band_df.sort_values(
        "frequency_band"
    )

    band_df.to_csv(
        BAND_PATH,
        index=False,
    )

    band_df.to_csv(
        FREQUENCY_DATA_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # CLASS x FREQUENCY IMPORTANCE
    # -----------------------------------------------------------------

    svm_classes = list(
        svm.classes_
    )

    class_band_rows = []

    for class_index, class_name in enumerate(
        svm_classes
    ):

        class_feature_importance = (
            np.abs(
                all_shap[
                    :,
                    :,
                    class_index,
                ]
            ).mean(
                axis=0
            )
        )

        start_index = 0

        for low, high in bands:

            stop_index = (
                start_index
                + n_components
            )

            band_importance = float(
                class_feature_importance[
                    start_index:stop_index
                ].mean()
            )

            class_band_rows.append(
                {
                    "class": (
                        class_name
                    ),
                    "frequency_band": (
                        format_band(
                            low,
                            high,
                        )
                    ),
                    "mean_abs_shap": (
                        band_importance
                    ),
                }
            )

            start_index = (
                stop_index
            )

    class_band_df = pd.DataFrame(
        class_band_rows
    )

    class_band_df.to_csv(
        CLASS_BAND_PATH,
        index=False,
    )

    # Wide form for dissertation figure data.
    heatmap_df = (
        class_band_df.pivot(
            index="class",
            columns="frequency_band",
            values="mean_abs_shap",
        )
        .reindex(
            index=CLASS_ORDER,
            columns=band_order,
        )
    )

    heatmap_df.to_csv(
        CLASS_FREQUENCY_DATA_PATH,
    )

    # -----------------------------------------------------------------
    # SUBJECT SUMMARY
    # -----------------------------------------------------------------

    pd.DataFrame(
        subject_summary_rows
    ).to_csv(
        SUBJECT_SUMMARY_PATH,
        index=False,
    )

    # -----------------------------------------------------------------
    # FIGURE 1 — GLOBAL FEATURE IMPORTANCE
    # -----------------------------------------------------------------

    top = (
        feature_df
        .head(20)
        .sort_values(
            "mean_abs_shap",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 7)
    )

    y_positions = np.arange(
        len(top)
    )

    ax.barh(
        y_positions,
        top["mean_abs_shap"],
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        top["feature"]
    )

    ax.set_xlabel(
        "Mean absolute SHAP value"
    )

    ax.set_ylabel(
        "FBCSP feature"
    )

    ax.set_title(
        "Global SHAP Feature Importance\n"
        "EA + FBCSP + SVM Cross-Session Model"
    )

    fig.tight_layout()

    fig.savefig(
        GLOBAL_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -----------------------------------------------------------------
    # FIGURE 2 — FREQUENCY BAND IMPORTANCE
    # -----------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    x = np.arange(
        len(band_df)
    )

    ax.bar(
        x,
        band_df[
            "mean_abs_shap"
        ],
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        band_df[
            "frequency_band"
        ],
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        "Mean absolute SHAP value"
    )

    ax.set_xlabel(
        "Frequency band"
    )

    ax.set_title(
        "SHAP Importance by Frequency Band\n"
        "EA + FBCSP + SVM Cross-Session Model"
    )

    fig.tight_layout()

    fig.savefig(
        FREQUENCY_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -----------------------------------------------------------------
    # FIGURE 3 — CLASS x FREQUENCY HEATMAP
    # -----------------------------------------------------------------

    matrix = heatmap_df.to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            len(band_order)
        )
    )

    ax.set_xticklabels(
        band_order,
        rotation=35,
        ha="right",
    )

    readable_classes = [
        label.replace(
            "_",
            " ",
        ).title()
        for label in CLASS_ORDER
    ]

    ax.set_yticks(
        np.arange(
            len(CLASS_ORDER)
        )
    )

    ax.set_yticklabels(
        readable_classes
    )

    ax.set_xlabel(
        "Frequency band"
    )

    ax.set_ylabel(
        "Motor-imagery class"
    )

    ax.set_title(
        "Class-Specific SHAP Frequency-Band Importance\n"
        "EA + FBCSP + SVM Cross-Session Model"
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Mean absolute SHAP value"
    )

    fig.tight_layout()

    fig.savefig(
        CLASS_FREQUENCY_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -----------------------------------------------------------------
    # METADATA
    # -----------------------------------------------------------------

    duration = (
        time.perf_counter()
        - start
    )

    metadata = {
        "experiment": (
            "final_cross_session_shap"
        ),
        "model": (
            "EA + FBCSP + RBF-SVM"
        ),
        "training_sessions": (
            "A01T-A09T"
        ),
        "evaluation_sessions": (
            "A01E-A09E"
        ),
        "evaluation_protocol": (
            "pooled_cross_session"
        ),
        "alignment": (
            "session-wise Euclidean Alignment"
        ),
        "frequency_bands": [
            [
                float(low),
                float(high),
            ]
            for low, high in bands
        ],
        "n_frequency_bands": (
            len(bands)
        ),
        "csp_components_per_band": (
            n_components
        ),
        "fbcsp_feature_count": (
            expected_features
        ),
        "csp_regularization": (
            "ledoit_wolf"
        ),
        "pca_variance": (
            float(
                classifier.named_steps[
                    "pca"
                ].n_components
            )
        ),
        "pca_components_retained": (
            int(
                pca.n_components_
            )
        ),
        "svm_kernel": (
            svm.kernel
        ),
        "svm_C": (
            float(
                svm.C
            )
        ),
        "svm_gamma": (
            str(
                svm.gamma
            )
        ),
        "mean_accuracy_percent": (
            float(
                mean_accuracy
            )
        ),
        "mean_kappa": (
            float(
                mean_kappa
            )
        ),
        "pooled_accuracy_percent": (
            float(
                pooled_accuracy
                * 100
            )
        ),
        "pooled_kappa": (
            float(
                pooled_kappa
            )
        ),
        "background_samples": (
            len(background)
        ),
        "examples_per_class_per_subject": (
            EXAMPLES_PER_CLASS_PER_SUBJECT
        ),
        "total_explained_trials": (
            int(
                len(
                    all_features
                )
            )
        ),
        "shap_nsamples": (
            SHAP_NSAMPLES
        ),
        "shap_explainer": (
            "KernelExplainer"
        ),
        "shap_input_space": (
            "original_regularized_FBCSP_features"
        ),
        "downstream_model_explained": (
            "PCA90 -> StandardScaler -> RBF-SVM"
        ),
        "duration_seconds": (
            float(
                duration
            )
        ),
        "alignment_identity_errors": (
            alignment_errors
        ),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    # -----------------------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL SHAP/XAI COMPLETE")
    print("=" * 78)

    print()
    print(
        "Model: EA + FBCSP + RBF-SVM"
    )

    print(
        f"Accuracy: {mean_accuracy:.2f}%"
    )

    print(
        f"Kappa: {mean_kappa:.3f}"
    )

    print(
        "Explained trials: "
        f"{len(all_features)}"
    )

    print(
        "Interpretable FBCSP features: "
        f"{expected_features}"
    )

    print()
    print("Top 10 global SHAP features:")
    print(
        feature_df[
            [
                "feature",
                "frequency_band",
                "mean_abs_shap",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print("Frequency-band ranking:")
    print(
        band_df.sort_values(
            "mean_abs_shap",
            ascending=False,
        ).to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(
        f"  {MASTER_FEATURE_PATH}"
    )
    print(
        f"  {BAND_PATH}"
    )
    print(
        f"  {CLASS_BAND_PATH}"
    )
    print(
        f"  {SUBJECT_SUMMARY_PATH}"
    )
    print(
        f"  {METADATA_PATH}"
    )

    print()
    print("Dissertation figures:")
    print(
        f"  {GLOBAL_FIGURE_PATH}"
    )
    print(
        f"  {FREQUENCY_FIGURE_PATH}"
    )
    print(
        f"  {CLASS_FREQUENCY_FIGURE_PATH}"
    )


if __name__ == "__main__":
    main()
