"""
Run SHAP explainability on the best FBCSP + LDA pipeline.

Outputs:
    results/explainability/fbcsp/shap_fbcsp_bar.png
    results/explainability/fbcsp/shap_fbcsp_beeswarm.png
    results/explainability/fbcsp/shap_fbcsp_feature_importance.csv

Run:
    python -m bci_wheelchair.explainability.shap
"""

import warnings
import mne
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.data.preprocessing import preprocess_raw
from bci_wheelchair.models import make_fbcsp_lda, DEFAULT_BANDS


warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


SUBJECT = "A07T"
N_COMPONENTS = 4
N_EXPLAIN = 40

BAR_OUTPUT = "results/explainability/fbcsp/shap_fbcsp_bar.png"
BEESWARM_OUTPUT = "results/explainability/fbcsp/shap_fbcsp_beeswarm.png"
CSV_OUTPUT = "results/explainability/fbcsp/shap_fbcsp_feature_importance.csv"


def build_feature_names():
    """
    Build feature names matching FBCSP output order.
    Each frequency band produces N_COMPONENTS CSP features.
    """

    feature_names = []

    for lo, hi in DEFAULT_BANDS:
        for component in range(1, N_COMPONENTS + 1):
            feature_names.append(f"{lo}-{hi}Hz_CSP{component}")

    return feature_names


def prepare_signed_shap_values(shap_values):
    """
    Convert SHAP output into one signed matrix:
    samples × features.

    For multiclass output, average signed SHAP values across classes.

    Signed SHAP values are required for beeswarm plots because
    they show whether a feature pushes the prediction up or down.
    """

    if isinstance(shap_values, list):
        return np.mean(np.array(shap_values), axis=0)

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        return np.mean(shap_values, axis=2)

    return shap_values


def main():
    print("\n========================================")
    print("SHAP Explainability for FBCSP + LDA")
    print("========================================")

    print(f"\nLoading {SUBJECT}...")

    raw = load_raw_gdf(f"data/raw/{SUBJECT}.gdf")
    X, y = preprocess_raw(raw, fmin=4.0, fmax=40.0, tmin=0.5, tmax=2.5)

    print(f"Loaded {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTraining FBCSP + LDA model...")

    model = make_fbcsp_lda(n_components=N_COMPONENTS)
    model.fit(X_train, y_train)

    fbcsp = model.named_steps["fbcsp"]
    lda = model.named_steps["lda"]

    print("Extracting FBCSP features...")

    X_train_features = fbcsp.transform(X_train)
    X_test_features = fbcsp.transform(X_test)

    feature_names = build_feature_names()

    print(f"FBCSP feature shape: {X_train_features.shape}")
    print(f"Number of feature names: {len(feature_names)}")

    if X_train_features.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature mismatch: FBCSP produced {X_train_features.shape[1]} features, "
            f"but {len(feature_names)} feature names were created."
        )

    print("\nRunning SHAP KernelExplainer on LDA classifier...")

    background = shap.sample(
        X_train_features,
        min(50, X_train_features.shape[0]),
        random_state=42,
    )

    X_explain = X_test_features[:N_EXPLAIN]

    explainer = shap.KernelExplainer(
        lda.predict_proba,
        background,
    )

    raw_shap_values = explainer.shap_values(X_explain)

    signed_shap_matrix = prepare_signed_shap_values(raw_shap_values)

    importance_matrix = np.abs(signed_shap_matrix)
    mean_abs_importance = np.mean(importance_matrix, axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(CSV_OUTPUT, index=False)

    print(f"\nFeature importance saved to: {CSV_OUTPUT}")

    print("\nTop 10 SHAP features:")
    print(importance_df.head(10).to_string(index=False))

    print("\nGenerating SHAP bar plot...")

    top_features = importance_df.head(15).iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("FBCSP feature")
    plt.title(f"Top FBCSP Features by SHAP Importance ({SUBJECT})")
    plt.tight_layout()
    plt.savefig(BAR_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SHAP bar plot saved to: {BAR_OUTPUT}")

    print("\nGenerating SHAP beeswarm summary plot...")

    shap.summary_plot(
        signed_shap_matrix,
        X_explain,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )

    plt.tight_layout()
    plt.savefig(BEESWARM_OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"SHAP beeswarm plot saved to: {BEESWARM_OUTPUT}")

    print("\nDone.")


if __name__ == "__main__":
    main()