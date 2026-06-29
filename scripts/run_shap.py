"""
Run SHAP explainability on the best FBCSP + LDA pipeline.

This explains the LDA classifier using FBCSP-extracted features.

Run:
    python -m scripts.run_shap
"""

import warnings
import mne
import shap
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from bci_wheelchair.data_loading import load_raw_gdf
from bci_wheelchair.preprocessing import preprocess_raw
from bci_wheelchair.models import make_fbcsp_lda, DEFAULT_BANDS


warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


SUBJECT = "A07T"
OUTPUT_PATH = "results/shap_fbcsp_summary.png"


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

    model = make_fbcsp_lda(n_components=4)
    model.fit(X_train, y_train)

    fbcsp = model.named_steps["fbcsp"]
    lda = model.named_steps["lda"]

    print("Extracting FBCSP features...")

    X_train_features = fbcsp.transform(X_train)
    X_test_features = fbcsp.transform(X_test)

    print(f"FBCSP feature shape: {X_train_features.shape}")

    feature_names = []

    for lo, hi in DEFAULT_BANDS:
        for component in range(1, 5):
            feature_names.append(f"{lo}-{hi}Hz_CSP{component}")

    print("\nRunning SHAP on LDA classifier...")

    background = shap.sample(X_train_features, min(50, X_train_features.shape[0]))

    explainer = shap.KernelExplainer(
        lda.predict_proba,
        background,
    )

    X_explain = X_test_features[:20]

    shap_values = explainer.shap_values(X_explain)

    print("\nGenerating SHAP summary plot...")

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSHAP summary plot saved to: {OUTPUT_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()