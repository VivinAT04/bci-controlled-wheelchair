"""
Run SHAP explanation for CSP + RBF-SVM on one subject.

Run:
    python -m bci_wheelchair.explainability.shap_svm
"""

import warnings
import mne
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from mne.decoding import CSP

from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.data.preprocessing import preprocess_raw


warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


SUBJECT = "A01T"
N_BACKGROUND = 40
N_EXPLAIN = 40


def build_csp_svm():
    return Pipeline([
        (
            "csp",
            CSP(
                n_components=6,
                reg=None,
                log=True,
                rank={"eeg": 22},
            ),
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=42,
            ),
        ),
    ])


def main():
    print("\n========================================")
    print("SHAP Explanation for CSP + RBF-SVM")
    print("========================================")

    path = f"data/raw/{SUBJECT}.gdf"

    print(f"\nLoading {SUBJECT}...")
    raw = load_raw_gdf(path)
    X, y = preprocess_raw(raw)

    print(
        f"Loaded {X.shape[0]} trials, "
        f"{X.shape[1]} EEG channels, "
        f"{X.shape[2]} samples/trial"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("\nTraining CSP + SVM model...")
    clf = build_csp_svm()
    clf.fit(X_train, y_train)

    print("Transforming EEG trials into CSP features...")
    csp = clf.named_steps["csp"]
    svm = clf.named_steps["svm"]

    X_train_csp = csp.transform(X_train)
    X_test_csp = csp.transform(X_test)

    print(f"CSP feature shape: {X_train_csp.shape}")

    background = shap.sample(X_train_csp, N_BACKGROUND, random_state=42)
    X_explain = X_test_csp[:N_EXPLAIN]

    print("\nRunning Kernel SHAP for nonlinear SVM...")
    explainer = shap.KernelExplainer(svm.predict_proba, background)
    shap_values = explainer.shap_values(X_explain)

    feature_names = [
        f"CSP_Component_{i + 1}"
        for i in range(X_train_csp.shape[1])
    ]

    print("\nSaving SHAP summary plot...")

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
    )

    output_path = "results/explainability/csp/shap_csp_svm_summary.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved SHAP plot to: {output_path}")

    print("\nMean absolute SHAP importance:")

    if isinstance(shap_values, list):
        mean_abs_shap = np.mean(
            [np.abs(class_values).mean(axis=0) for class_values in shap_values],
            axis=0,
        )

    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))

    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    ranked_indices = np.argsort(mean_abs_shap)[::-1]

    for idx in ranked_indices:
        idx = int(idx)
        print(f"{feature_names[idx]}: {mean_abs_shap[idx]:.6f}")


if __name__ == "__main__":
    main()