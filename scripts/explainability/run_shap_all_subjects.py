"""
Run SHAP explainability for FBCSP + LDA across all subjects.

This script generates:
1. Overall SHAP feature importance per subject
2. Per-class SHAP feature importance per subject
3. A master CSV summary across all subjects/classes

Run:
    python -m scripts.explainability.run_shap_all_subjects
"""

import os
import re
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


SUBJECTS = [
    "A01T", "A02T", "A03T", "A04T", "A05T",
    "A06T", "A07T", "A08T", "A09T",
]

N_COMPONENTS = 4
N_EXPLAIN = 40
BACKGROUND_SIZE = 50

OUTPUT_DIR = "results/explainability/all_subjects/shap_all_subjects"
MASTER_SUMMARY_CSV = "results/explainability/all_subjects/shap_all_subjects/shap_master_summary.csv"


def safe_name(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def build_feature_names():
    feature_names = []

    for lo, hi in DEFAULT_BANDS:
        for component in range(1, N_COMPONENTS + 1):
            feature_names.append(f"{lo}-{hi}Hz_CSP{component}")

    return feature_names


def extract_class_shap_matrices(raw_shap_values, n_classes):
    """
    Returns list of class-specific SHAP matrices.
    Each matrix has shape:
        samples × features
    """

    if isinstance(raw_shap_values, list):
        return [np.array(v) for v in raw_shap_values]

    values = np.array(raw_shap_values)

    if values.ndim == 3:
        # New SHAP format: samples × features × classes
        if values.shape[2] == n_classes:
            return [values[:, :, i] for i in range(n_classes)]

        # Older/alternative format: classes × samples × features
        if values.shape[0] == n_classes:
            return [values[i, :, :] for i in range(n_classes)]

    if values.ndim == 2 and n_classes == 1:
        return [values]

    raise ValueError(f"Unexpected SHAP output shape: {values.shape}")


def save_importance_csv(feature_names, shap_matrix, output_csv):
    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    df.to_csv(output_csv, index=False)
    return df


def save_bar_plot(importance_df, title, output_path):
    top_features = importance_df.head(15).iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("FBCSP feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_beeswarm_plot(shap_matrix, X_explain, feature_names, title, output_path):
    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_matrix,
        X_explain,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_subject(subject):
    print("\n========================================")
    print(f"Running SHAP for {subject}")
    print("========================================")

    subject_file = f"data/raw/{subject}.gdf"

    if not os.path.exists(subject_file):
        print(f"Skipping {subject}: file not found at {subject_file}")
        return []

    subject_dir = os.path.join(OUTPUT_DIR, subject)
    os.makedirs(subject_dir, exist_ok=True)

    raw = load_raw_gdf(subject_file)
    X, y = preprocess_raw(raw, fmin=4.0, fmax=40.0, tmin=0.5, tmax=2.5)

    print(f"Loaded {X.shape[0]} trials, {X.shape[1]} channels, {X.shape[2]} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_fbcsp_lda(n_components=N_COMPONENTS)
    model.fit(X_train, y_train)

    fbcsp = model.named_steps["fbcsp"]
    lda = model.named_steps["lda"]

    X_train_features = fbcsp.transform(X_train)
    X_test_features = fbcsp.transform(X_test)

    feature_names = build_feature_names()

    if X_train_features.shape[1] != len(feature_names):
        raise ValueError(
            f"{subject}: FBCSP produced {X_train_features.shape[1]} features, "
            f"but {len(feature_names)} feature names were created."
        )

    background = shap.sample(
        X_train_features,
        min(BACKGROUND_SIZE, X_train_features.shape[0]),
        random_state=42,
    )

    X_explain = X_test_features[:N_EXPLAIN]

    explainer = shap.KernelExplainer(
        lda.predict_proba,
        background,
    )

    raw_shap_values = explainer.shap_values(X_explain)

    class_names = list(lda.classes_)
    class_shap_matrices = extract_class_shap_matrices(
        raw_shap_values,
        n_classes=len(class_names),
    )

    summary_rows = []

    # Overall SHAP importance
    overall_dir = os.path.join(subject_dir, "overall")
    os.makedirs(overall_dir, exist_ok=True)

    overall_importance_matrix = np.mean(
        np.abs(np.stack(class_shap_matrices, axis=2)),
        axis=2,
    )

    overall_df = save_importance_csv(
        feature_names,
        overall_importance_matrix,
        os.path.join(overall_dir, "importance.csv"),
    )

    save_bar_plot(
        overall_df,
        f"Overall SHAP Feature Importance ({subject})",
        os.path.join(overall_dir, "bar.png"),
    )

    top_overall = overall_df.iloc[0]

    summary_rows.append(
        {
            "subject": subject,
            "class": "overall",
            "top_feature": top_overall["feature"],
            "mean_abs_shap": top_overall["mean_abs_shap"],
        }
    )

    # Class-wise SHAP
    for class_name, class_matrix in zip(class_names, class_shap_matrices):
        class_folder = safe_name(class_name)
        class_dir = os.path.join(subject_dir, class_folder)
        os.makedirs(class_dir, exist_ok=True)

        class_df = save_importance_csv(
            feature_names,
            class_matrix,
            os.path.join(class_dir, "importance.csv"),
        )

        save_bar_plot(
            class_df,
            f"SHAP Feature Importance for {class_name} ({subject})",
            os.path.join(class_dir, "bar.png"),
        )

        save_beeswarm_plot(
            class_matrix,
            X_explain,
            feature_names,
            f"SHAP Beeswarm for {class_name} ({subject})",
            os.path.join(class_dir, "beeswarm.png"),
        )

        top_class = class_df.iloc[0]

        summary_rows.append(
            {
                "subject": subject,
                "class": class_name,
                "top_feature": top_class["feature"],
                "mean_abs_shap": top_class["mean_abs_shap"],
            }
        )

        print(
            f"{subject} | {class_name} | Top feature: "
            f"{top_class['feature']} ({top_class['mean_abs_shap']:.6f})"
        )

    return summary_rows


def main():
    print("\n========================================")
    print("SHAP for All Subjects and All Classes")
    print("========================================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_summary_rows = []

    for subject in SUBJECTS:
        rows = run_subject(subject)
        all_summary_rows.extend(rows)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(MASTER_SUMMARY_CSV, index=False)

    print("\n========================================")
    print("SHAP analysis complete")
    print("========================================")
    print(f"Master summary saved to: {MASTER_SUMMARY_CSV}")
    print(f"All SHAP outputs saved inside: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()