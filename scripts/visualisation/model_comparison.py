"""
Generate final dissertation model-comparison figures.

Source of truth:
results/dissertation_figure_data/canonical_20_method_results.csv

No experimental metrics are hard-coded into the plotting logic.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "results/dissertation_figure_data/"
    "canonical_20_method_results.csv"
)

OUTPUT_DIR = Path("dissertation_figures")

DATA_DIR = Path(
    "results/dissertation_figure_data"
)


MODEL_ORDER = [
    "CSP + LDA",
    "CSP + RBF-SVM",
    "Tuned CSP + LDA",
    "Tuned CSP + RBF-SVM",
    "FBCSP + LDA",
    "FBCSP + RBF-SVM",
    "Riemannian MDM",
    "Riemannian TS + Shrinkage LDA",
    "Riemannian TS + RBF-SVM",
    "Filter-Bank Riemannian + Shrinkage LDA",
    "Filter-Bank Riemannian + RBF-SVM",
    "Autoencoder + LDA",
    "Autoencoder + RBF-SVM",
    "Supervised Autoencoder + LDA",
    "Supervised Autoencoder + RBF-SVM",
    "EEGNet",
    "EA + CSP + Shrinkage LDA",
    "EA + CSP + RBF-SVM",
    "EA + FBCSP + Shrinkage LDA",
    "EA + FBCSP + RBF-SVM",
]


SHORT = {
    "CSP + LDA": "CSP + LDA",
    "CSP + RBF-SVM": "CSP + RBF-SVM",
    "Tuned CSP + LDA": "Tuned CSP + LDA",
    "Tuned CSP + RBF-SVM": "Tuned CSP + RBF-SVM",
    "FBCSP + LDA": "FBCSP + LDA",
    "FBCSP + RBF-SVM": "FBCSP + RBF-SVM",
    "Riemannian MDM": "Riemannian MDM",
    "Riemannian TS + Shrinkage LDA":
        "Riemannian TS + Shrinkage LDA",
    "Riemannian TS + RBF-SVM":
        "Riemannian TS + RBF-SVM",
    "Filter-Bank Riemannian + Shrinkage LDA":
        "FB Riemannian + Shrinkage LDA",
    "Filter-Bank Riemannian + RBF-SVM":
        "FB Riemannian + RBF-SVM",
    "Autoencoder + LDA":
        "Autoencoder + LDA",
    "Autoencoder + RBF-SVM":
        "Autoencoder + RBF-SVM",
    "Supervised Autoencoder + LDA":
        "Supervised Autoencoder + LDA",
    "Supervised Autoencoder + RBF-SVM":
        "Supervised Autoencoder + RBF-SVM",
    "EEGNet":
        "EEGNet",
    "EA + CSP + Shrinkage LDA":
        "EA + CSP + Shrinkage LDA",
    "EA + CSP + RBF-SVM":
        "EA + CSP + RBF-SVM",
    "EA + FBCSP + Shrinkage LDA":
        "EA + FBCSP + Shrinkage LDA",
    "EA + FBCSP + RBF-SVM":
        "EA + FBCSP + RBF-SVM",
}


BEST_CROSS_SESSION = "EA + FBCSP + RBF-SVM"
BEST_CROSS_SUBJECT = "EA + CSP + Shrinkage LDA"


def load_results():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing canonical table: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required = [
        "Method",
        "Cross-Session Accuracy",
        "Cross-Session Kappa",
        "Cross-Subject Accuracy",
        "Cross-Subject Kappa",
    ]

    missing_columns = [
        col for col in required
        if col not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )

    if len(df) != 20:
        raise RuntimeError(
            f"Expected 20 methods, found {len(df)}."
        )

    if df[required].isna().any().any():
        raise RuntimeError(
            "Canonical table contains missing values."
        )

    actual = set(df["Method"])
    expected = set(MODEL_ORDER)

    if actual != expected:
        raise RuntimeError(
            "Method mismatch.\n"
            f"Missing: {sorted(expected - actual)}\n"
            f"Extra: {sorted(actual - expected)}"
        )

    df["Method"] = pd.Categorical(
        df["Method"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    return (
        df.sort_values("Method")
        .reset_index(drop=True)
    )


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.20)


def add_labels(
    ax,
    bars,
    values,
    suffix="",
    decimals=2,
):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.{decimals}f}{suffix}",
            va="center",
            fontsize=7.5,
        )


def cross_session_figure(df):
    data = df.sort_values(
        "Cross-Session Accuracy"
    )

    values = data[
        "Cross-Session Accuracy"
    ].to_numpy()

    labels = [
        SHORT[str(x)]
        for x in data["Method"]
    ]

    y = np.arange(len(data))

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    bars = ax.barh(
        y,
        values,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        labels,
        fontsize=8.5,
    )

    ax.set_xlabel(
        "Mean accuracy (%)"
    )

    ax.set_ylabel(
        "Classification method"
    )

    ax.set_title(
        "Cross-Session Classification Accuracy"
    )

    ax.set_xlim(
        20,
        values.max() + 6,
    )

    clean_axis(ax)

    add_labels(
        ax,
        bars,
        values,
        suffix="%",
    )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "model_comparison_cross_session.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", path)


def cross_subject_figure(df):
    data = df.sort_values(
        "Cross-Subject Accuracy"
    )

    values = data[
        "Cross-Subject Accuracy"
    ].to_numpy()

    labels = [
        SHORT[str(x)]
        for x in data["Method"]
    ]

    y = np.arange(len(data))

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    bars = ax.barh(
        y,
        values,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        labels,
        fontsize=8.5,
    )

    ax.set_xlabel(
        "Mean accuracy (%)"
    )

    ax.set_ylabel(
        "Classification method"
    )

    ax.set_title(
        "Cross-Subject Classification Accuracy"
    )

    ax.set_xlim(
        20,
        values.max() + 6,
    )

    clean_axis(ax)

    add_labels(
        ax,
        bars,
        values,
        suffix="%",
    )

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "model_comparison_cross_subject.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", path)


def combined_accuracy_figure(df):
    labels = [
        SHORT[str(x)]
        for x in df["Method"]
    ]

    session = df[
        "Cross-Session Accuracy"
    ].to_numpy()

    subject = df[
        "Cross-Subject Accuracy"
    ].to_numpy()

    y = np.arange(len(df))
    height = 0.36

    fig, ax = plt.subplots(
        figsize=(13, 11)
    )

    ax.barh(
        y - height / 2,
        session,
        height,
        label="Cross-session",
    )

    ax.barh(
        y + height / 2,
        subject,
        height,
        label="Cross-subject",
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        labels,
        fontsize=8.3,
    )

    ax.set_xlabel(
        "Mean accuracy (%)"
    )

    ax.set_ylabel(
        "Classification method"
    )

    ax.set_title(
        "Classification Performance Across Evaluation Protocols"
    )

    ax.set_xlim(
        20,
        max(
            session.max(),
            subject.max(),
        ) + 5,
    )

    ax.legend()

    clean_axis(ax)

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "model_comparison_accuracy.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", path)


def combined_kappa_figure(df):
    labels = [
        SHORT[str(x)]
        for x in df["Method"]
    ]

    session = df[
        "Cross-Session Kappa"
    ].to_numpy()

    subject = df[
        "Cross-Subject Kappa"
    ].to_numpy()

    y = np.arange(len(df))
    height = 0.36

    fig, ax = plt.subplots(
        figsize=(13, 11)
    )

    ax.barh(
        y - height / 2,
        session,
        height,
        label="Cross-session",
    )

    ax.barh(
        y + height / 2,
        subject,
        height,
        label="Cross-subject",
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        labels,
        fontsize=8.3,
    )

    ax.set_xlabel(
        r"Cohen's $\kappa$"
    )

    ax.set_ylabel(
        "Classification method"
    )

    ax.set_title(
        "Cohen's Kappa Across Evaluation Protocols"
    )

    ax.set_xlim(
        0,
        max(
            session.max(),
            subject.max(),
        ) + 0.06,
    )

    ax.legend()

    clean_axis(ax)

    fig.tight_layout()

    path = (
        OUTPUT_DIR
        / "model_comparison_kappa.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:", path)


def save_winners(df):
    cs = df.loc[
        df[
            "Cross-Session Accuracy"
        ].idxmax()
    ]

    xs = df.loc[
        df[
            "Cross-Subject Accuracy"
        ].idxmax()
    ]

    output = pd.DataFrame(
        [
            {
                "evaluation":
                    "Cross-Session",
                "best_method":
                    str(cs["Method"]),
                "accuracy_percent":
                    float(
                        cs[
                            "Cross-Session Accuracy"
                        ]
                    ),
                "kappa":
                    float(
                        cs[
                            "Cross-Session Kappa"
                        ]
                    ),
            },
            {
                "evaluation":
                    "Cross-Subject",
                "best_method":
                    str(xs["Method"]),
                "accuracy_percent":
                    float(
                        xs[
                            "Cross-Subject Accuracy"
                        ]
                    ),
                "kappa":
                    float(
                        xs[
                            "Cross-Subject Kappa"
                        ]
                    ),
            },
        ]
    )

    path = (
        DATA_DIR
        / "model_comparison_winners.csv"
    )

    output.to_csv(
        path,
        index=False,
    )

    print("Saved:", path)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_results()

    cs = df.loc[
        df[
            "Cross-Session Accuracy"
        ].idxmax()
    ]

    xs = df.loc[
        df[
            "Cross-Subject Accuracy"
        ].idxmax()
    ]

    if str(cs["Method"]) != BEST_CROSS_SESSION:
        raise RuntimeError(
            "Wrong cross-session winner: "
            f"{cs['Method']}"
        )

    if str(xs["Method"]) != BEST_CROSS_SUBJECT:
        raise RuntimeError(
            "Wrong cross-subject winner: "
            f"{xs['Method']}"
        )

    print()
    print("=" * 78)
    print("FINAL 20-METHOD MODEL COMPARISON")
    print("=" * 78)

    print()
    print("Source:")
    print(" ", INPUT_PATH)

    print()
    print("Methods:", len(df))

    print()
    print(
        "Cross-Session winner:"
    )

    print(
        f"  {cs['Method']} | "
        f"{cs['Cross-Session Accuracy']:.2f}% | "
        f"kappa={cs['Cross-Session Kappa']:.3f}"
    )

    print()
    print(
        "Cross-Subject winner:"
    )

    print(
        f"  {xs['Method']} | "
        f"{xs['Cross-Subject Accuracy']:.2f}% | "
        f"kappa={xs['Cross-Subject Kappa']:.3f}"
    )

    print()

    cross_session_figure(df)
    cross_subject_figure(df)
    combined_accuracy_figure(df)
    combined_kappa_figure(df)
    save_winners(df)

    print()
    print("=" * 78)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
