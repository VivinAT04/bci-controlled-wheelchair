"""
Generate dissertation figure for the target-subject calibration experiment.

The figure is generated directly from:
results/cross_subject/riemannian/riemannian_calibration/
riemannian_calibration_overall_summary.csv

No experiments are rerun.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path(
    "results/cross_subject/riemannian/"
    "riemannian_calibration/"
    "riemannian_calibration_overall_summary.csv"
)

OUTPUT_PATH = Path(
    "dissertation_figures/"
    "target_subject_calibration.png"
)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Calibration summary not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "calibration_percent",
        "mean_accuracy_percent",
        "std_between_subjects_percent",
        "mean_kappa",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.sort_values(
        "calibration_percent"
    ).reset_index(drop=True)

    x = df["calibration_percent"]
    accuracy = df["mean_accuracy_percent"]
    sd = df["std_between_subjects_percent"]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.errorbar(
        x,
        accuracy,
        yerr=sd,
        marker="o",
        markersize=7,
        linewidth=2,
        capsize=5,
    )

    for x_value, accuracy_value in zip(
        x,
        accuracy,
    ):
        ax.annotate(
            f"{accuracy_value:.2f}%",
            (
                x_value,
                accuracy_value,
            ),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    ax.axhline(
        25,
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_title(
        "Effect of Target-Subject Calibration on "
        "Cross-Subject Classification"
    )

    ax.set_xlabel(
        "Target-subject calibration data (%)"
    )

    ax.set_ylabel(
        "Mean classification accuracy (%)"
    )

    ax.set_xticks(
        x
    )

    ax.set_xlim(
        -2,
        32,
    )

    ax.set_ylim(
        20,
        max(
            accuracy + sd
        ) + 5,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("=" * 65)
    print("TARGET-SUBJECT CALIBRATION RESULTS")
    print("=" * 65)

    print(
        df[
            [
                "calibration_percent",
                "mean_accuracy_percent",
                "std_between_subjects_percent",
                "mean_kappa",
                "accuracy_improvement_percent_points",
            ]
        ].to_string(index=False)
    )

    baseline = df.iloc[0]
    best = df.loc[
        df["mean_accuracy_percent"].idxmax()
    ]

    print()
    print(
        f"Baseline: "
        f"{baseline['mean_accuracy_percent']:.2f}%"
    )

    print(
        f"Best: "
        f"{best['mean_accuracy_percent']:.2f}% "
        f"at {best['calibration_percent']:.0f}% calibration"
    )

    print(
        "Improvement: "
        f"{best['accuracy_improvement_percent_points']:.2f} "
        "percentage points"
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
