"""
Visualise final 20-method dissertation statistical analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(
    "dissertation_figures"
)

CS_MCNEMAR = Path(
    "results/statistical_analysis/cross_session/"
    "cross_session_mcnemar_results.csv"
)

CS_WILCOXON = Path(
    "results/statistical_analysis/cross_session/"
    "cross_session_wilcoxon_results.csv"
)

XS_WILCOXON = Path(
    "results/statistical_analysis/cross_subject/"
    "cross_subject_wilcoxon_results.csv"
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
    "CSP + LDA":
        "CSP+LDA",

    "CSP + RBF-SVM":
        "CSP+SVM",

    "Tuned CSP + LDA":
        "Tuned CSP+LDA",

    "Tuned CSP + RBF-SVM":
        "Tuned CSP+SVM",

    "FBCSP + LDA":
        "FBCSP+LDA",

    "FBCSP + RBF-SVM":
        "FBCSP+SVM",

    "Riemannian MDM":
        "Riem. MDM",

    "Riemannian TS + Shrinkage LDA":
        "Riem. TS+LDA",

    "Riemannian TS + RBF-SVM":
        "Riem. TS+SVM",

    "Filter-Bank Riemannian + Shrinkage LDA":
        "FB Riem.+LDA",

    "Filter-Bank Riemannian + RBF-SVM":
        "FB Riem.+SVM",

    "Autoencoder + LDA":
        "AE+LDA",

    "Autoencoder + RBF-SVM":
        "AE+SVM",

    "Supervised Autoencoder + LDA":
        "Sup.AE+LDA",

    "Supervised Autoencoder + RBF-SVM":
        "Sup.AE+SVM",

    "EEGNet":
        "EEGNet",

    "EA + CSP + Shrinkage LDA":
        "EA+CSP+LDA",

    "EA + CSP + RBF-SVM":
        "EA+CSP+SVM",

    "EA + FBCSP + Shrinkage LDA":
        "EA+FBCSP+LDA",

    "EA + FBCSP + RBF-SVM":
        "EA+FBCSP+SVM",
}


BEST_CS = (
    "EA + FBCSP + RBF-SVM"
)

BEST_XS = (
    "EA + CSP + Shrinkage LDA"
)


def matrix_from_results(df):

    matrix = pd.DataFrame(
        np.ones(
            (
                len(MODEL_ORDER),
                len(MODEL_ORDER),
            )
        ),
        index=MODEL_ORDER,
        columns=MODEL_ORDER,
        dtype=float,
    )

    for _, row in df.iterrows():

        a = row["model_a"]
        b = row["model_b"]

        p = float(
            row[
                "holm_adjusted_p"
            ]
        )

        matrix.loc[
            a,
            b,
        ] = p

        matrix.loc[
            b,
            a,
        ] = p

    return matrix


def significance_label(p):

    if p < 0.001:
        return "***"

    if p < 0.01:
        return "**"

    if p < 0.05:
        return "*"

    return "ns"


def heatmap(
    df,
    title,
    output_name,
):

    matrix = matrix_from_results(
        df
    )

    values = matrix.to_numpy()

    score = -np.log10(
        np.clip(
            values,
            1e-16,
            1.0,
        )
    )

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    im = ax.imshow(
        score,
        aspect="auto",
        vmin=0,
        vmax=max(
            2.0,
            float(
                np.nanmax(
                    score
                )
            ),
        ),
    )

    labels = [
        SHORT[m]
        for m in MODEL_ORDER
    ]

    ax.set_xticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=60,
        ha="right",
        fontsize=7,
    )

    ax.set_yticks(
        np.arange(
            len(labels)
        )
    )

    ax.set_yticklabels(
        labels,
        fontsize=7,
    )

    ax.set_xlabel(
        "Classification method"
    )

    ax.set_ylabel(
        "Classification method"
    )

    ax.set_title(
        title
    )

    for i in range(
        len(MODEL_ORDER)
    ):
        for j in range(
            len(MODEL_ORDER)
        ):

            if i == j:
                text = "—"

            else:
                text = significance_label(
                    values[i, j]
                )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=5.5,
            )

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        r"$-\log_{10}$(Holm-adjusted $p$)"
    )

    fig.text(
        0.5,
        0.01,
        "*** p < .001     ** p < .01     "
        "* p < .05     ns = not significant",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(
            0,
            0.03,
            1,
            1,
        )
    )

    path = (
        OUTPUT_DIR
        / output_name
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved:",
        path,
    )


def best_model_data(
    df,
    winner,
    test,
):

    records = []

    for _, row in df.iterrows():

        a = row[
            "model_a"
        ]

        b = row[
            "model_b"
        ]

        if (
            a != winner
            and b != winner
        ):
            continue

        if test == "mcnemar":

            if a == winner:

                other = b

                winner_acc = row[
                    "model_a_accuracy_percent"
                ]

                other_acc = row[
                    "model_b_accuracy_percent"
                ]

            else:

                other = a

                winner_acc = row[
                    "model_b_accuracy_percent"
                ]

                other_acc = row[
                    "model_a_accuracy_percent"
                ]

        else:

            if a == winner:

                other = b

                winner_acc = row[
                    "model_a_mean_accuracy_percent"
                ]

                other_acc = row[
                    "model_b_mean_accuracy_percent"
                ]

            else:

                other = a

                winner_acc = row[
                    "model_b_mean_accuracy_percent"
                ]

                other_acc = row[
                    "model_a_mean_accuracy_percent"
                ]

        records.append(
            {
                "model":
                    other,

                "advantage":
                    float(
                        winner_acc
                        - other_acc
                    ),

                "p":
                    float(
                        row[
                            "holm_adjusted_p"
                        ]
                    ),
            }
        )

    return (
        pd.DataFrame(
            records
        )
        .sort_values(
            "advantage",
            ascending=True,
        )
    )


def best_model_plot(
    df,
    winner,
    test,
    title,
    output_name,
):

    data = best_model_data(
        df,
        winner,
        test,
    )

    labels = [
        SHORT[x]
        for x in data[
            "model"
        ]
    ]

    values = data[
        "advantage"
    ].to_numpy()

    p_values = data[
        "p"
    ].to_numpy()

    y = np.arange(
        len(data)
    )

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

    bars = ax.barh(
        y,
        values,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        labels,
        fontsize=8,
    )

    ax.set_xlabel(
        "Winner accuracy advantage "
        "(percentage points)"
    )

    ax.set_ylabel(
        "Comparison method"
    )

    ax.set_title(
        title
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    for bar, value, p in zip(
        bars,
        values,
        p_values,
    ):

        label = (
            f"{value:+.2f} pp, "
            f"{significance_label(p)}"
        )

        x = (
            value + 0.25
            if value >= 0
            else value - 0.25
        )

        ha = (
            "left"
            if value >= 0
            else "right"
        )

        ax.text(
            x,
            bar.get_y()
            + bar.get_height() / 2,
            label,
            va="center",
            ha=ha,
            fontsize=7,
        )

    fig.text(
        0.5,
        0.01,
        "Significance labels use "
        "Holm-adjusted p-values",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=(
            0,
            0.03,
            1,
            1,
        )
    )

    path = (
        OUTPUT_DIR
        / output_name
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved:",
        path,
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cs_mcnemar = pd.read_csv(
        CS_MCNEMAR
    )

    cs_wilcoxon = pd.read_csv(
        CS_WILCOXON
    )

    xs_wilcoxon = pd.read_csv(
        XS_WILCOXON
    )

    expected_pairs = 190

    for name, df in [
        (
            "Cross-Session McNemar",
            cs_mcnemar,
        ),
        (
            "Cross-Session Wilcoxon",
            cs_wilcoxon,
        ),
        (
            "Cross-Subject Wilcoxon",
            xs_wilcoxon,
        ),
    ]:

        if len(df) != expected_pairs:
            raise RuntimeError(
                f"{name}: expected "
                f"{expected_pairs} pairs, "
                f"found {len(df)}."
            )

    heatmap(
        cs_mcnemar,
        (
            "Cross-Session Pairwise Significance\n"
            "Exact McNemar Test with Holm Correction"
        ),
        "statistical_cross_session_mcnemar.png",
    )

    heatmap(
        cs_wilcoxon,
        (
            "Cross-Session Subject-Level Significance\n"
            "Wilcoxon Signed-Rank Test with Holm Correction"
        ),
        "statistical_cross_session_wilcoxon.png",
    )

    heatmap(
        xs_wilcoxon,
        (
            "Cross-Subject Pairwise Significance\n"
            "Wilcoxon Signed-Rank Test with Holm Correction"
        ),
        "statistical_cross_subject_wilcoxon.png",
    )

    best_model_plot(
        cs_mcnemar,
        BEST_CS,
        "mcnemar",
        (
            "Cross-Session Winner Comparisons\n"
            "EA + FBCSP + RBF-SVM — Exact McNemar Test"
        ),
        "statistical_cross_session_best_model_mcnemar.png",
    )

    best_model_plot(
        cs_wilcoxon,
        BEST_CS,
        "wilcoxon",
        (
            "Cross-Session Winner Comparisons\n"
            "EA + FBCSP + RBF-SVM — Wilcoxon Test"
        ),
        "statistical_cross_session_best_model_wilcoxon.png",
    )

    best_model_plot(
        xs_wilcoxon,
        BEST_XS,
        "wilcoxon",
        (
            "Cross-Subject Winner Comparisons\n"
            "EA + CSP + Shrinkage LDA — Wilcoxon Test"
        ),
        "statistical_cross_subject_best_model_wilcoxon.png",
    )

    print()
    print("=" * 80)
    print("✅ FINAL STATISTICAL FIGURES GENERATED")
    print("=" * 80)


if __name__ == "__main__":
    main()
