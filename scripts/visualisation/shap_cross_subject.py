"""
Generate dissertation figures for the final Cross-Subject SHAP analysis.

Protocol:
    Cross-Subject T-to-E LOSO

Final model:
    EA + CSP + Shrinkage LDA

Final performance:
    Accuracy = 52.74%
    Cohen's kappa = 0.370

Inputs:
    results/dissertation_figure_data/
        shap_cross_subject_global_feature_importance.csv
        shap_cross_subject_class_feature_heatmap.csv

Outputs:
    dissertation_figures/
        shap_cross_subject_global_feature_importance.png
        shap_cross_subject_class_feature_heatmap.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path(
    "results/dissertation_figure_data"
)

FIGURE_DIR = Path(
    "dissertation_figures"
)

GLOBAL_INPUT = (
    DATA_DIR
    / "shap_cross_subject_global_feature_importance.csv"
)

CLASS_INPUT = (
    DATA_DIR
    / "shap_cross_subject_class_feature_heatmap.csv"
)

GLOBAL_OUTPUT = (
    FIGURE_DIR
    / "shap_cross_subject_global_feature_importance.png"
)

CLASS_OUTPUT = (
    FIGURE_DIR
    / "shap_cross_subject_class_feature_heatmap.png"
)


def validate_inputs():
    """Check required SHAP data files."""

    required = [
        GLOBAL_INPUT,
        CLASS_INPUT,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

    print("Input files verified:")
    for path in required:
        print(f"  ✅ {path}")


def detect_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    """Find a column using accepted candidate names."""

    for column in candidates:
        if column in dataframe.columns:
            return column

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns: "
        f"{dataframe.columns.tolist()}"
    )


def plot_global_importance():
    """Plot mean absolute SHAP importance of CSP features."""

    dataframe = pd.read_csv(
        GLOBAL_INPUT
    )

    print()
    print("Global importance columns:")
    print(dataframe.columns.tolist())

    feature_column = detect_column(
        dataframe,
        [
            "feature",
            "Feature",
            "feature_name",
        ],
        "feature",
    )

    importance_column = detect_column(
        dataframe,
        [
            "mean_abs_shap",
            "mean_absolute_shap",
            "importance",
            "shap_importance",
        ],
        "SHAP importance",
    )

    dataframe = dataframe[
        [
            feature_column,
            importance_column,
        ]
    ].copy()

    dataframe[importance_column] = (
        pd.to_numeric(
            dataframe[importance_column],
            errors="raise",
        )
    )

    dataframe = dataframe.sort_values(
        importance_column,
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 6.5)
    )

    ax.barh(
        dataframe[feature_column],
        dataframe[importance_column],
    )

    ax.set_xlabel(
        "Mean absolute SHAP value",
        fontsize=12,
    )

    ax.set_ylabel(
        "CSP feature",
        fontsize=12,
    )

    ax.set_title(
        "Cross-Subject SHAP Global Feature Importance\n"
        "EA + CSP + Shrinkage LDA",
        fontsize=14,
        pad=14,
    )

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        GLOBAL_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"✅ Saved: {GLOBAL_OUTPUT}"
    )


def prepare_heatmap(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the class-feature SHAP CSV into a feature x class matrix.

    Supports either:
    1. long format:
       class, feature, mean_abs_shap

    or

    2. matrix/wide format:
       feature, left_hand, right_hand, feet, tongue
    """

    print()
    print("Class-feature columns:")
    print(dataframe.columns.tolist())

    columns = set(
        dataframe.columns
    )

    feature_candidates = [
        "feature",
        "Feature",
        "feature_name",
    ]

    class_candidates = [
        "class",
        "Class",
        "class_name",
        "label",
    ]

    value_candidates = [
        "mean_abs_shap",
        "mean_absolute_shap",
        "importance",
        "shap_importance",
    ]

    feature_column = next(
        (
            column
            for column in feature_candidates
            if column in columns
        ),
        None,
    )

    class_column = next(
        (
            column
            for column in class_candidates
            if column in columns
        ),
        None,
    )

    value_column = next(
        (
            column
            for column in value_candidates
            if column in columns
        ),
        None,
    )

    # --------------------------------------------------
    # LONG FORMAT
    # --------------------------------------------------

    if (
        feature_column is not None
        and class_column is not None
        and value_column is not None
    ):
        matrix = dataframe.pivot_table(
            index=feature_column,
            columns=class_column,
            values=value_column,
            aggfunc="mean",
        )

        return matrix

    # --------------------------------------------------
    # WIDE FORMAT
    # --------------------------------------------------

    if feature_column is not None:
        matrix = dataframe.set_index(
            feature_column
        )

        numeric_columns = []

        for column in matrix.columns:
            converted = pd.to_numeric(
                matrix[column],
                errors="coerce",
            )

            if converted.notna().all():
                matrix[column] = converted
                numeric_columns.append(
                    column
                )

        if not numeric_columns:
            raise ValueError(
                "No numeric class importance "
                "columns found in heatmap CSV."
            )

        return matrix[
            numeric_columns
        ]

    # --------------------------------------------------
    # ALREADY MATRIX-LIKE
    # --------------------------------------------------

    numeric = dataframe.select_dtypes(
        include=[np.number]
    )

    if numeric.empty:
        raise ValueError(
            "Could not interpret "
            "class-feature heatmap CSV."
        )

    return numeric


def normalise_class_names(
    name: str,
) -> str:
    """Use readable motor-imagery class labels."""

    mapping = {
        "left_hand": "Left Hand",
        "right_hand": "Right Hand",
        "feet": "Feet",
        "tongue": "Tongue",
        "Left Hand": "Left Hand",
        "Right Hand": "Right Hand",
        "Feet": "Feet",
        "Tongue": "Tongue",
    }

    return mapping.get(
        str(name),
        str(name).replace(
            "_",
            " ",
        ).title(),
    )


def plot_class_feature_heatmap():
    """Plot class-specific CSP SHAP importance."""

    dataframe = pd.read_csv(
        CLASS_INPUT
    )

    matrix = prepare_heatmap(
        dataframe
    )

    preferred_features = [
        f"CSP{i}"
        for i in range(1, 11)
    ]

    if all(
        feature in matrix.index
        for feature in preferred_features
    ):
        matrix = matrix.loc[
            preferred_features
        ]

    preferred_classes = [
        "left_hand",
        "right_hand",
        "feet",
        "tongue",
    ]

    if all(
        class_name in matrix.columns
        for class_name in preferred_classes
    ):
        matrix = matrix[
            preferred_classes
        ]

    matrix.columns = [
        normalise_class_names(
            column
        )
        for column in matrix.columns
    ]

    values = matrix.to_numpy(
        dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 7.5)
    )

    image = ax.imshow(
        values,
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(
            len(matrix.columns)
        )
    )

    ax.set_xticklabels(
        matrix.columns,
        fontsize=11,
    )

    ax.set_yticks(
        np.arange(
            len(matrix.index)
        )
    )

    ax.set_yticklabels(
        matrix.index,
        fontsize=11,
    )

    ax.set_xlabel(
        "Motor imagery class",
        fontsize=12,
    )

    ax.set_ylabel(
        "CSP feature",
        fontsize=12,
    )

    ax.set_title(
        "Cross-Subject Class-Specific SHAP Importance\n"
        "EA + CSP + Shrinkage LDA",
        fontsize=14,
        pad=14,
    )

    colourbar = fig.colorbar(
        image,
        ax=ax,
    )

    colourbar.set_label(
        "Mean absolute SHAP value",
        fontsize=11,
    )

    threshold = (
        np.nanmax(values)
        + np.nanmin(values)
    ) / 2.0

    for row in range(
        values.shape[0]
    ):
        for column in range(
            values.shape[1]
        ):
            value = values[
                row,
                column,
            ]

            if np.isnan(value):
                continue

            text_colour = (
                "white"
                if value > threshold
                else "black"
            )

            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_colour,
            )

    fig.tight_layout()

    fig.savefig(
        CLASS_OUTPUT,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"✅ Saved: {CLASS_OUTPUT}"
    )


def main():
    """Generate final Cross-Subject SHAP figures."""

    print(
        "=" * 76
    )
    print(
        "CROSS-SUBJECT SHAP FIGURES"
    )
    print(
        "EA + CSP + SHRINKAGE LDA"
    )
    print(
        "=" * 76
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_inputs()

    plot_global_importance()
    plot_class_feature_heatmap()

    print()
    print(
        "=" * 76
    )
    print(
        "✅ CROSS-SUBJECT SHAP FIGURES COMPLETE"
    )
    print(
        "=" * 76
    )


if __name__ == "__main__":
    main()
