"""Generate dissertation-level SHAP feature-importance visualisations."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


INPUT_ROOT = Path(
    "results/explainability/all_subjects/shap_all_subjects"
)

OUTPUT_DIR = Path(
    "dissertation_figures"
)

SUBJECTS = [
    "A01T", "A02T", "A03T", "A04T", "A05T",
    "A06T", "A07T", "A08T", "A09T",
]

CLASSES = [
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
]


def load_overall_importance() -> pd.DataFrame:
    """Load overall SHAP importance for every subject."""

    frames = []

    for subject in SUBJECTS:
        path = (
            INPUT_ROOT
            / subject
            / "overall"
            / "importance.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing SHAP file: {path}"
            )

        df = pd.read_csv(path)
        df["subject"] = subject
        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def load_class_importance() -> pd.DataFrame:
    """Load class-wise SHAP importance across all subjects."""

    frames = []

    for subject in SUBJECTS:
        for class_name in CLASSES:

            path = (
                INPUT_ROOT
                / subject
                / class_name
                / "importance.csv"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing SHAP file: {path}"
                )

            df = pd.read_csv(path)
            df["subject"] = subject
            df["class"] = class_name
            frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def extract_frequency_band(feature_name: str) -> str:
    """Extract frequency band from a feature such as 24-28Hz_CSP2."""

    match = re.match(
        r"(\d+-\d+Hz)_CSP\d+",
        feature_name,
    )

    if match is None:
        raise ValueError(
            f"Unexpected feature name: {feature_name}"
        )

    return match.group(1)


def global_feature_importance(
    overall_df: pd.DataFrame,
) -> pd.DataFrame:
    """Average SHAP importance for each FBCSP feature across subjects."""

    result = (
        overall_df
        .groupby("feature", as_index=False)
        ["mean_abs_shap"]
        .mean()
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
    )

    return result


def frequency_band_importance(
    overall_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate SHAP importance by frequency band."""

    df = overall_df.copy()

    df["frequency_band"] = (
        df["feature"]
        .map(extract_frequency_band)
    )

    result = (
        df.groupby(
            ["subject", "frequency_band"],
            as_index=False,
        )["mean_abs_shap"]
        .mean()
        .groupby(
            "frequency_band",
            as_index=False,
        )["mean_abs_shap"]
        .mean()
    )

    result["low_frequency"] = (
        result["frequency_band"]
        .str.extract(r"(\d+)-")[0]
        .astype(int)
    )

    result = (
        result
        .sort_values("low_frequency")
        .drop(columns="low_frequency")
    )

    return result


def class_frequency_matrix(
    class_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create class-by-frequency-band SHAP matrix."""

    df = class_df.copy()

    df["frequency_band"] = (
        df["feature"]
        .map(extract_frequency_band)
    )

    grouped = (
        df.groupby(
            [
                "subject",
                "class",
                "frequency_band",
            ],
            as_index=False,
        )["mean_abs_shap"]
        .mean()
    )

    grouped = (
        grouped.groupby(
            [
                "class",
                "frequency_band",
            ],
            as_index=False,
        )["mean_abs_shap"]
        .mean()
    )

    matrix = grouped.pivot(
        index="class",
        columns="frequency_band",
        values="mean_abs_shap",
    )

    ordered_bands = sorted(
        matrix.columns,
        key=lambda value: int(
            value.split("-")[0]
        ),
    )

    matrix = matrix[ordered_bands]

    matrix = matrix.reindex(CLASSES)

    return matrix


def save_global_feature_plot(
    dataframe: pd.DataFrame,
) -> None:
    """Save global FBCSP SHAP ranking."""

    top = (
        dataframe
        .head(15)
        .sort_values("mean_abs_shap")
    )

    figure, axis = plt.subplots(
        figsize=(9, 6.5)
    )

    axis.barh(
        top["feature"],
        top["mean_abs_shap"],
    )

    axis.set_xlabel(
        "Mean absolute SHAP value"
    )

    axis.set_ylabel(
        "FBCSP feature"
    )

    axis.set_title(
        "Global FBCSP Feature Importance Across Subjects"
    )

    figure.tight_layout()

    output = (
        OUTPUT_DIR
        / "shap_cross_session_global_feature_importance.png"
    )

    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved {output}")


def save_frequency_plot(
    dataframe: pd.DataFrame,
) -> None:
    """Save aggregated SHAP importance by frequency band."""

    figure, axis = plt.subplots(
        figsize=(8.5, 5.5)
    )

    axis.bar(
        dataframe["frequency_band"],
        dataframe["mean_abs_shap"],
    )

    axis.set_xlabel(
        "Frequency band"
    )

    axis.set_ylabel(
        "Mean absolute SHAP value"
    )

    axis.set_title(
        "SHAP Importance by EEG Frequency Band"
    )

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    figure.tight_layout()

    output = (
        OUTPUT_DIR
        / "shap_cross_session_frequency_band_importance.png"
    )

    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved {output}")


def save_class_frequency_heatmap(
    matrix: pd.DataFrame,
) -> None:
    """Save class-by-frequency-band SHAP heatmap."""

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    image = axis.imshow(
        matrix.values,
        aspect="auto",
    )

    axis.set_xticks(
        range(len(matrix.columns))
    )

    axis.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
    )

    class_labels = {
        "left_hand": "Left hand",
        "right_hand": "Right hand",
        "feet": "Feet",
        "tongue": "Tongue",
    }

    axis.set_yticks(
        range(len(matrix.index))
    )

    axis.set_yticklabels(
        [
            class_labels[value]
            for value in matrix.index
        ]
    )

    axis.set_xlabel(
        "Frequency band"
    )

    axis.set_ylabel(
        "Motor-imagery class"
    )

    axis.set_title(
        "SHAP Importance Across Motor-Imagery Classes and Frequency Bands"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Mean absolute SHAP value"
    )

    figure.tight_layout()

    output = (
        OUTPUT_DIR
        / "shap_cross_session_class_frequency_heatmap.png"
    )

    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved {output}")


def main() -> None:
    """Generate dissertation-level SHAP figures."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_df = (
        load_overall_importance()
    )

    class_df = (
        load_class_importance()
    )

    global_df = (
        global_feature_importance(
            overall_df
        )
    )

    frequency_df = (
        frequency_band_importance(
            overall_df
        )
    )

    heatmap_df = (
        class_frequency_matrix(
            class_df
        )
    )

    global_df.to_csv(
        OUTPUT_DIR
        / "shap_cross_session_global_feature_importance.csv",
        index=False,
    )

    frequency_df.to_csv(
        OUTPUT_DIR
        / "shap_cross_session_frequency_band_importance.csv",
        index=False,
    )

    heatmap_df.to_csv(
        OUTPUT_DIR
        / "shap_cross_session_class_frequency_heatmap.csv",
    )

    print("\nTop 15 global SHAP features:")
    print(
        global_df
        .head(15)
        .to_string(index=False)
    )

    print("\nFrequency-band SHAP importance:")
    print(
        frequency_df
        .to_string(index=False)
    )

    print("\nClass × frequency-band matrix:")
    print(
        heatmap_df
        .round(6)
        .to_string()
    )

    save_global_feature_plot(
        global_df
    )

    save_frequency_plot(
        frequency_df
    )

    save_class_frequency_heatmap(
        heatmap_df
    )


if __name__ == "__main__":
    main()
