"""
Plot grid-size experiment results.

Run from the project root:

    python -m scripts.plot_grid_size_experiments
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path(
    "results/grid_size_experiment_summary.csv"
)

OUTPUT_DIRECTORY = Path(
    "results/grid_size_experiment_figures"
)


def save_line_plot(
    dataframe: pd.DataFrame,
    y_column: str,
    y_label: str,
    filename: str,
) -> None:
    """Create and save one grid-size line plot."""

    plt.figure(figsize=(7, 5))

    plt.plot(
        dataframe["grid_size"],
        dataframe[y_column],
        marker="o",
        linewidth=2,
    )

    plt.xlabel("Grid Size")
    plt.ylabel(y_label)

    plt.xticks(
        dataframe["grid_size"],
        dataframe["grid_label"],
    )

    plt.grid(True)
    plt.tight_layout()

    output_path = OUTPUT_DIRECTORY / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def main() -> None:
    """Generate all grid-size experiment figures."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Summary file not found: {INPUT_PATH}"
        )

    dataframe = pd.read_csv(INPUT_PATH)

    required_columns = {
        "grid_size",
        "grid_label",
        "success_rate_percent",
        "mean_initial_distance",
        "mean_steps_all",
        "mean_final_distance",
        "mean_blocked_moves",
        "mean_stop_commands",
    }

    missing_columns = (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    dataframe = dataframe.sort_values(
        "grid_size"
    ).reset_index(drop=True)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_line_plot(
        dataframe=dataframe,
        y_column="success_rate_percent",
        y_label="Navigation Success Rate (%)",
        filename="success_rate_vs_grid_size.png",
    )

    save_line_plot(
        dataframe=dataframe,
        y_column="mean_steps_all",
        y_label="Mean Steps",
        filename="mean_steps_vs_grid_size.png",
    )

    save_line_plot(
        dataframe=dataframe,
        y_column="mean_initial_distance",
        y_label="Mean Initial Distance",
        filename="initial_distance_vs_grid_size.png",
    )

    save_line_plot(
        dataframe=dataframe,
        y_column="mean_final_distance",
        y_label="Mean Final Distance",
        filename="final_distance_vs_grid_size.png",
    )

    save_line_plot(
        dataframe=dataframe,
        y_column="mean_blocked_moves",
        y_label="Mean Blocked Moves",
        filename="blocked_moves_vs_grid_size.png",
    )

    save_line_plot(
        dataframe=dataframe,
        y_column="mean_stop_commands",
        y_label="Mean Stop Commands",
        filename="stop_commands_vs_grid_size.png",
    )

    print("\nAll figures saved to:")
    print(OUTPUT_DIRECTORY.resolve())


if __name__ == "__main__":
    main()
