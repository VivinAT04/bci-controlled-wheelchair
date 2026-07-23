"""
Plot grid-size experiment results.

Run from the project root:

    python -m scripts.simulation.plot_grid_size_experiments
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path("results/simulation/grid_size/grid_size_experiment_summary.csv")
OUTPUT_DIRECTORY = Path("results/simulation/grid_size/grid_size_experiment_figures")


def save_line_plot(
    dataframe: pd.DataFrame,
    y_column: str,
    y_label: str,
    filename: str,
) -> None:
    """Create and save one grid-size line plot."""

    print(f"Creating: {filename}")

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.plot(
        dataframe["grid_size"],
        dataframe[y_column],
        marker="o",
        linewidth=2,
    )

    axis.set_xlabel("Grid Size")
    axis.set_ylabel(y_label)

    axis.set_xticks(
        dataframe["grid_size"],
        dataframe["grid_label"],
    )

    axis.grid(True)
    figure.tight_layout()

    output_path = OUTPUT_DIRECTORY / filename

    print(f"Saving: {output_path}")

    figure.savefig(
        output_path,
        dpi=150,
        format="png",
    )

    plt.close(figure)

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

    missing_columns = required_columns - set(dataframe.columns)

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
