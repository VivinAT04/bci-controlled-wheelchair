"""
Create trajectory-density heatmaps for EEG-classifier and random-baseline
wheelchair navigation.

Run from the project root:

    python -m scripts.simulation.plot_trajectory_heatmaps
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRID_ROWS = 20
GRID_COLS = 20

START_POSITION = (18, 1)
TARGET_POSITION = (1, 18)

INPUT_PATH = Path("results/simulation/trajectories/multiple_trajectory_data.csv")
OUTPUT_DIRECTORY = Path("results/simulation/trajectories/trajectory_figures")


def build_visit_density(
    dataframe: pd.DataFrame,
    controller: str,
) -> np.ndarray:
    """
    Count how many unique trajectories visited each grid cell.

    A cell is counted at most once per trajectory, even if the wheelchair
    remained there for several consecutive steps.
    """
    controller_data = dataframe[
        dataframe["controller"] == controller
    ].copy()

    if controller_data.empty:
        raise ValueError(
            f"No trajectory data found for controller: {controller}"
        )

    density = np.zeros(
        (GRID_ROWS, GRID_COLS),
        dtype=int,
    )

    for _, trajectory in controller_data.groupby("trajectory_id"):
        visited_cells = set(
            zip(
                trajectory["row"].astype(int),
                trajectory["column"].astype(int),
            )
        )

        for row, column in visited_cells:
            if 0 <= row < GRID_ROWS and 0 <= column < GRID_COLS:
                density[row, column] += 1

    return density


def configure_axis(ax, title: str) -> None:
    """Configure the grid and label start and target points."""
    ax.set_xlim(-0.5, GRID_COLS - 0.5)
    ax.set_ylim(GRID_ROWS - 0.5, -0.5)

    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))

    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.set_title(title)

    ax.set_aspect("equal")

    start_row, start_column = START_POSITION
    target_row, target_column = TARGET_POSITION

    ax.scatter(
        start_column,
        start_row,
        s=180,
        marker="o",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label="Point A — Start",
    )

    ax.scatter(
        target_column,
        target_row,
        s=260,
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        zorder=5,
        label="Point B — Target",
    )

    ax.annotate(
        "A",
        xy=(start_column, start_row),
        xytext=(-12, 10),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
    )

    ax.annotate(
        "B",
        xy=(target_column, target_row),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
    )

    ax.legend(loc="upper left")


def save_single_heatmap(
    density: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Save one controller's trajectory-density heatmap."""
    figure, ax = plt.subplots(figsize=(10, 9))

    image = ax.imshow(
        density,
        origin="upper",
        interpolation="nearest",
        vmin=0,
        vmax=100,
    )

    configure_axis(ax, title)

    colorbar = figure.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Number of trajectories visiting cell"
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def save_comparison_heatmap(
    classifier_density: np.ndarray,
    random_density: np.ndarray,
    output_path: Path,
) -> None:
    """Save classifier and random heatmaps side by side."""
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(18, 9),
    )

    maximum_density = max(
        int(classifier_density.max()),
        int(random_density.max()),
    )

    classifier_image = axes[0].imshow(
        classifier_density,
        origin="upper",
        interpolation="nearest",
        vmin=0,
        vmax=maximum_density,
    )

    axes[1].imshow(
        random_density,
        origin="upper",
        interpolation="nearest",
        vmin=0,
        vmax=maximum_density,
    )

    configure_axis(
        axes[0],
        "EEG Classifier\nTrajectory Density",
    )

    configure_axis(
        axes[1],
        "Random Baseline\nTrajectory Density",
    )

    colorbar = figure.colorbar(
        classifier_image,
        ax=axes,
        fraction=0.025,
        pad=0.02,
    )

    colorbar.set_label(
        "Number of trajectories visiting cell"
    )

    figure.suptitle(
        (
            "Wheelchair Trajectory Density from Point A to Point B\n"
            f"A {START_POSITION} → B {TARGET_POSITION}, n=100 per controller"
        ),
        fontsize=16,
        fontweight="bold",
    )

    figure.subplots_adjust(
        top=0.86,
        wspace=0.20,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def print_density_summary(
    controller_name: str,
    density: np.ndarray,
) -> None:
    """Print a compact density summary."""
    visited_cells = int(np.count_nonzero(density))
    maximum_visit_count = int(density.max())

    print("\n----------------------------------------")
    print(controller_name)
    print("----------------------------------------")
    print(f"Grid cells visited: {visited_cells}")
    print(f"Maximum trajectories through one cell: {maximum_visit_count}")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. "
            "Run `python -m scripts.simulation.plot_multiple_trajectories` first."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(INPUT_PATH)

    required_columns = {
        "controller",
        "trajectory_id",
        "row",
        "column",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Trajectory CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    classifier_density = build_visit_density(
        dataframe=dataframe,
        controller="EEG_classifier",
    )

    random_density = build_visit_density(
        dataframe=dataframe,
        controller="Random_baseline",
    )

    print_density_summary(
        controller_name="EEG Classifier",
        density=classifier_density,
    )

    print_density_summary(
        controller_name="Random Baseline",
        density=random_density,
    )

    save_single_heatmap(
        density=classifier_density,
        title=(
            "EEG Classifier Trajectory Density\n"
            f"A {START_POSITION} → B {TARGET_POSITION}, n=100"
        ),
        output_path=(
            OUTPUT_DIRECTORY
            / "eeg_classifier_trajectory_heatmap.png"
        ),
    )

    save_single_heatmap(
        density=random_density,
        title=(
            "Random Baseline Trajectory Density\n"
            f"A {START_POSITION} → B {TARGET_POSITION}, n=100"
        ),
        output_path=(
            OUTPUT_DIRECTORY
            / "random_baseline_trajectory_heatmap.png"
        ),
    )

    save_comparison_heatmap(
        classifier_density=classifier_density,
        random_density=random_density,
        output_path=(
            OUTPUT_DIRECTORY
            / "classifier_vs_random_trajectory_heatmaps.png"
        ),
    )

    print("\n========================================")
    print("Trajectory heatmaps completed")
    print("========================================")


if __name__ == "__main__":
    main()