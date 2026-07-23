"""
Generate visible wheelchair navigation paths from a fixed point A to B.

This script compares:

1. EEG classifier-controlled navigation
2. Uniform random baseline navigation
3. A side-by-side path comparison

Run from the project root:

    python -m scripts.simulation.plot_multiple_trajectories
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from bci_wheelchair.simulation import EEGPredictionSampler
from bci_wheelchair.simulation import RandomPredictionSampler
from bci_wheelchair.simulation import (
    GridEnvironment,
    Position,
    WheelchairState,
)
from bci_wheelchair.simulation import (
    SimulationResult,
    run_classifier_simulation,
)


# ============================================================
# Configuration
# ============================================================

GRID_ROWS = 20
GRID_COLS = 20

START_POSITION: Position = (18, 1)
TARGET_POSITION: Position = (1, 18)
START_HEADING = 0

N_TRAJECTORIES = 100
N_DISPLAYED_TRAJECTORIES = 20
MAX_STEPS = 500

CLASSIFIER_SEED = 42
RANDOM_BASELINE_SEED = 42

PREDICTIONS_PATH = Path("results/within_subject/predictions/test_predicted_commands.csv")
OUTPUT_DIRECTORY = Path("results/simulation/trajectories/trajectory_figures")
TRAJECTORY_DATA_PATH = Path(
    "results/simulation/trajectories/multiple_trajectory_data.csv"
)


# ============================================================
# Simulation
# ============================================================

def run_trajectories(
    controller_name: str,
    sampler,
    environment: GridEnvironment,
    n_trajectories: int,
) -> list[SimulationResult]:
    """
    Run multiple navigation simulations using the same A and B points.
    """
    results: list[SimulationResult] = []

    for trajectory_id in range(1, n_trajectories + 1):
        start_state = WheelchairState(
            position=START_POSITION,
            heading=START_HEADING,
        )

        result = run_classifier_simulation(
            environment=environment,
            sampler=sampler,
            start_state=start_state,
            target=TARGET_POSITION,
            max_steps=MAX_STEPS,
        )

        results.append(result)

        if trajectory_id % 10 == 0:
            print(
                f"{controller_name}: completed "
                f"{trajectory_id}/{n_trajectories}"
            )

    return results


# ============================================================
# Data export
# ============================================================

def create_trajectory_records(
    controller_name: str,
    results: Sequence[SimulationResult],
) -> list[dict]:
    """
    Convert trajectories into long-form CSV rows.
    """
    records: list[dict] = []

    for trajectory_id, result in enumerate(results, start=1):
        for path_step, position in enumerate(result.path):
            row, column = position

            records.append(
                {
                    "controller": controller_name,
                    "trajectory_id": trajectory_id,
                    "path_step": path_step,
                    "row": row,
                    "column": column,
                    "reached_target": result.reached_target,
                    "total_steps": result.steps,
                    "final_distance": result.final_distance,
                }
            )

    return records


# ============================================================
# Summary
# ============================================================

def print_summary(
    controller_name: str,
    results: Sequence[SimulationResult],
) -> None:
    """
    Print navigation statistics.
    """
    number_of_runs = len(results)

    successful_results = [
        result
        for result in results
        if result.reached_target
    ]

    success_rate = (
        len(successful_results) / number_of_runs * 100
        if number_of_runs
        else 0.0
    )

    mean_steps_all = (
        sum(result.steps for result in results) / number_of_runs
        if number_of_runs
        else float("nan")
    )

    mean_final_distance = (
        sum(result.final_distance for result in results)
        / number_of_runs
        if number_of_runs
        else float("nan")
    )

    if successful_results:
        mean_successful_steps = (
            sum(result.steps for result in successful_results)
            / len(successful_results)
        )
    else:
        mean_successful_steps = float("nan")

    print("\n----------------------------------------")
    print(controller_name)
    print("----------------------------------------")
    print(f"Trajectories: {number_of_runs}")
    print(f"Successful trajectories: {len(successful_results)}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Mean steps, all runs: {mean_steps_all:.2f}")
    print(
        "Mean steps, successful runs: "
        f"{mean_successful_steps:.2f}"
    )
    print(
        f"Mean final distance: {mean_final_distance:.2f}"
    )


# ============================================================
# Plot configuration
# ============================================================

def configure_grid(
    ax,
    title: str,
) -> None:
    """
    Configure a clear 20 x 20 navigation grid.
    """
    ax.set_xlim(-0.5, GRID_COLS - 0.5)
    ax.set_ylim(GRID_ROWS - 0.5, -0.5)

    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))

    ax.grid(
        True,
        which="major",
        linewidth=0.8,
        alpha=0.7,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.set_title(title)

    start_row, start_column = START_POSITION
    target_row, target_column = TARGET_POSITION

    ax.scatter(
        start_column,
        start_row,
        marker="o",
        s=180,
        edgecolors="black",
        linewidths=1.2,
        zorder=10,
        label="Point A — Start",
    )

    ax.scatter(
        target_column,
        target_row,
        marker="*",
        s=260,
        edgecolors="black",
        linewidths=1.2,
        zorder=10,
        label="Point B — Target",
    )

    ax.annotate(
        "A",
        xy=(start_column, start_row),
        xytext=(-12, 10),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        zorder=11,
    )

    ax.annotate(
        "B",
        xy=(target_column, target_row),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        zorder=11,
    )


# ============================================================
# Individual controller plot
# ============================================================

def plot_controller_trajectories(
    results: Sequence[SimulationResult],
    controller_title: str,
    output_path: Path,
) -> None:
    """
    Plot a visible sample of navigation paths for one controller.
    """
    displayed_results = list(
        results[:N_DISPLAYED_TRAJECTORIES]
    )

    figure, ax = plt.subplots(figsize=(10, 10))

    successful_count = sum(
        result.reached_target
        for result in displayed_results
    )

    for result in displayed_results:
        rows = [
            position[0]
            for position in result.path
        ]

        columns = [
            position[1]
            for position in result.path
        ]

        ax.plot(
            columns,
            rows,
            linewidth=1.8,
            alpha=0.65,
            marker="o",
            markersize=2.5,
            markevery=4,
            zorder=3,
        )

    success_rate = (
        successful_count
        / len(displayed_results)
        * 100
        if displayed_results
        else 0.0
    )

    configure_grid(
        ax,
        (
            f"{controller_title}: Multiple Navigation Paths\n"
            f"A {START_POSITION} → B {TARGET_POSITION} | "
            f"{len(displayed_results)} displayed | "
            f"Success={success_rate:.1f}%"
        ),
    )

    ax.legend(loc="upper left")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


# ============================================================
# Comparison plot
# ============================================================

def plot_comparison(
    classifier_results: Sequence[SimulationResult],
    random_results: Sequence[SimulationResult],
    output_path: Path,
) -> None:
    """
    Plot visible classifier and random paths side by side.
    """
    classifier_displayed = list(
        classifier_results[:N_DISPLAYED_TRAJECTORIES]
    )

    random_displayed = list(
        random_results[:N_DISPLAYED_TRAJECTORIES]
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(18, 9),
    )

    datasets = [
        (
            axes[0],
            classifier_displayed,
            "EEG Classifier",
        ),
        (
            axes[1],
            random_displayed,
            "Random Baseline",
        ),
    ]

    for ax, results, title in datasets:
        successful_count = sum(
            result.reached_target
            for result in results
        )

        for result in results:
            rows = [
                position[0]
                for position in result.path
            ]

            columns = [
                position[1]
                for position in result.path
            ]

            ax.plot(
                columns,
                rows,
                linewidth=1.8,
                alpha=0.65,
                marker="o",
                markersize=2.5,
                markevery=4,
                zorder=3,
            )

        success_rate = (
            successful_count
            / len(results)
            * 100
            if results
            else 0.0
        )

        configure_grid(
            ax,
            (
                f"{title}\n"
                f"{len(results)} displayed trajectories | "
                f"Success={success_rate:.1f}%"
            ),
        )

        ax.legend(loc="upper left")

    figure.suptitle(
        (
            "Multiple Wheelchair Navigation Paths "
            "from Point A to Point B\n"
            f"A {START_POSITION} → B {TARGET_POSITION}"
        ),
        fontsize=16,
        fontweight="bold",
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


# ============================================================
# Successful classifier plot
# ============================================================

def plot_successful_classifier_trajectories(
    results: Sequence[SimulationResult],
    output_path: Path,
) -> None:
    """
    Plot visible successful classifier paths only.
    """
    successful_results = [
        result
        for result in results
        if result.reached_target
    ]

    displayed_results = successful_results[
        :N_DISPLAYED_TRAJECTORIES
    ]

    if not displayed_results:
        print(
            "No successful classifier trajectories available."
        )
        return

    figure, ax = plt.subplots(figsize=(10, 10))

    for result in displayed_results:
        rows = [
            position[0]
            for position in result.path
        ]

        columns = [
            position[1]
            for position in result.path
        ]

        ax.plot(
            columns,
            rows,
            linewidth=1.8,
            alpha=0.65,
            marker="o",
            markersize=2.5,
            markevery=4,
            zorder=3,
        )

    configure_grid(
        ax,
        (
            "EEG Classifier: Successful Paths Only\n"
            f"A {START_POSITION} → B {TARGET_POSITION} | "
            f"{len(displayed_results)} displayed"
        ),
    )

    ax.legend(loc="upper left")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Run simulations and generate all trajectory figures.
    """
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {PREDICTIONS_PATH}. "
            "Run `python -m scripts.within_subject.export_predictions` first."
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    environment = GridEnvironment(
        rows=GRID_ROWS,
        cols=GRID_COLS,
    )

    classifier_sampler = EEGPredictionSampler(
        csv_path=PREDICTIONS_PATH,
        random_seed=CLASSIFIER_SEED,
    )

    random_sampler = RandomPredictionSampler(
        random_seed=RANDOM_BASELINE_SEED,
    )

    print("\n========================================")
    print("Running Multiple A-to-B Trajectories")
    print("========================================")
    print(f"Grid: {GRID_ROWS} x {GRID_COLS}")
    print(f"Point A: {START_POSITION}")
    print(f"Point B: {TARGET_POSITION}")
    print(
        "Total trajectories per controller: "
        f"{N_TRAJECTORIES}"
    )
    print(
        "Displayed trajectories per figure: "
        f"{N_DISPLAYED_TRAJECTORIES}"
    )
    print(f"Maximum steps: {MAX_STEPS}")

    classifier_results = run_trajectories(
        controller_name="EEG classifier",
        sampler=classifier_sampler,
        environment=environment,
        n_trajectories=N_TRAJECTORIES,
    )

    random_results = run_trajectories(
        controller_name="Random baseline",
        sampler=random_sampler,
        environment=environment,
        n_trajectories=N_TRAJECTORIES,
    )

    print_summary(
        controller_name="EEG Classifier",
        results=classifier_results,
    )

    print_summary(
        controller_name="Random Baseline",
        results=random_results,
    )

    trajectory_records = (
        create_trajectory_records(
            controller_name="EEG_classifier",
            results=classifier_results,
        )
        + create_trajectory_records(
            controller_name="Random_baseline",
            results=random_results,
        )
    )

    trajectory_dataframe = pd.DataFrame(
        trajectory_records
    )

    TRAJECTORY_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_dataframe.to_csv(
        TRAJECTORY_DATA_PATH,
        index=False,
    )

    print(f"\nSaved: {TRAJECTORY_DATA_PATH}")

    plot_controller_trajectories(
        results=classifier_results,
        controller_title="EEG Classifier",
        output_path=(
            OUTPUT_DIRECTORY
            / "eeg_classifier_multiple_trajectories.png"
        ),
    )

    plot_controller_trajectories(
        results=random_results,
        controller_title="Random Baseline",
        output_path=(
            OUTPUT_DIRECTORY
            / "random_baseline_multiple_trajectories.png"
        ),
    )

    plot_successful_classifier_trajectories(
        results=classifier_results,
        output_path=(
            OUTPUT_DIRECTORY
            / "eeg_classifier_successful_trajectories.png"
        ),
    )

    plot_comparison(
        classifier_results=classifier_results,
        random_results=random_results,
        output_path=(
            OUTPUT_DIRECTORY
            / "classifier_vs_random_trajectories.png"
        ),
    )

    print("\n========================================")
    print("Trajectory analysis completed")
    print("========================================")
    print(f"Figures directory: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()