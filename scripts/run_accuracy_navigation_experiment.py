"""
Empirically evaluate how simulated classification accuracy affects
wheelchair navigation efficiency.

This experiment reuses the navigation logic from
scripts.interactive_grid_size_demo.

Run:
    python -m scripts.run_accuracy_navigation_experiment
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.interactive_grid_size_demo import (
    apply_command,
    choose_intended_command,
)


GRID_SIZE = 20
START_POSITION = (GRID_SIZE - 2, 1)
TARGET_POSITION = (1, GRID_SIZE - 2)
INITIAL_HEADING = 0

ACCURACY_LEVELS = (
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
)

RUNS_PER_ACCURACY = 1000
MAX_STEPS = 500
RANDOM_SEED = 42

COMMANDS = (
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
)

OUTPUT_DIRECTORY = Path(
    "results/accuracy_navigation_analysis"
)

SUMMARY_CSV_PATH = (
    OUTPUT_DIRECTORY / "accuracy_vs_navigation.csv"
)

TRIAL_CSV_PATH = (
    OUTPUT_DIRECTORY / "accuracy_navigation_trials.csv"
)

MEAN_STEPS_GRAPH_PATH = (
    OUTPUT_DIRECTORY / "accuracy_vs_mean_steps.png"
)

SUCCESS_RATE_GRAPH_PATH = (
    OUTPUT_DIRECTORY / "accuracy_vs_success_rate.png"
)


def simulate_classifier_prediction(
    intended_command: str,
    accuracy: float,
    rng: random.Random,
) -> str:
    """
    Simulate a four-class classifier.

    The intended command is returned with probability equal to the chosen
    accuracy. When an error occurs, one of the three incorrect commands is
    selected uniformly.
    """
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(
            f"Accuracy must be between 0 and 1, received {accuracy}."
        )

    if intended_command not in COMMANDS:
        raise ValueError(
            f"Unsupported intended command: {intended_command}"
        )

    if rng.random() < accuracy:
        return intended_command

    incorrect_commands = [
        command
        for command in COMMANDS
        if command != intended_command
    ]

    return rng.choice(incorrect_commands)


def run_navigation_episode(
    accuracy: float,
    rng: random.Random,
) -> tuple[int, bool]:
    """
    Run one navigation episode using the simulated classifier.

    Returns:
        A tuple containing:
        - number of commands issued
        - whether the target was reached
    """
    position = START_POSITION
    heading = INITIAL_HEADING

    for step_number in range(1, MAX_STEPS + 1):
        if position == TARGET_POSITION:
            return step_number - 1, True

        intended_command = choose_intended_command(
            position=position,
            heading=heading,
            target=TARGET_POSITION,
        )

        predicted_command = simulate_classifier_prediction(
            intended_command=intended_command,
            accuracy=accuracy,
            rng=rng,
        )

        position, heading = apply_command(
            position=position,
            heading=heading,
            command=predicted_command,
            grid_size=GRID_SIZE,
        )

        if position == TARGET_POSITION:
            return step_number, True

    return MAX_STEPS, False


def calculate_summary(
    accuracy: float,
    step_values: list[int],
    success_values: list[bool],
) -> dict[str, int | float]:
    """Calculate summary statistics for one accuracy level."""
    steps_array = np.asarray(
        step_values,
        dtype=float,
    )

    success_array = np.asarray(
        success_values,
        dtype=bool,
    )

    successful_steps = steps_array[success_array]

    successful_runs = int(success_array.sum())
    failed_runs = int(len(success_array) - successful_runs)
    success_rate = float(success_array.mean())

    if successful_runs > 0:
        mean_successful_steps = float(
            successful_steps.mean()
        )
        median_successful_steps = float(
            np.median(successful_steps)
        )
        standard_deviation_successful_steps = float(
            successful_steps.std(ddof=1)
            if successful_runs > 1
            else 0.0
        )
        minimum_successful_steps = int(
            successful_steps.min()
        )
        maximum_successful_steps = int(
            successful_steps.max()
        )
    else:
        mean_successful_steps = float("nan")
        median_successful_steps = float("nan")
        standard_deviation_successful_steps = float("nan")
        minimum_successful_steps = -1
        maximum_successful_steps = -1

    return {
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "runs": len(step_values),
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "success_rate": success_rate,
        "success_rate_percent": success_rate * 100,
        "mean_steps_all_runs": float(
            steps_array.mean()
        ),
        "median_steps_all_runs": float(
            np.median(steps_array)
        ),
        "standard_deviation_all_steps": float(
            steps_array.std(ddof=1)
            if len(steps_array) > 1
            else 0.0
        ),
        "mean_steps_successful_runs": (
            mean_successful_steps
        ),
        "median_steps_successful_runs": (
            median_successful_steps
        ),
        "standard_deviation_successful_steps": (
            standard_deviation_successful_steps
        ),
        "minimum_successful_steps": (
            minimum_successful_steps
        ),
        "maximum_successful_steps": (
            maximum_successful_steps
        ),
        "maximum_steps_allowed": MAX_STEPS,
        "grid_size": GRID_SIZE,
    }


def save_trial_results(
    trial_rows: list[dict[str, int | float | bool]],
) -> None:
    """Save every individual navigation episode."""
    if not trial_rows:
        raise ValueError("No trial results were generated.")

    with TRIAL_CSV_PATH.open(
        "w",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(trial_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(trial_rows)


def save_summary_results(
    summary_rows: list[dict[str, int | float]],
) -> None:
    """Save one summary row per accuracy level."""
    if not summary_rows:
        raise ValueError("No summary results were generated.")

    with SUMMARY_CSV_PATH.open(
        "w",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(summary_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary_rows)


def plot_mean_steps(
    summary_rows: list[dict[str, int | float]],
) -> None:
    """Plot classification accuracy against mean navigation steps."""
    accuracies = [
        float(row["accuracy_percent"])
        for row in summary_rows
    ]

    mean_all_steps = [
        float(row["mean_steps_all_runs"])
        for row in summary_rows
    ]

    mean_successful_steps = [
        float(row["mean_steps_successful_runs"])
        for row in summary_rows
    ]

    plt.figure(figsize=(9, 6))

    plt.plot(
        accuracies,
        mean_all_steps,
        marker="o",
        label="Mean steps across all runs",
    )

    plt.plot(
        accuracies,
        mean_successful_steps,
        marker="s",
        label="Mean steps for successful runs",
    )

    plt.xlabel(
        "Simulated Classification Accuracy (%)"
    )

    plt.ylabel(
        "Number of Navigation Steps"
    )

    plt.title(
        "Effect of Classification Accuracy on Navigation Steps"
    )

    plt.xticks(accuracies)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        MEAN_STEPS_GRAPH_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_success_rate(
    summary_rows: list[dict[str, int | float]],
) -> None:
    """Plot classification accuracy against target success rate."""
    accuracies = [
        float(row["accuracy_percent"])
        for row in summary_rows
    ]

    success_rates = [
        float(row["success_rate_percent"])
        for row in summary_rows
    ]

    plt.figure(figsize=(9, 6))

    plt.plot(
        accuracies,
        success_rates,
        marker="o",
    )

    plt.xlabel(
        "Simulated Classification Accuracy (%)"
    )

    plt.ylabel(
        "Navigation Success Rate (%)"
    )

    plt.title(
        "Effect of Classification Accuracy on Navigation Success"
    )

    plt.xticks(accuracies)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        SUCCESS_RATE_GRAPH_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def print_summary_table(
    summary_rows: list[dict[str, int | float]],
) -> None:
    """Print compact terminal results."""
    print("\n" + "=" * 92)
    print("Summary")
    print("=" * 92)

    print(
        f"{'Accuracy':>10} "
        f"{'Success':>12} "
        f"{'Mean all':>12} "
        f"{'Mean success':>15} "
        f"{'Median success':>17}"
    )

    print("-" * 92)

    for row in summary_rows:
        print(
            f"{float(row['accuracy_percent']):>9.0f}% "
            f"{float(row['success_rate_percent']):>11.1f}% "
            f"{float(row['mean_steps_all_runs']):>12.1f} "
            f"{float(row['mean_steps_successful_runs']):>15.1f} "
            f"{float(row['median_steps_successful_runs']):>17.1f}"
        )


def main() -> None:
    """Run the complete accuracy-versus-navigation experiment."""
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = random.Random(RANDOM_SEED)

    summary_rows: list[dict[str, int | float]] = []
    trial_rows: list[dict[str, int | float | bool]] = []

    print("=" * 72)
    print("Simulated Classifier Accuracy vs Navigation Steps")
    print("=" * 72)

    print(f"Grid size:             {GRID_SIZE}x{GRID_SIZE}")
    print(f"Start position:        {START_POSITION}")
    print(f"Target position:       {TARGET_POSITION}")
    print(f"Initial heading:       {INITIAL_HEADING}")
    print(f"Runs per accuracy:     {RUNS_PER_ACCURACY}")
    print(f"Maximum steps:         {MAX_STEPS}")
    print(f"Random seed:           {RANDOM_SEED}")

    for accuracy in ACCURACY_LEVELS:
        accuracy_percentage = accuracy * 100

        print(
            f"\nRunning {RUNS_PER_ACCURACY} episodes at "
            f"{accuracy_percentage:.0f}% accuracy..."
        )

        step_values: list[int] = []
        success_values: list[bool] = []

        for run_number in range(
            1,
            RUNS_PER_ACCURACY + 1,
        ):
            steps, success = run_navigation_episode(
                accuracy=accuracy,
                rng=rng,
            )

            step_values.append(steps)
            success_values.append(success)

            trial_rows.append(
                {
                    "accuracy": accuracy,
                    "accuracy_percent": (
                        accuracy_percentage
                    ),
                    "run": run_number,
                    "steps": steps,
                    "success": success,
                    "grid_size": GRID_SIZE,
                    "maximum_steps_allowed": MAX_STEPS,
                }
            )

        summary = calculate_summary(
            accuracy=accuracy,
            step_values=step_values,
            success_values=success_values,
        )

        summary_rows.append(summary)

        print(
            f"Success rate: "
            f"{float(summary['success_rate_percent']):.1f}%"
        )

        print(
            f"Mean steps across all runs: "
            f"{float(summary['mean_steps_all_runs']):.1f}"
        )

        print(
            f"Mean steps for successful runs: "
            f"{float(summary['mean_steps_successful_runs']):.1f}"
        )

    save_trial_results(trial_rows)
    save_summary_results(summary_rows)
    plot_mean_steps(summary_rows)
    plot_success_rate(summary_rows)
    print_summary_table(summary_rows)

    print("\n" + "=" * 72)
    print("Experiment complete")
    print("=" * 72)

    print(f"Summary CSV:\n{SUMMARY_CSV_PATH}")
    print(f"\nIndividual trials CSV:\n{TRIAL_CSV_PATH}")
    print(f"\nMean steps graph:\n{MEAN_STEPS_GRAPH_PATH}")
    print(f"\nSuccess-rate graph:\n{SUCCESS_RATE_GRAPH_PATH}")


if __name__ == "__main__":
    main()
