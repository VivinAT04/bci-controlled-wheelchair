"""
Empirically analyse how simulated EEG classification accuracy affects
wheelchair navigation efficiency.

The simulated classifier outputs the correct navigation command with a
specified probability. Otherwise, it outputs one of the three incorrect
commands at random.

Run:
    python -m scripts.simulation.analyse_accuracy_vs_navigation
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GRID_ROWS = 20
GRID_COLS = 20

START_POSITION = (18, 1)
TARGET_POSITION = (1, 18)

ACCURACY_LEVELS = [
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

RUNS_PER_ACCURACY = 500
MAX_STEPS = 1000
RANDOM_SEED = 42

OUTPUT_DIRECTORY = Path("results/simulation/navigation/accuracy_navigation_analysis")
CSV_OUTPUT_PATH = OUTPUT_DIRECTORY / "accuracy_vs_navigation.csv"
STEPS_FIGURE_PATH = OUTPUT_DIRECTORY / "accuracy_vs_mean_steps.png"
SUCCESS_FIGURE_PATH = OUTPUT_DIRECTORY / "accuracy_vs_success_rate.png"

COMMANDS = [
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
]


def choose_ideal_command(
    position: tuple[int, int],
    target: tuple[int, int],
) -> str:
    """
    Select an ideal command that reduces Manhattan distance to the target.

    Horizontal and vertical movement are alternated randomly when both
    directions would reduce the distance.
    """
    row, column = position
    target_row, target_column = target

    possible_commands: list[str] = []

    if row > target_row:
        possible_commands.append("UP")
    elif row < target_row:
        possible_commands.append("DOWN")

    if column > target_column:
        possible_commands.append("LEFT")
    elif column < target_column:
        possible_commands.append("RIGHT")

    if not possible_commands:
        raise ValueError("Position is already at the target.")

    return possible_commands[
        np.random.randint(len(possible_commands))
    ]


def simulate_classifier_command(
    ideal_command: str,
    accuracy: float,
) -> str:
    """
    Return the ideal command with probability equal to classifier accuracy.

    When a classification error occurs, select uniformly from the three
    incorrect commands.
    """
    if np.random.random() < accuracy:
        return ideal_command

    incorrect_commands = [
        command
        for command in COMMANDS
        if command != ideal_command
    ]

    return incorrect_commands[
        np.random.randint(len(incorrect_commands))
    ]


def apply_command(
    position: tuple[int, int],
    command: str,
) -> tuple[int, int]:
    """Apply one command while keeping the wheelchair inside the grid."""
    row, column = position

    if command == "UP":
        row -= 1
    elif command == "DOWN":
        row += 1
    elif command == "LEFT":
        column -= 1
    elif command == "RIGHT":
        column += 1
    else:
        raise ValueError(f"Unknown command: {command}")

    row = int(np.clip(row, 0, GRID_ROWS - 1))
    column = int(np.clip(column, 0, GRID_COLS - 1))

    return row, column


def run_single_navigation(
    accuracy: float,
) -> tuple[int, bool]:
    """Run one navigation episode for a selected classifier accuracy."""
    position = START_POSITION

    for step_number in range(1, MAX_STEPS + 1):
        if position == TARGET_POSITION:
            return step_number - 1, True

        ideal_command = choose_ideal_command(
            position,
            TARGET_POSITION,
        )

        predicted_command = simulate_classifier_command(
            ideal_command,
            accuracy,
        )

        position = apply_command(
            position,
            predicted_command,
        )

        if position == TARGET_POSITION:
            return step_number, True

    return MAX_STEPS, False


def main() -> None:
    np.random.seed(RANDOM_SEED)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows: list[dict[str, float | int]] = []

    print("=" * 72)
    print("Classification Accuracy vs Navigation Efficiency")
    print("=" * 72)

    print(f"Grid size: {GRID_ROWS}x{GRID_COLS}")
    print(f"Start: {START_POSITION}")
    print(f"Target: {TARGET_POSITION}")
    print(f"Runs per accuracy: {RUNS_PER_ACCURACY}")
    print(f"Maximum steps: {MAX_STEPS}")

    for accuracy in ACCURACY_LEVELS:
        steps_results: list[int] = []
        success_results: list[bool] = []

        print(
            f"\nRunning simulated classifier at "
            f"{accuracy * 100:.0f}% accuracy..."
        )

        for _ in range(RUNS_PER_ACCURACY):
            steps, success = run_single_navigation(
                accuracy
            )

            steps_results.append(steps)
            success_results.append(success)

        steps_array = np.asarray(steps_results)
        success_array = np.asarray(success_results)

        successful_steps = steps_array[success_array]

        success_rate = float(success_array.mean())

        if len(successful_steps) > 0:
            mean_successful_steps = float(
                successful_steps.mean()
            )
            median_successful_steps = float(
                np.median(successful_steps)
            )
            standard_deviation = float(
                successful_steps.std()
            )
            minimum_steps = int(successful_steps.min())
            maximum_steps = int(successful_steps.max())
        else:
            mean_successful_steps = float("nan")
            median_successful_steps = float("nan")
            standard_deviation = float("nan")
            minimum_steps = MAX_STEPS
            maximum_steps = MAX_STEPS

        mean_all_steps = float(steps_array.mean())

        summary_rows.append(
            {
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100,
                "runs": RUNS_PER_ACCURACY,
                "successful_runs": int(success_array.sum()),
                "failed_runs": int(
                    RUNS_PER_ACCURACY - success_array.sum()
                ),
                "success_rate": success_rate,
                "success_rate_percent": success_rate * 100,
                "mean_steps_all_runs": mean_all_steps,
                "mean_steps_successful_runs": mean_successful_steps,
                "median_steps_successful_runs": median_successful_steps,
                "standard_deviation_successful_steps": standard_deviation,
                "minimum_successful_steps": minimum_steps,
                "maximum_successful_steps": maximum_steps,
                "max_steps_allowed": MAX_STEPS,
            }
        )

        print(
            f"Success rate: {success_rate * 100:.1f}%"
        )
        print(
            f"Mean steps across all runs: "
            f"{mean_all_steps:.1f}"
        )

        if len(successful_steps) > 0:
            print(
                f"Mean steps for successful runs: "
                f"{mean_successful_steps:.1f}"
            )

    with CSV_OUTPUT_PATH.open(
        "w",
        newline="",
    ) as output_file:
        fieldnames = list(summary_rows[0].keys())

        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    accuracy_percentages = [
        float(row["accuracy_percent"])
        for row in summary_rows
    ]

    mean_steps = [
        float(row["mean_steps_all_runs"])
        for row in summary_rows
    ]

    success_rates = [
        float(row["success_rate_percent"])
        for row in summary_rows
    ]

    plt.figure(figsize=(9, 6))
    plt.plot(
        accuracy_percentages,
        mean_steps,
        marker="o",
    )
    plt.xlabel("Simulated Classification Accuracy (%)")
    plt.ylabel("Mean Number of Steps")
    plt.title(
        "Effect of Classification Accuracy on Navigation Steps"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        STEPS_FIGURE_PATH,
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(
        accuracy_percentages,
        success_rates,
        marker="o",
    )
    plt.xlabel("Simulated Classification Accuracy (%)")
    plt.ylabel("Navigation Success Rate (%)")
    plt.title(
        "Effect of Classification Accuracy on Navigation Success"
    )
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        SUCCESS_FIGURE_PATH,
        dpi=300,
    )
    plt.close()

    print("\n" + "=" * 72)
    print("Analysis complete")
    print("=" * 72)

    print(f"CSV results: {CSV_OUTPUT_PATH}")
    print(f"Steps graph: {STEPS_FIGURE_PATH}")
    print(f"Success graph: {SUCCESS_FIGURE_PATH}")


if __name__ == "__main__":
    main()
