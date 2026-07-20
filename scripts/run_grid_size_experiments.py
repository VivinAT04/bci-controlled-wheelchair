"""
Evaluate EEG-controlled wheelchair navigation across different grid sizes.

Uses predictions from:
    results/evaluation_predictions.csv

Run from the project root:

    python -m scripts.run_grid_size_experiments
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bci_wheelchair.eeg_sampler import EEGPredictionSampler
from bci_wheelchair.experiment_utils import generate_scenarios
from bci_wheelchair.simulation import GridEnvironment
from bci_wheelchair.simulator import run_classifier_simulation


PREDICTIONS_PATH = Path(
    "results/evaluation_predictions.csv"
)

DETAILED_OUTPUT_PATH = Path(
    "results/grid_size_experiment_results.csv"
)

SUMMARY_OUTPUT_PATH = Path(
    "results/grid_size_experiment_summary.csv"
)

GRID_SIZES = [
    10,
    20,
    30,
    40,
]

N_SIMULATIONS = 1000
MAX_STEPS = 500

SCENARIO_SEED = 42
CLASSIFIER_SEED = 100


def run_grid_experiment(
    grid_size: int,
) -> list[dict]:
    """Run all navigation scenarios for one grid size."""

    print("\n========================================")
    print(f"Running grid size: {grid_size} x {grid_size}")
    print("========================================")

    environment = GridEnvironment(
        rows=grid_size,
        cols=grid_size,
    )

    scenarios = generate_scenarios(
        n_simulations=N_SIMULATIONS,
        rows=grid_size,
        cols=grid_size,
        random_seed=SCENARIO_SEED + grid_size,
    )

    sampler = EEGPredictionSampler(
        PREDICTIONS_PATH,
        random_seed=CLASSIFIER_SEED + grid_size,
    )

    records: list[dict] = []

    for simulation_id, (start_state, target) in enumerate(
        scenarios,
        start=1,
    ):
        result = run_classifier_simulation(
            environment=environment,
            sampler=sampler,
            start_state=start_state,
            target=target,
            max_steps=MAX_STEPS,
        )

        total_predictions = (
            result.correct_predictions
            + result.incorrect_predictions
        )

        empirical_accuracy = (
            result.correct_predictions / total_predictions
            if total_predictions > 0
            else float("nan")
        )

        records.append(
            {
                "grid_size": grid_size,
                "grid_rows": grid_size,
                "grid_cols": grid_size,
                "simulation_id": simulation_id,
                "start_row": start_state.position[0],
                "start_col": start_state.position[1],
                "start_heading": start_state.heading,
                "target_row": target[0],
                "target_col": target[1],
                "initial_distance": result.initial_distance,
                "reached_target": result.reached_target,
                "steps": result.steps,
                "final_row": result.final_state.position[0],
                "final_col": result.final_state.position[1],
                "final_heading": result.final_state.heading,
                "final_distance": result.final_distance,
                "correct_predictions": (
                    result.correct_predictions
                ),
                "incorrect_predictions": (
                    result.incorrect_predictions
                ),
                "empirical_accuracy": empirical_accuracy,
                "blocked_moves": result.blocked_moves,
                "stop_commands": result.stop_commands,
            }
        )

        if simulation_id % 100 == 0:
            print(
                f"{grid_size}x{grid_size}: completed "
                f"{simulation_id}/{N_SIMULATIONS}"
            )

    return records


def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Build one summary row for each grid size."""

    summary_records: list[dict] = []

    for grid_size, group in results.groupby(
        "grid_size",
        sort=True,
    ):
        successful = group[
            group["reached_target"]
        ]

        total_correct = int(
            group["correct_predictions"].sum()
        )

        total_incorrect = int(
            group["incorrect_predictions"].sum()
        )

        total_predictions = (
            total_correct + total_incorrect
        )

        empirical_accuracy = (
            total_correct / total_predictions
            if total_predictions > 0
            else float("nan")
        )

        summary_records.append(
            {
                "grid_size": grid_size,
                "grid_label": (
                    f"{grid_size}x{grid_size}"
                ),
                "simulations": len(group),
                "successful_simulations": int(
                    group["reached_target"].sum()
                ),
                "success_rate_percent": (
                    group["reached_target"].mean()
                    * 100
                ),
                "empirical_accuracy_percent": (
                    empirical_accuracy * 100
                ),
                "mean_initial_distance": (
                    group["initial_distance"].mean()
                ),
                "mean_steps_all": (
                    group["steps"].mean()
                ),
                "mean_steps_successful": (
                    successful["steps"].mean()
                    if not successful.empty
                    else float("nan")
                ),
                "mean_final_distance": (
                    group["final_distance"].mean()
                ),
                "mean_blocked_moves": (
                    group["blocked_moves"].mean()
                ),
                "mean_stop_commands": (
                    group["stop_commands"].mean()
                ),
            }
        )

    return pd.DataFrame(summary_records)


def print_summary(
    summary: pd.DataFrame,
) -> None:
    """Print grid-size experiment results."""

    print("\n========================================")
    print("Grid Size Experiment Summary")
    print("========================================")

    display_columns = [
        "grid_label",
        "empirical_accuracy_percent",
        "success_rate_percent",
        "mean_initial_distance",
        "mean_steps_all",
        "mean_steps_successful",
        "mean_final_distance",
        "mean_blocked_moves",
        "mean_stop_commands",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            formatters={
                "empirical_accuracy_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "success_rate_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "mean_initial_distance": (
                    lambda value: f"{value:.2f}"
                ),
                "mean_steps_all": (
                    lambda value: f"{value:.2f}"
                ),
                "mean_steps_successful": (
                    lambda value: f"{value:.2f}"
                ),
                "mean_final_distance": (
                    lambda value: f"{value:.2f}"
                ),
                "mean_blocked_moves": (
                    lambda value: f"{value:.2f}"
                ),
                "mean_stop_commands": (
                    lambda value: f"{value:.2f}"
                ),
            },
        )
    )


def main() -> None:
    """Run experiments across all grid sizes."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{PREDICTIONS_PATH}"
        )

    print("Grid Size Navigation Experiment")
    print(
        "Using real evaluation predictions from: "
        f"{PREDICTIONS_PATH}"
    )
    print(
        f"Simulations per grid: {N_SIMULATIONS}"
    )
    print(
        f"Maximum steps: {MAX_STEPS}"
    )
    print(
        "Grid sizes: "
        + ", ".join(
            f"{size}x{size}"
            for size in GRID_SIZES
        )
    )

    all_records: list[dict] = []

    for grid_size in GRID_SIZES:
        records = run_grid_experiment(
            grid_size=grid_size,
        )
        all_records.extend(records)

    results = pd.DataFrame(all_records)
    summary = build_summary(results)

    DETAILED_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        DETAILED_OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )

    print_summary(summary)

    print(
        f"\nSaved {len(results)} detailed rows to "
        f"{DETAILED_OUTPUT_PATH}"
    )

    print(
        f"Saved {len(summary)} summary rows to "
        f"{SUMMARY_OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
