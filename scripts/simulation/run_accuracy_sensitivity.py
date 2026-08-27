"""Measure how classifier accuracy affects wheelchair navigation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bci_wheelchair.simulation.scenarios import generate_scenarios
from bci_wheelchair.simulation import GridEnvironment
from bci_wheelchair.simulation import run_classifier_simulation


N_SIMULATIONS = 1000
GRID_ROWS = 20
GRID_COLS = 20
MAX_STEPS = 500

SCENARIO_SEED = 42
SAMPLER_SEED = 100

ACCURACY_LEVELS = [
    0.25,
    0.40,
    0.50,
    0.6049,
    0.70,
    0.80,
    0.90,
    1.00,
]

VALID_CLASSES = (
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
)

DETAILED_OUTPUT_PATH = Path(
    "results/simulation/accuracy_sensitivity/accuracy_sensitivity_results.csv"
)

SUMMARY_OUTPUT_PATH = Path(
    "results/simulation/accuracy_sensitivity/accuracy_sensitivity_summary.csv"
)


class ControlledAccuracySampler:
    """
    Simulate a classifier with a specified prediction accuracy.

    With probability `accuracy`, the intended class is returned.
    Otherwise, one of the other three classes is selected uniformly.
    """

    def __init__(
        self,
        accuracy: float,
        random_seed: int | None = None,
    ) -> None:
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(
                "Accuracy must be between 0.0 and 1.0."
            )

        self.accuracy = accuracy
        self.rng = np.random.default_rng(random_seed)

    def sample_prediction(
        self,
        intended_class: str,
    ) -> str:
        """Return a correct or incorrect classifier prediction."""

        if intended_class not in VALID_CLASSES:
            raise ValueError(
                f"Unknown intended class: {intended_class}"
            )

        if self.rng.random() < self.accuracy:
            return intended_class

        incorrect_classes = [
            class_name
            for class_name in VALID_CLASSES
            if class_name != intended_class
        ]

        return str(
            self.rng.choice(incorrect_classes)
        )


def run_accuracy_experiment(
    accuracy: float,
    scenarios,
    environment: GridEnvironment,
) -> list[dict]:
    """Run all scenarios for one controlled accuracy."""

    seed = SAMPLER_SEED + int(
        round(accuracy * 1000)
    )

    sampler = ControlledAccuracySampler(
        accuracy=accuracy,
        random_seed=seed,
    )

    records: list[dict] = []

    accuracy_percent = accuracy * 100

    print(
        "\n----------------------------------------"
    )
    print(
        f"Running target accuracy: "
        f"{accuracy_percent:.0f}%"
    )
    print(
        "----------------------------------------"
    )

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
            result.correct_predictions
            / total_predictions
            if total_predictions > 0
            else float("nan")
        )

        records.append(
            {
                "target_accuracy": accuracy,
                "target_accuracy_percent": (
                    accuracy_percent
                ),
                "simulation_id": simulation_id,
                "grid_rows": GRID_ROWS,
                "grid_cols": GRID_COLS,
                "start_row": start_state.position[0],
                "start_col": start_state.position[1],
                "start_heading": start_state.heading,
                "target_row": target[0],
                "target_col": target[1],
                "initial_distance": (
                    result.initial_distance
                ),
                "reached_target": (
                    result.reached_target
                ),
                "steps": result.steps,
                "final_row": (
                    result.final_state.position[0]
                ),
                "final_col": (
                    result.final_state.position[1]
                ),
                "final_heading": (
                    result.final_state.heading
                ),
                "final_distance": (
                    result.final_distance
                ),
                "correct_predictions": (
                    result.correct_predictions
                ),
                "incorrect_predictions": (
                    result.incorrect_predictions
                ),
                "empirical_accuracy": (
                    empirical_accuracy
                ),
                "stop_commands": (
                    result.stop_commands
                ),
                "blocked_moves": (
                    result.blocked_moves
                ),
            }
        )

        if simulation_id % 100 == 0:
            print(
                f"{accuracy_percent:.0f}% accuracy: "
                f"completed {simulation_id}/"
                f"{len(scenarios)}"
            )

    return records


def build_summary(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Create one summary row for each accuracy level."""

    summary_records: list[dict] = []

    for accuracy, group in results.groupby(
        "target_accuracy",
        sort=True,
    ):
        successful = group[
            group["reached_target"]
        ]

        total_correct = (
            group["correct_predictions"].sum()
        )

        total_incorrect = (
            group["incorrect_predictions"].sum()
        )

        total_predictions = (
            total_correct + total_incorrect
        )

        overall_empirical_accuracy = (
            total_correct / total_predictions
            if total_predictions > 0
            else float("nan")
        )

        summary_records.append(
            {
                "target_accuracy": accuracy,
                "target_accuracy_percent": (
                    accuracy * 100
                ),
                "empirical_accuracy": (
                    overall_empirical_accuracy
                ),
                "empirical_accuracy_percent": (
                    overall_empirical_accuracy * 100
                ),
                "simulations": len(group),
                "successful_simulations": (
                    int(group["reached_target"].sum())
                ),
                "success_rate_percent": (
                    group["reached_target"].mean()
                    * 100
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
                "mean_initial_distance": (
                    group["initial_distance"].mean()
                ),
            }
        )

    return pd.DataFrame(summary_records)


def print_summary(
    summary: pd.DataFrame,
) -> None:
    """Print the sensitivity-analysis summary."""

    print("\n========================================")
    print("Accuracy Sensitivity Summary")
    print("========================================")

    display_columns = [
        "target_accuracy_percent",
        "empirical_accuracy_percent",
        "success_rate_percent",
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
                "target_accuracy_percent": (
                    lambda value: f"{value:.0f}%"
                ),
                "empirical_accuracy_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "success_rate_percent": (
                    lambda value: f"{value:.2f}%"
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
    """Run the complete accuracy-sensitivity experiment."""

    print("Accuracy Sensitivity Experiment")
    print(
        f"Grid: {GRID_ROWS} x {GRID_COLS}"
    )
    print(
        f"Simulations per accuracy: "
        f"{N_SIMULATIONS}"
    )
    print(
        f"Maximum steps: {MAX_STEPS}"
    )
    print(
        "Accuracy levels: "
        + ", ".join(
            f"{accuracy * 100:.0f}%"
            for accuracy in ACCURACY_LEVELS
        )
    )

    environment = GridEnvironment(
        rows=GRID_ROWS,
        cols=GRID_COLS,
    )

    scenarios = generate_scenarios(
        n_simulations=N_SIMULATIONS,
        rows=GRID_ROWS,
        cols=GRID_COLS,
        random_seed=SCENARIO_SEED,
    )

    all_records: list[dict] = []

    for accuracy in ACCURACY_LEVELS:
        records = run_accuracy_experiment(
            accuracy=accuracy,
            scenarios=scenarios,
            environment=environment,
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
