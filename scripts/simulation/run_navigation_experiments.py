"""Run large-scale wheelchair navigation experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bci_wheelchair.simulation import EEGPredictionSampler
from bci_wheelchair.simulation.scenarios import generate_scenarios
from bci_wheelchair.simulation import RandomPredictionSampler
from bci_wheelchair.simulation import GridEnvironment
from bci_wheelchair.simulation import run_classifier_simulation


N_SIMULATIONS = 1000
GRID_ROWS = 20
GRID_COLS = 20
MAX_STEPS = 500

SCENARIO_SEED = 42
CLASSIFIER_SEED = 42
RANDOM_BASELINE_SEED = 42

PREDICTION_PATH = Path(
    "results/cross_session/euclidean_alignment/fbcsp/svm/"
    "ea_fbcsp_svm_cross_session_predictions.csv"
)

OUTPUT_PATH = Path(
    "results/simulation/navigation/"
    "ea_fbcsp_cross_session/"
    "navigation_simulation_results.csv"
)


def run_experiment(
    controller_name: str,
    sampler,
    scenarios,
    environment: GridEnvironment,
) -> list[dict]:
    """Run all navigation scenarios for one controller."""

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

        records.append(
            {
                "controller": controller_name,
                "evaluation": "Cross-Session",
                "method": (
                    "EA + FBCSP + RBF-SVM"
                ),
                "prediction_source": str(
                    PREDICTION_PATH
                ),
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
                "stop_commands": result.stop_commands,
                "blocked_moves": result.blocked_moves,
            }
        )

        if simulation_id % 100 == 0:
            print(
                f"{controller_name}: completed "
                f"{simulation_id}/{len(scenarios)}"
            )

    return records


def print_summary(results: pd.DataFrame) -> None:
    """Print navigation metrics for each controller."""

    print("\n========================================")
    print("Navigation Experiment Summary")
    print("========================================")

    for controller, group in results.groupby(
        "controller"
    ):
        successful = group[
            group["reached_target"]
        ]

        success_rate = (
            group["reached_target"].mean() * 100
        )

        mean_steps_all = group["steps"].mean()

        if successful.empty:
            mean_steps_successful = float("nan")
        else:
            mean_steps_successful = (
                successful["steps"].mean()
            )

        mean_final_distance = (
            group["final_distance"].mean()
        )

        mean_blocked_moves = (
            group["blocked_moves"].mean()
        )

        mean_stop_commands = (
            group["stop_commands"].mean()
        )

        print(f"\nController: {controller}")
        print(f"Simulations: {len(group)}")
        print(
            f"Success rate: {success_rate:.2f}%"
        )
        print(
            f"Mean steps (all): "
            f"{mean_steps_all:.2f}"
        )
        print(
            "Mean steps (successful only): "
            f"{mean_steps_successful:.2f}"
        )
        print(
            f"Mean final distance: "
            f"{mean_final_distance:.2f}"
        )
        print(
            f"Mean blocked moves: "
            f"{mean_blocked_moves:.2f}"
        )
        print(
            f"Mean stop commands: "
            f"{mean_stop_commands:.2f}"
        )


def main() -> None:
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation prediction file not found: "
            f"{PREDICTION_PATH}"
        )

    print(
        f"Using classifier predictions from: "
        f"{PREDICTION_PATH}"
    )

    print(
        f"Grid: {GRID_ROWS} x {GRID_COLS}"
    )

    print(
        f"Simulations per controller: "
        f"{N_SIMULATIONS}"
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

    classifier_sampler = EEGPredictionSampler(
        PREDICTION_PATH,
        random_seed=CLASSIFIER_SEED,
    )

    random_sampler = RandomPredictionSampler(
        random_seed=RANDOM_BASELINE_SEED,
    )

    classifier_records = run_experiment(
        controller_name=(
            "EA_FBCSP_RBF_SVM"
        ),
        sampler=classifier_sampler,
        scenarios=scenarios,
        environment=environment,
    )

    random_records = run_experiment(
        controller_name="Random_baseline",
        sampler=random_sampler,
        scenarios=scenarios,
        environment=environment,
    )

    results = pd.DataFrame(
        classifier_records + random_records
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_summary(results)

    print(
        f"\nSaved {len(results)} rows to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
