"""Analyse and visualise wheelchair navigation experiment results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_PATH = Path("results/simulation/navigation/navigation_simulation_results.csv")
OUTPUT_DIR = Path("results/cross_subject/predictions")


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Create controller-level navigation summary metrics."""
    summary = (
        results.groupby("controller")
        .agg(
            simulations=("simulation_id", "count"),
            successful_runs=("reached_target", "sum"),
            success_rate=("reached_target", "mean"),
            mean_commands_all=("steps", "mean"),
            median_commands_all=("steps", "median"),
            mean_final_distance=("final_distance", "mean"),
            mean_incorrect_predictions=("incorrect_predictions", "mean"),
            mean_stop_commands=("stop_commands", "mean"),
            mean_blocked_moves=("blocked_moves", "mean"),
        )
    )

    successful = (
        results[results["reached_target"]]
        .groupby("controller")
        .agg(
            mean_commands_successful=("steps", "mean"),
            median_commands_successful=("steps", "median"),
            minimum_commands_successful=("steps", "min"),
            maximum_commands_successful=("steps", "max"),
        )
    )

    summary = summary.join(successful)
    summary["success_rate"] *= 100

    return summary


def save_success_rate_plot(results: pd.DataFrame) -> None:
    controller_order = [
        "EEG_classifier",
        "Random_baseline",
    ]

    success_rates = (
        results.groupby("controller")["reached_target"]
        .mean()
        .reindex(controller_order)
        * 100
    )

    plt.figure(figsize=(7, 5))
    success_rates.plot(kind="bar")

    plt.ylabel("Success rate (%)")
    plt.xlabel("Controller")
    plt.title(
        "Wheelchair Navigation Success Rate\n"
        "1,000 Simulations per Controller"
    )
    plt.xticks(rotation=0)
    plt.ylim(0, 110)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "navigation_success_rate.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved {output_path}")


def save_commands_boxplot(results: pd.DataFrame) -> None:
    controller_order = [
        "EEG_classifier",
        "Random_baseline",
    ]

    command_data = [
        results[
            results["controller"] == controller
        ]["steps"]
        for controller in controller_order
    ]

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        command_data,
        tick_labels=[
            "EEG classifier",
            "Random baseline",
        ],
    )

    plt.ylabel("Total commands")
    plt.xlabel("Controller")
    plt.title(
        "Commands Used Across All Navigation Simulations"
    )
    plt.tight_layout()

    output_path = OUTPUT_DIR / "navigation_commands_boxplot.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved {output_path}")


def save_successful_commands_boxplot(
    results: pd.DataFrame,
) -> None:
    successful = results[results["reached_target"]]

    controller_order = [
        "EEG_classifier",
        "Random_baseline",
    ]

    command_data = [
        successful[
            successful["controller"] == controller
        ]["steps"]
        for controller in controller_order
    ]

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        command_data,
        tick_labels=[
            "EEG classifier",
            "Random baseline",
        ],
    )

    plt.ylabel("Commands to reach target")
    plt.xlabel("Controller")
    plt.title(
        "Commands Required for Successful Navigation"
    )
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "navigation_successful_commands_boxplot.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved {output_path}")


def save_final_distance_boxplot(
    results: pd.DataFrame,
) -> None:
    controller_order = [
        "EEG_classifier",
        "Random_baseline",
    ]

    final_distance_data = [
        results[
            results["controller"] == controller
        ]["final_distance"]
        for controller in controller_order
    ]

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        final_distance_data,
        tick_labels=[
            "EEG classifier",
            "Random baseline",
        ],
    )

    plt.ylabel("Final Manhattan distance from target")
    plt.xlabel("Controller")
    plt.title(
        "Final Distance from Target After Simulation"
    )
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "navigation_final_distance_boxplot.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved {output_path}")


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = pd.read_csv(RESULTS_PATH)

    required_columns = {
        "controller",
        "simulation_id",
        "reached_target",
        "steps",
        "final_distance",
        "incorrect_predictions",
        "stop_commands",
        "blocked_moves",
    }

    missing_columns = (
        required_columns - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Results CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    summary = build_summary(results)

    summary_path = (
        OUTPUT_DIR
        / "navigation_simulation_summary.csv"
    )

    summary.to_csv(summary_path)

    print("\nNavigation Simulation Summary")
    print("=" * 100)
    print(summary.round(2).to_string())
    print(f"\nSaved {summary_path}")

    save_success_rate_plot(results)
    save_commands_boxplot(results)
    save_successful_commands_boxplot(results)
    save_final_distance_boxplot(results)


if __name__ == "__main__":
    main()
