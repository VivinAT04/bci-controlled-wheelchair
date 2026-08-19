"""
Plot accuracy sensitivity experiment results.

Run with:

    python -m scripts.simulation.plot_accuracy_sensitivity
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = Path("results/simulation/accuracy_sensitivity/accuracy_sensitivity_summary.csv")
OUTPUT_DIR = Path(
    "results/dissertation"
)


def save_plot(
    x: pd.Series,
    y: pd.Series,
    ylabel: str,
    filename: str,
) -> None:
    """Create and save one accuracy-sensitivity figure."""

    print(f"Creating: {filename}")

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.plot(
        x,
        y,
        marker="o",
        linewidth=2,
    )

    axis.set_xlabel("Classifier Accuracy (%)")
    axis.set_ylabel(ylabel)
    axis.grid(True)

    figure.tight_layout()

    output_path = OUTPUT_DIR / filename

    print(f"Saving: {output_path}")

    figure.savefig(
        output_path,
        dpi=150,
    )

    plt.close(figure)

    print(f"Saved: {output_path}")


def main() -> None:
    """Load the experiment results and generate all figures."""

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT.resolve()}\n"
            "Run the accuracy sensitivity experiment first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(INPUT)
    x = dataframe["target_accuracy_percent"]

    save_plot(
        x,
        dataframe["success_rate_percent"],
        "Navigation Success Rate (%)",
        "success_rate_vs_accuracy.png",
    )

    save_plot(
        x,
        dataframe["mean_steps_all"],
        "Mean Steps",
        "mean_steps_vs_accuracy.png",
    )

    save_plot(
        x,
        dataframe["mean_final_distance"],
        "Mean Final Distance",
        "final_distance_vs_accuracy.png",
    )

    save_plot(
        x,
        dataframe["mean_blocked_moves"],
        "Mean Blocked Moves",
        "blocked_moves_vs_accuracy.png",
    )

    save_plot(
        x,
        dataframe["mean_stop_commands"],
        "Mean Stop Commands",
        "stop_commands_vs_accuracy.png",
    )

    print("\nFigures saved to:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
