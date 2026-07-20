"""
Plot accuracy sensitivity experiment results.

Run with:

python -m scripts.plot_accuracy_sensitivity
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT = Path("results/accuracy_sensitivity_summary.csv")
OUTPUT_DIR = Path("results/accuracy_sensitivity_figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

x = df["target_accuracy_percent"]


def save_plot(y, ylabel, filename):
    plt.figure(figsize=(7,5))

    plt.plot(
        x,
        y,
        marker="o",
        linewidth=2,
    )

    plt.xlabel("Classifier Accuracy (%)")
    plt.ylabel(ylabel)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


save_plot(
    df["success_rate_percent"],
    "Navigation Success Rate (%)",
    "success_rate_vs_accuracy.png",
)

save_plot(
    df["mean_steps_all"],
    "Mean Steps",
    "mean_steps_vs_accuracy.png",
)

save_plot(
    df["mean_final_distance"],
    "Mean Final Distance",
    "final_distance_vs_accuracy.png",
)

save_plot(
    df["mean_blocked_moves"],
    "Mean Blocked Moves",
    "blocked_moves_vs_accuracy.png",
)

save_plot(
    df["mean_stop_commands"],
    "Mean Stop Commands",
    "stop_commands_vs_accuracy.png",
)

print("\nFigures saved to:")
print(OUTPUT_DIR.resolve())
