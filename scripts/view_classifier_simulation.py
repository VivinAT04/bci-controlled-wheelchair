"""Visualise one heading-based classifier-driven wheelchair simulation."""

import matplotlib.pyplot as plt

from bci_wheelchair.eeg_sampler import EEGPredictionSampler
from bci_wheelchair.simulation import (
    GridEnvironment,
    WheelchairState,
)
from bci_wheelchair.simulator import run_classifier_simulation


def main():
    grid = GridEnvironment(
        rows=20,
        cols=20,
    )

    sampler = EEGPredictionSampler(
        "results/predicted_commands.csv",
        random_seed=42,
    )

    result = run_classifier_simulation(
        environment=grid,
        sampler=sampler,
        start_state=WheelchairState(
            position=(10, 10),
            heading=0,
        ),
        target=(5, 15),
        max_steps=200,
    )

    rows = [
        position[0]
        for position in result.path
    ]

    columns = [
        position[1]
        for position in result.path
    ]

    plt.figure(figsize=(8, 8))

    plt.plot(
        columns,
        rows,
        marker="o",
        linewidth=2,
        label="Classifier-driven path",
    )

    plt.scatter(
        result.start_state.position[1],
        result.start_state.position[0],
        marker="s",
        s=150,
        label="Start",
    )

    plt.scatter(
        result.target[1],
        result.target[0],
        marker="*",
        s=250,
        label="Target",
    )

    plt.scatter(
        result.final_state.position[1],
        result.final_state.position[0],
        marker="X",
        s=150,
        label="Final position",
    )

    heading_vectors = {
        0: (0, -0.35),
        90: (0.35, 0),
        180: (0, 0.35),
        270: (-0.35, 0),
    }

    for index in range(
        0,
        len(result.path),
        max(1, len(result.path) // 12),
    ):
        position = result.path[index]
        heading = result.headings[index]

        dx, dy = heading_vectors[heading]

        plt.arrow(
            position[1],
            position[0],
            dx,
            dy,
            head_width=0.15,
            head_length=0.15,
            length_includes_head=True,
        )

    plt.xlim(-0.5, grid.cols - 0.5)
    plt.ylim(grid.rows - 0.5, -0.5)

    plt.xticks(range(grid.cols))
    plt.yticks(range(grid.rows))

    plt.grid(True)
    plt.xlabel("Column")
    plt.ylabel("Row")

    status = (
        "Reached target"
        if result.reached_target
        else "Failed"
    )

    plt.title(
        "Classifier-Driven Wheelchair Simulation\n"
        f"{status} | "
        f"Commands: {result.steps} | "
        f"Errors: {result.incorrect_predictions} | "
        f"Stops: {result.stop_commands}"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()