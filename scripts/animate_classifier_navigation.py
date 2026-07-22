"""Animate one classifier-driven wheelchair navigation simulation."""

import time

import matplotlib.pyplot as plt

from bci_wheelchair.eeg_sampler import EEGPredictionSampler
from bci_wheelchair.simulation import (
    GridEnvironment,
    WheelchairState,
)
from bci_wheelchair.simulator import run_classifier_simulation


ANIMATION_DELAY = 0.5


def action_name(action: str) -> str:
    """Return a presentation-friendly command name."""
    names = {
        "left_hand": "Turn left",
        "right_hand": "Turn right",
        "feet": "Move forward",
        "tongue": "Stop",
    }

    return names.get(action, action)


def main() -> None:
    grid = GridEnvironment(
        rows=20,
        cols=20,
    )

    sampler = EEGPredictionSampler(
        "results/test_predicted_commands.csv",
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

    heading_vectors = {
        0: (0, -0.6),
        90: (0.6, 0),
        180: (0, 0.6),
        270: (-0.6, 0),
    }

    plt.ion()

    figure, axis = plt.subplots(
        figsize=(8, 8),
    )

    axis.set_xlim(
        -0.5,
        grid.cols - 0.5,
    )
    axis.set_ylim(
        grid.rows - 0.5,
        -0.5,
    )

    axis.set_xticks(
        range(grid.cols)
    )
    axis.set_yticks(
        range(grid.rows)
    )

    axis.grid(True)
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")

    start_position = result.start_state.position

    axis.scatter(
        start_position[1],
        start_position[0],
        marker="s",
        s=150,
        label="Start",
    )

    axis.scatter(
        result.target[1],
        result.target[0],
        marker="*",
        s=250,
        label="Target",
    )

    path_line, = axis.plot(
        [],
        [],
        marker="o",
        linewidth=2,
        label="Classifier-driven path",
    )

    wheelchair_marker, = axis.plot(
        [],
        [],
        marker="o",
        markersize=12,
        linestyle="None",
        label="Wheelchair",
    )

    heading_arrow = None

    axis.legend(
        loc="upper right"
    )

    displayed_path = [
        start_position
    ]

    for step_index, step in enumerate(
        result.trace,
        start=1,
    ):
        displayed_path.append(
            step.position_after
        )

        rows = [
            position[0]
            for position in displayed_path
        ]

        columns = [
            position[1]
            for position in displayed_path
        ]

        path_line.set_data(
            columns,
            rows,
        )

        wheelchair_marker.set_data(
            [step.position_after[1]],
            [step.position_after[0]],
        )

        if heading_arrow is not None:
            heading_arrow.remove()

        dx, dy = heading_vectors[
            step.heading_after
        ]

        heading_arrow = axis.arrow(
            step.position_after[1],
            step.position_after[0],
            dx,
            dy,
            head_width=0.18,
            head_length=0.18,
            length_includes_head=True,
        )

        correctness = (
            "Correct"
            if step.prediction_correct
            else "Incorrect"
        )

        axis.set_title(
            "Classifier-Driven Wheelchair Navigation\n"
            f"Step {step_index}/{result.steps} | "
            f"Intended: {action_name(step.intended_action)} | "
            f"Predicted: {action_name(step.predicted_action)} | "
            f"{correctness}"
        )

        figure.canvas.draw()
        figure.canvas.flush_events()

        time.sleep(
            ANIMATION_DELAY
        )

    status = (
        "Reached target"
        if result.reached_target
        else "Failed to reach target"
    )

    axis.set_title(
        "Classifier-Driven Wheelchair Navigation\n"
        f"{status} | "
        f"Commands: {result.steps} | "
        f"Errors: {result.incorrect_predictions} | "
        f"Stops: {result.stop_commands}"
    )

    axis.scatter(
        result.final_state.position[1],
        result.final_state.position[0],
        marker="X",
        s=180,
        label="Final position",
    )

    axis.legend(
        loc="upper right"
    )

    figure.canvas.draw()
    figure.canvas.flush_events()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
