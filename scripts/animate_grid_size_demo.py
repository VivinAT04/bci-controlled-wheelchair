"""
Automatically demonstrate EEG-controlled wheelchair navigation across
10x10, 20x20, 30x30, and 40x40 grids.

The animation uses classifier predictions sampled from:

    results/evaluation_predictions.csv

Run from the project root:

    python -m scripts.animate_grid_size_demo
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from bci_wheelchair.eeg_sampler import EEGPredictionSampler


PREDICTIONS_PATH = Path(
    "results/evaluation_predictions.csv"
)

OUTPUT_DIRECTORY = Path(
    "results/grid_size_demo"
)

OUTPUT_GIF = OUTPUT_DIRECTORY / (
    "grid_size_navigation_demo.gif"
)

GRID_SIZES = [10, 20, 30, 40]

MAX_STEPS = 500
FRAME_INTERVAL_MS = 100
PAUSE_FRAMES = 15

HEADINGS = {
    0: "North",
    90: "East",
    180: "South",
    270: "West",
}

COMMAND_LABELS = {
    "left_hand": "Turn Left",
    "right_hand": "Turn Right",
    "feet": "Move Forward",
    "tongue": "Stop",
}


@dataclass
class DemoStep:
    """Store one animation step."""

    row: int
    col: int
    heading: int
    intended_command: str
    predicted_command: str
    reached_target: bool


@dataclass
class GridDemo:
    """Store a complete trajectory for one grid size."""

    grid_size: int
    start: tuple[int, int]
    target: tuple[int, int]
    steps: list[DemoStep]


def turn_left(heading: int) -> int:
    """Rotate the wheelchair 90 degrees left."""

    return (heading - 90) % 360


def turn_right(heading: int) -> int:
    """Rotate the wheelchair 90 degrees right."""

    return (heading + 90) % 360


def heading_delta(
    heading: int,
) -> tuple[int, int]:
    """Return the row and column movement for a heading."""

    movement = {
        0: (-1, 0),
        90: (0, 1),
        180: (1, 0),
        270: (0, -1),
    }

    return movement[heading]


def desired_heading(
    position: tuple[int, int],
    target: tuple[int, int],
) -> int | None:
    """
    Determine the desired heading.

    Horizontal movement is completed first, followed by vertical movement.
    """

    current_row, current_col = position
    target_row, target_col = target

    if position == target:
        return None

    if target_col > current_col:
        return 90

    if target_col < current_col:
        return 270

    if target_row > current_row:
        return 180

    return 0


def intended_command(
    position: tuple[int, int],
    heading: int,
    target: tuple[int, int],
) -> str:
    """Choose the ideal motor-imagery command."""

    required_heading = desired_heading(
        position=position,
        target=target,
    )

    if required_heading is None:
        return "tongue"

    if heading == required_heading:
        return "feet"

    difference = (
        required_heading - heading
    ) % 360

    if difference == 90:
        return "right_hand"

    if difference == 270:
        return "left_hand"

    # When facing the opposite direction, use a right turn.
    # The following replanning step will turn again.
    return "right_hand"


def apply_prediction(
    position: tuple[int, int],
    heading: int,
    predicted_command: str,
    grid_size: int,
) -> tuple[tuple[int, int], int]:
    """Apply a classifier prediction to the wheelchair."""

    row, col = position

    if predicted_command == "left_hand":
        return position, turn_left(heading)

    if predicted_command == "right_hand":
        return position, turn_right(heading)

    if predicted_command == "tongue":
        return position, heading

    if predicted_command != "feet":
        raise ValueError(
            f"Unknown predicted command: "
            f"{predicted_command}"
        )

    row_delta, col_delta = heading_delta(
        heading
    )

    new_row = row + row_delta
    new_col = col + col_delta

    if (
        0 <= new_row < grid_size
        and 0 <= new_col < grid_size
    ):
        return (new_row, new_col), heading

    return position, heading


def build_demo(
    grid_size: int,
    random_seed: int,
) -> GridDemo:
    """Generate one closed-loop navigation trajectory."""

    sampler = EEGPredictionSampler(
        PREDICTIONS_PATH,
        random_seed=random_seed,
    )

    start = (
        grid_size - 2,
        1,
    )

    target = (
        1,
        grid_size - 2,
    )

    position = start
    heading = 0

    trajectory: list[DemoStep] = [
        DemoStep(
            row=position[0],
            col=position[1],
            heading=heading,
            intended_command="tongue",
            predicted_command="tongue",
            reached_target=False,
        )
    ]

    for _ in range(MAX_STEPS):
        if position == target:
            break

        correct_command = intended_command(
            position=position,
            heading=heading,
            target=target,
        )

        classifier_command = (
            sampler.sample_prediction(
                correct_command
            )
        )

        position, heading = apply_prediction(
            position=position,
            heading=heading,
            predicted_command=classifier_command,
            grid_size=grid_size,
        )

        trajectory.append(
            DemoStep(
                row=position[0],
                col=position[1],
                heading=heading,
                intended_command=correct_command,
                predicted_command=classifier_command,
                reached_target=(
                    position == target
                ),
            )
        )

    return GridDemo(
        grid_size=grid_size,
        start=start,
        target=target,
        steps=trajectory,
    )


def main() -> None:
    """Create, save, and display the animation."""

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{PREDICTIONS_PATH}"
        )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    demos = [
        build_demo(
            grid_size=grid_size,
            random_seed=100 + grid_size,
        )
        for grid_size in GRID_SIZES
    ]

    print("\nGenerated demonstration trajectories:")

    for demo in demos:
        reached = (
            demo.steps[-1].reached_target
        )

        print(
            f"{demo.grid_size}x{demo.grid_size}: "
            f"{len(demo.steps) - 1} steps, "
            f"target reached = {reached}"
        )

    frame_map: list[
        tuple[int, int]
    ] = []

    for demo_index, demo in enumerate(demos):
        for step_index in range(
            len(demo.steps)
        ):
            frame_map.append(
                (demo_index, step_index)
            )

        for _ in range(PAUSE_FRAMES):
            frame_map.append(
                (
                    demo_index,
                    len(demo.steps) - 1,
                )
            )

    figure, axis = plt.subplots(
        figsize=(8, 8)
    )

    def update(
        frame_index: int,
    ):
        """Draw one animation frame."""

        demo_index, step_index = (
            frame_map[frame_index]
        )

        demo = demos[demo_index]
        step = demo.steps[step_index]

        axis.clear()

        size = demo.grid_size

        axis.set_xlim(-0.5, size - 0.5)
        axis.set_ylim(size - 0.5, -0.5)

        axis.set_xticks(
            range(size)
        )

        axis.set_yticks(
            range(size)
        )

        if size >= 30:
            axis.set_xticklabels(
                [
                    str(value)
                    if value % 5 == 0
                    else ""
                    for value in range(size)
                ]
            )

            axis.set_yticklabels(
                [
                    str(value)
                    if value % 5 == 0
                    else ""
                    for value in range(size)
                ]
            )

        axis.grid(True)

        path_rows = [
            trajectory_step.row
            for trajectory_step
            in demo.steps[: step_index + 1]
        ]

        path_cols = [
            trajectory_step.col
            for trajectory_step
            in demo.steps[: step_index + 1]
        ]

        axis.plot(
            path_cols,
            path_rows,
            linewidth=2,
            alpha=0.75,
            label="Wheelchair path",
        )

        axis.scatter(
            demo.start[1],
            demo.start[0],
            marker="s",
            s=160,
            label="Start",
            zorder=4,
        )

        axis.scatter(
            demo.target[1],
            demo.target[0],
            marker="*",
            s=260,
            label="Target",
            zorder=4,
        )

        axis.scatter(
            step.col,
            step.row,
            marker="o",
            s=220,
            label="Wheelchair",
            zorder=5,
        )

        intended_text = COMMAND_LABELS.get(
            step.intended_command,
            step.intended_command,
        )

        predicted_text = COMMAND_LABELS.get(
            step.predicted_command,
            step.predicted_command,
        )

        status = (
            "Target reached"
            if step.reached_target
            else "Navigating"
        )

        axis.set_title(
            f"EEG Wheelchair Navigation — "
            f"{size}×{size} Grid\n"
            f"Step {step_index}/"
            f"{len(demo.steps) - 1} | "
            f"Heading: {HEADINGS[step.heading]} | "
            f"{status}"
        )

        axis.set_xlabel(
            f"Intended command: {intended_text}    "
            f"Classifier prediction: "
            f"{predicted_text}"
        )

        axis.set_ylabel("Grid row")

        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )

        axis.set_aspect(
            "equal",
            adjustable="box",
        )

        figure.tight_layout()

        return []

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frame_map),
        interval=FRAME_INTERVAL_MS,
        repeat=True,
        blit=False,
    )

    print(
        f"\nSaving animation to: "
        f"{OUTPUT_GIF}"
    )

    animation.save(
        OUTPUT_GIF,
        writer=PillowWriter(fps=10),
        dpi=120,
    )

    print("Animation saved successfully.")
    print(
        "\nA window will now open and play "
        "the animation."
    )

    plt.show()


if __name__ == "__main__":
    main()
