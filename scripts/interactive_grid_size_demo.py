"""
Interactive EEG wheelchair navigation demonstration.

Select a grid size using buttons:

    10x10
    20x20
    30x30
    40x40

The demonstration uses predictions sampled from:

    results/evaluation_predictions.csv

Run from the project root:

    python -m scripts.interactive_grid_size_demo
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

from bci_wheelchair.eeg_sampler import EEGPredictionSampler


PREDICTIONS_PATH = Path("results/evaluation_predictions.csv")

GRID_SIZES = [10, 20, 30, 40]

MAX_STEPS = 500
FRAME_INTERVAL_MS = 120

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
    """Store one navigation step."""

    row: int
    col: int
    heading: int
    intended_command: str
    predicted_command: str
    reached_target: bool


@dataclass
class GridDemo:
    """Store one complete trajectory."""

    grid_size: int
    start: tuple[int, int]
    target: tuple[int, int]
    steps: list[DemoStep]


def turn_left(heading: int) -> int:
    """Rotate 90 degrees left."""

    return (heading - 90) % 360


def turn_right(heading: int) -> int:
    """Rotate 90 degrees right."""

    return (heading + 90) % 360


def heading_delta(
    heading: int,
) -> tuple[int, int]:
    """Return movement for the current heading."""

    movements = {
        0: (-1, 0),
        90: (0, 1),
        180: (1, 0),
        270: (0, -1),
    }

    return movements[heading]


def desired_heading(
    position: tuple[int, int],
    target: tuple[int, int],
) -> int | None:
    """Return the heading required to move towards the target."""

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

    difference = (required_heading - heading) % 360

    if difference == 90:
        return "right_hand"

    if difference == 270:
        return "left_hand"

    return "right_hand"


def apply_prediction(
    position: tuple[int, int],
    heading: int,
    predicted_command: str,
    grid_size: int,
) -> tuple[tuple[int, int], int]:
    """Apply a classifier command to the wheelchair."""

    if predicted_command == "left_hand":
        return position, turn_left(heading)

    if predicted_command == "right_hand":
        return position, turn_right(heading)

    if predicted_command == "tongue":
        return position, heading

    if predicted_command != "feet":
        raise ValueError(
            f"Unknown predicted command: {predicted_command}"
        )

    row, col = position
    row_delta, col_delta = heading_delta(heading)

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
    """Generate one trajectory using evaluation-data predictions."""

    sampler = EEGPredictionSampler(
        PREDICTIONS_PATH,
        random_seed=random_seed,
    )

    start = (grid_size - 2, 1)
    target = (1, grid_size - 2)

    position = start
    heading = 0

    steps = [
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

        predicted_command = sampler.sample_prediction(
            correct_command
        )

        position, heading = apply_prediction(
            position=position,
            heading=heading,
            predicted_command=predicted_command,
            grid_size=grid_size,
        )

        steps.append(
            DemoStep(
                row=position[0],
                col=position[1],
                heading=heading,
                intended_command=correct_command,
                predicted_command=predicted_command,
                reached_target=(position == target),
            )
        )

    return GridDemo(
        grid_size=grid_size,
        start=start,
        target=target,
        steps=steps,
    )


class InteractiveGridDemo:
    """Interactive grid-size navigation UI."""

    def __init__(self) -> None:
        if not PREDICTIONS_PATH.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {PREDICTIONS_PATH}"
            )

        self.figure, self.axis = plt.subplots(
            figsize=(10, 8)
        )

        self.figure.subplots_adjust(
            left=0.08,
            right=0.78,
            bottom=0.12,
            top=0.90,
        )

        self.current_demo: GridDemo | None = None
        self.current_frame = 0
        self.animation: FuncAnimation | None = None
        self.is_paused = False

        self.create_buttons()
        self.show_welcome_screen()

    def create_buttons(self) -> None:
        """Create grid selection and animation buttons."""

        button_width = 0.14
        button_height = 0.055
        button_left = 0.82

        button_positions = [
            ("10×10", 0.78, 10),
            ("20×20", 0.70, 20),
            ("30×30", 0.62, 30),
            ("40×40", 0.54, 40),
        ]

        self.grid_buttons = []

        for label, bottom, grid_size in button_positions:
            button_axis = self.figure.add_axes(
                [
                    button_left,
                    bottom,
                    button_width,
                    button_height,
                ]
            )

            button = Button(
                button_axis,
                label,
            )

            button.on_clicked(
                lambda event, size=grid_size:
                self.start_demo(size)
            )

            self.grid_buttons.append(button)

        replay_axis = self.figure.add_axes(
            [
                button_left,
                0.38,
                button_width,
                button_height,
            ]
        )

        self.replay_button = Button(
            replay_axis,
            "Replay",
        )

        self.replay_button.on_clicked(
            self.replay_demo
        )

        pause_axis = self.figure.add_axes(
            [
                button_left,
                0.30,
                button_width,
                button_height,
            ]
        )

        self.pause_button = Button(
            pause_axis,
            "Pause",
        )

        self.pause_button.on_clicked(
            self.toggle_pause
        )

    def show_welcome_screen(self) -> None:
        """Show instructions before a grid is selected."""

        self.axis.clear()
        self.axis.axis("off")

        self.axis.text(
            0.5,
            0.62,
            "EEG-Controlled Wheelchair",
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
            transform=self.axis.transAxes,
        )

        self.axis.text(
            0.5,
            0.48,
            "Select a grid size using the buttons",
            ha="center",
            va="center",
            fontsize=15,
            transform=self.axis.transAxes,
        )

        self.axis.text(
            0.5,
            0.38,
            "The wheelchair will navigate using\n"
            "predictions sampled from the unseen evaluation data.",
            ha="center",
            va="center",
            fontsize=12,
            transform=self.axis.transAxes,
        )

        self.figure.canvas.draw_idle()

    def start_demo(
        self,
        grid_size: int,
    ) -> None:
        """Generate and start a selected grid demonstration."""

        if self.animation is not None:
            event_source = self.animation.event_source

            if event_source is not None:
                event_source.stop()

            self.animation = None

        self.current_demo = build_demo(
            grid_size=grid_size,
            random_seed=100 + grid_size,
        )

        self.current_frame = 0
        self.is_paused = False
        self.pause_button.label.set_text("Pause")

        total_steps = len(self.current_demo.steps) - 1
        reached = self.current_demo.steps[-1].reached_target

        print(
            f"\nSelected {grid_size}x{grid_size}"
        )
        print(
            f"Steps: {total_steps}"
        )
        print(
            f"Target reached: {reached}"
        )

        self.animation = FuncAnimation(
            self.figure,
            self.update_frame,
            frames=len(self.current_demo.steps),
            interval=FRAME_INTERVAL_MS,
            repeat=False,
            blit=False,
        )

        self.figure.canvas.draw_idle()

    def replay_demo(
        self,
        event,
    ) -> None:
        """Replay the currently selected grid."""

        if self.current_demo is None:
            return

        self.start_demo(
            self.current_demo.grid_size
        )

    def toggle_pause(
        self,
        event,
    ) -> None:
        """Pause or resume the animation."""

        if self.animation is None:
            return

        event_source = self.animation.event_source

        if event_source is None:
            return

        if self.is_paused:
            event_source.start()
            self.pause_button.label.set_text(
                "Pause"
            )
            self.is_paused = False
        else:
            event_source.stop()
            self.pause_button.label.set_text(
                "Resume"
            )
            self.is_paused = True

        self.figure.canvas.draw_idle()

    def update_frame(
        self,
        frame_index: int,
    ):
        """Draw one animation frame."""

        if self.current_demo is None:
            return []

        self.current_frame = frame_index

        demo = self.current_demo
        step = demo.steps[frame_index]
        size = demo.grid_size

        self.axis.clear()

        self.axis.set_xlim(-0.5, size - 0.5)
        self.axis.set_ylim(size - 0.5, -0.5)

        self.axis.set_xticks(
            range(size)
        )
        self.axis.set_yticks(
            range(size)
        )

        if size >= 30:
            self.axis.set_xticklabels(
                [
                    str(value)
                    if value % 5 == 0
                    else ""
                    for value in range(size)
                ]
            )

            self.axis.set_yticklabels(
                [
                    str(value)
                    if value % 5 == 0
                    else ""
                    for value in range(size)
                ]
            )

        self.axis.grid(True)

        path_rows = [
            item.row
            for item in demo.steps[: frame_index + 1]
        ]

        path_cols = [
            item.col
            for item in demo.steps[: frame_index + 1]
        ]

        self.axis.plot(
            path_cols,
            path_rows,
            linewidth=2,
            label="Wheelchair path",
        )

        self.axis.scatter(
            demo.start[1],
            demo.start[0],
            marker="s",
            s=150,
            label="Start",
            zorder=4,
        )

        self.axis.scatter(
            demo.target[1],
            demo.target[0],
            marker="*",
            s=250,
            label="Target",
            zorder=4,
        )

        self.axis.scatter(
            step.col,
            step.row,
            marker="o",
            s=200,
            label="Wheelchair",
            zorder=5,
        )

        status = (
            "Target reached"
            if step.reached_target
            else "Navigating"
        )

        intended_text = COMMAND_LABELS.get(
            step.intended_command,
            step.intended_command,
        )

        predicted_text = COMMAND_LABELS.get(
            step.predicted_command,
            step.predicted_command,
        )

        self.axis.set_title(
            f"EEG Wheelchair Navigation — "
            f"{size}×{size} Grid\n"
            f"Step {frame_index}/"
            f"{len(demo.steps) - 1} | "
            f"Heading: {HEADINGS[step.heading]} | "
            f"{status}"
        )

        self.axis.set_xlabel(
            f"Intended: {intended_text}    "
            f"Predicted: {predicted_text}"
        )

        self.axis.set_ylabel(
            "Grid row"
        )

        self.axis.legend(
            loc="upper left",
        )

        self.axis.set_aspect(
            "equal",
            adjustable="box",
        )

        self.figure.canvas.draw_idle()

        return []

    def run(self) -> None:
        """Open the interactive interface."""

        plt.show()


def main() -> None:
    """Run the interactive UI."""

    application = InteractiveGridDemo()
    application.run()


if __name__ == "__main__":
    main()
