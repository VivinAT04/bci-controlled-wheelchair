"""
Interactive EEG-controlled wheelchair grid-size demonstration.

Run from the project root:

    python -m scripts.interactive_grid_size_demo
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Select an interactive backend before importing pyplot.
if sys.platform == "darwin":
    try:
        matplotlib.use("MacOSX")
    except ImportError:
        matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from bci_wheelchair.eeg_sampler import EEGPredictionSampler


PREDICTIONS_PATH = Path("results/cross_subject_a09_predictions.csv")

GRID_SIZES = (10, 20, 30, 40)
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


@dataclass(frozen=True)
class DemoStep:
    """Represent one navigation step."""

    row: int
    column: int
    heading: int
    intended_command: str
    predicted_command: str
    reached_target: bool


@dataclass(frozen=True)
class GridDemo:
    """Represent one complete grid trajectory."""

    grid_size: int
    start: tuple[int, int]
    target: tuple[int, int]
    steps: list[DemoStep]


def turn_left(heading: int) -> int:
    """Turn 90 degrees left."""

    return (heading - 90) % 360


def turn_right(heading: int) -> int:
    """Turn 90 degrees right."""

    return (heading + 90) % 360


def get_heading_delta(heading: int) -> tuple[int, int]:
    """Return row and column changes for a heading."""

    movements = {
        0: (-1, 0),
        90: (0, 1),
        180: (1, 0),
        270: (0, -1),
    }

    if heading not in movements:
        raise ValueError(f"Unsupported heading: {heading}")

    return movements[heading]


def get_desired_heading(
    position: tuple[int, int],
    target: tuple[int, int],
) -> int | None:
    """Determine the direction required to approach the target."""

    current_row, current_column = position
    target_row, target_column = target

    if position == target:
        return None

    if target_column > current_column:
        return 90

    if target_column < current_column:
        return 270

    if target_row > current_row:
        return 180

    return 0


def choose_intended_command(
    position: tuple[int, int],
    heading: int,
    target: tuple[int, int],
) -> str:
    """Choose the ideal motor-imagery command."""

    desired_heading = get_desired_heading(
        position=position,
        target=target,
    )

    if desired_heading is None:
        return "tongue"

    if heading == desired_heading:
        return "feet"

    difference = (desired_heading - heading) % 360

    if difference == 90:
        return "right_hand"

    if difference == 270:
        return "left_hand"

    return "right_hand"


def apply_command(
    position: tuple[int, int],
    heading: int,
    command: str,
    grid_size: int,
) -> tuple[tuple[int, int], int]:
    """Apply a predicted command to the wheelchair."""

    if command == "left_hand":
        return position, turn_left(heading)

    if command == "right_hand":
        return position, turn_right(heading)

    if command == "tongue":
        return position, heading

    if command != "feet":
        raise ValueError(f"Unknown command: {command}")

    row, column = position
    row_change, column_change = get_heading_delta(heading)

    new_row = row + row_change
    new_column = column + column_change

    if (
        0 <= new_row < grid_size
        and 0 <= new_column < grid_size
    ):
        return (new_row, new_column), heading

    return position, heading


def build_demo(
    grid_size: int,
    random_seed: int,
) -> GridDemo:
    """Generate a trajectory using evaluation predictions."""

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
            column=position[1],
            heading=heading,
            intended_command="tongue",
            predicted_command="tongue",
            reached_target=False,
        )
    ]

    for _ in range(MAX_STEPS):
        if position == target:
            break

        intended_command = choose_intended_command(
            position=position,
            heading=heading,
            target=target,
        )

        predicted_command = sampler.sample_prediction(
            intended_command
        )

        position, heading = apply_command(
            position=position,
            heading=heading,
            command=predicted_command,
            grid_size=grid_size,
        )

        steps.append(
            DemoStep(
                row=position[0],
                column=position[1],
                heading=heading,
                intended_command=intended_command,
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
    """Manage the interactive animation interface."""

    def __init__(self) -> None:
        if not PREDICTIONS_PATH.exists():
            raise FileNotFoundError(
                f"Prediction file not found: "
                f"{PREDICTIONS_PATH.resolve()}"
            )

        self.figure, self.axis = plt.subplots(figsize=(11, 8))

        self.figure.subplots_adjust(
            left=0.08,
            right=0.77,
            bottom=0.13,
            top=0.90,
        )

        try:
            self.figure.canvas.manager.set_window_title(
                "EEG Wheelchair Navigation"
            )
        except AttributeError:
            pass

        self.current_demo: GridDemo | None = None
        self.current_frame = 0
        self.timer = None
        self.is_paused = False

        self.grid_buttons: list[Button] = []

        self.create_buttons()
        self.show_welcome_screen()

    def create_button(
        self,
        label: str,
        bottom: float,
        callback,
    ) -> Button:
        """Create one interface button."""

        button_axis = self.figure.add_axes(
            [0.81, bottom, 0.15, 0.06]
        )

        button = Button(button_axis, label)
        button.on_clicked(callback)

        return button

    def create_buttons(self) -> None:
        """Create grid selection and playback controls."""

        button_data = [
            ("10 × 10", 0.79, 10),
            ("20 × 20", 0.71, 20),
            ("30 × 30", 0.63, 30),
            ("40 × 40", 0.55, 40),
        ]

        for label, bottom, grid_size in button_data:
            button = self.create_button(
                label=label,
                bottom=bottom,
                callback=(
                    lambda event, size=grid_size:
                    self.start_demo(size)
                ),
            )

            self.grid_buttons.append(button)

        self.replay_button = self.create_button(
            label="Replay",
            bottom=0.39,
            callback=self.replay_demo,
        )

        self.pause_button = self.create_button(
            label="Pause",
            bottom=0.31,
            callback=self.toggle_pause,
        )

        self.stop_button = self.create_button(
            label="Stop",
            bottom=0.23,
            callback=self.stop_demo,
        )

    def show_welcome_screen(self) -> None:
        """Display the initial instructions."""

        self.axis.clear()
        self.axis.axis("off")

        self.axis.text(
            0.5,
            0.65,
            "EEG-Controlled Wheelchair",
            ha="center",
            va="center",
            fontsize=23,
            fontweight="bold",
            transform=self.axis.transAxes,
        )

        self.axis.text(
            0.5,
            0.50,
            "Select a grid size using the buttons",
            ha="center",
            va="center",
            fontsize=16,
            transform=self.axis.transAxes,
        )

        self.axis.text(
            0.5,
            0.36,
            "The wheelchair uses classifier predictions\n"
            "sampled from the unseen evaluation dataset.",
            ha="center",
            va="center",
            fontsize=13,
            transform=self.axis.transAxes,
        )

        self.figure.canvas.draw_idle()

    def stop_timer(self) -> None:
        """Safely stop the current animation timer."""

        if self.timer is not None:
            self.timer.stop()
            self.timer = None

    def start_demo(self, grid_size: int) -> None:
        """Start the selected grid-size demonstration."""

        self.stop_timer()

        print(f"\nGenerating {grid_size}x{grid_size} trajectory...")

        self.current_demo = build_demo(
            grid_size=grid_size,
            random_seed=100 + grid_size,
        )

        self.current_frame = 0
        self.is_paused = False
        self.pause_button.label.set_text("Pause")

        total_steps = len(self.current_demo.steps) - 1
        target_reached = (
            self.current_demo.steps[-1].reached_target
        )

        print(f"Grid size: {grid_size}x{grid_size}")
        print(f"Steps: {total_steps}")
        print(f"Target reached: {target_reached}")

        self.draw_frame(0)

        self.timer = self.figure.canvas.new_timer(
            interval=FRAME_INTERVAL_MS
        )

        self.timer.add_callback(self.advance_frame)
        self.timer.start()

    def advance_frame(self):
        """Advance the animation by one step."""

        if self.current_demo is None:
            return False

        next_frame = self.current_frame + 1

        if next_frame >= len(self.current_demo.steps):
            self.stop_timer()
            print("Animation completed.")
            return False

        self.current_frame = next_frame
        self.draw_frame(self.current_frame)

        return True

    def replay_demo(self, event=None) -> None:
        """Replay the selected demonstration."""

        if self.current_demo is None:
            print("Select a grid size first.")
            return

        self.start_demo(self.current_demo.grid_size)

    def toggle_pause(self, event=None) -> None:
        """Pause or resume the animation."""

        if self.current_demo is None:
            print("Select a grid size first.")
            return

        if self.timer is None:
            if self.current_frame >= len(self.current_demo.steps) - 1:
                print("Animation finished. Press Replay.")
                return

            self.timer = self.figure.canvas.new_timer(
                interval=FRAME_INTERVAL_MS
            )
            self.timer.add_callback(self.advance_frame)

        if self.is_paused:
            self.timer.start()
            self.is_paused = False
            self.pause_button.label.set_text("Pause")
            print("Animation resumed.")
        else:
            self.timer.stop()
            self.is_paused = True
            self.pause_button.label.set_text("Resume")
            print("Animation paused.")

        self.figure.canvas.draw_idle()

    def stop_demo(self, event=None) -> None:
        """Stop the current animation."""

        self.stop_timer()
        self.is_paused = False
        self.pause_button.label.set_text("Pause")

        print("Animation stopped.")

    def draw_frame(self, frame_index: int) -> None:
        """Draw one animation frame."""

        if self.current_demo is None:
            return

        demo = self.current_demo
        step = demo.steps[frame_index]
        grid_size = demo.grid_size

        self.axis.clear()

        self.axis.set_xlim(-0.5, grid_size - 0.5)
        self.axis.set_ylim(grid_size - 0.5, -0.5)

        tick_interval = 1 if grid_size <= 20 else 5
        major_ticks = range(0, grid_size, tick_interval)

        self.axis.set_xticks(major_ticks)
        self.axis.set_yticks(major_ticks)

        self.axis.set_xticks(range(grid_size), minor=True)
        self.axis.set_yticks(range(grid_size), minor=True)

        self.axis.grid(
            which="minor",
            linewidth=0.4,
            alpha=0.4,
        )

        self.axis.grid(
            which="major",
            linewidth=0.8,
            alpha=0.8,
        )

        visible_steps = demo.steps[: frame_index + 1]

        path_rows = [
            trajectory_step.row
            for trajectory_step in visible_steps
        ]

        path_columns = [
            trajectory_step.column
            for trajectory_step in visible_steps
        ]

        self.axis.plot(
            path_columns,
            path_rows,
            linewidth=2.2,
            label="Wheelchair path",
            zorder=2,
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
            s=270,
            label="Target",
            zorder=4,
        )

        self.axis.scatter(
            step.column,
            step.row,
            marker="o",
            s=210,
            label="Wheelchair",
            zorder=5,
        )

        status = (
            "Target reached"
            if step.reached_target
            else "Navigating"
        )

        intended_label = COMMAND_LABELS.get(
            step.intended_command,
            step.intended_command,
        )

        predicted_label = COMMAND_LABELS.get(
            step.predicted_command,
            step.predicted_command,
        )

        self.axis.set_title(
            f"EEG Wheelchair Navigation — "
            f"{grid_size} × {grid_size} Grid\n"
            f"Step {frame_index}/"
            f"{len(demo.steps) - 1} | "
            f"Heading: {HEADINGS[step.heading]} | "
            f"{status}",
            fontsize=14,
        )

        self.axis.set_xlabel(
            f"Intended: {intended_label}     "
            f"Predicted: {predicted_label}",
            fontsize=11,
        )

        self.axis.set_ylabel("Grid row")

        self.axis.legend(
            loc="upper left",
            fontsize=9,
        )

        self.axis.set_aspect(
            "equal",
            adjustable="box",
        )

        self.figure.canvas.draw_idle()

    def run(self) -> None:
        """Open the interactive interface."""

        print(f"Matplotlib backend: {matplotlib.get_backend()}")
        print("Opening interactive demonstration...")

        plt.show(block=True)


def main() -> None:
    """Run the application."""

    print("Starting interactive grid-size demo...")

    application = InteractiveGridDemo()
    application.run()


if __name__ == "__main__":
    main()
