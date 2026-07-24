"""
Interactive BCI-controlled wheelchair demonstration.

Demonstration modes:

1. Unseen recording session
   Train on A09T and test on unseen A09E.

2. Unseen person
   Train on A01T-A08T and test on unseen A09T.

3. Simulated classifier accuracy
   Artificially control prediction accuracy to study its effect on
   wheelchair navigation.

Run:

    python -m scripts.simulation.run_navigation_demo
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

import matplotlib
import numpy as np

if sys.platform == "darwin":
    try:
        matplotlib.use("MacOSX")
    except ImportError:
        matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from bci_wheelchair.simulation import (
    EEGPredictionSampler,
    GridEnvironment,
    SimulationResult,
    WheelchairState,
    run_classifier_simulation,
)


UNSEEN_SESSION_PREDICTIONS = Path(
    "results/within_subject/predictions/"
    "test_predicted_commands.csv"
)

UNSEEN_PERSON_PREDICTIONS = Path(
    "results/cross_subject/csp_fbcsp/"
    "cross_subject_a09_predictions.csv"
)

GRID_SIZES = (10, 20, 30, 40)

ACCURACY_LEVELS = (
    0.25,
    0.40,
    0.50,
    0.60,
    0.66,
    0.70,
    0.80,
    0.90,
    1.00,
)

VALID_COMMANDS = (
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
)

COMMAND_LABELS = {
    "left_hand": "Turn left",
    "right_hand": "Turn right",
    "feet": "Move forward",
    "tongue": "Stop",
}

HEADING_LABELS = {
    0: "North",
    90: "East",
    180: "South",
    270: "West",
}

MAX_STEPS = 500
FRAME_INTERVAL_MS = 80
RANDOM_SEED = 42


class PredictionSampler(Protocol):
    """Interface required by the navigation runner."""

    def sample_prediction(
        self,
        intended_class: str,
    ) -> str:
        """Return one predicted motor-imagery command."""


class SimulatedAccuracySampler:
    """Generate predictions at a selected classification accuracy."""

    def __init__(
        self,
        accuracy: float,
        random_seed: int | None = None,
    ) -> None:
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(
                "Accuracy must be between 0 and 1."
            )

        self.accuracy = accuracy
        self.rng = np.random.default_rng(random_seed)

    def sample_prediction(
        self,
        intended_class: str,
    ) -> str:
        """Return either the intended command or an incorrect command."""

        if intended_class not in VALID_COMMANDS:
            raise ValueError(
                f"Unknown command: {intended_class}"
            )

        if self.rng.random() < self.accuracy:
            return intended_class

        incorrect_commands = [
            command
            for command in VALID_COMMANDS
            if command != intended_class
        ]

        return str(
            self.rng.choice(incorrect_commands)
        )


class NavigationDemo:
    """Manage the interactive wheelchair demonstration."""

    def __init__(self) -> None:
        self.figure = plt.figure(
            figsize=(14, 8.5)
        )

        try:
            self.figure.canvas.manager.set_window_title(
                "BCI Wheelchair Demonstration"
            )
        except AttributeError:
            pass

        self.navigation_axis = self.figure.add_axes(
            [0.05, 0.10, 0.62, 0.82]
        )

        self.information_axis = self.figure.add_axes(
            [0.72, 0.04, 0.26, 0.25]
        )

        self.information_axis.axis("off")

        self.grid_size = 20
        self.mode = "unseen_session"
        self.simulated_accuracy = 0.66

        self.result: SimulationResult | None = None
        self.current_frame = 0
        self.timer = None
        self.paused = False

        self.mode_buttons: dict[str, Button] = {}
        self.grid_buttons: dict[int, Button] = {}
        self.accuracy_buttons: dict[float, Button] = {}

        self._create_interface()
        self._update_button_styles()
        self._show_welcome_screen()
        self._show_current_selection()

    def _create_button(
        self,
        label: str,
        left: float,
        bottom: float,
        width: float,
        height: float,
        callback,
    ) -> Button:
        """Create one interface button."""

        axis = self.figure.add_axes(
            [left, bottom, width, height]
        )

        button = Button(
            axis,
            label,
            hovercolor="0.85",
        )

        button.on_clicked(callback)

        return button

    def _create_interface(self) -> None:
        """Create all labels and controls."""

        self.figure.text(
            0.72,
            0.93,
            "Prediction source",
            fontsize=13,
            fontweight="bold",
        )

        self.mode_buttons["unseen_session"] = (
            self._create_button(
                "Unseen session: A09T → A09E",
                0.72,
                0.865,
                0.26,
                0.045,
                lambda event:
                self._set_mode("unseen_session"),
            )
        )

        self.mode_buttons["unseen_person"] = (
            self._create_button(
                "Unseen person: A01-A08 → A09",
                0.72,
                0.81,
                0.26,
                0.045,
                lambda event:
                self._set_mode("unseen_person"),
            )
        )

        self.mode_buttons["simulated"] = (
            self._create_button(
                "Simulated classifier accuracy",
                0.72,
                0.755,
                0.26,
                0.045,
                lambda event:
                self._set_mode("simulated"),
            )
        )

        self.figure.text(
            0.72,
            0.705,
            "Grid size",
            fontsize=13,
            fontweight="bold",
        )

        grid_layout = {
            10: (0.72, 0.645),
            20: (0.855, 0.645),
            30: (0.72, 0.59),
            40: (0.855, 0.59),
        }

        for size, (left, bottom) in grid_layout.items():
            self.grid_buttons[size] = self._create_button(
                f"{size} × {size}",
                left,
                bottom,
                0.125,
                0.045,
                lambda event, selected=size:
                self._set_grid_size(selected),
            )

        self.figure.text(
            0.72,
            0.54,
            "Simulated accuracy",
            fontsize=13,
            fontweight="bold",
        )

        accuracy_layout = {
            0.25: (0.72, 0.48),
            0.40: (0.81, 0.48),
            0.50: (0.90, 0.48),
            0.60: (0.72, 0.425),
            0.66: (0.81, 0.425),
            0.70: (0.90, 0.425),
            0.80: (0.72, 0.37),
            0.90: (0.81, 0.37),
            1.00: (0.90, 0.37),
        }

        for accuracy, (
            left,
            bottom,
        ) in accuracy_layout.items():
            self.accuracy_buttons[accuracy] = (
                self._create_button(
                    f"{accuracy * 100:.0f}%",
                    left,
                    bottom,
                    0.08,
                    0.043,
                    lambda event, selected=accuracy:
                    self._set_accuracy(selected),
                )
            )

        self.run_button = self._create_button(
            "Run demonstration",
            0.72,
            0.305,
            0.26,
            0.045,
            self._run_demo,
        )

        self.pause_button = self._create_button(
            "Pause",
            0.72,
            0.25,
            0.08,
            0.043,
            self._toggle_pause,
        )

        self.replay_button = self._create_button(
            "Replay",
            0.81,
            0.25,
            0.08,
            0.043,
            self._replay,
        )

        self.stop_button = self._create_button(
            "Reset",
            0.90,
            0.25,
            0.08,
            0.043,
            self._reset,
        )

    def _set_mode(
        self,
        mode: str,
    ) -> None:
        """Select a prediction source."""

        self.mode = mode
        self._update_button_styles()
        self._show_current_selection()

    def _set_grid_size(
        self,
        grid_size: int,
    ) -> None:
        """Select a grid size."""

        self.grid_size = grid_size
        self._update_button_styles()
        self._show_current_selection()

    def _set_accuracy(
        self,
        accuracy: float,
    ) -> None:
        """Select simulated accuracy and activate simulated mode."""

        self.simulated_accuracy = accuracy
        self.mode = "simulated"

        self._update_button_styles()
        self._show_current_selection()

    def _update_button_styles(self) -> None:
        """Visually identify selected controls."""

        normal_colour = "0.90"
        selected_colour = "0.72"

        for mode, button in self.mode_buttons.items():
            button.ax.set_facecolor(
                selected_colour
                if mode == self.mode
                else normal_colour
            )

        for size, button in self.grid_buttons.items():
            button.ax.set_facecolor(
                selected_colour
                if size == self.grid_size
                else normal_colour
            )

        for accuracy, button in self.accuracy_buttons.items():
            selected = (
                self.mode == "simulated"
                and accuracy == self.simulated_accuracy
            )

            button.ax.set_facecolor(
                selected_colour
                if selected
                else normal_colour
            )

        self.figure.canvas.draw_idle()

    def _mode_title(self) -> str:
        """Return a readable mode name."""

        titles = {
            "unseen_session": "Unseen recording session",
            "unseen_person": "Unseen person",
            "simulated": "Simulated classifier",
        }

        return titles[self.mode]

    def _mode_details(self) -> str:
        """Describe the selected prediction source."""

        details = {
            "unseen_session": (
                "The classifier is trained on A09T.\n"
                "It is tested on the separate A09E session.\n"
                "A09E is not used during training."
            ),
            "unseen_person": (
                "The classifier is trained on A01T-A08T.\n"
                "It is tested on the unseen person A09T.\n"
                "A09 is not used during training."
            ),
            "simulated": (
                "Artificial classifier predictions are used.\n"
                f"Selected accuracy: "
                f"{self.simulated_accuracy * 100:.0f}%.\n"
                "This measures accuracy versus navigation steps."
            ),
        }

        return details[self.mode]

    def _show_current_selection(self) -> None:
        """Show selected settings in the information panel."""

        self.information_axis.clear()
        self.information_axis.axis("off")

        information = (
            f"CURRENT SELECTION\n\n"
            f"Mode: {self._mode_title()}\n"
            f"Grid: {self.grid_size} × {self.grid_size}\n\n"
            f"{self._mode_details()}"
        )

        self.information_axis.text(
            0.0,
            1.0,
            information,
            va="top",
            fontsize=10.5,
            linespacing=1.4,
            transform=self.information_axis.transAxes,
        )

        self.figure.canvas.draw_idle()

    def _show_welcome_screen(self) -> None:
        """Show initial instructions."""

        self.navigation_axis.clear()
        self.navigation_axis.axis("off")

        self.navigation_axis.text(
            0.5,
            0.72,
            "BCI-Controlled Wheelchair",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            transform=self.navigation_axis.transAxes,
        )

        self.navigation_axis.text(
            0.5,
            0.52,
            "Choose a prediction source, grid size,\n"
            "and simulated accuracy when required.",
            ha="center",
            va="center",
            fontsize=15,
            linespacing=1.5,
            transform=self.navigation_axis.transAxes,
        )

        self.navigation_axis.text(
            0.5,
            0.31,
            "Then press Run demonstration.",
            ha="center",
            va="center",
            fontsize=13,
            transform=self.navigation_axis.transAxes,
        )

        self.figure.canvas.draw_idle()

    def _build_sampler(self) -> PredictionSampler:
        """Create the selected prediction sampler."""

        if self.mode == "unseen_session":
            if not UNSEEN_SESSION_PREDICTIONS.exists():
                raise FileNotFoundError(
                    "Unseen-session prediction file is missing:\n"
                    f"{UNSEEN_SESSION_PREDICTIONS.resolve()}"
                )

            return EEGPredictionSampler(
                UNSEEN_SESSION_PREDICTIONS,
                random_seed=RANDOM_SEED,
            )

        if self.mode == "unseen_person":
            if not UNSEEN_PERSON_PREDICTIONS.exists():
                raise FileNotFoundError(
                    "Unseen-person prediction file is missing:\n"
                    f"{UNSEEN_PERSON_PREDICTIONS.resolve()}"
                )

            return EEGPredictionSampler(
                UNSEEN_PERSON_PREDICTIONS,
                random_seed=RANDOM_SEED,
            )

        return SimulatedAccuracySampler(
            accuracy=self.simulated_accuracy,
            random_seed=RANDOM_SEED,
        )

    def _build_result(self) -> SimulationResult:
        """Run one complete navigation simulation."""

        environment = GridEnvironment(
            rows=self.grid_size,
            cols=self.grid_size,
        )

        start_state = WheelchairState(
            position=(self.grid_size - 2, 1),
            heading=0,
        )

        target = (
            1,
            self.grid_size - 2,
        )

        return run_classifier_simulation(
            environment=environment,
            sampler=self._build_sampler(),
            start_state=start_state,
            target=target,
            max_steps=MAX_STEPS,
        )

    def _run_demo(
        self,
        event=None,
    ) -> None:
        """Build and animate the selected demonstration."""

        try:
            self.result = self._build_result()
        except Exception as error:
            self._show_error(str(error))
            return

        self.current_frame = 0
        self.paused = False
        self.pause_button.label.set_text("Pause")

        if self.timer is not None:
            self.timer.stop()

        self.timer = self.figure.canvas.new_timer(
            interval=FRAME_INTERVAL_MS
        )

        self.timer.add_callback(
            self._advance_frame
        )

        self._draw_frame()
        self.timer.start()

    def _advance_frame(self) -> None:
        """Advance the animation by one navigation step."""

        if self.paused or self.result is None:
            return

        if self.current_frame < len(self.result.path) - 1:
            self.current_frame += 1
            self._draw_frame()
        elif self.timer is not None:
            self.timer.stop()

    def _draw_frame(self) -> None:
        """Draw the current navigation state."""

        if self.result is None:
            return

        axis = self.navigation_axis
        axis.clear()

        axis.set_xlim(
            -0.5,
            self.grid_size - 0.5,
        )

        axis.set_ylim(
            self.grid_size - 0.5,
            -0.5,
        )

        axis.set_aspect("equal")

        axis.set_xticks(
            np.arange(
                -0.5,
                self.grid_size,
                1,
            ),
            minor=True,
        )

        axis.set_yticks(
            np.arange(
                -0.5,
                self.grid_size,
                1,
            ),
            minor=True,
        )

        axis.grid(
            which="minor",
            linewidth=0.5,
            alpha=0.45,
        )

        axis.tick_params(
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

        visible_path = self.result.path[
            : self.current_frame + 1
        ]

        rows = [
            position[0]
            for position in visible_path
        ]

        columns = [
            position[1]
            for position in visible_path
        ]

        axis.plot(
            columns,
            rows,
            linewidth=2,
            label="Wheelchair path",
        )

        start = self.result.start_state.position
        target = self.result.target
        current_position = visible_path[-1]

        axis.scatter(
            start[1],
            start[0],
            marker="s",
            s=130,
            label="Start",
        )

        axis.scatter(
            target[1],
            target[0],
            marker="*",
            s=220,
            label="Target",
        )

        axis.scatter(
            current_position[1],
            current_position[0],
            marker="o",
            s=150,
            label="Wheelchair",
        )

        axis.set_title(
            f"{self.grid_size} × {self.grid_size} grid"
            f"   |   Step {self.current_frame}",
            fontsize=16,
            pad=12,
        )

        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=3,
        )

        self._update_result_information()

        self.figure.canvas.draw_idle()

    def _update_result_information(self) -> None:
        """Display clear metrics without overlapping controls."""

        if self.result is None:
            return

        if self.current_frame == 0:
            intended_command = "Not started"
            predicted_command = "Not started"
        else:
            trace_step = self.result.trace[
                self.current_frame - 1
            ]

            intended_command = COMMAND_LABELS[
                trace_step.intended_action
            ]

            predicted_command = COMMAND_LABELS[
                trace_step.predicted_action
            ]

        current_heading = self.result.headings[
            self.current_frame
        ]

        total_predictions = (
            self.result.correct_predictions
            + self.result.incorrect_predictions
        )

        empirical_accuracy = (
            self.result.correct_predictions
            / total_predictions
            * 100
            if total_predictions > 0
            else 0.0
        )

        reached_target = (
            "Yes"
            if self.result.reached_target
            else "No"
        )

        information = (
            f"{self._mode_title().upper()}\n"
            f"{self._mode_details()}\n\n"
            f"NAVIGATION RESULT\n"
            f"Grid: {self.grid_size} × {self.grid_size}\n"
            f"Current heading: "
            f"{HEADING_LABELS[current_heading]}\n"
            f"Intended command: {intended_command}\n"
            f"Predicted command: {predicted_command}\n\n"
            f"Reached target: {reached_target}\n"
            f"Total steps: {self.result.steps}\n"
            f"Initial distance: "
            f"{self.result.initial_distance}\n"
            f"Final distance: "
            f"{self.result.final_distance}\n"
            f"Empirical accuracy: "
            f"{empirical_accuracy:.1f}%\n"
            f"Incorrect predictions: "
            f"{self.result.incorrect_predictions}\n"
            f"Blocked moves: "
            f"{self.result.blocked_moves}\n"
            f"Stop commands: "
            f"{self.result.stop_commands}"
        )

        self.information_axis.clear()
        self.information_axis.axis("off")

        self.information_axis.text(
            0.0,
            1.0,
            information,
            va="top",
            fontsize=9.2,
            linespacing=1.25,
            transform=self.information_axis.transAxes,
        )

    def _toggle_pause(
        self,
        event=None,
    ) -> None:
        """Pause or resume the animation."""

        if self.result is None:
            return

        self.paused = not self.paused

        self.pause_button.label.set_text(
            "Resume"
            if self.paused
            else "Pause"
        )

        self.figure.canvas.draw_idle()

    def _replay(
        self,
        event=None,
    ) -> None:
        """Replay the current simulation result."""

        if self.result is None:
            self._run_demo()
            return

        self.current_frame = 0
        self.paused = False
        self.pause_button.label.set_text("Pause")

        if self.timer is not None:
            self.timer.stop()

        self.timer = self.figure.canvas.new_timer(
            interval=FRAME_INTERVAL_MS
        )

        self.timer.add_callback(
            self._advance_frame
        )

        self._draw_frame()
        self.timer.start()

    def _reset(
        self,
        event=None,
    ) -> None:
        """Stop the simulation and return to the welcome screen."""

        if self.timer is not None:
            self.timer.stop()

        self.result = None
        self.current_frame = 0
        self.paused = False

        self.pause_button.label.set_text("Pause")

        self._show_welcome_screen()
        self._show_current_selection()

    def _show_error(
        self,
        message: str,
    ) -> None:
        """Display an error inside the demonstration window."""

        if self.timer is not None:
            self.timer.stop()

        self.navigation_axis.clear()
        self.navigation_axis.axis("off")

        self.navigation_axis.text(
            0.5,
            0.60,
            "Unable to run demonstration",
            ha="center",
            fontsize=18,
            fontweight="bold",
            transform=self.navigation_axis.transAxes,
        )

        self.navigation_axis.text(
            0.5,
            0.40,
            message,
            ha="center",
            fontsize=11,
            wrap=True,
            transform=self.navigation_axis.transAxes,
        )

        self.figure.canvas.draw_idle()


def main() -> None:
    """Launch the interactive BCI wheelchair demonstration."""

    NavigationDemo()
    plt.show()


if __name__ == "__main__":
    main()
