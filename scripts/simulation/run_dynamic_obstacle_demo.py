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

    python -m scripts.simulation.run_dynamic_obstacle_demo
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
from collections import deque
from typing import Protocol

import matplotlib
import numpy as np

if sys.platform == "darwin":
    try:
        matplotlib.use("MacOSX")
    except ImportError:
        matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

from bci_wheelchair.simulation import (
    EEGPredictionSampler,
    GridEnvironment,
    SimulationResult,
    WheelchairState,
    run_classifier_simulation,
)



METHODS = (
    "CSP + LDA",
    "CSP + SVM",

    "Tuned CSP + LDA",
    "Tuned CSP + SVM",

    "FBCSP + LDA",
    "FBCSP + SVM",

    "Riemannian MDM",

    "Riemannian TS + LDA",
    "Riemannian TS + SVM",

    "Filter-Bank Riemannian + LDA",
    "Filter-Bank Riemannian + SVM",

    "Autoencoder + LDA",
    "Autoencoder + SVM",

    "Supervised Autoencoder + LDA",
    "Supervised Autoencoder + SVM",

    "EEGNet",

    "EA + CSP + LDA",
    "EA + CSP + SVM",

    "EA + FBCSP + LDA",
    "EA + FBCSP + SVM",
)

DEFAULT_METHOD = (
    "EA + FBCSP + SVM"
)


PREDICTION_FILES = {
    "Cross-Session": {

        "CSP + LDA": Path(
            "results/cross_session/csp_lda/"
            "csp_lda_cross_session_predictions.csv"
        ),

        "CSP + SVM": Path(
            "results/cross_session/csp_rbf_svm/"
            "csp_rbf_svm_cross_session_predictions.csv"
        ),

        "Tuned CSP + LDA": Path(
            "results/cross_session/tuned_csp_lda/"
            "tuned_csp_lda_cross_session_predictions.csv"
        ),

        "Tuned CSP + SVM": Path(
            "results/cross_session/tuned_csp_rbf_svm/"
            "tuned_csp_rbf_svm_cross_session_predictions.csv"
        ),

        "FBCSP + LDA": Path(
            "results/cross_session/fbcsp_lda/"
            "fbcsp_lda_cross_session_predictions.csv"
        ),

        "FBCSP + SVM": Path(
            "results/cross_session/fbcsp_rbf_svm/"
            "fbcsp_rbf_svm_cross_session_predictions.csv"
        ),

        "Riemannian MDM": Path(
            "results/cross_session/riemannian/mdm/"
            "riemannian_mdm_cross_session_predictions.csv"
        ),

        "Riemannian TS + LDA": Path(
            "results/cross_session/riemannian/tangent_lda/"
            "tangent_lda_cross_session_predictions.csv"
        ),

        "Riemannian TS + SVM": Path(
            "results/cross_session/riemannian/tangent_rbf_svm/"
            "tangent_rbf_svm_cross_session_predictions.csv"
        ),

        "Filter-Bank Riemannian + LDA": Path(
            "results/cross_session/riemannian/filterbank/lda/"
            "filterbank_riemannian_lda_cross_session_predictions.csv"
        ),

        "Filter-Bank Riemannian + SVM": Path(
            "results/cross_session/riemannian/filterbank/svm/"
            "filterbank_riemannian_svm_cross_session_predictions.csv"
        ),

        "Autoencoder + LDA": Path(
            "results/cross_session/autoencoder_lda/"
            "autoencoder_lda_cross_session_predictions.csv"
        ),

        "Autoencoder + SVM": Path(
            "results/cross_session/autoencoder_rbf_svm/"
            "autoencoder_rbf_svm_cross_session_predictions.csv"
        ),

        "Supervised Autoencoder + LDA": Path(
            "results/cross_session/supervised_autoencoder_lda/"
            "supervised_autoencoder_lda_cross_session_predictions.csv"
        ),

        "Supervised Autoencoder + SVM": Path(
            "results/cross_session/supervised_autoencoder_rbf_svm/"
            "supervised_autoencoder_rbf_svm_cross_session_predictions.csv"
        ),

        "EEGNet": Path(
            "results/cross_session/eegnet/"
            "eegnet_cross_session_predictions.csv"
        ),

        "EA + CSP + LDA": Path(
            "results/cross_session/euclidean_alignment/csp/lda/"
            "ea_csp_lda_cross_session_predictions.csv"
        ),

        "EA + CSP + SVM": Path(
            "results/cross_session/euclidean_alignment/csp/svm/"
            "ea_csp_svm_cross_session_predictions.csv"
        ),

        "EA + FBCSP + LDA": Path(
            "results/cross_session/euclidean_alignment/fbcsp/lda/"
            "ea_fbcsp_lda_cross_session_predictions.csv"
        ),

        "EA + FBCSP + SVM": Path(
            "results/cross_session/euclidean_alignment/fbcsp/svm/"
            "ea_fbcsp_svm_cross_session_predictions.csv"
        ),
    },

    "Cross-Subject": {

        "CSP + LDA": Path(
            "results/cross_subject/csp_lda/"
            "csp_lda_cross_subject_predictions.csv"
        ),

        "CSP + SVM": Path(
            "results/cross_subject/csp_rbf_svm/"
            "csp_rbf_svm_cross_subject_predictions.csv"
        ),

        "Tuned CSP + LDA": Path(
            "results/cross_subject/tuned_csp_lda/"
            "tuned_csp_lda_cross_subject_predictions.csv"
        ),

        "Tuned CSP + SVM": Path(
            "results/cross_subject/tuned_csp_rbf_svm/"
            "tuned_csp_rbf_svm_cross_subject_predictions.csv"
        ),

        "FBCSP + LDA": Path(
            "results/cross_subject/fbcsp_lda/"
            "fbcsp_lda_cross_subject_predictions.csv"
        ),

        "FBCSP + SVM": Path(
            "results/cross_subject/fbcsp_rbf_svm/"
            "fbcsp_rbf_svm_cross_subject_predictions.csv"
        ),

        "Riemannian MDM": Path(
            "results/cross_subject/riemannian_mdm/"
            "riemannian_mdm_cross_subject_predictions.csv"
        ),

        "Riemannian TS + LDA": Path(
            "results/cross_subject/riemannian_tangent_lda/"
            "riemannian_tangent_lda_cross_subject_predictions.csv"
        ),

        "Riemannian TS + SVM": Path(
            "results/cross_subject/riemannian_tangent_rbf_svm/"
            "riemannian_tangent_rbf_svm_cross_subject_predictions.csv"
        ),

        "Filter-Bank Riemannian + LDA": Path(
            "results/cross_subject/riemannian/filterbank/lda/"
            "filterbank_riemannian_lda_cross_subject_predictions.csv"
        ),

        "Filter-Bank Riemannian + SVM": Path(
            "results/cross_subject/riemannian/filterbank/svm/"
            "filterbank_riemannian_svm_cross_subject_predictions.csv"
        ),

        "Autoencoder + LDA": Path(
            "results/cross_subject/autoencoder_lda/"
            "autoencoder_lda_cross_subject_predictions.csv"
        ),

        "Autoencoder + SVM": Path(
            "results/cross_subject/autoencoder_rbf_svm/"
            "autoencoder_rbf_svm_cross_subject_predictions.csv"
        ),

        "Supervised Autoencoder + LDA": Path(
            "results/cross_subject/supervised_autoencoder_lda/"
            "supervised_autoencoder_lda_cross_subject_predictions.csv"
        ),

        "Supervised Autoencoder + SVM": Path(
            "results/cross_subject/supervised_autoencoder_rbf_svm/"
            "supervised_autoencoder_rbf_svm_cross_subject_predictions.csv"
        ),

        "EEGNet": Path(
            "results/cross_subject/eegnet/"
            "eegnet_loso_improved/predictions.csv"
        ),

        "EA + CSP + LDA": Path(
            "results/cross_subject/euclidean_alignment/csp/lda/"
            "ea_csp_lda_cross_subject_predictions.csv"
        ),

        "EA + CSP + SVM": Path(
            "results/cross_subject/euclidean_alignment/csp/svm/"
            "ea_csp_svm_cross_subject_predictions.csv"
        ),

        "EA + FBCSP + LDA": Path(
            "results/cross_subject/euclidean_alignment/fbcsp/lda/"
            "ea_fbcsp_lda_cross_subject_predictions.csv"
        ),

        "EA + FBCSP + SVM": Path(
            "results/cross_subject/euclidean_alignment/fbcsp/svm/"
            "ea_fbcsp_svm_cross_subject_predictions.csv"
        ),
    },
}


FINAL_RESULTS = {
    "Cross-Session": {

        "CSP + LDA": (47.80, 0.304),
        "CSP + SVM": (49.31, 0.324),

        "Tuned CSP + LDA": (47.96, 0.306),
        "Tuned CSP + SVM": (51.31, 0.351),

        "FBCSP + LDA": (45.37, 0.272),
        "FBCSP + SVM": (55.90, 0.412),

        "Riemannian MDM": (36.50, 0.153),

        "Riemannian TS + LDA": (52.16, 0.362),
        "Riemannian TS + SVM": (55.59, 0.408),

        "Filter-Bank Riemannian + LDA": (49.85, 0.331),
        "Filter-Bank Riemannian + SVM": (55.36, 0.405),

        "Autoencoder + LDA": (27.24, 0.030),
        "Autoencoder + SVM": (28.94, 0.052),

        "Supervised Autoencoder + LDA": (25.42, 0.006),
        "Supervised Autoencoder + SVM": (27.28, 0.030),

        "EEGNet": (52.55, 0.367),

        "EA + CSP + LDA": (57.56, 0.434),
        "EA + CSP + SVM": (58.45, 0.446),

        "EA + FBCSP + LDA": (57.60, 0.435),
        "EA + FBCSP + SVM": (60.49, 0.473),
    },

    "Cross-Subject": {

        "CSP + LDA": (43.94, 0.253),
        "CSP + SVM": (38.66, 0.182),

        "Tuned CSP + LDA": (42.82, 0.238),
        "Tuned CSP + SVM": (36.15, 0.149),

        "FBCSP + LDA": (39.08, 0.188),
        "FBCSP + SVM": (37.58, 0.168),

        "Riemannian MDM": (34.92, 0.132),

        "Riemannian TS + LDA": (38.50, 0.180),
        "Riemannian TS + SVM": (37.69, 0.169),

        "Filter-Bank Riemannian + LDA": (37.77, 0.170),
        "Filter-Bank Riemannian + SVM": (39.12, 0.188),

        "Autoencoder + LDA": (26.97, 0.026),
        "Autoencoder + SVM": (27.70, 0.036),

        "Supervised Autoencoder + LDA": (26.20, 0.016),
        "Supervised Autoencoder + SVM": (26.04, 0.014),

        "EEGNet": (43.06, 0.241),

        "EA + CSP + LDA": (52.74, 0.370),
        "EA + CSP + SVM": (52.70, 0.369),

        "EA + FBCSP + LDA": (48.34, 0.311),
        "EA + FBCSP + SVM": (47.03, 0.294),
    },
}


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
            figsize=(16, 9.5)
        )

        try:
            self.figure.canvas.manager.set_window_title(
                "BCI Wheelchair Demonstration"
            )
        except AttributeError:
            pass

        self.navigation_axis = self.figure.add_axes(
            [0.04, 0.12, 0.62, 0.80]
        )

        # Hide Matplotlib's live "(x, y)" cursor coordinate display.
        self.navigation_axis.format_coord = (
            lambda x, y: ""
        )


        self.information_axis = self.figure.add_axes(
            [0.69, 0.055, 0.29, 0.075]
        )

        self.information_axis.axis("off")

        self.grid_size = 20

        # Real classifier predictions are now the default.
        self.mode = "real"

        # Default to the strongest final method.
        self.method = DEFAULT_METHOD

        # Supervisor Method 1 by default.
        self.evaluation = "Cross-Session"

        # Retain the existing accuracy-sensitivity mode.
        self.simulated_accuracy = 0.66

        self.result: SimulationResult | None = None
        self.current_frame = 0
        self.timer = None
        self.paused = False

        # True while a navigation animation is running.
        # Configuration controls are locked during this time.
        self.run_active = False

        # Dropdown click protection.
        # Prevents a classifier-selection mouse release from
        # falling through to the Run demonstration button.
        self.dropdown_is_open = False
        self.run_click_block_until = 0.0

        # Static obstacle used by the obstacle demo.
        self.static_obstacle_position = None
        self.obstacle_enabled = False

        # Original classifier-generated path.
        # Obstacle avoidance may temporarily deviate from this
        # path, but should rejoin it afterwards.
        self.baseline_classifier_path = None

        # Guaranteed obstacle encounter state.
        self.obstacle_crossing_cell = None
        self.obstacle_crossing_frame = None
        self.obstacle_crossing_done = False

        # Interception behaviour:
        # X moves independently toward a FUTURE wheelchair
        # path cell and arrives before the wheelchair.
        self.obstacle_intercept_cell = None
        self.obstacle_intercept_frame = None
        self.obstacle_intercept_arrived = False

        # Dynamic obstacle state machine:
        #
        # approach -> blocked -> detouring -> stationary
        #
        # approach:
        #   X moves toward a future classifier path cell.
        #
        # blocked:
        #   X remains stationary on that cell until the
        #   wheelchair performs its safety detour.
        #
        # released:
        #   X resumes natural wandering.
        self.obstacle_phase = "approach"

        # Safety pause before deviation.
        # This is a visual simulation pause, not a classifier change.
        self.obstacle_safety_pause_done = False
        self.obstacle_safety_pause_seconds = 1.0

        self.obstacle_interception_cell = None

        self.obstacle_hold_counter = 0
        self.obstacle_hold_frames = 8

        self.obstacle_crossing_route = []
        self.obstacle_crossing_index = 0


        # Moving obstacle state.
        self.moving_obstacle_path = []
        self.moving_obstacle_index = 0
        self.moving_obstacle_direction = 1
        self.moving_obstacle_position = None

        # Obstacle moves more slowly than the wheelchair
        # and wanders naturally around the grid.
        # X moves once every two wheelchair frames.
        # This is slow enough to be visually clear, but fast
        # enough to reach the selected path cell before the
        # wheelchair in the demonstration.
        self.obstacle_move_interval = 1
        self.obstacle_frame_counter = 0

        # Independent counter for alternating X movement
        # between row and column directions.
        self.obstacle_axis_counter = 0

        self.obstacle_rng = np.random.default_rng(12345)




        # --------------------------------------------------
        # DYNAMIC OBSTACLE SIMULATION
        # --------------------------------------------------

        self.dynamic_obstacle_position = None
        self.dynamic_obstacle_path = []
        self.dynamic_obstacle_time = 0
        self.dynamic_obstacle_start_time = 0
        self.dynamic_obstacle_crossing_frame = 0


        self.mode_buttons: dict[str, Button] = {}
        self.grid_buttons: dict[int, Button] = {}
        self.accuracy_buttons: dict[float, Button] = {}

        # Compact dropdown-style selectors.
        self.dropdown_option_axes = []
        self.dropdown_option_buttons = []

        # Opaque background used behind an open dropdown.
        self.dropdown_background_axis = None

        # Axes temporarily hidden while a dropdown is open.
        self.dropdown_hidden_axes = []

        # Compact dropdown-style selectors.
        self.dropdown_option_axes = []
        self.dropdown_option_buttons = []

        # Compact dropdown-style selectors.
        self.dropdown_option_axes = []
        self.dropdown_option_buttons = []

        self._create_interface()
        self._update_button_styles()
        self._update_obstacle_radio_style()
        self._show_welcome_screen()

    def _create_button(
        self,
        label: str,
        left: float,
        bottom: float,
        width: float,
        height: float,
        callback,
    ) -> Button:
        """Create a button with consistent hover behaviour."""

        axis = self.figure.add_axes(
            [
                left,
                bottom,
                width,
                height,
            ]
        )

        button = Button(
            axis,
            label,
            color="0.90",
            hovercolor="0.72",
        )

        button.label.set_fontsize(
            9.5
        )

        button.on_clicked(
            callback
        )

        return button


    def _create_interface(self) -> None:
        """Create compact dropdown-style simulation controls."""

        # -----------------------------------------------------
        # CLASSIFIER METHOD DROPDOWN
        # -----------------------------------------------------

        self.figure.text(
            0.695,
            0.925,
            "Classifier method",
            fontsize=12,
            fontweight="bold",
        )

        self.method_dropdown = self._create_button(
            f"{self.method}    ▼",
            0.69,
            0.865,
            0.29,
            0.045,
            self._toggle_method_dropdown,
        )

        # -----------------------------------------------------
        # EVALUATION DROPDOWN
        # -----------------------------------------------------

        self.figure.text(
            0.695,
            0.825,
            "Evaluation",
            fontsize=11.5,
            fontweight="bold",
        )

        self.evaluation_dropdown = self._create_button(
            f"{self.evaluation}    ▼",
            0.69,
            0.770,
            0.29,
            0.042,
            self._toggle_evaluation_dropdown,
        )

        # -----------------------------------------------------
        # PREDICTION MODE
        # -----------------------------------------------------

        self.figure.text(
            0.695,
            0.730,
            "Prediction mode",
            fontsize=11.5,
            fontweight="bold",
        )

        self.real_mode_button = self._create_button(
            "Real classifier",
            0.69,
            0.675,
            0.14,
            0.042,
            lambda event:
            self._select_prediction_mode(
                "real"
            ),
        )

        self.simulated_mode_button = self._create_button(
            "Simulated accuracy",
            0.84,
            0.675,
            0.14,
            0.042,
            lambda event:
            self._select_prediction_mode(
                "simulated"
            ),
        )

        # -----------------------------------------------------
        # GRID SIZE
        # -----------------------------------------------------

        self.figure.text(
            0.695,
            0.635,
            "Grid size",
            fontsize=11.5,
            fontweight="bold",
        )

        grid_layout = {
            10: (0.69, 0.585),
            20: (0.84, 0.585),
            30: (0.69, 0.535),
            40: (0.84, 0.535),
        }

        for size, (left, bottom) in grid_layout.items():

            self.grid_buttons[size] = (
                self._create_button(
                    f"{size} × {size}",
                    left,
                    bottom,
                    0.14,
                    0.038,
                    lambda event, selected=size:
                    self._set_grid_size(selected),
                )
            )

        # -----------------------------------------------------
        # SIMULATED ACCURACY
        # -----------------------------------------------------

        self.figure.text(
            0.695,
            0.435,
            "Simulated classifier accuracy",
            fontsize=11.5,
            fontweight="bold",
        )

        accuracy_layout = {
            0.25: (0.69, 0.385),
            0.40: (0.79, 0.385),
            0.50: (0.89, 0.385),

            0.60: (0.69, 0.338),
            0.66: (0.79, 0.338),
            0.70: (0.89, 0.338),

            0.80: (0.69, 0.291),
            0.90: (0.79, 0.291),
            1.00: (0.89, 0.291),
        }

        for accuracy, (left, bottom) in accuracy_layout.items():

            self.accuracy_buttons[accuracy] = (
                self._create_button(
                    f"{accuracy * 100:.0f}%",
                    left,
                    bottom,
                    0.09,
                    0.036,
                    lambda event, selected=accuracy:
                    self._set_accuracy(selected),
                )
            )

        # -----------------------------------------------------
        # RUN / PLAYBACK
        # -----------------------------------------------------

        # -----------------------------------------------------
        # OBSTACLE MODE
        # -----------------------------------------------------

        # -----------------------------------------------------
        # OBSTACLE MODE DROPDOWN
        # -----------------------------------------------------

        self.figure.text(
            0.695,
            0.505,
            "Obstacle",
            fontsize=11.5,
            fontweight="bold",
            va="center",
        )

        self.obstacle_off_button = self._create_button(
            "● OFF",
            0.79,
            0.485,
            0.08,
            0.038,
            lambda event:
            self._set_obstacle_mode(False),
        )

        self.obstacle_on_button = self._create_button(
            "○ ON",
            0.88,
            0.485,
            0.08,
            0.038,
            lambda event:
            self._set_obstacle_mode(True),
        )

        self.run_button = self._create_button(
            "Run demonstration",
            0.69,
            0.225,
            0.29,
            0.045,
            self._run_demo,
        )

        self.pause_button = self._create_button(
            "Pause",
            0.69,
            0.170,
            0.09,
            0.040,
            self._toggle_pause,
        )

        self.replay_button = self._create_button(
            "Replay",
            0.79,
            0.170,
            0.09,
            0.040,
            self._replay,
        )

        self.stop_button = self._create_button(
            "Reset",
            0.89,
            0.170,
            0.09,
            0.040,
            self._reset,
        )

        # Remove Matplotlib's bottom-right cursor coordinates.
        self._disable_coordinate_display()

    def _disable_coordinate_display(self) -> None:
        """Disable Matplotlib cursor coordinate/status text."""

        for axis in self.figure.axes:

            axis.format_coord = (
                lambda x, y: ""
            )

            # Modern Matplotlib may use this method for
            # toolbar status text instead of format_coord.
            if hasattr(
                axis,
                "format_cursor_data",
            ):
                axis.format_cursor_data = (
                    lambda data: ""
                )


    def _close_dropdowns(self) -> None:
        """Close the active dropdown and restore controls."""

        self.dropdown_is_open = False

        # Disconnect the temporary popup click handler.
        click_cid = getattr(
            self,
            "dropdown_click_cid",
            None,
        )

        if click_cid is not None:
            try:
                self.figure.canvas.mpl_disconnect(
                    click_cid
                )
            except Exception:
                pass

        self.dropdown_click_cid = None

        # Remove old per-option axes if any remain from
        # previous dropdown implementations.
        for axis in list(
            getattr(
                self,
                "dropdown_option_axes",
                [],
            )
        ):
            try:
                axis.remove()
            except Exception:
                pass

        self.dropdown_option_axes = []
        self.dropdown_option_buttons = []

        # Remove the single popup axis.
        popup_axis = getattr(
            self,
            "dropdown_background_axis",
            None,
        )

        if popup_axis is not None:
            try:
                popup_axis.remove()
            except Exception:
                pass

        self.dropdown_background_axis = None

        # Restore every control hidden while popup was open.
        for axis in list(
            getattr(
                self,
                "dropdown_hidden_axes",
                [],
            )
        ):
            try:
                axis.set_visible(True)
            except Exception:
                pass

        self.dropdown_hidden_axes = []

        self.figure.canvas.draw_idle()


    def _open_dropdown(
        self,
        options,
        selected_value,
        left,
        top,
        width,
        callback,
        font_size=8.0,
    ) -> None:
        """
        Open a smooth single-column dropdown.

        One popup axis is used instead of creating a separate
        Matplotlib Button for every classifier. This avoids
        sticky behaviour on macOS/TkAgg.
        """

        if self.run_active:
            return

        self._close_dropdowns()

        options = list(options)

        if not options:
            return

        self.dropdown_is_open = True

        number_of_options = len(options)

        option_height = 0.034

        menu_height = (
            number_of_options
            * option_height
        )

        menu_bottom = (
            top - menu_height
        )

        # --------------------------------------------------
        # HIDE ALL OTHER INTERACTIVE CONTROLS
        # --------------------------------------------------

        keep_visible = {
            self.navigation_axis,
            self.information_axis,
        }

        # Keep ONLY the selector that opened this popup visible.
        #
        # Any controls underneath the popup, including the
        # Evaluation selector, prediction-mode buttons, grid
        # buttons, obstacle controls, accuracy buttons and
        # Run/Replay/Reset controls, must disappear while the
        # classifier menu is open.
        if selected_value in METHODS:
            active_control = getattr(
                self,
                "method_dropdown",
                None,
            )
        else:
            active_control = getattr(
                self,
                "evaluation_dropdown",
                None,
            )

        if active_control is not None:
            keep_visible.add(
                active_control.ax
            )

        self.dropdown_hidden_axes = []

        # --------------------------------------------------
        # HIDE ONLY CONTROLS PHYSICALLY UNDER THE POPUP
        # --------------------------------------------------
        #
        # Do not blank the entire control panel.
        #
        # Classifier dropdown:
        #   long popup -> overlapping controls disappear.
        #
        # Evaluation dropdown:
        #   short popup -> only nearby overlapping controls
        #   disappear.
        # --------------------------------------------------

        menu_left = left
        menu_right = left + width
        menu_top = top
        menu_bottom_edge = menu_bottom

        for axis in list(
            self.figure.axes
        ):

            if axis in keep_visible:
                continue

            if not axis.get_visible():
                continue

            position = axis.get_position()

            horizontal_overlap = (
                position.x1 > menu_left
                and
                position.x0 < menu_right
            )

            vertical_overlap = (
                position.y1 > menu_bottom_edge
                and
                position.y0 < menu_top
            )

            if (
                horizontal_overlap
                and vertical_overlap
            ):
                axis.set_visible(
                    False
                )

                self.dropdown_hidden_axes.append(
                    axis
                )

        # --------------------------------------------------
        # ONE POPUP AXIS
        # --------------------------------------------------

        popup = self.figure.add_axes(
            [
                left,
                menu_bottom,
                width,
                menu_height,
            ]
        )

        popup.set_zorder(10000)
        popup.set_facecolor("0.96")

        popup.set_xlim(
            0,
            1,
        )

        # Row 0 at top.
        popup.set_ylim(
            number_of_options,
            0,
        )

        popup.set_xticks([])
        popup.set_yticks([])

        popup.format_coord = (
            lambda x, y: ""
        )

        # Border.
        for spine in popup.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

        # --------------------------------------------------
        # DRAW EACH OPTION AS TEXT
        # --------------------------------------------------

        for index, option in enumerate(
            options
        ):

            selected = (
                option == selected_value
            )

            if selected:
                popup.axhspan(
                    index,
                    index + 1,
                    facecolor="0.72",
                    edgecolor="none",
                    zorder=0,
                )

            popup.text(
                0.5,
                index + 0.5,
                str(option),
                ha="center",
                va="center",
                fontsize=font_size,
                transform=popup.transData,
                zorder=2,
            )

            # Row separator.
            popup.plot(
                [0, 1],
                [index + 1, index + 1],
                linewidth=0.7,
                color="0.35",
                zorder=1,
            )

        self.dropdown_background_axis = popup

        # --------------------------------------------------
        # ONE CLICK HANDLER FOR THE WHOLE LIST
        # --------------------------------------------------

        def dropdown_click(event):

            # Click somewhere outside popup:
            # simply close it.
            if event.inaxes is not popup:
                self._close_dropdowns()
                return

            if event.ydata is None:
                return

            index = int(
                event.ydata
            )

            if not (
                0 <= index < number_of_options
            ):
                return

            selected_option = (
                options[index]
            )

            # IMPORTANT:
            # This runs on mouse PRESS, not on a Matplotlib
            # Button mouse-release callback. Therefore the
            # underlying Run button can never receive the same
            # click.
            callback(
                selected_option
            )

        self.dropdown_click_cid = (
            self.figure.canvas.mpl_connect(
                "button_press_event",
                dropdown_click,
            )
        )

        self.figure.canvas.draw_idle()


    def _toggle_method_dropdown(
        self,
        event,
    ) -> None:
        """Open classifier-method dropdown."""

        if self.run_active:
            return

        self._open_dropdown(
            options=METHODS,
            selected_value=self.method,
            left=0.69,
            top=0.865,
            width=0.29,
            callback=self._select_method_dropdown,
            font_size=7.5,
        )


    def _select_method_dropdown(
        self,
        method,
    ) -> None:
        """
        Select classifier only.

        Navigation never starts from classifier selection.
        """

        if self.run_active:
            return

        if method not in METHODS:
            return

        # Stop and clear any previous simulation state.
        self.run_active = False
        self.paused = False

        if self.timer is not None:
            try:
                self.timer.stop()
            except Exception:
                pass

        self.timer = None

        self.result = None
        self.current_frame = 0
        self.baseline_classifier_path = None

        # Reset temporary obstacle animation state.
        self.dynamic_obstacle_position = None
        self.dynamic_obstacle_path = []
        self.dynamic_obstacle_time = 0

        self.moving_obstacle_position = None
        self.obstacle_crossing_done = False
        self.obstacle_intercept_arrived = False
        self.obstacle_safety_pause_done = False
        self.obstacle_phase = "approach"

        # Change classifier.
        self.method = method
        self.mode = "real"

        # Keep a brief Run guard as an additional safety layer.
        self.run_click_block_until = (
            time.monotonic() + 0.20
        )

        self.method_dropdown.label.set_text(
            f"{self.method}    ▼"
        )

        if hasattr(
            self,
            "pause_button",
        ):
            self.pause_button.label.set_text(
                "Pause"
            )

        self._update_button_styles()

        # Close immediately.
        # This is safe because selection occurred on mouse PRESS.
        self._close_dropdowns()

        # Return to clean screen.
        self._show_welcome_screen()

        self.figure.canvas.draw_idle()


    def _toggle_evaluation_dropdown(
        self,
        event,
    ) -> None:
        """Open evaluation dropdown."""
        if self.run_active:
            return


        self._open_dropdown(
            options=(
                "Cross-Session",
                "Cross-Subject",
            ),
            selected_value=self.evaluation,
            left=0.69,
            top=0.770,
            width=0.29,
            callback=self._select_evaluation_dropdown,
            font_size=8.5,
        )


    def _select_evaluation_dropdown(
        self,
        evaluation,
    ) -> None:
        """Select evaluation protocol."""

        self._close_dropdowns()

        self._set_evaluation(
            evaluation
        )

        self.evaluation_dropdown.label.set_text(
            f"{self.evaluation}    ▼"
        )

        self.figure.canvas.draw_idle()


    def _select_prediction_mode(
        self,
        mode: str,
    ) -> None:
        """Switch prediction mode immediately."""
        if self.run_active:
            return


        self._close_dropdowns()

        self._set_mode(
            mode
        )

        self._update_button_styles()


    def _toggle_mode_dropdown(
        self,
        event,
    ) -> None:
        """Open prediction-mode dropdown."""

        selected = (
            "Real classifier"
            if self.mode == "real"
            else "Simulated accuracy"
        )

        self._open_dropdown(
            options=(
                "Real classifier",
                "Simulated accuracy",
            ),
            selected_value=selected,
            left=0.69,
            top=0.675,
            width=0.29,
            callback=self._select_mode_dropdown,
            font_size=8.5,
        )


    def _select_mode_dropdown(
        self,
        selected,
    ) -> None:
        """Select prediction mode."""

        self._close_dropdowns()

        if selected == "Real classifier":

            self._set_mode(
                "real"
            )

        else:

            self._set_mode(
                "simulated"
            )

        self.figure.canvas.draw_idle()


    def _set_mode(
        self,
        mode: str,
    ) -> None:
        """Select real or simulated prediction mode."""
        if self.run_active:
            return


        if mode not in {
            "real",
            "simulated",
        }:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        self.mode = mode

        self._update_button_styles()


    def _set_mode_from_label(
        self,
        label: str,
    ) -> None:
        """Handle prediction-mode radio selection."""

        if label == "Real classifier":
            self._set_mode("real")
        else:
            self._set_mode(
                "simulated"
            )


    def _set_method(
        self,
        method: str,
    ) -> None:
        """Select one trained EEG classifier."""
        if self.run_active:
            return


        if method not in METHODS:
            raise ValueError(
                f"Unknown method: {method}"
            )

        self.method = method
        self.mode = "real"

        if (
            hasattr(
                self,
                "mode_radio",
            )
            and self.mode_radio.value_selected
            != "Real classifier"
        ):
            self.mode_radio.set_active(
                0
            )

        self._update_button_styles()


    def _set_evaluation(
        self,
        evaluation: str,
    ) -> None:
        """Select cross-session or cross-subject predictions."""
        if self.run_active:
            return


        if evaluation not in {
            "Cross-Session",
            "Cross-Subject",
        }:
            raise ValueError(
                f"Unknown evaluation: "
                f"{evaluation}"
            )

        self.evaluation = evaluation
        self.mode = "real"

        if (
            hasattr(
                self,
                "mode_radio",
            )
            and self.mode_radio.value_selected
            != "Real classifier"
        ):
            self.mode_radio.set_active(
                0
            )

        self._update_button_styles()


    def _set_grid_size(
        self,
        grid_size: int,
    ) -> None:
        """Select a grid size."""
        if self.run_active:
            return


        self.grid_size = grid_size
        self._update_button_styles()

    def _set_accuracy(
        self,
        accuracy: float,
    ) -> None:
        """Select simulated accuracy and activate simulated mode."""
        if self.run_active:
            return


        self.simulated_accuracy = (
            accuracy
        )

        self.mode = "simulated"

        if (
            hasattr(
                self,
                "mode_radio",
            )
            and self.mode_radio.value_selected
            != "Simulated accuracy"
        ):
            self.mode_radio.set_active(
                1
            )

        self._update_button_styles()


    def _update_button_styles(self) -> None:
        """Apply deterministic selected/unselected button colours."""

        normal_colour = "0.90"
        selected_colour = "0.62"

        # --------------------------------------------------
        # GRID SIZE
        # --------------------------------------------------

        for size, button in self.grid_buttons.items():

            selected = (
                size == self.grid_size
            )

            colour = (
                selected_colour
                if selected
                else normal_colour
            )

            button.ax.set_facecolor(
                colour
            )

            button.color = colour

        # --------------------------------------------------
        # SIMULATED ACCURACY
        # --------------------------------------------------

        for accuracy, button in self.accuracy_buttons.items():

            selected = (
                self.mode == "simulated"
                and abs(
                    accuracy
                    - self.simulated_accuracy
                ) < 1e-9
            )

            colour = (
                selected_colour
                if selected
                else normal_colour
            )

            button.ax.set_facecolor(
                colour
            )

            button.color = colour

        # --------------------------------------------------
        # PREDICTION MODE
        # --------------------------------------------------

        if hasattr(
            self,
            "real_mode_button",
        ):

            colour = (
                selected_colour
                if self.mode == "real"
                else normal_colour
            )

            self.real_mode_button.ax.set_facecolor(
                colour
            )

            self.real_mode_button.color = (
                colour
            )

        if hasattr(
            self,
            "simulated_mode_button",
        ):

            colour = (
                selected_colour
                if self.mode == "simulated"
                else normal_colour
            )

            self.simulated_mode_button.ax.set_facecolor(
                colour
            )

            self.simulated_mode_button.color = (
                colour
            )

        self.figure.canvas.draw_idle()


    def _mode_title(self) -> str:
        """Return readable prediction mode."""

        if self.mode == "real":
            return "Real EEG classifier"

        return "Simulated classifier"


    def _mode_details(self) -> str:
        """Describe the current prediction source."""

        if self.mode == "simulated":

            return (
                "Artificial predictions | "
                f"accuracy="
                f"{self.simulated_accuracy * 100:.0f}%"
            )

        accuracy, kappa = (
            FINAL_RESULTS[
                self.evaluation
            ][
                self.method
            ]
        )

        return (
            f"{self.method} | "
            f"{self.evaluation} | "
            f"Acc={accuracy:.2f}% | "
            f"κ={kappa:.3f}"
        )


    def _show_current_selection(self) -> None:
        """Show current settings below the controls."""

        self.information_axis.clear()
        self.information_axis.axis(
            "off"
        )

        if self.mode == "real":

            accuracy, kappa = (
                FINAL_RESULTS[
                    self.evaluation
                ][
                    self.method
                ]
            )

            information = (
                "CURRENT SELECTION\n"
                f"Method: {self.method}\n"
                f"Evaluation: {self.evaluation}\n"
                "Mode: Real classifier\n"
                f"Accuracy: {accuracy:.2f}%\n"
                f"Kappa: {kappa:.3f}\n"
                f"Grid: {self.grid_size} × "
                f"{self.grid_size}"
            )

        else:

            information = (
                "CURRENT SELECTION\n"
                "Mode: Simulated accuracy\n"
                f"Accuracy: "
                f"{self.simulated_accuracy * 100:.0f}%\n"
                f"Grid: {self.grid_size} × "
                f"{self.grid_size}"
            )

        self.information_axis.text(
            0.0,
            1.0,
            information,
            va="top",
            ha="left",
            fontsize=9.0,
            linespacing=1.18,
            transform=(
                self.information_axis.transAxes
            ),
        )

        self.figure.canvas.draw_idle()


    def _show_welcome_screen(self) -> None:
        """Show initial instructions."""

        self.navigation_axis.clear()
        self.navigation_axis.axis("off")

        # Keep the information area empty before a run starts.
        self.information_axis.clear()
        self.information_axis.axis("off")

        self.navigation_axis.text(
            0.5,
            0.68,
            "BCI-Controlled Wheelchair",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            transform=self.navigation_axis.transAxes,
        )

        welcome_text = (
            "Choose a classifier and evaluation protocol\n"
            "for real EEG predictions, or select a simulated\n"
            "classifier accuracy for sensitivity analysis."
        )

        self.navigation_axis.text(
            0.5,
            0.48,
            welcome_text,
            ha="center",
            va="center",
            fontsize=13.5,
            linespacing=1.45,
            transform=self.navigation_axis.transAxes,
        )

        self.navigation_axis.text(
            0.5,
            0.28,
            "Select the grid size and press Run demonstration.",
            ha="center",
            va="center",
            fontsize=12.5,
            transform=self.navigation_axis.transAxes,
        )

        self.figure.canvas.draw_idle()


    def _build_sampler(self) -> PredictionSampler:
        """Build the selected real or simulated prediction sampler."""

        if self.mode == "simulated":

            return SimulatedAccuracySampler(
                accuracy=(
                    self.simulated_accuracy
                ),
                random_seed=RANDOM_SEED,
            )

        prediction_path = (
            PREDICTION_FILES[
                self.evaluation
            ][
                self.method
            ]
        )

        if not prediction_path.exists():
            raise FileNotFoundError(
                "Prediction file is missing:\n"
                f"{prediction_path.resolve()}"
            )

        print()
        print("=" * 72)
        print("REAL CLASSIFIER NAVIGATION")
        print("=" * 72)

        print(
            f"Method:     "
            f"{self.method}"
        )

        print(
            f"Evaluation: "
            f"{self.evaluation}"
        )

        accuracy, kappa = (
            FINAL_RESULTS[
                self.evaluation
            ][
                self.method
            ]
        )

        print(
            f"Accuracy:   "
            f"{accuracy:.2f}%"
        )

        print(
            f"Kappa:      "
            f"{kappa:.3f}"
        )

        print(
            f"Predictions:"
            f" {prediction_path}"
        )

        return EEGPredictionSampler(
            prediction_path,
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

    def _find_dynamic_route(
        self,
        start,
        target,
        blocked_position,
    ):
        """
        Find a safe grid route around the current moving obstacle.

        Breadth-first search is used because every grid movement
        has equal cost.

        The moving obstacle's CURRENT cell is treated as blocked.
        """

        if start == target:
            return [start]

        blocked = set()

        if blocked_position is not None:
            blocked.add(
                blocked_position
            )

        queue = deque(
            [start]
        )

        parents = {
            start: None
        }

        # Four-directional wheelchair movement.
        directions = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        )

        while queue:

            current = queue.popleft()

            if current == target:
                break

            for dr, dc in directions:

                neighbour = (
                    current[0] + dr,
                    current[1] + dc,
                )

                row, col = neighbour

                if not (
                    0 <= row < self.grid_size
                    and
                    0 <= col < self.grid_size
                ):
                    continue

                if neighbour in blocked:
                    continue

                if neighbour in parents:
                    continue

                parents[
                    neighbour
                ] = current

                queue.append(
                    neighbour
                )

        if target not in parents:
            return None

        route = []

        current = target

        while current is not None:

            route.append(
                current
            )

            current = parents[
                current
            ]

        route.reverse()

        return route


    def _reroute_around_dynamic_obstacle(
        self,
    ) -> bool:
        """
        Replace the remaining wheelchair path with a safe route
        around the moving obstacle.

        Returns True when a new route was found.
        """

        if self.result is None:
            return False

        if (
            self.dynamic_obstacle_position
            is None
        ):
            return False

        current_position = (
            self.result.path[
                self.current_frame
            ]
        )

        target = self.result.target

        new_route = self._find_dynamic_route(
            start=current_position,
            target=target,
            blocked_position=(
                self.dynamic_obstacle_position
            ),
        )

        if (
            new_route is None
            or len(new_route) < 2
        ):
            return False

        # Keep everything already travelled.
        travelled_path = (
            self.result.path[
                :self.current_frame + 1
            ]
        )

        # new_route[0] is the current position,
        # therefore avoid adding it twice.
        self.result.path = (
            travelled_path
            + new_route[1:]
        )

        print(
            "DYNAMIC OBSTACLE DETECTED:",
            self.dynamic_obstacle_position,
        )

        print(
            "AI SAFETY CONTROLLER:",
            "route replanned",
        )

        print(
            "NEW REMAINING ROUTE:",
            len(new_route) - 1,
            "steps",
        )

        return True


    def _prepare_dynamic_obstacle(self) -> None:
        """
        Create one simulated moving pedestrian.

        The pedestrian is deliberately positioned to cross
        the wheelchair's planned route during the run.
        """

        self.dynamic_obstacle_position = None
        self.dynamic_obstacle_path = []
        self.dynamic_obstacle_time = 0

        if (
            self.result is None
            or len(self.result.path) < 10
        ):
            return

        last_frame = len(
            self.result.path
        ) - 1

        # Cross approximately halfway through the journey.
        crossing_frame = max(
            4,
            min(
                last_frame - 4,
                int(last_frame * 0.55),
            ),
        )

        self.dynamic_obstacle_crossing_frame = (
            crossing_frame
        )

        crossing_position = (
            self.result.path[
                crossing_frame
            ]
        )

        crossing_row = (
            crossing_position[0]
        )

        crossing_col = (
            crossing_position[1]
        )

        # Work out the direction of the wheelchair path
        # around the crossing point.
        previous_position = (
            self.result.path[
                max(
                    0,
                    crossing_frame - 2,
                )
            ]
        )

        next_position = (
            self.result.path[
                min(
                    last_frame,
                    crossing_frame + 2,
                )
            ]
        )

        row_change = abs(
            next_position[0]
            - previous_position[0]
        )

        col_change = abs(
            next_position[1]
            - previous_position[1]
        )

        obstacle_path = []

        # --------------------------------------------------
        # If wheelchair is mainly travelling horizontally,
        # pedestrian crosses vertically.
        # --------------------------------------------------

        if col_change >= row_change:

            for offset in range(
                -5,
                6,
            ):

                row = (
                    crossing_row
                    + offset
                )

                col = crossing_col

                if (
                    0 <= row < self.grid_size
                    and
                    0 <= col < self.grid_size
                ):
                    obstacle_path.append(
                        (row, col)
                    )

        # --------------------------------------------------
        # Otherwise pedestrian crosses horizontally.
        # --------------------------------------------------

        else:

            for offset in range(
                -5,
                6,
            ):

                row = crossing_row

                col = (
                    crossing_col
                    + offset
                )

                if (
                    0 <= row < self.grid_size
                    and
                    0 <= col < self.grid_size
                ):
                    obstacle_path.append(
                        (row, col)
                    )

        if not obstacle_path:
            return

        self.dynamic_obstacle_path = (
            obstacle_path
        )

        # Arrange timing so that the middle of the
        # pedestrian path reaches the wheelchair path
        # at approximately the crossing frame.
        middle_index = (
            len(obstacle_path) // 2
        )

        self.dynamic_obstacle_start_time = max(
            0,
            crossing_frame
            - middle_index,
        )


    def _update_dynamic_obstacle(self) -> None:
        """Move the simulated pedestrian by one time step."""

        self.dynamic_obstacle_position = None

        if not self.dynamic_obstacle_path:
            return

        path_index = (
            self.dynamic_obstacle_time
            - self.dynamic_obstacle_start_time
        )

        if (
            0
            <= path_index
            < len(
                self.dynamic_obstacle_path
            )
        ):

            self.dynamic_obstacle_position = (
                self.dynamic_obstacle_path[
                    path_index
                ]
            )


    def _set_obstacle_mode(
        self,
        enabled: bool,
    ) -> None:
        """Enable or disable moving obstacle mode."""

        if getattr(self, "run_active", False):
            return

        self.obstacle_enabled = enabled

        if enabled:

            self._prepare_moving_obstacle()

            # Show obstacle before run.
            self._draw_obstacle_preview()

        else:

            self.static_obstacle_position = None
            self.moving_obstacle_position = None
            self.moving_obstacle_path = []

            self._show_welcome_screen()

        self._update_obstacle_radio_style()

        self.figure.canvas.draw_idle()


    def _update_obstacle_radio_style(
        self,
    ) -> None:
        """Update obstacle OFF/ON radio appearance."""

        if not hasattr(
            self,
            "obstacle_off_button",
        ):
            return

        selected_colour = "0.62"
        normal_colour = "0.90"

        if self.obstacle_enabled:

            self.obstacle_off_button.label.set_text(
                "○ OFF"
            )

            self.obstacle_on_button.label.set_text(
                "● ON"
            )

            self.obstacle_off_button.ax.set_facecolor(
                normal_colour
            )

            self.obstacle_on_button.ax.set_facecolor(
                selected_colour
            )

            self.obstacle_off_button.color = normal_colour
            self.obstacle_on_button.color = selected_colour

        else:

            self.obstacle_off_button.label.set_text(
                "● OFF"
            )

            self.obstacle_on_button.label.set_text(
                "○ ON"
            )

            self.obstacle_off_button.ax.set_facecolor(
                selected_colour
            )

            self.obstacle_on_button.ax.set_facecolor(
                normal_colour
            )

            self.obstacle_off_button.color = selected_colour
            self.obstacle_on_button.color = normal_colour

        self.figure.canvas.draw_idle()


    def _draw_obstacle_preview(self) -> None:
        """Draw the selected grid with the static obstacle."""

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

        start = (
            self.grid_size - 2,
            1,
        )

        target = (
            1,
            self.grid_size - 2,
        )

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

        if self.moving_obstacle_position is not None:

            row, col = (
                self.moving_obstacle_position
            )

            axis.scatter(
                col,
                row,
                marker="X",
                s=300,
                color="red",
                linewidths=2.2,
                label="Obstacle",
                zorder=10,
            )

        axis.set_title(
            f"{self.grid_size} × {self.grid_size} grid",
            fontsize=16,
            pad=12,
        )

        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=3,
        )

        self.figure.canvas.draw_idle()


    def _prepare_guaranteed_crossing(self) -> None:
        """
        Choose one FUTURE cell from the ORIGINAL classifier path.

        IMPORTANT:
        This function does NOT change the wheelchair trajectory.

        X moves independently toward that future path cell and
        stops there before the wheelchair arrives.
        """

        self.obstacle_phase = "approach"
        self.obstacle_crossing_done = False
        self.obstacle_intercept_arrived = False
        self.obstacle_safety_pause_done = False
        self.obstacle_hold_counter = 0

        self.obstacle_interception_cell = None
        self.obstacle_intercept_cell = None
        self.obstacle_intercept_frame = None

        if not self.obstacle_enabled:
            return

        if not self.baseline_classifier_path:
            return

        # COPY ONLY.
        # Never rewrite baseline_classifier_path.
        original_path = list(
            self.baseline_classifier_path
        )

        if len(original_path) < 15:
            return

        # ----------------------------------------------------
        # CHOOSE A FUTURE POINT NEAR THE TARGET
        # ----------------------------------------------------
        #
        # X should ultimately stop on the genuine classifier
        # trajectory approximately 4-5 GRID CELLS before the
        # final target.
        #
        # We therefore search backward through the original
        # classifier path and select a cell whose Manhattan
        # distance from the target is around 5 cells.
        # ----------------------------------------------------

        target_position = original_path[-1]

        intercept_index = None

        # Place X farther from the final target so the
        # obstacle interaction occurs around the middle of the
        # final approach rather than immediately beside the star.
        #
        # Aim for roughly 8-10 grid cells from the target.
        preferred_distance = 9

        for index in range(
            len(original_path) - 2,
            7,
            -1,
        ):
            cell = original_path[index]

            distance_to_target = (
                abs(
                    cell[0]
                    - target_position[0]
                )
                +
                abs(
                    cell[1]
                    - target_position[1]
                )
            )

            if (
                12
                <= distance_to_target
                <= 14
            ):
                intercept_index = index
                break

        # Defensive fallback for unusual classifier paths.
        if intercept_index is None:
            intercept_index = max(
                8,
                len(original_path) - 8,
            )

        # Prefer an actual movement cell.
        for index in range(
            intercept_index,
            min(
                len(original_path) - 2,
                intercept_index + 12,
            ),
        ):

            previous = original_path[
                index - 1
            ]

            current = original_path[
                index
            ]

            following = original_path[
                index + 1
            ]

            if (
                current != previous
                and
                current != following
            ):
                intercept_index = index
                break

        intercept_cell = original_path[
            intercept_index
        ]

        self.obstacle_interception_cell = (
            intercept_cell
        )

        self.obstacle_intercept_cell = (
            intercept_cell
        )

        self.obstacle_intercept_frame = (
            intercept_index
        )

        # ----------------------------------------------------
        # START X AWAY FROM THAT CELL
        # ----------------------------------------------------

        row, col = intercept_cell

        # Scale X's starting distance with the grid.
        #
        # It must be far enough to visibly move toward the
        # classifier path, but close enough to arrive before
        # the wheelchair.
        if self.grid_size <= 10:
            travel_distance = 2

        else:
            # Keep X close enough to reach the future
            # classifier path well before the wheelchair.
            travel_distance = 5

        candidates = [
            (
                max(
                    1,
                    row - travel_distance,
                ),
                col,
            ),
            (
                min(
                    self.grid_size - 2,
                    row + travel_distance,
                ),
                col,
            ),
            (
                row,
                max(
                    1,
                    col - travel_distance,
                ),
            ),
            (
                row,
                min(
                    self.grid_size - 2,
                    col + travel_distance,
                ),
            ),
        ]

        candidates = [
            position
            for position in candidates
            if position != intercept_cell
        ]

        # --------------------------------------------------
        # OBSTACLE START POSITION
        # --------------------------------------------------
        #
        # X begins from the wheelchair's ORIGINAL blue Start
        # cell and slowly travels toward the selected future
        # classifier-path interception cell.
        #
        # The wheelchair will already be moving away from the
        # Start while X follows behind toward the future path.
        # --------------------------------------------------

        start_position = tuple(
            self.baseline_classifier_path[0]
        )

        start_row, start_col = start_position

        # X must start CLOSE TO the wheelchair Start,
        # but never on the exact same cell.
        possible_starts = [
            (start_row, start_col + 1),
            (start_row - 1, start_col),
            (start_row, start_col - 1),
            (start_row + 1, start_col),
        ]

        valid_starts = [
            cell
            for cell in possible_starts
            if (
                1 <= cell[0] <= self.grid_size - 2
                and
                1 <= cell[1] <= self.grid_size - 2
                and
                cell != start_position
            )
        ]

        if valid_starts:
            self.moving_obstacle_position = valid_starts[0]
        else:
            self.moving_obstacle_position = (
                max(
                    1,
                    min(
                        self.grid_size - 2,
                        start_row - 1,
                    ),
                ),
                start_col,
            )

        # X must actively travel toward the selected future
        # classifier-path cell.
        self.obstacle_phase = "approach"
        self.obstacle_frame_counter = 0
        self.obstacle_axis_counter = 0
        self.obstacle_intercept_arrived = False
        self.obstacle_crossing_done = False

        print()
        print("=" * 70)
        print("ORIGINAL CLASSIFIER PATH PRESERVED")
        print("=" * 70)
        print(
            "Future path cell selected:",
            intercept_cell,
        )
        print(
            "Original path index:",
            intercept_index,
        )
        print(
            "X starts near wheelchair Start:",
            self.moving_obstacle_position,
        )
        print(
            "X will move slowly toward:",
            intercept_cell,
        )
        print(
            "X will stop permanently once it reaches "
            "the classifier path."
        )
        print(
            "Wheelchair path was NOT modified."
        )
        print("=" * 70)


    def _prepare_moving_obstacle(self) -> None:
        """Place wandering obstacle near the centre of the grid."""

        self.moving_obstacle_path = []
        self.moving_obstacle_index = 0
        self.moving_obstacle_direction = 1
        self.obstacle_frame_counter = 0

        if not self.obstacle_enabled:
            self.moving_obstacle_position = None
            return

        row = self.grid_size // 2
        col = self.grid_size // 2

        row = max(
            2,
            min(
                self.grid_size - 3,
                row,
            ),
        )

        col = max(
            2,
            min(
                self.grid_size - 3,
                col,
            ),
        )

        self.moving_obstacle_position = (
            row,
            col,
        )


    def _move_obstacle_once(self) -> None:
        """
        Move the obstacle through three genuine phases:

        approach:
            X moves one grid cell at a time toward the
            selected future classifier path cell.

        blocked:
            X remains stationary on that path cell.

        released:
            after wheelchair avoidance, X wanders freely.
        """

        if not self.obstacle_enabled:
            self.moving_obstacle_position = None
            return

        if self.moving_obstacle_position is None:
            return

        current_row, current_col = (
            self.moving_obstacle_position
        )

        # ====================================================
        # APPROACH
        # ====================================================

        if self.obstacle_phase == "approach":

            if self.obstacle_interception_cell is None:
                return

            target_row, target_col = (
                self.obstacle_interception_cell
            )

            if (
                self.moving_obstacle_position
                == self.obstacle_interception_cell
            ):
                self.obstacle_phase = "blocked"
                self.obstacle_intercept_arrived = True

                print()
                print("=" * 70)
                print(
                    ">>> X REACHED FUTURE WHEELCHAIR PATH <<<"
                )
                print(
                    "X permanently stopped at:",
                    self.moving_obstacle_position,
                )
                print("=" * 70)

                return

            row_difference = (
                target_row - current_row
            )

            col_difference = (
                target_col - current_col
            )

            # ------------------------------------------------
            # CONTROLLED ZIG-ZAG MOTION
            # ------------------------------------------------
            #
            # Most movements progress toward the interception
            # cell, but every few moves X takes a safe
            # perpendicular step. The next movements naturally
            # correct this offset.
            #
            # This creates a visible pedestrian-like zig-zag
            # rather than a perfectly straight trajectory.
            # ------------------------------------------------

            move_number = self.obstacle_axis_counter

            possible_moves = []

            # Normal progress toward target.
            if row_difference != 0:
                possible_moves.append(
                    (
                        current_row
                        + (
                            1
                            if row_difference > 0
                            else -1
                        ),
                        current_col,
                    )
                )

            if col_difference != 0:
                possible_moves.append(
                    (
                        current_row,
                        current_col
                        + (
                            1
                            if col_difference > 0
                            else -1
                        ),
                    )
                )

            # Every third movement, deliberately add a
            # perpendicular side-step if there is enough
            # remaining distance to safely recover.
            if (
                move_number % 3 == 2
                and
                (
                    abs(row_difference)
                    + abs(col_difference)
                ) > 5
            ):

                side_candidates = []

                # If travelling mainly vertically,
                # weave left/right.
                if (
                    abs(row_difference)
                    >= abs(col_difference)
                ):

                    direction = (
                        1
                        if (
                            (move_number // 3) % 2 == 0
                        )
                        else -1
                    )

                    side_candidates.append(
                        (
                            current_row,
                            current_col + direction,
                        )
                    )

                # If travelling mainly horizontally,
                # weave up/down.
                else:

                    direction = (
                        1
                        if (
                            (move_number // 3) % 2 == 0
                        )
                        else -1
                    )

                    side_candidates.append(
                        (
                            current_row + direction,
                            current_col,
                        )
                    )

                valid_side_moves = [
                    cell
                    for cell in side_candidates
                    if (
                        1
                        <= cell[0]
                        <= self.grid_size - 2
                        and
                        1
                        <= cell[1]
                        <= self.grid_size - 2
                    )
                ]

                if valid_side_moves:
                    next_position = valid_side_moves[0]
                elif possible_moves:
                    next_position = possible_moves[0]
                else:
                    next_position = (
                        current_row,
                        current_col,
                    )

            else:

                if len(possible_moves) >= 2:

                    # Alternate row/column progress.
                    next_position = possible_moves[
                        move_number % 2
                    ]

                elif possible_moves:

                    next_position = possible_moves[0]

                else:

                    next_position = (
                        current_row,
                        current_col,
                    )

            self.obstacle_axis_counter += 1

            self.moving_obstacle_position = (
                next_position
            )

            print(
                "X zig-zag moving:",
                self.moving_obstacle_position,
                "->",
                self.obstacle_interception_cell,
            )

            # ------------------------------------------------
            # ARRIVED
            # ------------------------------------------------

            if (
                self.moving_obstacle_position
                == self.obstacle_interception_cell
            ):

                self.obstacle_phase = "blocked"
                self.obstacle_intercept_arrived = True

                print()
                print("=" * 70)
                print(
                    ">>> X ARRIVED BEFORE WHEELCHAIR <<<"
                )
                print(
                    "X permanently blocks:",
                    self.moving_obstacle_position,
                )
                print(
                    "Approximately 12-14 cells "
                    "before target."
                )
                print("=" * 70)

            return

        # ====================================================
        # BLOCKED
        # ====================================================

        if self.obstacle_phase in (
            "blocked",
            "detouring",
            "stationary",
        ):

            # X intentionally remains stationary.
            #
            # blocked:
            #   X has reached the classifier path and waits
            #   for the wheelchair.
            #
            # detouring:
            #   wheelchair is currently navigating around X.
            #
            # stationary:
            #   wheelchair has safely passed X, but the
            #   obstacle remains permanently in the same cell.
            #
            # X NEVER resumes wandering.
            return

        # ====================================================
        # NO RELEASED / WANDERING PHASE
        # ====================================================
        #
        # Once X reaches the classifier path, it remains there
        # for the rest of the demonstration.
        #
        # This gives a clear persistent obstacle which forces
        # the wheelchair safety policy to perform a genuine
        # local avoidance manoeuvre.
        # ====================================================

        return


    def _moving_obstacle_is_near_route(
        self,
        lookahead: int = 3,
    ) -> bool:
        """
        Detect an upcoming stationary moving-obstacle conflict.

        40x40:
            Wait until the wheelchair is exactly one movement
            before X. This avoids premature large-grid rerouting.

        Other grids:
            Preserve the existing look-ahead behaviour.
        """

        if self.result is None:
            return False

        if self.moving_obstacle_position is None:
            return False

        if self.obstacle_phase != "blocked":
            return False

        next_index = (
            self.current_frame + 1
        )

        if next_index >= len(
            self.result.path
        ):
            return False

        obstacle = (
            self.moving_obstacle_position
        )

        # ====================================================
        # 40x40 SPECIAL CASE
        # ====================================================
        #
        # Do NOT react while X is merely nearby.
        #
        # Continue genuine classifier playback until the very
        # next wheelchair path cell is occupied by X.
        #
        # At that point the wheelchair is exactly one movement
        # before the obstacle.
        # ====================================================

        if self.grid_size >= 40:

            next_position = (
                self.result.path[
                    next_index
                ]
            )

            if next_position == obstacle:

                print()
                print("=" * 70)
                print(
                    ">>> 40x40: WHEELCHAIR ONE STEP BEFORE X <<<"
                )
                print(
                    "Wheelchair:",
                    self.result.path[
                        self.current_frame
                    ],
                )
                print(
                    "Next classifier cell:",
                    next_position,
                )
                print(
                    "X:",
                    obstacle,
                )
                print(
                    "Stopping before collision."
                )
                print("=" * 70)

                return True

            return False

        # ====================================================
        # EXISTING BEHAVIOUR FOR 10 / 20 / 30
        # ====================================================

        start_index = (
            self.current_frame + 1
        )

        if self.grid_size <= 10:
            effective_lookahead = 2
        else:
            effective_lookahead = 3

        end_index = min(
            len(self.result.path),
            start_index
            + effective_lookahead,
        )

        obstacle_row, obstacle_col = (
            obstacle
        )

        for future_cell in (
            self.result.path[
                start_index:end_index
            ]
        ):

            row, col = future_cell

            distance = (
                abs(
                    row
                    - obstacle_row
                )
                +
                abs(
                    col
                    - obstacle_col
                )
            )

            if distance <= 1:

                print()
                print(
                    ">>> WHEELCHAIR APPROACHING "
                    "STATIONARY X <<<"
                )

                return True

        return False


    def _reroute_40x40_obstacle(
        self,
    ) -> bool:
        """
        Simple reliable 40x40 safety bypass.

        The classifier path is untouched until X occupies the
        next classifier cell.

        Then the safety controller:
        - stops before X,
        - moves three cells perpendicular to the route,
        - travels past X,
        - rejoins the original classifier path,
        - keeps X stationary until three additional wheelchair
          movements have completed.
        """

        if self.result is None:
            return False

        if self.moving_obstacle_position is None:
            return False

        if not self.baseline_classifier_path:
            return False

        current = self.result.path[
            self.current_frame
        ]

        obstacle = (
            self.moving_obstacle_position
        )

        baseline = (
            self.baseline_classifier_path
        )

        cr, cc = current
        OR, OC = obstacle

        # ----------------------------------------------------
        # USE EXACT INTERCEPTION INDEX
        # ----------------------------------------------------

        obstacle_index = getattr(
            self,
            "obstacle_intercept_frame",
            None,
        )

        if (
            obstacle_index is None
            or
            obstacle_index >= len(baseline)
            or
            baseline[obstacle_index] != obstacle
        ):

            occurrences = [
                i
                for i in range(len(baseline))
                if (
                    baseline[i] == obstacle
                    and
                    i > self.current_frame
                )
            ]

            if not occurrences:
                return False

            obstacle_index = occurrences[0]

        # ----------------------------------------------------
        # FIND A REAL CELL AFTER X
        # ----------------------------------------------------

        rejoin_index = None

        for i in range(
            obstacle_index + 1,
            len(baseline),
        ):

            cell = baseline[i]

            distance = (
                abs(cell[0] - OR)
                +
                abs(cell[1] - OC)
            )

            if (
                cell != obstacle
                and
                distance >= 4
            ):
                rejoin_index = i
                break

        if rejoin_index is None:

            for i in range(
                obstacle_index + 1,
                len(baseline),
            ):

                if baseline[i] != obstacle:
                    rejoin_index = i
                    break

        if rejoin_index is None:
            return False

        rejoin = baseline[
            rejoin_index
        ]

        rr, rc = rejoin

        # ----------------------------------------------------
        # DETERMINE DIRECTION INTO X
        # ----------------------------------------------------

        dr = OR - cr
        dc = OC - cc

        clearance = 3

        candidates = []

        # Mainly vertical movement into X:
        # detour horizontally.
        if abs(dr) >= abs(dc):

            candidates = [
                cc - clearance,
                cc + clearance,
            ]

            routes = []

            for side_col in candidates:

                if not (
                    1
                    <= side_col
                    <= self.grid_size - 2
                ):
                    continue

                waypoint_1 = (
                    cr,
                    side_col,
                )

                waypoint_2 = (
                    rr,
                    side_col,
                )

                routes.append(
                    (
                        waypoint_1,
                        waypoint_2,
                    )
                )

        # Mainly horizontal movement into X:
        # detour vertically.
        else:

            candidates = [
                cr - clearance,
                cr + clearance,
            ]

            routes = []

            for side_row in candidates:

                if not (
                    1
                    <= side_row
                    <= self.grid_size - 2
                ):
                    continue

                waypoint_1 = (
                    side_row,
                    cc,
                )

                waypoint_2 = (
                    side_row,
                    rc,
                )

                routes.append(
                    (
                        waypoint_1,
                        waypoint_2,
                    )
                )

        if not routes:
            return False

        # ----------------------------------------------------
        # ONE-CELL-AT-A-TIME CONNECTION
        # ----------------------------------------------------

        def connect(a, b):

            result = []

            row, col = a
            target_row, target_col = b

            while row != target_row:

                row += (
                    1
                    if target_row > row
                    else -1
                )

                result.append(
                    (row, col)
                )

            while col != target_col:

                col += (
                    1
                    if target_col > col
                    else -1
                )

                result.append(
                    (row, col)
                )

            return result

        # ----------------------------------------------------
        # TEST BOTH SIDES
        # ----------------------------------------------------

        chosen = None
        chosen_waypoints = None

        for waypoint_1, waypoint_2 in routes:

            candidate_route = []
            position = current
            valid = True

            for waypoint in (
                waypoint_1,
                waypoint_2,
                rejoin,
            ):

                segment = connect(
                    position,
                    waypoint,
                )

                for cell in segment:

                    r, c = cell

                    if cell == obstacle:
                        valid = False
                        break

                    if not (
                        1 <= r <= self.grid_size - 2
                        and
                        1 <= c <= self.grid_size - 2
                    ):
                        valid = False
                        break

                    if (
                        not candidate_route
                        or
                        candidate_route[-1] != cell
                    ):
                        candidate_route.append(
                            cell
                        )

                if not valid:
                    break

                position = waypoint

            if valid and candidate_route:

                chosen = candidate_route
                chosen_waypoints = (
                    waypoint_1,
                    waypoint_2,
                )

                break

        if not chosen:

            print(
                "40x40: no safe side route found."
            )

            return False

        # ----------------------------------------------------
        # INSERT LOCAL SAFETY ROUTE
        # ----------------------------------------------------

        travelled = self.result.path[
            :self.current_frame + 1
        ]

        remaining = baseline[
            rejoin_index + 1:
        ]

        self.result.path = (
            travelled
            + chosen
            + remaining
        )

        # X MUST STAY STILL.
        self.obstacle_phase = "detouring"

        self.obstacle_crossing_done = False

        self.obstacle_intercept_arrived = True

        # Rejoin occurs at the end of chosen.
        self.obstacle_detour_rejoin_frame = (
            self.current_frame
            + len(chosen)
        )

        # X moves only after wheelchair has travelled
        # another three cells.
        self.obstacle_release_frame = (
            self.obstacle_detour_rejoin_frame
            + 3
        )

        print()
        print("=" * 72)
        print("40x40 SAFETY BYPASS GENERATED")
        print("=" * 72)
        print(
            "Wheelchair stopped at:",
            current,
        )
        print(
            "X:",
            obstacle,
        )
        print(
            "Waypoint 1:",
            chosen_waypoints[0],
        )
        print(
            "Waypoint 2:",
            chosen_waypoints[1],
        )
        print(
            "Rejoin:",
            rejoin,
        )
        print(
            "Detour length:",
            len(chosen),
        )
        print(
            "X remains stationary."
        )
        print("=" * 72)

        return True


    def _reroute_around_moving_obstacle(
        self,
    ) -> bool:
        """
        Insert a visible local safety detour around moving X.

        The original classifier trajectory is preserved before
        the safety intervention.

        For 40x40, a dedicated one-step safety bypass is used.
        Other grid sizes keep the existing reroute behaviour.
        """

        if self.grid_size >= 40:
            return self._reroute_40x40_obstacle()

        if self.result is None:
            return False

        if self.moving_obstacle_position is None:
            return False

        if not self.baseline_classifier_path:
            return False

        current = self.result.path[
            self.current_frame
        ]

        obstacle = self.moving_obstacle_position
        baseline = self.baseline_classifier_path

        cr, cc = current
        OR, OC = obstacle

        # ====================================================
        # FIND THE OBSTACLE ON THE ORIGINAL CLASSIFIER PATH
        # ====================================================

        # Use the EXACT future occurrence selected when X
        # was prepared. This matters because a classifier path
        # can revisit the same grid cell many times.
        obstacle_index = getattr(
            self,
            "obstacle_intercept_frame",
            None,
        )

        # Validate the stored index.
        if (
            obstacle_index is None
            or obstacle_index < 0
            or obstacle_index >= len(baseline)
            or baseline[obstacle_index] != obstacle
        ):

            # Fallback:
            # find an occurrence that is still ahead of the
            # wheelchair rather than taking the first occurrence
            # anywhere in the trajectory.
            future_occurrences = [
                i
                for i in range(
                    max(
                        self.current_frame + 1,
                        0,
                    ),
                    len(baseline),
                )
                if baseline[i] == obstacle
            ]

            if future_occurrences:
                obstacle_index = (
                    future_occurrences[0]
                )
            else:
                obstacle_index = None

        if obstacle_index is None:
            print(
                "Could not locate the selected future "
                "obstacle occurrence on classifier path."
            )
            return False

        # ====================================================
        # FIND A REJOIN POINT AFTER THE OBSTACLE
        # ====================================================

        # Move sufficiently beyond X before rejoining.
        if self.grid_size <= 10:
            offset = 4
            clearance = 1

        else:
            # Use the proven 20x20 local-avoidance geometry
            # on 20x20, 30x30 and 40x40.
            offset = 7
            clearance = 3

        # ====================================================
        # FIND A REAL SPATIAL REJOIN POINT
        # ====================================================
        #
        # "offset" classifier frames are not necessarily
        # "offset" grid cells because the classifier path can
        # contain repeated positions.
        #
        # Search forward until the wheelchair has genuinely
        # progressed away from X.
        # ====================================================

        minimum_rejoin_distance = (
            2
            if self.grid_size <= 10
            else 4
        )

        search_start = min(
            obstacle_index + offset,
            len(baseline) - 1,
        )

        rejoin_index = None

        OR, OC = obstacle

        for candidate_index in range(
            search_start,
            len(baseline),
        ):

            candidate = baseline[
                candidate_index
            ]

            if candidate == obstacle:
                continue

            candidate_distance = (
                abs(candidate[0] - OR)
                + abs(candidate[1] - OC)
            )

            if (
                candidate_distance
                >= minimum_rejoin_distance
            ):
                rejoin_index = candidate_index
                break

        # If the classifier never gets four cells away,
        # use the first later cell that is simply not X.
        if rejoin_index is None:

            for candidate_index in range(
                obstacle_index + 1,
                len(baseline),
            ):

                if (
                    baseline[candidate_index]
                    != obstacle
                ):
                    rejoin_index = (
                        candidate_index
                    )
                    break

        if rejoin_index is None:

            print()
            print(
                "No valid classifier-path rejoin "
                "point found."
            )

            return False

        rejoin = baseline[
            rejoin_index
        ]

        rr, rc = rejoin

        # ====================================================
        # DETERMINE DIRECTION OF ORIGINAL PATH
        # ====================================================

        before_index = max(
            0,
            obstacle_index - 2,
        )

        after_index = min(
            len(baseline) - 1,
            obstacle_index + 2,
        )

        br, bc = baseline[
            before_index
        ]

        ar, ac = baseline[
            after_index
        ]

        row_motion = abs(ar - br)
        col_motion = abs(ac - bc)

        # ====================================================
        # CHOOSE A REAL PERPENDICULAR SIDE
        # ====================================================

        candidates = []

        # ----------------------------------------------------
        # Mainly HORIZONTAL classifier route
        #
        # Move UP/DOWN around X.
        # ----------------------------------------------------

        if col_motion >= row_motion:

            side_rows = [
                OR - clearance,
                OR + clearance,
            ]

            for side_row in side_rows:

                if not (
                    1 <= side_row <= self.grid_size - 2
                ):
                    continue

                waypoint_1 = (
                    side_row,
                    cc,
                )

                waypoint_2 = (
                    side_row,
                    rc,
                )

                candidates.append(
                    (
                        waypoint_1,
                        waypoint_2,
                    )
                )

        # ----------------------------------------------------
        # Mainly VERTICAL classifier route
        #
        # Move LEFT/RIGHT around X.
        # ----------------------------------------------------

        else:

            side_cols = [
                OC - clearance,
                OC + clearance,
            ]

            for side_col in side_cols:

                if not (
                    1 <= side_col <= self.grid_size - 2
                ):
                    continue

                waypoint_1 = (
                    cr,
                    side_col,
                )

                waypoint_2 = (
                    rr,
                    side_col,
                )

                candidates.append(
                    (
                        waypoint_1,
                        waypoint_2,
                    )
                )

        if not candidates:
            print(
                "No valid side detour available."
            )
            return False

        # ====================================================
        # CONNECT CELLS ONE AT A TIME
        # ====================================================

        def connect(a, b):

            route = []

            row, col = a
            target_row, target_col = b

            # Move rows first.
            while row != target_row:

                if target_row > row:
                    row += 1
                else:
                    row -= 1

                route.append(
                    (row, col)
                )

            # Then columns.
            while col != target_col:

                if target_col > col:
                    col += 1
                else:
                    col -= 1

                route.append(
                    (row, col)
                )

            return route

        # ====================================================
        # TEST CANDIDATE DETOURS
        # ====================================================

        chosen_detour = None
        chosen_waypoints = None

        for waypoint_1, waypoint_2 in candidates:

            route = []
            position = current

            valid = True

            for waypoint in (
                waypoint_1,
                waypoint_2,
                rejoin,
            ):

                segment = connect(
                    position,
                    waypoint,
                )

                for cell in segment:

                    # Never enter X.
                    if cell == obstacle:
                        valid = False
                        break

                    # Stay inside usable grid.
                    r, c = cell

                    if not (
                        1 <= r <= self.grid_size - 2
                        and
                        1 <= c <= self.grid_size - 2
                    ):
                        valid = False
                        break

                    if (
                        not route
                        or route[-1] != cell
                    ):
                        route.append(cell)

                if not valid:
                    break

                position = waypoint

            if valid and route:

                # Extra check:
                # route must genuinely leave the original
                # obstacle line.
                if col_motion >= row_motion:

                    visible_side_step = any(
                        cell[0] != OR
                        for cell in route
                    )

                else:

                    visible_side_step = any(
                        cell[1] != OC
                        for cell in route
                    )

                if visible_side_step:
                    chosen_detour = route
                    chosen_waypoints = (
                        waypoint_1,
                        waypoint_2,
                    )
                    break

        if not chosen_detour:

            # ==================================================
            # GUARANTEED FALLBACK ROUTE
            # ==================================================
            #
            # The preferred policy is the visible perpendicular
            # side detour above.
            #
            # Some classifier trajectories contain repeated
            # positions, turns, or unusual geometry for which
            # that simple side-detour construction is not valid.
            #
            # In that case, DO NOT leave the wheelchair stopped
            # forever in front of X. Use the existing safe grid
            # route finder to reach a later point on the ORIGINAL
            # classifier trajectory while treating X as blocked.
            # ==================================================

            print()
            print("=" * 70)
            print("VISIBLE SIDE DETOUR NOT AVAILABLE")
            print("Trying safe fallback route...")
            print("=" * 70)

            fallback_route = None
            fallback_rejoin_index = None
            fallback_rejoin = None

            # Search progressively farther along the genuine
            # classifier path until a safe rejoin route exists.
            fallback_start = max(
                obstacle_index + 1,
                self.current_frame + 2,
            )

            for candidate_index in range(
                fallback_start,
                len(baseline),
            ):

                candidate_rejoin = baseline[
                    candidate_index
                ]

                # Never rejoin on the obstacle.
                if candidate_rejoin == obstacle:
                    continue

                # Prefer a point genuinely beyond X.
                distance_from_obstacle = (
                    abs(
                        candidate_rejoin[0]
                        - obstacle[0]
                    )
                    +
                    abs(
                        candidate_rejoin[1]
                        - obstacle[1]
                    )
                )

                if distance_from_obstacle < 2:
                    continue

                safe_route = self._find_dynamic_route(
                    start=current,
                    target=candidate_rejoin,
                    blocked_position=obstacle,
                )

                if (
                    safe_route is None
                    or len(safe_route) < 2
                ):
                    continue

                # _find_dynamic_route includes current position.
                safe_detour = safe_route[1:]

                # Absolute final safety check.
                if obstacle in safe_detour:
                    continue

                fallback_route = safe_detour
                fallback_rejoin_index = candidate_index
                fallback_rejoin = candidate_rejoin
                break

            if fallback_route is None:

                print()
                print("=" * 70)
                print("NO SAFE FALLBACK ROUTE FOUND")
                print("Wheelchair remains protected from collision.")
                print("=" * 70)

                return False

            chosen_detour = fallback_route
            chosen_waypoints = (
                fallback_rejoin,
                fallback_rejoin,
            )

            # IMPORTANT:
            # use the rejoin point found by the fallback,
            # rather than the earlier side-detour rejoin.
            rejoin_index = fallback_rejoin_index
            rejoin = fallback_rejoin

            print()
            print("=" * 70)
            print("SAFE FALLBACK DETOUR GENERATED")
            print("=" * 70)
            print(
                "Wheelchair:",
                current,
            )
            print(
                "Obstacle:",
                obstacle,
            )
            print(
                "Rejoin original classifier path:",
                rejoin,
            )
            print(
                "Fallback detour:",
                chosen_detour,
            )
            print(
                "Fallback length:",
                len(chosen_detour),
            )
            print("=" * 70)

        # ====================================================
        # IMPORTANT:
        # PRESERVE EVERYTHING ALREADY EXECUTED
        # ====================================================

        travelled = self.result.path[
            :self.current_frame + 1
        ]

        # Continue with ORIGINAL classifier path only AFTER
        # the rejoin point.
        remaining_original = baseline[
            rejoin_index + 1:
        ]

        self.result.path = (
            travelled
            + chosen_detour
            + remaining_original
        )

        # ====================================================
        # KEEP X STATIONARY WHILE DETOUR IS EXECUTED
        # ====================================================
        #
        # IMPORTANT:
        # Generating the detour does NOT mean the wheelchair
        # has already passed the obstacle.
        #
        # X therefore remains frozen until the wheelchair:
        #
        #   1. executes the detour,
        #   2. passes the X,
        #   3. travels 3 additional cells.
        #
        # Only then will X be released.
        # ====================================================

        self.obstacle_phase = "detouring"
        self.obstacle_crossing_done = False
        self.obstacle_intercept_arrived = True

        # Store where the safety route rejoins the ORIGINAL
        # classifier trajectory.
        #
        # Because travelled already contains current_frame,
        # the rejoin position occurs after len(chosen_detour)
        # newly inserted path cells.
        self.obstacle_detour_rejoin_frame = (
            self.current_frame
            + len(chosen_detour)
        )

        # Keep X frozen for another 3 wheelchair movements
        # after the wheelchair reaches the rejoin point.
        self.obstacle_release_frame = (
            self.obstacle_detour_rejoin_frame
            + 3
        )

        print()
        print("=" * 70)
        print("REAL SIDE DETOUR GENERATED")
        print("=" * 70)
        print(
            "Wheelchair:",
            current,
        )
        print(
            "Obstacle:",
            obstacle,
        )
        print(
            "Waypoint 1:",
            chosen_waypoints[0],
        )
        print(
            "Waypoint 2:",
            chosen_waypoints[1],
        )
        print(
            "Rejoin original classifier path:",
            rejoin,
        )
        print(
            "Detour:",
            chosen_detour,
        )
        print(
            "Detour length:",
            len(chosen_detour),
        )
        print("=" * 70)

        return True


    def _reroute_around_static_obstacle(
        self,
    ) -> bool:
        """
        Perform a LOCAL safety detour around the obstacle.

        The classifier-generated trajectory is preserved.
        Only the unsafe segment is replaced, after which the
        wheelchair rejoins the original classifier path.
        """

        if self.result is None:
            return False

        if not self.obstacle_enabled:
            return False

        if self.static_obstacle_position is None:
            return False

        if self.baseline_classifier_path is None:
            return False

        current_position = self.result.path[
            self.current_frame
        ]

        baseline = self.baseline_classifier_path

        obstacle = self.static_obstacle_position

        # --------------------------------------------------
        # FIND WHERE WE CURRENTLY ARE IN BASELINE PATH
        # --------------------------------------------------

        baseline_index = None

        # Search around the current animation point first.
        search_start = max(
            0,
            self.current_frame - 5,
        )

        for index in range(
            search_start,
            len(baseline),
        ):
            if baseline[index] == current_position:
                baseline_index = index
                break

        if baseline_index is None:
            return False

        # --------------------------------------------------
        # FIND A NEARBY REJOIN POINT
        #
        # Do NOT plan directly to the target.
        # We only bypass the obstacle and rejoin the original
        # classifier trajectory a few steps later.
        # --------------------------------------------------

        for offset in range(3, 12):

            rejoin_index = min(
                baseline_index + offset,
                len(baseline) - 1,
            )

            rejoin_position = baseline[
                rejoin_index
            ]

            if rejoin_position == obstacle:
                continue

            local_route = self._find_dynamic_route(
                start=current_position,
                target=rejoin_position,
                blocked_position=obstacle,
            )

            if (
                local_route is None
                or len(local_route) < 2
            ):
                continue

            # --------------------------------------------------
            # PRESERVE EVERYTHING ALREADY TRAVELLED
            # --------------------------------------------------

            travelled = self.result.path[
                :self.current_frame + 1
            ]

            # --------------------------------------------------
            # LOCAL DETOUR
            # --------------------------------------------------

            detour = local_route[1:]

            # --------------------------------------------------
            # REJOIN ORIGINAL CLASSIFIER PATH
            # --------------------------------------------------

            original_remaining_path = baseline[
                rejoin_index + 1:
            ]

            self.result.path = (
                travelled
                + detour
                + original_remaining_path
            )

            print()
            print("=" * 64)
            print("STATIC OBSTACLE DETECTED")
            print("=" * 64)
            print(
                "Obstacle position:",
                obstacle,
            )
            print(
                "Wheelchair position:",
                current_position,
            )
            print(
                "Safety action: LOCAL DETOUR"
            )
            print(
                "Rejoining classifier path at:",
                rejoin_position,
            )
            print("=" * 64)

            return True

        return False


    def _run_demo(
        self,
        event=None,
    ) -> None:
        """Build and play the selected demonstration reliably."""

        # --------------------------------------------------
        # HARD UI EVENT GUARD
        # --------------------------------------------------
        #
        # A dropdown selection is handled on mouse release.
        # Matplotlib can send that same release to an axis that
        # becomes visible underneath the popup.
        #
        # Therefore Run is blocked:
        #   1. while any dropdown is open, and
        #   2. briefly after a dropdown item was selected.
        #
        # Only a later deliberate click on Run demonstration
        # is allowed to start navigation.
        # --------------------------------------------------

        if getattr(self, "dropdown_is_open", False):
            return

        if (
            time.monotonic()
            < getattr(
                self,
                "run_click_block_until",
                0.0,
            )
        ):
            print(
                "Ignored dropdown click-through; "
                "Run demonstration was not started."
            )
            return

        if getattr(self, "run_active", False):
            return

        self._close_dropdowns()

        # Stop any old timer left from previous versions.
        if self.timer is not None:
            try:
                self.timer.stop()
            except Exception:
                pass

        self.timer = None

        try:
            self.result = self._build_result()

            # Save the exact classifier-generated trajectory.
            # Obstacle ON must behave exactly like OFF until
            # an actual collision risk is encountered.
            self.baseline_classifier_path = list(
                self.result.path
            )

            if self.obstacle_enabled:

                # Initialise counters FIRST.
                self._prepare_moving_obstacle()

                # THEN choose the genuine future classifier
                # interception point and X starting position.
                #
                # This order is important:
                # _prepare_moving_obstacle() must NOT overwrite
                # the X position selected from the classifier path.
                self._prepare_guaranteed_crossing()

            

        except Exception as error:
            self.run_active = False
            self._show_error(str(error))
            return

        self.current_frame = 0
        self.paused = False
        self.run_active = True


        self.pause_button.label.set_text(
            "Pause"
        )

        # Draw initial frame.
        self._draw_frame()
        self.figure.canvas.draw_idle()

        # --------------------------------------------------
        # RELIABLE PLAYBACK LOOP
        # --------------------------------------------------

        last_frame = len(self.result.path) - 1

        while (
            self.run_active
            and self.result is not None
            and self.current_frame < last_frame
        ):

            # Pause keeps GUI responsive.
            while (
                self.paused
                and self.run_active
                and self.result is not None
            ):
                plt.pause(0.05)

            # Reset may have been pressed while paused.
            if (
                not self.run_active
                or self.result is None
            ):
                return

            next_frame = self.current_frame + 1

            # Defensive guard because obstacle rerouting may
            # replace the path with a shorter trajectory.
            if next_frame >= len(self.result.path):
                break

            next_position = self.result.path[
                next_frame
            ]

            # ----------------------------------------------
            # SAFETY CONTROLLER
            #
            # If a pedestrian currently occupies the next
            # wheelchair cell, do NOT move the wheelchair.
            # The pedestrian continues moving and the
            # wheelchair waits until the path is safe.
            # ----------------------------------------------

            # Safe to move.
            # ----------------------------------------------
            # FAST MOVING OBSTACLE
            # ----------------------------------------------

            if self.obstacle_enabled:

                self.obstacle_frame_counter += 1

                # Move only once every few wheelchair frames.
                if (
                    self.obstacle_frame_counter
                    >= self.obstacle_move_interval
                ):
                    self._move_obstacle_once()
                    self.obstacle_frame_counter = 0

                # If moving obstacle threatens the next
                # few classifier-driven cells, make a local
                # detour only.
                if self._moving_obstacle_is_near_route():

                    # --------------------------------------
                    # SAFETY STOP
                    # --------------------------------------
                    #
                    # Simulates the wheelchair detecting the
                    # obstruction and pausing before executing
                    # the local avoidance manoeuvre.
                    #
                    # No classifier prediction is changed here.
                    # --------------------------------------

                    if not self.obstacle_safety_pause_done:

                        print()
                        print(
                            ">>> SAFETY STOP <<<"
                        )
                        print(
                            "Wheelchair paused before deviation."
                        )

                        self._draw_frame()

                        plt.pause(
                            self.obstacle_safety_pause_seconds
                        )

                        self.obstacle_safety_pause_done = True

                    rerouted = (
                        self._reroute_around_moving_obstacle()
                    )

                    if rerouted:

                        last_frame = (
                            len(self.result.path) - 1
                        )

                        self._draw_frame()

                        plt.pause(0.10)

                        continue

            # ----------------------------------------------
            # REACTIVE OBSTACLE DETECTION
            # ----------------------------------------------

            next_frame = (
                self.current_frame + 1
            )

            if (
                self.obstacle_enabled
                and
                self.static_obstacle_position is not None
                and
                next_frame < len(self.result.path)
            ):

                next_position = (
                    self.result.path[
                        next_frame
                    ]
                )

                if (
                    next_position
                    == self.static_obstacle_position
                ):

                    # Do not enter obstacle cell.
                    # Re-plan only NOW, when the hazard
                    # actually blocks the route.
                    rerouted = (
                        self._reroute_around_static_obstacle()
                    )

                    if rerouted:

                        # Path length may have changed.
                        last_frame = (
                            len(self.result.path) - 1
                        )

                        self._draw_frame()

                        # Small visual reaction delay.
                        plt.pause(0.20)

                        # Start next loop iteration using
                        # the newly planned route.
                        continue

            # ----------------------------------------------
            # HARD COLLISION GUARD
            # ----------------------------------------------
            #
            # The wheelchair is NEVER allowed to enter the
            # moving obstacle's current grid cell.
            # ----------------------------------------------

            next_frame = (
                self.current_frame + 1
            )

            if (
                self.obstacle_enabled
                and
                self.moving_obstacle_position is not None
                and
                next_frame < len(self.result.path)
            ):

                next_position = (
                    self.result.path[
                        next_frame
                    ]
                )

                if (
                    next_position
                    == self.moving_obstacle_position
                ):

                    print()
                    print(
                        ">>> DIRECT COLLISION PREVENTED <<<"
                    )

                    rerouted = (
                        self._reroute_around_moving_obstacle()
                    )

                    self._draw_frame()

                    # Short reaction pause.
                    plt.pause(0.12)

                    if rerouted:

                        # Rerouting changes self.result.path.
                        # Refresh the final valid frame index
                        # before continuing playback.
                        last_frame = (
                            len(self.result.path) - 1
                        )

                        continue

                    # No safe route yet:
                    # remain stationary instead of crossing X.
                    continue

            # ----------------------------------------------
            # NORMAL CLASSIFIER TRAJECTORY
            # ----------------------------------------------

            # ==============================================
            # HARD MOVING-OBSTACLE SAFETY GATE
            # ==============================================
            #
            # IMPORTANT:
            #
            # self.result.path is still the genuine classifier
            # trajectory.
            #
            # We inspect ONLY the next cell before executing it.
            # If stationary X occupies that cell, the classifier
            # movement is temporarily prevented and the safety
            # controller must generate a local detour.
            # ==============================================

            next_frame = self.current_frame + 1

            if (
                self.obstacle_enabled
                and
                self.obstacle_phase == "blocked"
                and
                self.moving_obstacle_position is not None
                and
                next_frame < len(self.result.path)
            ):

                next_position = (
                    self.result.path[
                        next_frame
                    ]
                )

                obstacle_position = (
                    self.moving_obstacle_position
                )

                # ------------------------------------------
                # DIRECT COLLISION
                # ------------------------------------------

                if next_position == obstacle_position:

                    print()
                    print("=" * 70)
                    print(
                        ">>> HARD COLLISION PREVENTION <<<"
                    )
                    print(
                        "Wheelchair:",
                        self.result.path[
                            self.current_frame
                        ],
                    )
                    print(
                        "Next classifier cell:",
                        next_position,
                    )
                    print(
                        "Obstacle:",
                        obstacle_position,
                    )
                    print(
                        "NORMAL MOVE BLOCKED"
                    )
                    print("=" * 70)

                    # --------------------------------------
                    # STOP FIRST
                    # --------------------------------------

                    if not self.obstacle_safety_pause_done:

                        self._draw_frame()

                        print(
                            "Wheelchair stopping..."
                        )

                        plt.pause(
                            self.obstacle_safety_pause_seconds
                        )

                        self.obstacle_safety_pause_done = True

                    # --------------------------------------
                    # FORCE LOCAL DETOUR
                    # --------------------------------------

                    rerouted = (
                        self._reroute_around_moving_obstacle()
                    )

                    if rerouted:

                        print(
                            "Safety detour generated."
                        )

                        # IMPORTANT:
                        # Do NOT increment current_frame here.
                        #
                        # The next loop iteration will execute
                        # the first SAFE detour cell instead.
                        self._draw_frame()

                        plt.pause(0.15)

                        continue

                    # --------------------------------------
                    # NO SAFE DETOUR?
                    #
                    # WAIT. NEVER CROSS X.
                    # --------------------------------------

                    print(
                        "No safe detour available yet."
                    )
                    print(
                        "Wheelchair remains stopped."
                    )

                    self._draw_frame()

                    plt.pause(0.20)

                    continue

            # ==============================================
            # NORMAL CLASSIFIER MOVEMENT
            # ==============================================
            #
            # This happens only when the next cell is safe.
            # ==============================================

            self.current_frame += 1

            # ==============================================
            # RELEASE X ONLY AFTER THE WHEELCHAIR HAS
            # COMPLETED THE DETOUR AND MOVED 3 MORE CELLS
            # ==============================================

            if (
                self.obstacle_enabled
                and
                getattr(
                    self,
                    "obstacle_phase",
                    None,
                ) == "detouring"
            ):

                release_frame = getattr(
                    self,
                    "obstacle_release_frame",
                    None,
                )

                if (
                    release_frame is not None
                    and
                    self.current_frame >= release_frame
                ):

                    # Wheelchair has physically executed the
                    # detour and travelled the required extra
                    # frames. X can now move again.
                    self.obstacle_phase = "stationary"
                    self.obstacle_crossing_done = True
                    self.obstacle_intercept_arrived = False
                    self.obstacle_safety_pause_done = False

                    print()
                    print("=" * 70)
                    print(
                        ">>> WHEELCHAIR SAFELY PASSED X <<<"
                    )
                    print(
                        "Wheelchair completed the obstacle "
                        "avoidance manoeuvre."
                    )
                    print(
                        "X REMAINS PERMANENTLY STATIONARY."
                    )
                    print(
                        "Wheelchair continues on the original "
                        "classifier-driven route."
                    )
                    print("=" * 70)

            self._draw_frame()

            plt.pause(
                max(
                    FRAME_INTERVAL_MS / 1000.0,
                    0.001,
                )
            )

        # --------------------------------------------------
        # FINISHED
        # --------------------------------------------------

        if (
            self.result is not None
            and self.current_frame >= last_frame
        ):
            self.current_frame = last_frame

            self._draw_frame()

            self.run_active = False
            self.paused = False

            self.pause_button.label.set_text(
                "Pause"
            )

            self.figure.canvas.draw_idle()


    def _advance_frame(self) -> None:
        """Legacy callback retained for compatibility."""
        return


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

        # --------------------------------------------------
        # STATIC GREEN OBSTACLE
        # --------------------------------------------------

        if (
            self.obstacle_enabled
            and self.moving_obstacle_position is not None
        ):
            obstacle_row, obstacle_col = (
                self.moving_obstacle_position
            )

            axis.scatter(
                obstacle_col,
                obstacle_row,
                marker="X",
                s=300,
                color="red",
                linewidths=2.2,
                label="Obstacle",
                zorder=10,
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
            ncol=4,
        )

        self._update_result_information()

        self.figure.canvas.draw_idle()

    def _update_result_information(self) -> None:
        """Display minimal navigation status."""

        if self.result is None:
            return

        last_frame = len(self.result.path) - 1

        animation_complete = (
            self.current_frame >= last_frame
        )

        if animation_complete:
            status = "NAVIGATION COMPLETE"

            reached = (
                "YES"
                if self.result.reached_target
                else "NO"
            )
        else:
            status = "NAVIGATION IN PROGRESS"
            reached = "In progress"

        information = (
            f"{status}\n"
            f"Step: {self.current_frame} / {last_frame}\n"
            f"Reached target: {reached}"
        )

        self.information_axis.clear()
        self.information_axis.axis("off")

        self.information_axis.text(
            0.0,
            1.0,
            information,
            va="top",
            ha="left",
            fontsize=10.0,
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
        self.run_active = True
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

        self._close_dropdowns()

        if self.timer is not None:
            self.timer.stop()

        self.timer = None
        self.result = None
        self.current_frame = 0
        self.paused = False
        self.run_active = False

        self.pause_button.label.set_text(
            "Pause"
        )

        self._show_welcome_screen()

        self.figure.canvas.draw_idle()


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
