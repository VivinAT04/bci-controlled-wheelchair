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
    "Tuned CSP + LDA",
    "FBCSP + LDA",
    "CSP + RBF-SVM",
    "FBCSP + RBF-SVM",
    "Riemannian MDM",
    "Riemannian TS + Shrinkage LDA",
    "Filter-Bank Riemannian",
    "Autoencoder + RBF-SVM",
    "Supervised Autoencoder + RBF-SVM",
    "EEGNet",
    "EA + FBCSP + Shrinkage LDA",
)

DEFAULT_METHOD = (
    "EA + FBCSP + Shrinkage LDA"
)

PREDICTION_FILES = {
    "Cross-Session": {
        "CSP + LDA": Path(
            "results/cross_session/csp_lda/"
            "csp_lda_cross_session_predictions.csv"
        ),
        "Tuned CSP + LDA": Path(
            "results/cross_session/tuned_csp_lda/"
            "tuned_csp_lda_cross_session_predictions.csv"
        ),
        "FBCSP + LDA": Path(
            "results/cross_session/fbcsp_lda/"
            "fbcsp_lda_cross_session_predictions.csv"
        ),
        "CSP + RBF-SVM": Path(
            "results/cross_session/csp_rbf_svm/"
            "csp_rbf_svm_cross_session_predictions.csv"
        ),
        "FBCSP + RBF-SVM": Path(
            "results/cross_session/fbcsp_rbf_svm/"
            "fbcsp_rbf_svm_cross_session_predictions.csv"
        ),
        "Riemannian MDM": Path(
            "results/cross_session/riemannian/mdm/"
            "riemannian_mdm_cross_session_predictions.csv"
        ),
        "Riemannian TS + Shrinkage LDA": Path(
            "results/cross_session/riemannian/tangent_lda/"
            "tangent_lda_cross_session_predictions.csv"
        ),
        "Filter-Bank Riemannian": Path(
            "results/cross_session/riemannian/filterbank/"
            "filterbank_riemannian_cross_session_predictions.csv"
        ),
        "Autoencoder + RBF-SVM": Path(
            "results/cross_session/autoencoder_rbf_svm/"
            "autoencoder_rbf_svm_cross_session_predictions.csv"
        ),
        "Supervised Autoencoder + RBF-SVM": Path(
            "results/cross_session/supervised_autoencoder_rbf_svm/"
            "supervised_autoencoder_rbf_svm_cross_session_predictions.csv"
        ),
        "EEGNet": Path(
            "results/cross_session/eegnet/"
            "eegnet_cross_session_predictions.csv"
        ),
        "EA + FBCSP + Shrinkage LDA": Path(
            "results/cross_session/ea_fbcsp_lda/"
            "ea_fbcsp_lda_cross_session_predictions.csv"
        ),
    },

    "Cross-Subject": {
        "CSP + LDA": Path(
            "results/cross_subject/csp_lda/"
            "csp_lda_cross_subject_predictions.csv"
        ),
        "Tuned CSP + LDA": Path(
            "results/cross_subject/tuned_csp_lda/"
            "tuned_csp_lda_cross_subject_predictions.csv"
        ),
        "FBCSP + LDA": Path(
            "results/cross_subject/fbcsp_lda/"
            "fbcsp_lda_cross_subject_predictions.csv"
        ),
        "CSP + RBF-SVM": Path(
            "results/cross_subject/csp_rbf_svm/"
            "csp_rbf_svm_cross_subject_predictions.csv"
        ),
        "FBCSP + RBF-SVM": Path(
            "results/cross_subject/fbcsp_rbf_svm/"
            "fbcsp_rbf_svm_cross_subject_predictions.csv"
        ),
        "Riemannian MDM": Path(
            "results/cross_subject/riemannian_mdm/"
            "riemannian_mdm_cross_subject_predictions.csv"
        ),
        "Riemannian TS + Shrinkage LDA": Path(
            "results/cross_subject/riemannian_tangent_lda/"
            "riemannian_tangent_lda_cross_subject_predictions.csv"
        ),
        "Filter-Bank Riemannian": Path(
            "results/cross_subject/filterbank_riemannian/"
            "filterbank_riemannian_cross_subject_predictions.csv"
        ),
        "Autoencoder + RBF-SVM": Path(
            "results/cross_subject/autoencoder_rbf_svm/"
            "autoencoder_rbf_svm_cross_subject_predictions.csv"
        ),
        "Supervised Autoencoder + RBF-SVM": Path(
            "results/cross_subject/supervised_autoencoder_rbf_svm/"
            "supervised_autoencoder_rbf_svm_cross_subject_predictions.csv"
        ),
        "EEGNet": Path(
            "results/cross_subject/eegnet/"
            "eegnet_loso_improved/predictions.csv"
        ),
        "EA + FBCSP + Shrinkage LDA": Path(
            "results/cross_subject/ea_fbcsp_lda/"
            "ea_fbcsp_lda_cross_subject_predictions.csv"
        ),
    },
}


FINAL_RESULTS = {
    "Cross-Session": {
        "CSP + LDA": (47.80, 0.304),
        "Tuned CSP + LDA": (47.96, 0.306),
        "FBCSP + LDA": (45.37, 0.272),
        "CSP + RBF-SVM": (49.31, 0.324),
        "FBCSP + RBF-SVM": (55.90, 0.412),
        "Riemannian MDM": (36.50, 0.153),
        "Riemannian TS + Shrinkage LDA": (52.16, 0.362),
        "Filter-Bank Riemannian": (49.85, 0.331),
        "Autoencoder + RBF-SVM": (28.94, 0.052),
        "Supervised Autoencoder + RBF-SVM": (27.28, 0.030),
        "EEGNet": (52.55, 0.367),
        "EA + FBCSP + Shrinkage LDA": (57.60, 0.435),
    },

    "Cross-Subject": {
        "CSP + LDA": (43.94, 0.253),
        "Tuned CSP + LDA": (42.82, 0.238),
        "FBCSP + LDA": (39.08, 0.188),
        "CSP + RBF-SVM": (38.66, 0.182),
        "FBCSP + RBF-SVM": (37.58, 0.168),
        "Riemannian MDM": (34.92, 0.132),
        "Riemannian TS + Shrinkage LDA": (38.50, 0.180),
        "Filter-Bank Riemannian": (37.77, 0.170),
        "Autoencoder + RBF-SVM": (27.70, 0.036),
        "Supervised Autoencoder + RBF-SVM": (26.04, 0.014),
        "EEGNet": (43.06, 0.241),
        "EA + FBCSP + Shrinkage LDA": (48.34, 0.311),
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
            [0.69, 0.025, 0.29, 0.17]
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
            0.495,
            "Simulated classifier accuracy",
            fontsize=11.5,
            fontweight="bold",
        )

        accuracy_layout = {
            0.25: (0.69, 0.445),
            0.40: (0.79, 0.445),
            0.50: (0.89, 0.445),

            0.60: (0.69, 0.398),
            0.66: (0.79, 0.398),
            0.70: (0.89, 0.398),

            0.80: (0.69, 0.351),
            0.90: (0.79, 0.351),
            1.00: (0.89, 0.351),
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

        self.run_button = self._create_button(
            "Run demonstration",
            0.69,
            0.285,
            0.29,
            0.045,
            self._run_demo,
        )

        self.pause_button = self._create_button(
            "Pause",
            0.69,
            0.230,
            0.09,
            0.040,
            self._toggle_pause,
        )

        self.replay_button = self._create_button(
            "Replay",
            0.79,
            0.230,
            0.09,
            0.040,
            self._replay,
        )

        self.stop_button = self._create_button(
            "Reset",
            0.89,
            0.230,
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
        """Close dropdown and restore hidden controls."""

        for axis in self.dropdown_option_axes:
            try:
                axis.remove()
            except Exception:
                pass

        self.dropdown_option_axes.clear()
        self.dropdown_option_buttons.clear()

        if self.dropdown_background_axis is not None:
            try:
                self.dropdown_background_axis.remove()
            except Exception:
                pass

            self.dropdown_background_axis = None

        # Restore controls hidden while popup was open.
        for axis in self.dropdown_hidden_axes:
            try:
                axis.set_visible(True)
            except Exception:
                pass

        self.dropdown_hidden_axes.clear()

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
        """Open dropdown as a true modal-style popup."""

        self._close_dropdowns()

        option_height = 0.034
        number_of_options = len(options)

        menu_height = (
            option_height
            * number_of_options
        )

        menu_bottom = (
            top
            - menu_height
        )

        # --------------------------------------------------
        # HIDE ALL UNDERLYING CONTROL AXES THAT INTERSECT
        # THE DROPDOWN AREA.
        # --------------------------------------------------

        menu_left = left
        menu_right = left + width
        menu_top = top

        protected_axes = {
            getattr(
                self,
                "method_dropdown",
                None,
            ).ax
            if hasattr(
                self,
                "method_dropdown",
            )
            else None
        }

        for axis in list(
            self.figure.axes
        ):

            if axis in protected_axes:
                continue

            # Don't hide navigation display or info panel.
            if axis is self.navigation_axis:
                continue

            if axis is self.information_axis:
                continue

            position = axis.get_position()

            axis_left = position.x0
            axis_right = position.x1
            axis_bottom = position.y0
            axis_top = position.y1

            horizontal_overlap = (
                axis_right > menu_left
                and axis_left < menu_right
            )

            vertical_overlap = (
                axis_top > menu_bottom
                and axis_bottom < menu_top
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
        # OPAQUE POPUP BACKGROUND
        # --------------------------------------------------

        self.dropdown_background_axis = (
            self.figure.add_axes(
                [
                    left,
                    menu_bottom,
                    width,
                    menu_height,
                ]
            )
        )

        self.dropdown_background_axis.set_zorder(
            10000
        )

        self.dropdown_background_axis.set_facecolor(
            "white"
        )

        self.dropdown_background_axis.patch.set_alpha(
            1.0
        )

        self.dropdown_background_axis.set_xticks(
            []
        )

        self.dropdown_background_axis.set_yticks(
            []
        )

        self.dropdown_background_axis.format_coord = (
            lambda x, y: ""
        )

        # --------------------------------------------------
        # DROPDOWN OPTIONS
        # --------------------------------------------------

        for index, option in enumerate(
            options
        ):

            bottom = (
                top
                - (index + 1)
                * option_height
            )

            axis = self.figure.add_axes(
                [
                    left,
                    bottom,
                    width,
                    option_height,
                ]
            )

            axis.set_zorder(
                11000 + index
            )

            axis.format_coord = (
                lambda x, y: ""
            )

            is_selected = (
                option == selected_value
            )

            button = Button(
                axis,
                str(option),
                color=(
                    "0.70"
                    if is_selected
                    else "0.94"
                ),
                hovercolor="0.78",
            )

            button.label.set_fontsize(
                font_size
            )

            button.on_clicked(
                lambda event,
                value=option:
                callback(value)
            )

            self.dropdown_option_axes.append(
                axis
            )

            self.dropdown_option_buttons.append(
                button
            )

        self.figure.canvas.draw_idle()


    def _toggle_method_dropdown(
        self,
        event,
    ) -> None:
        """Open classifier-method dropdown."""

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
        """Select classifier from dropdown."""

        self._close_dropdowns()

        self._set_method(
            method
        )

        self.method_dropdown.label.set_text(
            f"{self.method}    ▼"
        )

        self.figure.canvas.draw_idle()


    def _toggle_evaluation_dropdown(
        self,
        event,
    ) -> None:
        """Open evaluation dropdown."""

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

        self._close_dropdowns()

        self._set_mode(
            mode
        )

        self._update_button_styles()
        self._show_current_selection()


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

        if mode not in {
            "real",
            "simulated",
        }:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        self.mode = mode

        self._update_button_styles()
        self._show_current_selection()


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
        self._show_current_selection()


    def _set_evaluation(
        self,
        evaluation: str,
    ) -> None:
        """Select cross-session or cross-subject predictions."""

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
        self._show_current_selection()


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
