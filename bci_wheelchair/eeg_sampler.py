"""Sample classifier outcomes for intended motor-imagery actions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


VALID_CLASSES = {
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
}


class EEGPredictionSampler:
    """
    Sample a classifier prediction conditioned on an intended EEG class.

    The CSV contains previously generated out-of-fold classifier predictions.
    Selecting a row by its true class represents sampling an EEG trial for
    the user's intended movement and passing it through the classifier.
    """

    def __init__(
        self,
        csv_path: str | Path,
        random_seed: int | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.rng = np.random.default_rng(random_seed)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {self.csv_path}"
            )

        self.predictions = pd.read_csv(self.csv_path)

        required_columns = {
            "true_class",
            "predicted_class",
        }

        missing_columns = required_columns - set(self.predictions.columns)

        if missing_columns:
            raise ValueError(
                "Prediction CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        unknown_true_classes = (
            set(self.predictions["true_class"].unique()) - VALID_CLASSES
        )

        unknown_predicted_classes = (
            set(self.predictions["predicted_class"].unique()) - VALID_CLASSES
        )

        if unknown_true_classes:
            raise ValueError(
                f"Unknown true classes: {sorted(unknown_true_classes)}"
            )

        if unknown_predicted_classes:
            raise ValueError(
                "Unknown predicted classes: "
                f"{sorted(unknown_predicted_classes)}"
            )

    def sample_prediction(self, intended_class: str) -> str:
        """
        Sample one predicted class for the requested intended class.

        Args:
            intended_class:
                The movement class the simulated user intends to perform.

        Returns:
            The class predicted by the EEG classifier for a randomly
            selected trial belonging to the intended class.
        """
        if intended_class not in VALID_CLASSES:
            raise ValueError(
                f"Unknown intended class: {intended_class}"
            )

        matching_trials = self.predictions[
            self.predictions["true_class"] == intended_class
        ]

        if matching_trials.empty:
            raise ValueError(
                f"No EEG trials found for class: {intended_class}"
            )

        selected_index = self.rng.integers(
            low=0,
            high=len(matching_trials),
        )

        selected_trial = matching_trials.iloc[selected_index]

        return str(selected_trial["predicted_class"])