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
    Sample classifier predictions conditioned on an intended EEG class.

    Supported CSV column formats:

    1. true_class, predicted_class
    2. true_label, predicted_label
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

        available_columns = set(self.predictions.columns)

        if {
            "true_class",
            "predicted_class",
        }.issubset(available_columns):
            self.true_column = "true_class"
            self.predicted_column = "predicted_class"

        elif {
            "true_label",
            "predicted_label",
        }.issubset(available_columns):
            self.true_column = "true_label"
            self.predicted_column = "predicted_label"

        else:
            raise ValueError(
                "Prediction CSV must contain either "
                "['true_class', 'predicted_class'] or "
                "['true_label', 'predicted_label']."
            )

        unknown_true_classes = (
            set(self.predictions[self.true_column].dropna().unique())
            - VALID_CLASSES
        )

        unknown_predicted_classes = (
            set(
                self.predictions[
                    self.predicted_column
                ].dropna().unique()
            )
            - VALID_CLASSES
        )

        if unknown_true_classes:
            raise ValueError(
                f"Unknown true classes: "
                f"{sorted(unknown_true_classes)}"
            )

        if unknown_predicted_classes:
            raise ValueError(
                f"Unknown predicted classes: "
                f"{sorted(unknown_predicted_classes)}"
            )

    def sample_prediction(self, intended_class: str) -> str:
        """Sample one prediction for the intended movement class."""

        if intended_class not in VALID_CLASSES:
            raise ValueError(
                f"Unknown intended class: {intended_class}"
            )

        matching_trials = self.predictions[
            self.predictions[self.true_column] == intended_class
        ]

        if matching_trials.empty:
            raise ValueError(
                f"No EEG trials found for class: {intended_class}"
            )

        selected_index = int(
            self.rng.integers(
                low=0,
                high=len(matching_trials),
            )
        )

        selected_trial = matching_trials.iloc[selected_index]

        return str(
            selected_trial[self.predicted_column]
        )
