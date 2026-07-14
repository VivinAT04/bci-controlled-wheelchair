"""Random command sampler for wheelchair baseline experiments."""

from __future__ import annotations

import numpy as np


VALID_CLASSES = [
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
]


class RandomPredictionSampler:
    """
    Produce uniformly random wheelchair commands.

    This baseline ignores the intended action and returns one of the
    four motor-imagery classes with equal probability.
    """

    def __init__(
        self,
        random_seed: int | None = None,
    ) -> None:
        self.rng = np.random.default_rng(random_seed)

    def sample_prediction(
        self,
        intended_class: str,
    ) -> str:
        """
        Return a random class.

        The intended_class argument is accepted so this sampler has the
        same interface as EEGPredictionSampler.
        """
        return str(self.rng.choice(VALID_CLASSES))