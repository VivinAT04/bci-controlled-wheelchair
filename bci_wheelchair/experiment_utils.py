"""Utilities for generating repeatable wheelchair simulation scenarios."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from bci_wheelchair.simulation import Position, WheelchairState


Scenario = Tuple[WheelchairState, Position]


def generate_scenarios(
    n_simulations: int,
    rows: int = 20,
    cols: int = 20,
    random_seed: int = 42,
) -> List[Scenario]:
    """
    Generate repeatable random start states and target positions.

    The same scenario list should be used for both the EEG classifier
    and random baseline so that the comparison is fair.
    """
    if n_simulations <= 0:
        raise ValueError("n_simulations must be greater than zero")

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be greater than zero")

    rng = np.random.default_rng(random_seed)

    headings = [0, 90, 180, 270]
    scenarios: List[Scenario] = []

    for _ in range(n_simulations):
        start_position = (
            int(rng.integers(0, rows)),
            int(rng.integers(0, cols)),
        )

        target = start_position

        while target == start_position:
            target = (
                int(rng.integers(0, rows)),
                int(rng.integers(0, cols)),
            )

        start_heading = int(rng.choice(headings))

        start_state = WheelchairState(
            position=start_position,
            heading=start_heading,
        )

        scenarios.append(
            (
                start_state,
                target,
            )
        )

    return scenarios