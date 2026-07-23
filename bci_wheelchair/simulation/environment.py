"""
Grid environment and navigation decision helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


Position = Tuple[int, int]

VALID_ACTIONS = {
    "left",
    "right",
    "forward",
    "stop",
}

VALID_HEADINGS = {
    "north",
    "east",
    "south",
    "west",
}


@dataclass
class WheelchairState:
    """Current wheelchair position and heading."""

    position: Position
    heading: str


class GridEnvironment:
    """Two-dimensional grid environment for wheelchair navigation."""

    def __init__(
        self,
        rows: int,
        columns: int,
        start_position: Position,
        target_position: Position,
        start_heading: str = "north",
    ) -> None:
        self.rows = rows
        self.columns = columns
        self.start_position = start_position
        self.target_position = target_position
        self.start_heading = start_heading

        if start_heading not in VALID_HEADINGS:
            raise ValueError(
                f"Invalid start heading: {start_heading}"
            )

        self._validate_position(
            start_position,
            "start_position",
        )
        self._validate_position(
            target_position,
            "target_position",
        )

    def _validate_position(
        self,
        position: Position,
        name: str,
    ) -> None:
        row, column = position

        if not (
            0 <= row < self.rows
            and 0 <= column < self.columns
        ):
            raise ValueError(
                f"{name} {position} is outside the grid."
            )

    def create_initial_state(self) -> WheelchairState:
        """Create the initial wheelchair state."""

        return WheelchairState(
            position=self.start_position,
            heading=self.start_heading,
        )

    def is_target_reached(
        self,
        state: WheelchairState,
    ) -> bool:
        """Return whether the wheelchair reached the target."""

        return state.position == self.target_position

    def apply_action(
        self,
        state: WheelchairState,
        action: str,
    ) -> tuple[WheelchairState, bool]:
        """
        Apply a navigation action.

        Returns:
            New state and whether movement was blocked.
        """

        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action: {action}"
            )

        if action == "stop":
            return state, False

        if action == "left":
            return WheelchairState(
                position=state.position,
                heading=self._turn_left(state.heading),
            ), False

        if action == "right":
            return WheelchairState(
                position=state.position,
                heading=self._turn_right(state.heading),
            ), False

        next_position = self._forward_position(state)

        if not self._is_inside_grid(next_position):
            return state, True

        return WheelchairState(
            position=next_position,
            heading=state.heading,
        ), False

    def _is_inside_grid(
        self,
        position: Position,
    ) -> bool:
        row, column = position

        return (
            0 <= row < self.rows
            and 0 <= column < self.columns
        )

    @staticmethod
    def _turn_left(heading: str) -> str:
        headings = [
            "north",
            "west",
            "south",
            "east",
        ]

        return headings[
            (headings.index(heading) + 1)
            % len(headings)
        ]

    @staticmethod
    def _turn_right(heading: str) -> str:
        headings = [
            "north",
            "east",
            "south",
            "west",
        ]

        return headings[
            (headings.index(heading) + 1)
            % len(headings)
        ]

    @staticmethod
    def _forward_position(
        state: WheelchairState,
    ) -> Position:
        row, column = state.position

        offsets = {
            "north": (-1, 0),
            "east": (0, 1),
            "south": (1, 0),
            "west": (0, -1),
        }

        row_offset, column_offset = offsets[state.heading]

        return (
            row + row_offset,
            column + column_offset,
        )


def manhattan_distance(
    first: Position,
    second: Position,
) -> int:
    """Calculate Manhattan distance between two grid positions."""

    return (
        abs(first[0] - second[0])
        + abs(first[1] - second[1])
    )


def desired_heading(
    current_position: Position,
    target_position: Position,
) -> Optional[str]:
    """Choose a heading that moves toward the target."""

    current_row, current_column = current_position
    target_row, target_column = target_position

    row_difference = target_row - current_row
    column_difference = target_column - current_column

    if row_difference == 0 and column_difference == 0:
        return None

    if abs(row_difference) >= abs(column_difference):
        return "south" if row_difference > 0 else "north"

    return "east" if column_difference > 0 else "west"


def choose_intended_action(
    state: WheelchairState,
    target_position: Position,
) -> str:
    """Choose the next ideal navigation action."""

    target_heading = desired_heading(
        state.position,
        target_position,
    )

    if target_heading is None:
        return "stop"

    if state.heading == target_heading:
        return "forward"

    right_turn = {
        "north": "east",
        "east": "south",
        "south": "west",
        "west": "north",
    }

    if right_turn[state.heading] == target_heading:
        return "right"

    return "left"
