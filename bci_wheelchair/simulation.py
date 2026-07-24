"""Grid-based wheelchair simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


Position = Tuple[int, int]

VALID_ACTIONS = {
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
}

VALID_HEADINGS = {
    0,
    90,
    180,
    270,
}


@dataclass(frozen=True)
class WheelchairState:
    """Wheelchair position and orientation."""

    position: Position
    heading: int


@dataclass
class GridEnvironment:
    """Rectangular grid environment for wheelchair navigation."""

    rows: int = 20
    cols: int = 20

    def is_valid_position(self, position: Position) -> bool:
        """Return True when a position is inside the grid."""
        row, column = position

        return (
            0 <= row < self.rows
            and 0 <= column < self.cols
        )

    def step(
        self,
        state: WheelchairState,
        action: str,
    ) -> WheelchairState:
        """
        Apply one motor-imagery command to the wheelchair.

        Mapping:
            left_hand  -> turn left
            right_hand -> turn right
            feet       -> move forward
            tongue     -> stop
        """
        if action not in VALID_ACTIONS:
            raise ValueError(f"Unknown action: {action}")

        if state.heading not in VALID_HEADINGS:
            raise ValueError(f"Invalid heading: {state.heading}")

        if action == "left_hand":
            return WheelchairState(
                position=state.position,
                heading=(state.heading - 90) % 360,
            )

        if action == "right_hand":
            return WheelchairState(
                position=state.position,
                heading=(state.heading + 90) % 360,
            )

        if action == "tongue":
            return state

        next_position = self._forward_position(
            state.position,
            state.heading,
        )

        if not self.is_valid_position(next_position):
            return state

        return WheelchairState(
            position=next_position,
            heading=state.heading,
        )

    @staticmethod
    def _forward_position(
        position: Position,
        heading: int,
    ) -> Position:
        """Return the position one cell ahead of the wheelchair."""
        row, column = position

        movement = {
            0: (-1, 0),
            90: (0, 1),
            180: (1, 0),
            270: (0, -1),
        }

        row_change, column_change = movement[heading]

        return (
            row + row_change,
            column + column_change,
        )


def manhattan_distance(
    position: Position,
    target: Position,
) -> int:
    """Calculate Manhattan distance between two positions."""
    return (
        abs(position[0] - target[0])
        + abs(position[1] - target[1])
    )


def desired_heading(
    position: Position,
    target: Position,
) -> Optional[int]:
    """
    Return a heading that moves the wheelchair towards the target.

    Horizontal movement is completed before vertical movement.
    """
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


def choose_intended_action(
    state: WheelchairState,
    target: Position,
) -> Optional[str]:
    """
    Choose the next motor-imagery command needed to approach the target.

    The agent first turns toward the required direction and then issues
    the feet command to move forward.
    """
    target_heading = desired_heading(
        state.position,
        target,
    )

    if target_heading is None:
        return None

    if state.heading == target_heading:
        return "feet"

    right_turn_heading = (state.heading + 90) % 360

    if right_turn_heading == target_heading:
        return "right_hand"

    return "left_hand"