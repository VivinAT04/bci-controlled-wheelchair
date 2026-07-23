"""
Grid environment and navigation decision helpers.

Motor-imagery commands are translated into wheelchair actions:

    left_hand  -> turn left
    right_hand -> turn right
    feet       -> move forward
    tongue     -> stop
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeAlias


Position: TypeAlias = tuple[int, int]
Heading: TypeAlias = int

NORTH = 0
EAST = 90
SOUTH = 180
WEST = 270

VALID_HEADINGS = {
    NORTH,
    EAST,
    SOUTH,
    WEST,
}

VALID_ACTIONS = {
    "left_hand",
    "right_hand",
    "feet",
    "tongue",
}

HEADING_OFFSETS = {
    NORTH: (-1, 0),
    EAST: (0, 1),
    SOUTH: (1, 0),
    WEST: (0, -1),
}


@dataclass(frozen=True)
class WheelchairState:
    """Current wheelchair position and heading."""

    position: Position
    heading: Heading


class GridEnvironment:
    """Two-dimensional grid used for wheelchair navigation."""

    def __init__(
        self,
        rows: int,
        cols: int | None = None,
        *,
        columns: int | None = None,
        start_position: Position | None = None,
        target_position: Position | None = None,
        start_heading: Heading = NORTH,
    ) -> None:
        """
        Create a grid environment.

        Both ``cols`` and ``columns`` are accepted so older and newer
        simulation scripts continue to work.
        """
        if cols is None and columns is None:
            raise ValueError(
                "Provide the grid width using cols or columns."
            )

        if cols is not None and columns is not None and cols != columns:
            raise ValueError(
                "cols and columns must contain the same value."
            )

        resolved_columns = (
            columns
            if columns is not None
            else cols
        )

        if rows <= 0:
            raise ValueError("rows must be greater than zero.")

        if resolved_columns is None or resolved_columns <= 0:
            raise ValueError(
                "Grid columns must be greater than zero."
            )

        if start_heading not in VALID_HEADINGS:
            raise ValueError(
                f"Invalid start heading: {start_heading}. "
                f"Expected one of {sorted(VALID_HEADINGS)}."
            )

        self.rows = rows
        self.cols = resolved_columns
        self.columns = resolved_columns

        self.start_position = start_position
        self.target_position = target_position
        self.start_heading = start_heading

        if start_position is not None:
            self._validate_position(
                start_position,
                "start_position",
            )

        if target_position is not None:
            self._validate_position(
                target_position,
                "target_position",
            )

    def _validate_position(
        self,
        position: Position,
        name: str,
    ) -> None:
        """Raise an error when a position lies outside the grid."""
        if not self.is_valid_position(position):
            raise ValueError(
                f"{name} {position} is outside the "
                f"{self.rows}x{self.columns} grid."
            )

    def is_valid_position(
        self,
        position: Position,
    ) -> bool:
        """Return whether a position is inside the grid."""
        row, column = position

        return (
            0 <= row < self.rows
            and 0 <= column < self.columns
        )

    def create_initial_state(self) -> WheelchairState:
        """Create the configured starting state."""
        if self.start_position is None:
            raise ValueError(
                "No start_position was supplied when the "
                "environment was created."
            )

        return WheelchairState(
            position=self.start_position,
            heading=self.start_heading,
        )

    def is_target_reached(
        self,
        state: WheelchairState,
        target: Position | None = None,
    ) -> bool:
        """Return whether the wheelchair reached the target."""
        resolved_target = (
            target
            if target is not None
            else self.target_position
        )

        if resolved_target is None:
            raise ValueError(
                "No target position was supplied."
            )

        return state.position == resolved_target

    def step(
        self,
        state: WheelchairState,
        action: str,
    ) -> WheelchairState:
        """
        Apply one motor-imagery command.

        Turning and stopping do not change the wheelchair position.
        Forward movement is blocked at the grid boundary.
        """
        if not self.is_valid_position(state.position):
            raise ValueError(
                f"State position {state.position} is outside the grid."
            )

        if state.heading not in VALID_HEADINGS:
            raise ValueError(
                f"Invalid heading: {state.heading}"
            )

        if action not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action: {action}. "
                f"Expected one of {sorted(VALID_ACTIONS)}."
            )

        if action == "left_hand":
            return WheelchairState(
                position=state.position,
                heading=turn_left(state.heading),
            )

        if action == "right_hand":
            return WheelchairState(
                position=state.position,
                heading=turn_right(state.heading),
            )

        if action == "tongue":
            return state

        next_position = forward_position(state)

        if not self.is_valid_position(next_position):
            return state

        return WheelchairState(
            position=next_position,
            heading=state.heading,
        )

    def apply_action(
        self,
        state: WheelchairState,
        action: str,
    ) -> tuple[WheelchairState, bool]:
        """
        Apply an action and report whether forward movement was blocked.

        This compatibility method is retained for older visualisation code.
        """
        next_state = self.step(state, action)

        blocked = (
            action == "feet"
            and next_state.position == state.position
        )

        return next_state, blocked


def turn_left(
    heading: Heading,
) -> Heading:
    """Turn the wheelchair 90 degrees to the left."""
    if heading not in VALID_HEADINGS:
        raise ValueError(f"Invalid heading: {heading}")

    return (heading - 90) % 360


def turn_right(
    heading: Heading,
) -> Heading:
    """Turn the wheelchair 90 degrees to the right."""
    if heading not in VALID_HEADINGS:
        raise ValueError(f"Invalid heading: {heading}")

    return (heading + 90) % 360


def forward_position(
    state: WheelchairState,
) -> Position:
    """Return the position one cell ahead of the wheelchair."""
    if state.heading not in VALID_HEADINGS:
        raise ValueError(
            f"Invalid heading: {state.heading}"
        )

    row_offset, column_offset = HEADING_OFFSETS[
        state.heading
    ]

    row, column = state.position

    return (
        row + row_offset,
        column + column_offset,
    )


def manhattan_distance(
    first: Position,
    second: Position,
) -> int:
    """Calculate Manhattan distance between two positions."""
    return (
        abs(first[0] - second[0])
        + abs(first[1] - second[1])
    )


def desired_heading(
    current_position: Position,
    target_position: Position,
) -> Optional[Heading]:
    """
    Choose a heading that moves toward the target.

    Vertical movement is selected first. Once the target row is reached,
    horizontal movement is selected.
    """
    current_row, current_column = current_position
    target_row, target_column = target_position

    if current_position == target_position:
        return None

    if current_row > target_row:
        return NORTH

    if current_row < target_row:
        return SOUTH

    if current_column < target_column:
        return EAST

    return WEST


def choose_intended_action(
    state: WheelchairState,
    target_position: Position,
) -> Optional[str]:
    """
    Choose the command required to move toward the target.

    Returns:
        ``feet`` when already facing the desired direction,
        ``left_hand`` or ``right_hand`` when turning is required,
        or ``None`` when the target has already been reached.
    """
    required_heading = desired_heading(
        state.position,
        target_position,
    )

    if required_heading is None:
        return None

    if state.heading == required_heading:
        return "feet"

    right_heading = turn_right(state.heading)

    if right_heading == required_heading:
        return "right_hand"

    left_heading = turn_left(state.heading)

    if left_heading == required_heading:
        return "left_hand"

    # The required direction is directly behind the wheelchair.
    # Two right turns will eventually face it toward the target.
    return "right_hand"
