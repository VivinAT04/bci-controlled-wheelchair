"""Run classifier-driven wheelchair navigation simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from bci_wheelchair.simulation.environment import (
    GridEnvironment,
    Position,
    WheelchairState,
    choose_intended_action,
    manhattan_distance,
)


class PredictionSampler(Protocol):
    """Interface for real and simulated EEG prediction samplers."""

    def sample_prediction(
        self,
        intended_class: str,
    ) -> str:
        """Return a predicted motor-imagery command."""


@dataclass
class SimulationStep:
    """Information recorded for one simulation step."""

    step: int
    position_before: Position
    heading_before: int

    # Goal-directed command requested by navigation planner.
    intended_action: str

    # Raw EEG/classifier output.
    predicted_action: str

    # Command actually executed by the wheelchair.
    executed_action: str

    position_after: Position
    heading_after: int

    prediction_correct: bool
    blocked_move: bool
    stop_command: bool

    # True when shared-control AI corrected the EEG command.
    ai_intervention: bool


@dataclass
class SimulationResult:
    """Results from one wheelchair navigation simulation."""

    reached_target: bool
    steps: int

    start_state: WheelchairState
    target: Position
    final_state: WheelchairState

    initial_distance: int
    final_distance: int

    correct_predictions: int
    incorrect_predictions: int

    blocked_moves: int
    stop_commands: int

    # Number of commands corrected by shared control.
    ai_interventions: int

    path: List[Position]
    headings: List[int]
    trace: List[SimulationStep]


def run_classifier_simulation(
    environment: GridEnvironment,
    sampler: PredictionSampler,
    start_state: WheelchairState,
    target: Position,
    max_steps: int = 200,
) -> SimulationResult:
    """
    Run one EEG-classifier-driven wheelchair navigation simulation.

    The simulation uses shared human-AI control.

    Normal behaviour
    ----------------
    The classifier prediction is executed directly.

    Recovery behaviour
    ------------------
    If repeated EEG predictions fail to make useful progress,
    the intelligent-navigation layer temporarily executes the
    goal-directed action instead.

    This means classifier accuracy still affects path efficiency,
    incorrect commands, blocked movements and the number of AI
    interventions, while the wheelchair can recover from prolonged
    poor prediction sequences.

    Near the end of the allowed simulation budget, recovery mode is
    strengthened so that an otherwise recoverable navigation task is
    not allowed to fail simply because of repeated classifier errors.
    """

    if not environment.is_valid_position(
        start_state.position
    ):
        raise ValueError(
            f"Invalid start position: "
            f"{start_state.position}"
        )

    if not environment.is_valid_position(
        target
    ):
        raise ValueError(
            f"Invalid target position: {target}"
        )

    if max_steps <= 0:
        raise ValueError(
            "max_steps must be greater than zero"
        )

    state = start_state

    path: List[Position] = [
        state.position
    ]

    headings: List[int] = [
        state.heading
    ]

    trace: List[SimulationStep] = []

    correct_predictions = 0
    incorrect_predictions = 0

    blocked_moves = 0
    stop_commands = 0
    ai_interventions = 0

    initial_distance = manhattan_distance(
        start_state.position,
        target,
    )

    steps_completed = 0

    # ---------------------------------------------------------
    # SHARED-CONTROL SETTINGS
    # ---------------------------------------------------------

    # Number of consecutive unproductive commands tolerated
    # before navigation AI intervenes.
    stall_threshold = 5

    # After several wrong predictions in succession, intervention
    # is also permitted even when orientation changes obscure
    # Manhattan-distance progress.
    error_streak_threshold = 4

    no_progress_steps = 0
    incorrect_streak = 0

    # Keep enough steps near the end for deterministic recovery.
    #
    # Reaching a target can require turning plus moving, therefore
    # use a conservative multiple of Manhattan distance.
    recovery_budget = (
        max(
            20,
            initial_distance * 4 + 12,
        )
    )

    recovery_start_step = max(
        1,
        max_steps - recovery_budget,
    )

    # ---------------------------------------------------------
    # SIMULATION LOOP
    # ---------------------------------------------------------

    for step in range(
        1,
        max_steps + 1,
    ):

        if state.position == target:
            break

        intended_action = (
            choose_intended_action(
                state,
                target,
            )
        )

        if intended_action is None:
            break

        predicted_action = (
            sampler.sample_prediction(
                intended_action
            )
        )

        prediction_correct = (
            predicted_action
            == intended_action
        )

        if prediction_correct:

            correct_predictions += 1
            incorrect_streak = 0

        else:

            incorrect_predictions += 1
            incorrect_streak += 1

        previous_state = state

        previous_distance = (
            manhattan_distance(
                previous_state.position,
                target,
            )
        )

        # -----------------------------------------------------
        # SHARED HUMAN-AI CONTROL
        # -----------------------------------------------------

        force_recovery = (
            step >= recovery_start_step
        )

        stalled = (
            no_progress_steps
            >= stall_threshold
        )

        repeated_errors = (
            incorrect_streak
            >= error_streak_threshold
        )

        ai_intervention = (
            force_recovery
            or stalled
            or repeated_errors
        )

        if ai_intervention:

            executed_action = (
                intended_action
            )

            ai_interventions += 1

        else:

            executed_action = (
                predicted_action
            )

        # -----------------------------------------------------
        # EXECUTE COMMAND
        # -----------------------------------------------------

        state = environment.step(
            state,
            executed_action,
        )

        stop_command = (
            predicted_action
            == "tongue"
        )

        blocked_move = (
            executed_action == "feet"
            and state.position
            == previous_state.position
        )

        if stop_command:
            stop_commands += 1

        if blocked_move:
            blocked_moves += 1

        # -----------------------------------------------------
        # PROGRESS MONITORING
        # -----------------------------------------------------

        new_distance = (
            manhattan_distance(
                state.position,
                target,
            )
        )

        # Movement toward target resets the stall counter.
        if new_distance < previous_distance:

            no_progress_steps = 0

        else:

            # Turning is sometimes necessary before movement.
            # It still counts as a non-distance-reducing step,
            # but intervention occurs only after several such
            # steps in succession.
            no_progress_steps += 1

        # Successful AI correction clears the error streak so
        # control can be returned to the EEG classifier.
        if (
            ai_intervention
            and not force_recovery
        ):
            incorrect_streak = 0

        # -----------------------------------------------------
        # RECORD TRACE
        # -----------------------------------------------------

        trace.append(
            SimulationStep(
                step=step,
                position_before=(
                    previous_state.position
                ),
                heading_before=(
                    previous_state.heading
                ),
                intended_action=(
                    intended_action
                ),
                predicted_action=(
                    predicted_action
                ),
                executed_action=(
                    executed_action
                ),
                position_after=(
                    state.position
                ),
                heading_after=(
                    state.heading
                ),
                prediction_correct=(
                    prediction_correct
                ),
                blocked_move=(
                    blocked_move
                ),
                stop_command=(
                    stop_command
                ),
                ai_intervention=(
                    ai_intervention
                ),
            )
        )

        path.append(
            state.position
        )

        headings.append(
            state.heading
        )

        steps_completed = step

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    reached_target = (
        state.position == target
    )

    # Safety check:
    # if navigation reports success, the final path position
    # must exactly match the target.
    if reached_target:
        if not path or path[-1] != target:
            path.append(target)
            headings.append(state.heading)

    return SimulationResult(
        reached_target=(
            reached_target
        ),
        steps=(
            steps_completed
        ),
        start_state=(
            start_state
        ),
        target=(
            target
        ),
        final_state=(
            state
        ),
        initial_distance=(
            initial_distance
        ),
        final_distance=(
            manhattan_distance(
                state.position,
                target,
            )
        ),
        correct_predictions=(
            correct_predictions
        ),
        incorrect_predictions=(
            incorrect_predictions
        ),
        blocked_moves=(
            blocked_moves
        ),
        stop_commands=(
            stop_commands
        ),
        ai_interventions=(
            ai_interventions
        ),
        path=(
            path
        ),
        headings=(
            headings
        ),
        trace=(
            trace
        ),
    )
