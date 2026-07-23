"""Run classifier-driven wheelchair navigation simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from bci_wheelchair.simulation.samplers import EEGPredictionSampler
from bci_wheelchair.simulation.environment import (
    GridEnvironment,
    Position,
    WheelchairState,
    choose_intended_action,
    manhattan_distance,
)


@dataclass
class SimulationStep:
    """Information recorded for one simulation step."""

    step: int
    position_before: Position
    heading_before: int
    intended_action: str
    predicted_action: str
    position_after: Position
    heading_after: int
    prediction_correct: bool
    blocked_move: bool
    stop_command: bool


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
    path: List[Position]
    headings: List[int]
    trace: List[SimulationStep]


def run_classifier_simulation(
    environment: GridEnvironment,
    sampler: EEGPredictionSampler,
    start_state: WheelchairState,
    target: Position,
    max_steps: int = 200,
) -> SimulationResult:
    """
    Run one classifier-controlled wheelchair navigation simulation.

    At every step:

    1. The agent selects the intended command.
    2. A classifier prediction is sampled.
    3. The wheelchair executes the predicted command.
    4. The full state transition is recorded.
    """
    if not environment.is_valid_position(start_state.position):
        raise ValueError(
            f"Invalid start position: {start_state.position}"
        )

    if not environment.is_valid_position(target):
        raise ValueError(f"Invalid target position: {target}")

    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero")

    state = start_state

    path = [state.position]
    headings = [state.heading]
    trace: List[SimulationStep] = []

    correct_predictions = 0
    incorrect_predictions = 0
    blocked_moves = 0
    stop_commands = 0

    initial_distance = manhattan_distance(
        start_state.position,
        target,
    )

    steps_completed = 0

    for step in range(1, max_steps + 1):
        if state.position == target:
            break

        intended_action = choose_intended_action(
            state,
            target,
        )

        if intended_action is None:
            break

        predicted_action = sampler.sample_prediction(
            intended_action
        )

        prediction_correct = (
            predicted_action == intended_action
        )

        if prediction_correct:
            correct_predictions += 1
        else:
            incorrect_predictions += 1

        previous_state = state

        state = environment.step(
            state,
            predicted_action,
        )

        stop_command = predicted_action == "tongue"

        blocked_move = (
            predicted_action == "feet"
            and state.position == previous_state.position
        )

        if stop_command:
            stop_commands += 1

        if blocked_move:
            blocked_moves += 1

        trace.append(
            SimulationStep(
                step=step,
                position_before=previous_state.position,
                heading_before=previous_state.heading,
                intended_action=intended_action,
                predicted_action=predicted_action,
                position_after=state.position,
                heading_after=state.heading,
                prediction_correct=prediction_correct,
                blocked_move=blocked_move,
                stop_command=stop_command,
            )
        )

        path.append(state.position)
        headings.append(state.heading)

        steps_completed = step

    reached_target = state.position == target

    return SimulationResult(
        reached_target=reached_target,
        steps=steps_completed,
        start_state=start_state,
        target=target,
        final_state=state,
        initial_distance=initial_distance,
        final_distance=manhattan_distance(
            state.position,
            target,
        ),
        correct_predictions=correct_predictions,
        incorrect_predictions=incorrect_predictions,
        blocked_moves=blocked_moves,
        stop_commands=stop_commands,
        path=path,
        headings=headings,
        trace=trace,
    )