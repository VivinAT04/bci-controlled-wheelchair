"""Visual demonstration of ideal grid-based wheelchair navigation."""

import matplotlib.pyplot as plt

from bci_wheelchair.simulation import (
    GridEnvironment,
    WheelchairState,
    choose_intended_action,
    manhattan_distance,
)


def main() -> None:
    grid = GridEnvironment(
        rows=20,
        cols=20,
    )

    start_state = WheelchairState(
        position=(10, 10),
        heading=0,
    )

    target = (5, 15)

    state = start_state
    path = [state.position]
    headings = [state.heading]

    max_steps = 200
    steps = 0

    while state.position != target and steps < max_steps:
        intended_action = choose_intended_action(
            state,
            target,
        )

        if intended_action is None:
            break

        state = grid.step(
            state,
            intended_action,
        )

        path.append(state.position)
        headings.append(state.heading)
        steps += 1

    rows = [
        position[0]
        for position in path
    ]

    columns = [
        position[1]
        for position in path
    ]

    plt.figure(figsize=(8, 8))

    plt.plot(
        columns,
        rows,
        marker="o",
        linewidth=2,
        label="Ideal navigation path",
    )

    plt.scatter(
        start_state.position[1],
        start_state.position[0],
        marker="s",
        s=150,
        label="Start",
    )

    plt.scatter(
        target[1],
        target[0],
        marker="*",
        s=250,
        label="Target",
    )

    plt.scatter(
        state.position[1],
        state.position[0],
        marker="X",
        s=150,
        label="Final position",
    )

    plt.xlim(
        -0.5,
        grid.cols - 0.5,
    )

    plt.ylim(
        grid.rows - 0.5,
        -0.5,
    )

    plt.xticks(
        range(grid.cols)
    )

    plt.yticks(
        range(grid.rows)
    )

    plt.grid(True)
    plt.xlabel("Column")
    plt.ylabel("Row")

    status = (
        "Reached target"
        if state.position == target
        else "Failed"
    )

    plt.title(
        "Ideal Grid-Based Wheelchair Navigation\n"
        f"{status} | "
        f"Commands: {steps} | "
        f"Initial Manhattan distance: "
        f"{manhattan_distance(start_state.position, target)}"
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
