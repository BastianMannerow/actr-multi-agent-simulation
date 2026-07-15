"""Deterministic virtual matrix used by the feature-complete demo GUI."""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence

from simulation.world.entities import Checkpoint, Goal, Wall


LEVEL_DIMENSIONS: dict[str, tuple[int, int]] = {
    "demo_matrix": (12, 16),  # (height, width)
}


def level_dimensions(level_type: str) -> tuple[int, int]:
    try:
        return LEVEL_DIMENSIONS[level_type]
    except KeyError as exc:
        raise ValueError(f"Unknown virtual level: {level_type!r}") from exc


def build_level(
    level_type: str,
    agents: Sequence[Any],
    rng: Optional[random.Random] = None,
) -> list[list[Any | None]]:
    if level_type != "demo_matrix":
        raise ValueError(f"Unknown virtual level: {level_type!r}")
    return _demo_matrix(agents, rng or random.Random())


def _demo_matrix(
    agents: Sequence[Any], rng: random.Random
) -> list[list[Any | None]]:
    height, width = level_dimensions("demo_matrix")
    matrix: list[list[Any | None]] = [
        [None for _ in range(width)] for _ in range(height)
    ]

    walls: set[tuple[int, int]] = set()
    for row in range(height):
        walls.add((row, 0))
        walls.add((row, width - 1))
    for column in range(width):
        walls.add((0, column))
        walls.add((height - 1, column))

    # Three compact barriers with deliberate gaps. The layout remains easy to
    # read while supporting collisions, partial perception and route changes.
    walls.update((row, 5) for row in range(2, 10) if row != 6)
    walls.update((4, column) for column in range(7, 14) if column != 10)
    walls.update((8, column) for column in range(8, 15) if column != 12)
    for row, column in walls:
        matrix[row][column] = Wall()

    goal_position = (2, 13)
    matrix[goal_position[0]][goal_position[1]] = Goal()
    for row, column in ((2, 2), (6, 10), (9, 13)):
        matrix[row][column] = Checkpoint()

    free_cells = [
        (row, column)
        for row in range(1, height - 1)
        for column in range(1, width - 1)
        if matrix[row][column] is None
    ]
    rng.shuffle(free_cells)
    if len(free_cells) < len(agents):
        raise ValueError("Demo Matrix has too few free cells for all agents.")
    for agent, (row, column) in zip(agents, free_cells):
        matrix[row][column] = agent
    return matrix
