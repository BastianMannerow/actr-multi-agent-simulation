import random
from typing import Any, List, Optional, Sequence, Tuple


def build_level(height: int, width: int, agents: Sequence[Any], rng: Optional[random.Random] = None) -> List[List[Optional[Any]]]:
    """
    Minimal grid builder: randomly place agents on an empty matrix.

    Intent
    ------
    - Provide a minimal example.
    - Fail fast if the grid cannot accommodate all agents.
    - Keep the API surface small and explicit.

    Parameters
    ----------
    height : int
        Number of rows in the matrix (must be > 0).
    width : int
        Number of columns in the matrix (must be > 0).
    agents : Sequence[Any]
        Agent objects to place. Each agent occupies exactly one cell.
    rng : Optional[random.Random]
        Optional RNG for deterministic placements in tests. Defaults to `random`.

    Returns
    -------
    List[List[Optional[Any]]]
        A `height × width` matrix. Cells contain either `None` or a single agent.

    Raises
    ------
    ValueError
        - If `height` or `width` is non-positive.
        - If `len(agents) > height * width` (not enough free cells).
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")

    total_cells = height * width
    num_agents = len(agents)
    if num_agents > total_cells:
        raise ValueError(
            f"Not enough space: {num_agents} agents for {total_cells} cells "
            f"({height}×{width})"
        )

    # Create an empty matrix
    matrix: List[List[Optional[Any]]] = [[None for _ in range(width)] for _ in range(height)]

    # Generate a shuffled list of all cell coordinates; take the first N for agents
    rng = rng or random
    coords: List[Tuple[int, int]] = [(r, c) for r in range(height) for c in range(width)]
    rng.shuffle(coords)

    for agent, (r, c) in zip(agents, coords[:num_agents]):
        matrix[r][c] = agent

    return matrix
