from typing import Any, List, Optional, Tuple
from gui.ExampleGUI import ExampleGUI

class Environment:
    """
    Minimal grid-based environment for agent movement.

    Purpose
    -------
    - Provide a *simple* matrix world where agents occupy cells and can move
      by discrete steps (up, down, left, right).
    - Serve as a narrow interface that `Middleman.py` (or controller if human) can call
      to mediate agent ↔ environment communication.

    Scope
    -----
    - **No obstacles:** Walls/Water and other terrain types are intentionally
      omitted to keep this a minimal example.
    - **No GUI by default:** This class does not create or manage any GUI.
      If you need a visual environment, *initialize the GUI inside this class*
      and call `_update_gui()` at state changes. The hook is provided.

    Notes
    -----
    - Cell contents are lists of objects; multiple agents may share a cell if
      the simulation allows it.
    - `remove_agent_from_game` removes an agent and lets the simulation proceed.
      Use this to continue after an agent hits an internal error.
    """

    def __init__(self, level_matrix: List[List[Any]], gui: Optional[Any] = None) -> None:
        """
        Initialize the grid.

        Parameters
        ----------
        level_matrix : 2D list
            Matrix of cells. Each cell may be a single object or a list of objects.
        gui : Optional[Any]
            Optional GUI handle. If provided, `_update_gui()` will invoke `gui.update()`.
            For a visible environment, create and wire the GUI here.
        """
        # Normalize cells to lists to allow multiple occupants per cell.
        self.level_matrix: List[List[List[Any]]] = [
            [cell if isinstance(cell, list) else [cell] for cell in row]
            for row in level_matrix
        ]

        # Optional GUI; not required for headless simulations.
        self.gui = ExampleGUI(self, gui)
        self._update_gui()

    # -----------------------------
    # Core utilities
    # -----------------------------
    def _update_gui(self) -> None:
        """No-op unless a GUI object was provided. Call after state changes."""
        if self.gui and hasattr(self.gui, "update"):
            self.gui.update()

    def find_agent(self, agent: Any) -> Optional[Tuple[int, int]]:
        """
        Locate an agent within the matrix.

        Returns
        -------
        Optional[Tuple[int, int]]
            (row, col) if found, else None.
        """
        for r, row in enumerate(self.level_matrix):
            for c, cell in enumerate(row):
                if agent in cell:
                    return r, c
        return None

    # -----------------------------
    # Movement API
    # -----------------------------
    def move_agent(self, agent: Any, dr: int, dc: int) -> bool:
        """
        Move an agent by a delta (dr, dc) if the target is in bounds.

        Rules
        -----
        - Bounds-checked only. No terrain or collision semantics in this minimal version.
        - Multiple agents per cell are allowed.

        Returns
        -------
        bool
            True if the move succeeded, False otherwise.
        """
        pos = self.find_agent(agent)
        if pos is None:
            return False

        r, c = pos
        nr, nc = r + dr, c + dc

        # Bounds check
        if not (0 <= nr < len(self.level_matrix) and 0 <= nc < len(self.level_matrix[0])):
            return False

        # Perform move
        try:
            self.level_matrix[r][c].remove(agent)
        except ValueError:
            # Agent vanished between checks; treat as failed move.
            return False

        self.level_matrix[nr][nc].append(agent)

        # Hook for visuals / tracing
        self._update_gui()
        return True

    def move_agent_top(self, agent: Any) -> bool:
        """Move agent one cell up."""
        return self.move_agent(agent, -1, 0)

    def move_agent_bottom(self, agent: Any) -> bool:
        """Move agent one cell down."""
        return self.move_agent(agent, 1, 0)

    def move_agent_left(self, agent: Any) -> bool:
        """Move agent one cell left."""
        return self.move_agent(agent, 0, -1)

    def move_agent_right(self, agent: Any) -> bool:
        """Move agent one cell right."""
        return self.move_agent(agent, 0, 1)

    # -----------------------------
    # Lifecycle
    # -----------------------------
    def remove_agent_from_game(self, agent: Any) -> None:
        """
        Remove an agent from the grid and continue the simulation.

        Intent
        ------
        - Acts as a fail-safe to proceed even if an agent experiences an internal error.
        - Leaves other agents and the environment state intact.

        Behavior
        --------
        - Removes the first occurrence found.
        - Silent if the agent is not present.
        """
        pos = self.find_agent(agent)
        if pos is None:
            self._update_gui()
            return

        r, c = pos
        try:
            self.level_matrix[r][c].remove(agent)
        except ValueError:
            # Already removed or inconsistent state; ignore.
            pass

        self._update_gui()

    # -----------------------------
    # GUI wiring
    # -----------------------------
    def set_gui(self, gui: Any) -> None:
        """
        Attach or replace the GUI object.

        Expectation
        -----------
        - The GUI object provides an `update()` method.
        - Initialize and wire your visual frontend here if you want on-screen output.
        """
        self.gui = gui
        self._update_gui()
