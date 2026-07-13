from typing import Any, List, Optional, Tuple


class Environment:
    """Minimal grid environment with an optional presentation adapter."""

    def __init__(self, level_matrix: List[List[Any]], gui: Optional[Any] = None) -> None:
        self.level_matrix: List[List[List[Any]]] = [
            [
                cell
                if isinstance(cell, list)
                else ([] if cell is None else [cell])
                for cell in row
            ]
            for row in level_matrix
        ]
        self.gui = gui
        self._update_gui()

    def _update_gui(self) -> None:
        """Notify the attached presentation adapter after state changes."""
        if self.gui is None:
            return
        refresh = getattr(self.gui, "refresh", None)
        if callable(refresh):
            refresh()
            return
        update = getattr(self.gui, "update", None)
        if callable(update):
            update()

    def find_agent(self, agent: Any) -> Optional[Tuple[int, int]]:
        for row_index, row in enumerate(self.level_matrix):
            for column_index, cell in enumerate(row):
                if agent in cell:
                    return row_index, column_index
        return None

    def move_agent(self, agent: Any, dr: int, dc: int) -> bool:
        position = self.find_agent(agent)
        if position is None or not self.level_matrix or not self.level_matrix[0]:
            return False

        row, column = position
        next_row, next_column = row + dr, column + dc
        if not (
            0 <= next_row < len(self.level_matrix)
            and 0 <= next_column < len(self.level_matrix[0])
        ):
            return False

        try:
            self.level_matrix[row][column].remove(agent)
        except ValueError:
            return False
        self.level_matrix[next_row][next_column].append(agent)
        self._update_gui()
        return True

    def move_agent_top(self, agent: Any) -> bool:
        return self.move_agent(agent, -1, 0)

    def move_agent_bottom(self, agent: Any) -> bool:
        return self.move_agent(agent, 1, 0)

    def move_agent_left(self, agent: Any) -> bool:
        return self.move_agent(agent, 0, -1)

    def move_agent_right(self, agent: Any) -> bool:
        return self.move_agent(agent, 0, 1)

    def remove_agent_from_game(self, agent: Any) -> None:
        position = self.find_agent(agent)
        if position is not None:
            row, column = position
            try:
                self.level_matrix[row][column].remove(agent)
            except ValueError:
                pass
        self._update_gui()

    def set_gui(self, gui: Any) -> None:
        self.gui = gui
        self._update_gui()
