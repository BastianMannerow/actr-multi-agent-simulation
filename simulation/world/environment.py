"""Collision-aware backend for the virtual matrix demonstration."""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from simulation.world.entities import SpatialEntity, Target


class Environment:
    """Collision-aware virtual grid environment."""

    backend_name = "virtual"

    def __init__(
        self,
        level_matrix: List[List[Any]],
        gui: Optional[Any] = None,
        *,
        simulation: Any | None = None,
    ) -> None:
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
        self.simulation = simulation
        self._update_gui()

    def _update_gui(self) -> None:
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

    def objects_at(self, row: int, column: int) -> list[Any]:
        if not (
            0 <= row < len(self.level_matrix)
            and self.level_matrix
            and 0 <= column < len(self.level_matrix[0])
        ):
            return []
        return list(self.level_matrix[row][column])

    def is_blocked(self, row: int, column: int) -> bool:
        return any(
            bool(getattr(item, "blocks_movement", False))
            for item in self.objects_at(row, column)
        )

    def target_positions(self) -> list[tuple[int, int]]:
        return [
            (row, column)
            for row, values in enumerate(self.level_matrix)
            for column, cell in enumerate(values)
            if any(isinstance(item, Target) or getattr(item, "is_target", False) for item in cell)
        ]

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
            self.register_bumping(agent, reason="boundary")
            return False
        if self.is_blocked(next_row, next_column):
            self.register_bumping(agent, reason="obstacle")
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

    def register_bumping(self, agent: Any, *, reason: str = "obstacle") -> None:
        middleman = getattr(agent, "middleman", None)
        detect = getattr(middleman, "detect_bump", None)
        if callable(detect):
            detect(agent, reason=reason)

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

    def close(self) -> None:
        return None
