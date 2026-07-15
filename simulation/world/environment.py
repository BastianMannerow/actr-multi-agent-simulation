"""Indexed grid-world backends for virtual and ROS-connected simulations."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from simulation.world.entities import SpatialAgent, SpatialEntity, Target


class Environment:
    """Collision-aware virtual grid with O(1) agent lookup and revisions."""

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
        self.world_revision = 0
        self.static_revision = 0
        self._positions: dict[int, tuple[int, int]] = {}
        self._agents_by_name: dict[str, Any] = {}
        self._target_positions_cache: tuple[tuple[int, int], ...] | None = None
        self._render_dirty_cells: set[tuple[int, int]] = set()
        self._render_full_redraw = True
        self._rebuild_indices()
        self._update_gui()

    def _rebuild_indices(self) -> None:
        self._positions.clear()
        self._agents_by_name.clear()
        for row_index, row in enumerate(self.level_matrix):
            for column_index, cell in enumerate(row):
                for item in cell:
                    if isinstance(item, SpatialAgent) or getattr(item, "name", None):
                        self._positions[id(item)] = (row_index, column_index)
                        name = str(getattr(item, "name", ""))
                        if name:
                            self._agents_by_name[name] = item

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

    @property
    def agent_count(self) -> int:
        return len(self._positions)

    def agent_by_name(self, name: str) -> Any | None:
        return self._agents_by_name.get(str(name))

    def positioned_agents(self) -> tuple[tuple[Any, tuple[int, int]], ...]:
        """Return indexed dynamic occupants without scanning the matrix."""
        return tuple(
            (agent, position)
            for agent in self._agents_by_name.values()
            if (position := self._positions.get(id(agent))) is not None
        )

    def consume_render_changes(self) -> set[tuple[int, int]] | None:
        """Return changed cells, or ``None`` when a full redraw is required."""
        if self._render_full_redraw:
            self._render_full_redraw = False
            self._render_dirty_cells.clear()
            return None
        dirty = set(self._render_dirty_cells)
        self._render_dirty_cells.clear()
        return dirty

    def find_agent(self, agent: Any) -> Optional[Tuple[int, int]]:
        position = self._positions.get(id(agent))
        if position is not None:
            return position
        # Repair the index if external code edited the matrix directly.
        for row_index, row in enumerate(self.level_matrix):
            for column_index, cell in enumerate(row):
                if agent in cell:
                    position = (row_index, column_index)
                    self._positions[id(agent)] = position
                    name = str(getattr(agent, "name", ""))
                    if name:
                        self._agents_by_name[name] = agent
                    return position
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
        if self._target_positions_cache is None:
            self._target_positions_cache = tuple(
                (row, column)
                for row, values in enumerate(self.level_matrix)
                for column, cell in enumerate(values)
                if any(
                    isinstance(item, Target) or getattr(item, "is_target", False)
                    for item in cell
                )
            )
        return list(self._target_positions_cache)

    def mark_static_changed(self) -> None:
        """Invalidate terrain caches after a deliberate level mutation."""
        self.static_revision += 1
        self.world_revision += 1
        self._target_positions_cache = None
        self._render_full_redraw = True
        self._render_dirty_cells.clear()
        self._rebuild_indices()
        self._mark_perception_dirty(None, None, None)
        self._update_gui()

    def _mark_perception_dirty(
        self,
        moved_agent: Any | None,
        old_position: tuple[int, int] | None,
        new_position: tuple[int, int] | None,
    ) -> None:
        agents = list(getattr(self.simulation, "spatial_agents", ()) or ())
        for candidate in agents:
            marker = getattr(candidate, "mark_perception_dirty", None)
            if not callable(marker):
                continue
            if candidate is moved_agent or old_position is None or new_position is None:
                marker()
                continue
            candidate_position = self.find_agent(candidate)
            if candidate_position is None:
                marker()
                continue
            los = max(0, int(getattr(candidate, "los", 0)))
            rows = len(self.level_matrix)
            columns = len(self.level_matrix[0]) if self.level_matrix else 0
            if los == 0 or los >= rows or los >= columns:
                marker()
                continue
            if any(
                max(
                    abs(candidate_position[0] - position[0]),
                    abs(candidate_position[1] - position[1]),
                ) <= los
                for position in (old_position, new_position)
            ):
                marker()

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
            self._positions.pop(id(agent), None)
            return False
        self.level_matrix[next_row][next_column].append(agent)
        old_position = (row, column)
        new_position = (next_row, next_column)
        self._positions[id(agent)] = new_position
        self._render_dirty_cells.update((old_position, new_position))
        name = str(getattr(agent, "name", ""))
        if name:
            self._agents_by_name[name] = agent
        self.world_revision += 1
        self._mark_perception_dirty(agent, old_position, new_position)
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
        self._positions.pop(id(agent), None)
        if position is not None:
            self._render_dirty_cells.add(position)
        name = str(getattr(agent, "name", ""))
        if self._agents_by_name.get(name) is agent:
            self._agents_by_name.pop(name, None)
        self.world_revision += 1
        self._mark_perception_dirty(agent, position, position)
        self._update_gui()

    def set_gui(self, gui: Any) -> None:
        self.gui = gui
        self._update_gui()

    def close(self) -> None:
        return None
