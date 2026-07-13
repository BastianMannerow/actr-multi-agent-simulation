"""Communication layer between ACT-R agents and the simple matrix world."""

from __future__ import annotations

from typing import Any

from simulation.runtime.agent_construct import AgentConstruct
from simulation.world.entities import SpatialAgent


class Middleman:
    """Translate matrix state into pyactr-safe perception and motor actions."""

    def __init__(self, simulation: Any, print_middleman: bool):
        self.simulation = simulation
        self.experiment_environment = None
        self.print_middleman = print_middleman

    def set_game_environment(self, experiment_environment: Any) -> None:
        self.experiment_environment = experiment_environment

    def motor_input(self, key: str, current_agent: AgentConstruct) -> bool:
        """Apply a W/A/S/D command to the simple matrix environment."""
        if self.experiment_environment is None:
            return False
        movement = {
            "W": self.experiment_environment.move_agent_top,
            "A": self.experiment_environment.move_agent_left,
            "S": self.experiment_environment.move_agent_bottom,
            "D": self.experiment_environment.move_agent_right,
        }.get(str(key).upper())
        if movement is None:
            return False
        moved = bool(movement(current_agent))
        if self.print_middleman:
            print(
                f"{current_agent.name}: motor {key} -> "
                f"{'accepted' if moved else 'blocked'}"
            )
        return moved

    def get_agent_stimulus(self, agent: AgentConstruct):
        """Build the current visual frame for one ACT-R agent.

        Only fields understood by pyactr are placed in the stimulus frame:
        ``text``, ``position`` and optionally ``vis_delay``. Application data
        such as class names and matrix/view coordinates is stored separately in
        ``agent.visual_metadata`` and can therefore never become an invalid
        pyactr chunk slot.

        The position tuple follows pyactr's screen-coordinate convention
        ``(x, y)``. For the matrix this means ``(column, row)``.
        """
        environment = self.experiment_environment
        if environment is None:
            agent.visual_stimuli = []
            agent.visual_metadata = {}
            return [set()], [{}]

        matrix = environment.level_matrix
        position = environment.find_agent(agent)
        if position is None or not matrix or not matrix[0]:
            agent.visual_stimuli = []
            agent.visual_metadata = {}
            return [set()], [{}]

        row, column = position
        agent_map = agent.get_agent_dictionary()
        line_of_sight = int(agent.los)
        rows, columns = len(matrix), len(matrix[0])

        if line_of_sight == 0 or line_of_sight >= max(rows, columns):
            window_width, window_height = columns, rows
            offset_x, offset_y = column, row
        else:
            window_width = window_height = 2 * line_of_sight + 1
            offset_x = offset_y = line_of_sight

        trigger_symbols: set[str] = set()
        frame: dict[str, dict[str, Any]] = {}
        metadata: dict[str, dict[str, Any]] = {}
        visible_matrix = [
            ["-" for _ in range(window_width)]
            for _ in range(window_height)
        ]

        for view_row in range(window_height):
            for view_column in range(window_width):
                matrix_row = row - offset_y + view_row
                matrix_column = column - offset_x + view_column
                if not (0 <= matrix_row < rows and 0 <= matrix_column < columns):
                    continue

                cell = matrix[matrix_row][matrix_column]
                if not cell:
                    continue

                symbols: list[str] = []
                for object_index, element in enumerate(cell):
                    symbol = self._symbol_for(element, agent_map)
                    if symbol is None:
                        continue

                    symbols.append(symbol)
                    trigger_symbols.add(symbol)
                    stimulus_id = (
                        f"r{matrix_row}_c{matrix_column}_i{object_index}_"
                        f"{type(element).__name__}"
                    )
                    frame[stimulus_id] = {
                        "text": symbol,
                        "position": (matrix_column, matrix_row),
                    }
                    metadata[stimulus_id] = {
                        "entity_class": type(element).__name__,
                        "display_name": str(
                            getattr(element, "name", type(element).__name__)
                        ),
                        "matrix_position": (matrix_row, matrix_column),
                        "view_position": (view_row, view_column),
                        "is_human_controlled": bool(
                            getattr(element, "is_human_controlled", False)
                        ),
                    }

                visible_matrix[view_row][view_column] = "".join(symbols) or "-"

        agent.visual_stimuli = visible_matrix
        agent.visual_metadata = metadata

        # pyactr expects exactly one trigger collection for one visual frame.
        # A list of strings would duplicate the same frame once per symbol.
        return [trigger_symbols], [frame]

    @staticmethod
    def _symbol_for(
        element: Any,
        agent_map: dict[str, dict[str, Any]],
    ) -> str | None:
        if isinstance(element, SpatialAgent):
            for candidate, info in agent_map.items():
                if info.get("agent") is element:
                    return str(candidate)
            return str(getattr(element, "symbol", "A"))
        symbol = getattr(element, "symbol", None)
        return str(symbol) if symbol is not None else None
