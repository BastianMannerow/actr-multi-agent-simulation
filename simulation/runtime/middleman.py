"""Communication layer between ACT-R agents and world backends."""

from __future__ import annotations

from typing import Any

from simulation.runtime.agent_construct import AgentConstruct
from simulation.world.entities import SpatialAgent, SpatialEntity


class Middleman:
    """Translate world state into pyactr-compatible perceptual input."""

    def __init__(self, simulation: Any, print_middleman: bool):
        self.simulation = simulation
        self.experiment_environment = None
        self.print_middleman = print_middleman

    def set_game_environment(self, experiment_environment: Any) -> None:
        self.experiment_environment = experiment_environment

    def motor_input(self, key: str, current_agent: AgentConstruct) -> bool:
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
        """Return a live visual frame using only pyactr-supported fields.

        pyactr treats all keys besides ``text``, ``position`` and
        ``vis_delay`` as chunk slots. Arbitrary application metadata therefore
        must not be included in the stimulus dictionary. Rich metadata is kept
        separately on ``agent.visual_metadata`` for GUI/debug use.
        """
        environment = self.experiment_environment
        if environment is None:
            return [set()], [{}]
        matrix = environment.level_matrix
        position = environment.find_agent(agent)
        if position is None:
            return [set()], [{}]

        row, column = position
        agent_map = agent.get_agent_dictionary()
        los = agent.los
        rows, columns = len(matrix), len(matrix[0])
        if los == 0 or los >= rows or los >= columns:
            window_width, window_height = columns, rows
            offset_x, offset_y = column, row
        else:
            window_width = window_height = 2 * los + 1
            offset_x = offset_y = los

        trigger_symbols: set[str] = set()
        frame: dict[str, dict[str, Any]] = {}
        metadata: dict[str, dict[str, Any]] = {}
        visual_stimuli = [
            ["-" for _ in range(window_width)]
            for _ in range(window_height)
        ]
        frame_origin = (row - offset_y, column - offset_x)
        valid_positions: set[tuple[int, int]] = set()

        for view_row in range(window_height):
            for view_column in range(window_width):
                matrix_row = row - offset_y + view_row
                matrix_column = column - offset_x + view_column
                if not (0 <= matrix_row < rows and 0 <= matrix_column < columns):
                    continue
                valid_positions.add((matrix_row, matrix_column))
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

                    # pyactr expects (screen_x, screen_y), hence column first.
                    frame[stimulus_id] = {
                        "text": symbol,
                        "position": (matrix_column, matrix_row),
                    }
                    metadata[stimulus_id] = {
                        "entity_class": type(element).__name__,
                        "display_name": str(
                            getattr(
                                element,
                                "display_name",
                                getattr(element, "name", type(element).__name__),
                            )
                        ),
                        "matrix_position": (matrix_row, matrix_column),
                        "view_position": (view_row, view_column),
                        "blocks_movement": bool(
                            getattr(element, "blocks_movement", False)
                        ),
                        "is_target": bool(getattr(element, "is_target", False)),
                    }

                visual_stimuli[view_row][view_column] = "".join(symbols) or "-"

        agent.visual_stimuli = visual_stimuli
        agent.visual_metadata = metadata
        agent.visual_frame_origin = frame_origin
        agent.visual_frame_valid_positions = valid_positions

        # One frame must have exactly one trigger collection. Returning a list
        # of individual strings would make pyactr duplicate the same frame once
        # per trigger.
        return [trigger_symbols], [frame]

    @staticmethod
    def _symbol_for(element: Any, agent_map: dict[str, Any]) -> str | None:
        if isinstance(element, SpatialAgent):
            for candidate, info in agent_map.items():
                if info["agent"] is element:
                    return str(candidate)
            return str(getattr(element, "symbol", "A"))
        if isinstance(element, SpatialEntity):
            return str(getattr(element, "symbol", "?"))
        if getattr(element, "symbol", None):
            return str(element.symbol)
        return None

    def detect_bump(self, agent: AgentConstruct, *, reason: str = "obstacle") -> None:
        adapter = getattr(agent, "actr_adapter", None)
        callback = getattr(adapter, "on_bump_detected", None)
        if callable(callback):
            callback(reason=reason)
        if self.print_middleman:
            print(f"{agent.name}: bump detected ({reason})")
