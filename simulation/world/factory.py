"""Create the virtual matrix backend."""

from __future__ import annotations

from typing import Any

from simulation.world.environment import Environment
from simulation.world.level_builder import build_level


def create_environment(config: Any, agents: list[Any], simulation: Any) -> Environment:
    matrix = build_level(config.virtual_level, agents)
    environment = Environment(matrix, simulation=simulation)
    level_labels = {"demo_matrix": "Demo Matrix"}
    environment.level_type = config.virtual_level
    environment.level_name = level_labels.get(config.virtual_level, config.virtual_level)
    return environment
