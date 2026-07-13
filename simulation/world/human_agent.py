"""Human-controlled spatial agent."""

from __future__ import annotations

from simulation.world.entities import SpatialAgent


class HumanAgent(SpatialAgent):
    """Grid entity moved directly through keyboard input rather than ACT-R events."""

    is_human_controlled = True
    actr_agent_type_name = "Human"

    def __init__(self, name: str) -> None:
        super().__init__(name)
