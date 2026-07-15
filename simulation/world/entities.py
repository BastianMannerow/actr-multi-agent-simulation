"""Spatial entities for the virtual matrix demonstration."""

from __future__ import annotations


class SpatialEntity:
    """Base class for every object that can occupy a matrix cell."""

    display_name = "Entity"
    symbol = "?"
    blocks_movement = False
    is_target = False

    def __repr__(self) -> str:
        return type(self).__name__ + "()"


class SpatialAgent(SpatialEntity):
    """Base class for ACT-R and human-controlled agents."""

    is_human_controlled = False
    symbol = "A"

    def __init__(self, name: str) -> None:
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("A spatial agent needs a non-empty name.")
        self.name = normalized
        self.name_number = normalized
        self.display_name = normalized

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class Wall(SpatialEntity):
    """A solid matrix obstacle."""

    display_name = "Wall"
    symbol = "X"
    blocks_movement = True


class Goal(SpatialEntity):
    """A non-blocking destination marker."""

    display_name = "Goal"
    symbol = "G"
    blocks_movement = False
    is_target = True


class Checkpoint(SpatialEntity):
    """A non-blocking observation marker used by the demo environment."""

    display_name = "Checkpoint"
    symbol = "C"
    blocks_movement = False


# Compatibility aliases used by generic inspection and older demo agents.
Target = Goal
DefinitelyAWall = Wall
FakeWall = Wall
BurningTree = Wall
FireTarget = Goal
