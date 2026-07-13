"""Shared spatial entity types used by the grid environment."""

from __future__ import annotations


class SpatialAgent:
    """Base class for named entities that can occupy and move on the grid."""

    is_human_controlled = False

    def __init__(self, name: str) -> None:
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("A spatial agent needs a non-empty name.")
        self.name = normalized
        self.name_number = normalized

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
