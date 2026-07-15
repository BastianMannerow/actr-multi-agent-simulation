"""Renderer-independent models used by all node/edge graph views."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DiagramNode:
    node_id: str
    title: str
    group: str = ""
    stereotype: str = ""
    attributes: tuple[tuple[str, str], ...] = ()
    width: float = 300.0
    height: float = 84.0
    priority: int = 0
    fill: str = "#334155"
    border: str = "#cbd5e1"
    kind: str = "card"
    tooltip: str = ""


@dataclass(frozen=True, slots=True)
class DiagramEdge:
    edge_id: str
    source_id: str
    target_id: str
    label: str = ""
    kind: str = "reference"
    color: str = "#38bdf8"
    dashed: bool = False
    weight: float = 1.0
    visible: bool = True
    tooltip: str = ""


@dataclass(slots=True)
class DiagramModel:
    title: str
    nodes: list[DiagramNode] = field(default_factory=list)
    edges: list[DiagramEdge] = field(default_factory=list)
    initial_node_id: str | None = None
    rank_hints: dict[str, int] = field(default_factory=dict)
