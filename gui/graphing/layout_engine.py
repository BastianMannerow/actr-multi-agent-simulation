"""Shared deterministic layered layout for all directed graph views.

The engine is a reusable facade over the bounded Sugiyama-style placement and
orthogonal channel router. State graphs, declarative-memory diagrams and future
node/edge views therefore share the same port allocation, feedback channels,
minimum lane spacing and runtime characteristics.
"""

from __future__ import annotations

from collections.abc import Iterable

from PyQt6.QtCore import QPointF, QRectF

from gui.graph_layout import LayoutEdge, LayoutNode, LayoutResult
from gui.ux_graph_layout import (
    _ConcreteStateGraphUXLayout,
    normalize_route_lanes,
)
from gui.graphing.models import DiagramEdge, DiagramNode


class _RankAwareLayeredLayout(_ConcreteStateGraphUXLayout):
    def __init__(self, node_by_id, *, rank_hints=None, **kwargs) -> None:
        super().__init__(node_by_id, **kwargs)
        self._rank_hints = {
            str(node_id): max(0, int(rank))
            for node_id, rank in (rank_hints or {}).items()
        }

    def _assign_ranks(self, nodes, edges, initial_node_id):
        ranks = super()._assign_ranks(nodes, edges, initial_node_id)
        ranks.update(
            {
                node_id: rank
                for node_id, rank in self._rank_hints.items()
                if any(node.node_id == node_id for node in nodes)
            }
        )
        return ranks


class ReusableLayeredGraphLayout:
    """One layout API for state, memory and other directed diagrams."""

    def __init__(
        self,
        *,
        minimum_lane_gap: float = 20.0,
        node_gap: float = 72.0,
        rank_gap: float = 112.0,
        port_stub: float = 18.0,
    ) -> None:
        self.minimum_lane_gap = max(16.0, float(minimum_lane_gap))
        self.node_gap = float(node_gap)
        self.rank_gap = float(rank_gap)
        self.port_stub = float(port_stub)

    def layout(
        self,
        nodes: Iterable[DiagramNode] | Iterable[LayoutNode],
        edges: Iterable[DiagramEdge] | Iterable[LayoutEdge],
        *,
        initial_node_id: str | None,
        rank_hints: dict[str, int] | None = None,
        offset: QPointF = QPointF(42.0, 150.0),
    ) -> LayoutResult:
        node_values = list(nodes)
        edge_values = list(edges)
        layout_nodes = [self._layout_node(node) for node in node_values]
        layout_edges = [self._layout_edge(edge) for edge in edge_values]
        if not layout_nodes:
            return LayoutResult({}, {}, "vertical", QRectF(), {})  # pragma: no cover
        initial = initial_node_id or layout_nodes[0].node_id
        node_by_id = {node.node_id: node for node in layout_nodes}
        result = _RankAwareLayeredLayout(
            node_by_id,
            rank_hints=rank_hints,
            node_gap=self.node_gap,
            base_rank_gap=self.rank_gap,
            lane_gap=max(22.0, self.minimum_lane_gap + 2.0),
            port_stub=self.port_stub,
        ).layout(
            layout_nodes,
            layout_edges,
            initial_node_id=initial,
            offset=offset,
        )
        return normalize_route_lanes(
            result,
            minimum_gap=self.minimum_lane_gap,
        )

    @staticmethod
    def _layout_node(node) -> LayoutNode:
        if isinstance(node, LayoutNode):
            return node
        return LayoutNode(
            node_id=node.node_id,
            label=node.title,
            group=node.group,
            width=node.width,
            height=node.height,
            priority=node.priority,
        )

    @staticmethod
    def _layout_edge(edge) -> LayoutEdge:
        if isinstance(edge, LayoutEdge):
            return edge
        return LayoutEdge(
            edge_id=edge.edge_id,
            source_id=edge.source_id,
            target_id=edge.target_id,
            kind=edge.kind,
            weight=edge.weight,
        )
