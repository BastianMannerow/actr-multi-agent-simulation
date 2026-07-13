"""Compact graph placement and obstacle-aware orthogonal edge routing.

The module is intentionally independent from QGraphicsScene.  It receives plain
nodes and edges and returns QRectF/QPointF geometry that the renderer can draw.
The implementation combines a phase-aware layered placement with an orthogonal
A* router over a compressed visibility grid.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable, Literal

from PyQt6.QtCore import QPointF, QRectF


Side = Literal["left", "right", "top", "bottom"]
Orientation = Literal["horizontal", "vertical", "folded"]
Direction = Literal["H", "V", "N"]


@dataclass(frozen=True, slots=True)
class LayoutNode:
    node_id: str
    label: str
    group: str
    width: float = 240.0
    height: float = 72.0
    priority: int = 0


@dataclass(frozen=True, slots=True)
class LayoutEdge:
    edge_id: str
    source_id: str
    target_id: str
    kind: str = "production"
    weight: float = 1.0


@dataclass(slots=True)
class NodePlacement:
    node: LayoutNode
    rect: QRectF
    rank: int
    group_order: int


@dataclass(slots=True)
class RoutedEdge:
    edge: LayoutEdge
    points: list[QPointF]
    source_side: Side
    target_side: Side
    label_position: QPointF


@dataclass(slots=True)
class LayoutResult:
    placements: dict[str, NodePlacement]
    routes: dict[str, RoutedEdge]
    orientation: Orientation
    bounds: QRectF
    group_headers: dict[str, QPointF] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _PortRequest:
    edge_id: str
    node_id: str
    other_id: str
    side: Side
    is_source: bool


@dataclass(slots=True)
class _Port:
    border: QPointF
    external: QPointF
    side: Side


class CompactLayeredLayout:
    """Place state nodes close to their graph neighbours.

    Placement is performed at phase/group level first.  The directed group graph
    is ranked from the initial group with shortest-path ranks; feedback edges are
    allowed and later routed around obstacles.  Three candidate arrangements are
    scored: horizontal, vertical, and serpentine folded.  The score combines area,
    aspect-ratio quality, total edge length, estimated crossings, and group spread.
    """

    def __init__(
        self,
        *,
        node_gap: float = 34.0,
        group_gap: float = 96.0,
        rank_gap: float = 150.0,
        header_height: float = 34.0,
        target_aspect: float = 1.65,
    ) -> None:
        self.node_gap = node_gap
        self.group_gap = group_gap
        self.rank_gap = rank_gap
        self.header_height = header_height
        self.target_aspect = target_aspect

    def place(
        self,
        nodes: Iterable[LayoutNode],
        edges: Iterable[LayoutEdge],
        *,
        initial_node_id: str,
        offset: QPointF = QPointF(40.0, 150.0),
    ) -> tuple[dict[str, NodePlacement], dict[str, QPointF], Orientation]:
        node_list = list(nodes)
        edge_list = list(edges)
        if not node_list:
            return {}, {}, "horizontal"
        node_by_id = {node.node_id: node for node in node_list}
        groups: dict[str, list[LayoutNode]] = defaultdict(list)
        for node in node_list:
            groups[node.group].append(node)

        initial_group = node_by_id.get(initial_node_id, node_list[0]).group
        group_edges = self._group_edges(edge_list, node_by_id)
        ordered_nodes = self._order_nodes_within_groups(
            groups, edge_list, node_by_id, initial_node_id
        )
        group_sizes = self._group_sizes(ordered_nodes)
        (
            component_groups,
            component_of,
            component_sizes,
            component_edges,
            component_ranks,
            local_offsets,
            initial_component,
        ) = self._component_model(
            list(groups), group_edges, group_sizes, initial_group
        )
        ranks = {
            group: component_ranks[component_of[group]]
            for group in groups
        }

        candidates: list[
            tuple[
                Orientation,
                dict[str, QPointF],
                dict[str, NodePlacement],
                dict[str, QPointF],
            ]
        ] = []
        for orientation in ("horizontal", "vertical", "folded"):
            component_origins = self._place_group_origins(
                groups=component_groups,
                ranks=component_ranks,
                sizes=component_sizes,
                group_edges=component_edges,
                orientation=orientation,
                initial_group=initial_component,
            )
            origins = {
                group: component_origins[component_of[group]] + local_offsets[group]
                for group in groups
            }
            placements, headers = self._materialize(
                ordered_nodes,
                group_sizes,
                origins,
                ranks,
                offset,
            )
            candidates.append((orientation, origins, placements, headers))

        orientation, _, placements, headers = min(
            candidates,
            key=lambda candidate: self._score(
                candidate[2], edge_list, candidate[0]
            ),
        )
        return placements, headers, orientation

    def _component_model(
        self,
        groups: list[str],
        group_edges: dict[tuple[str, str], float],
        group_sizes: dict[str, tuple[float, float]],
        initial_group: str,
    ) -> tuple[
        list[str],
        dict[str, str],
        dict[str, tuple[float, float]],
        dict[tuple[str, str], float],
        dict[str, int],
        dict[str, QPointF],
        str,
    ]:
        """Collapse cyclic phase regions and arrange each SCC as a compact block."""
        components = self._strongly_connected_components(groups, group_edges)
        component_of: dict[str, str] = {}
        component_groups: list[str] = []
        for index, component in enumerate(components):
            component_id = f"component:{index}"
            component_groups.append(component_id)
            for group in component:
                component_of[group] = component_id

        component_edges: dict[tuple[str, str], float] = defaultdict(float)
        for (source, target), weight in group_edges.items():
            source_component = component_of[source]
            target_component = component_of[target]
            if source_component != target_component:
                component_edges[(source_component, target_component)] += weight

        local_offsets: dict[str, QPointF] = {}
        component_sizes: dict[str, tuple[float, float]] = {}
        incoming_external: dict[str, float] = defaultdict(float)
        for (source, target), weight in group_edges.items():
            if component_of[source] != component_of[target]:
                incoming_external[target] += weight

        for index, component in enumerate(components):
            component_id = f"component:{index}"
            ordered = self._order_component_cycle(
                component, group_edges, incoming_external, initial_group
            )
            offsets, size = self._pack_component_groups(ordered, group_sizes)
            local_offsets.update(offsets)
            component_sizes[component_id] = size

        initial_component = component_of[initial_group]
        rank_groups = {component_id: [] for component_id in component_groups}
        component_ranks = self._group_ranks(
            rank_groups, dict(component_edges), initial_component
        )
        return (
            component_groups,
            component_of,
            component_sizes,
            dict(component_edges),
            component_ranks,
            local_offsets,
            initial_component,
        )

    @staticmethod
    def _strongly_connected_components(
        groups: list[str],
        group_edges: dict[tuple[str, str], float],
    ) -> list[list[str]]:
        outgoing: dict[str, list[str]] = defaultdict(list)
        for source, target in group_edges:
            outgoing[source].append(target)
        index = 0
        stack: list[str] = []
        on_stack: set[str] = set()
        indices: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        result: list[list[str]] = []

        def visit(node: str) -> None:
            nonlocal index
            indices[node] = index
            lowlink[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)
            for target in outgoing.get(node, []):
                if target not in indices:
                    visit(target)
                    lowlink[node] = min(lowlink[node], lowlink[target])
                elif target in on_stack:
                    lowlink[node] = min(lowlink[node], indices[target])
            if lowlink[node] == indices[node]:
                component: list[str] = []
                while stack:
                    value = stack.pop()
                    on_stack.remove(value)
                    component.append(value)
                    if value == node:
                        break
                result.append(sorted(component))

        for group in sorted(groups):
            if group not in indices:
                visit(group)
        return result

    @staticmethod
    def _order_component_cycle(
        component: list[str],
        group_edges: dict[tuple[str, str], float],
        incoming_external: dict[str, float],
        initial_group: str,
    ) -> list[str]:
        if len(component) <= 1:
            return list(component)
        remaining = set(component)
        start = max(
            component,
            key=lambda group: (
                1 if group == initial_group else 0,
                incoming_external.get(group, 0.0),
                sum(
                    weight
                    for (source, target), weight in group_edges.items()
                    if source == group and target in remaining
                ),
                group,
            ),
        )
        order = [start]
        remaining.remove(start)
        while remaining:
            current = order[-1]
            nxt = max(
                remaining,
                key=lambda group: (
                    group_edges.get((current, group), 0.0),
                    group_edges.get((group, current), 0.0),
                    incoming_external.get(group, 0.0),
                    group,
                ),
            )
            order.append(nxt)
            remaining.remove(nxt)
        return order

    def _pack_component_groups(
        self,
        groups: list[str],
        sizes: dict[str, tuple[float, float]],
    ) -> tuple[dict[str, QPointF], tuple[float, float]]:
        if len(groups) == 1:
            group = groups[0]
            return {group: QPointF(0.0, 0.0)}, sizes[group]
        columns = 2 if len(groups) <= 4 else math.ceil(math.sqrt(len(groups)))
        rows = math.ceil(len(groups) / columns)
        # Perimeter order preserves the dominant cycle: top-left -> top-right ->
        # bottom-right -> bottom-left. Larger SCCs continue in a serpentine grid.
        cells: list[tuple[int, int]] = []
        if len(groups) == 4 and columns == 2:
            cells = [(0, 0), (1, 0), (1, 1), (0, 1)]
        else:
            for row in range(rows):
                values = range(columns) if row % 2 == 0 else reversed(range(columns))
                for column in values:
                    cells.append((column, row))
        column_widths = [0.0] * columns
        row_heights = [0.0] * rows
        for group, (column, row) in zip(groups, cells):
            width, height = sizes[group]
            column_widths[column] = max(column_widths[column], width)
            row_heights[row] = max(row_heights[row], height)
        x_positions = [0.0]
        y_positions = [0.0]
        for width in column_widths[:-1]:
            x_positions.append(x_positions[-1] + width + self.group_gap)
        for height in row_heights[:-1]:
            y_positions.append(y_positions[-1] + height + self.group_gap)
        offsets: dict[str, QPointF] = {}
        for group, (column, row) in zip(groups, cells):
            width, height = sizes[group]
            offsets[group] = QPointF(
                x_positions[column] + (column_widths[column] - width) / 2.0,
                y_positions[row] + (row_heights[row] - height) / 2.0,
            )
        total_width = sum(column_widths) + self.group_gap * max(0, columns - 1)
        total_height = sum(row_heights) + self.group_gap * max(0, rows - 1)
        return offsets, (total_width, total_height)

    @staticmethod
    def _group_edges(
        edges: list[LayoutEdge], node_by_id: dict[str, LayoutNode]
    ) -> dict[tuple[str, str], float]:
        result: dict[tuple[str, str], float] = defaultdict(float)
        for edge in edges:
            source = node_by_id.get(edge.source_id)
            target = node_by_id.get(edge.target_id)
            if source is None or target is None or source.group == target.group:
                continue
            result[(source.group, target.group)] += max(0.25, edge.weight)
        return dict(result)

    @staticmethod
    def _group_ranks(
        groups: dict[str, list[LayoutNode]],
        group_edges: dict[tuple[str, str], float],
        initial_group: str,
    ) -> dict[str, int]:
        outgoing: dict[str, set[str]] = defaultdict(set)
        for source, target in group_edges:
            outgoing[source].add(target)
        ranks = {initial_group: 0}
        queue: deque[str] = deque([initial_group])
        while queue:
            source = queue.popleft()
            for target in sorted(outgoing.get(source, set())):
                candidate = ranks[source] + 1
                if target not in ranks or candidate < ranks[target]:
                    ranks[target] = candidate
                    queue.append(target)
        next_rank = max(ranks.values(), default=-1) + 1
        for group in sorted(groups):
            if group not in ranks:
                ranks[group] = next_rank
                next_rank += 1
        return ranks

    def _order_nodes_within_groups(
        self,
        groups: dict[str, list[LayoutNode]],
        edges: list[LayoutEdge],
        node_by_id: dict[str, LayoutNode],
        initial_node_id: str,
    ) -> dict[str, list[LayoutNode]]:
        neighbours: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            neighbours[edge.source_id].append(edge.target_id)
            neighbours[edge.target_id].append(edge.source_id)

        ordered: dict[str, list[LayoutNode]] = {
            group: sorted(
                values,
                key=lambda node: (
                    0 if node.node_id == initial_node_id else 1,
                    -node.priority,
                    node.label.casefold(),
                ),
            )
            for group, values in groups.items()
        }
        positions = {
            node.node_id: float(index)
            for values in ordered.values()
            for index, node in enumerate(values)
        }
        # Barycentric sweeps reduce crossings between adjacent groups without
        # destroying deterministic ordering when no useful neighbour exists.
        for _ in range(8):
            for group in sorted(ordered):
                values = ordered[group]
                decorated = []
                for index, node in enumerate(values):
                    neighbour_positions = [
                        positions[other]
                        for other in neighbours.get(node.node_id, [])
                        if other in positions
                        and node_by_id.get(other) is not None
                        and node_by_id[other].group != group
                    ]
                    barycenter = (
                        sum(neighbour_positions) / len(neighbour_positions)
                        if neighbour_positions
                        else float(index)
                    )
                    decorated.append(
                        (
                            0 if node.node_id == initial_node_id else 1,
                            barycenter,
                            -node.priority,
                            node.label.casefold(),
                            node,
                        )
                    )
                ordered[group] = [item[-1] for item in sorted(decorated)]
                for index, node in enumerate(ordered[group]):
                    positions[node.node_id] = float(index)
        return ordered

    def _group_sizes(
        self, ordered_nodes: dict[str, list[LayoutNode]]
    ) -> dict[str, tuple[float, float]]:
        result: dict[str, tuple[float, float]] = {}
        for group, nodes in ordered_nodes.items():
            width = max((node.width for node in nodes), default=240.0)
            height = self.header_height + sum(node.height for node in nodes)
            height += self.node_gap * max(0, len(nodes) - 1)
            result[group] = (width, height)
        return result

    def _place_group_origins(
        self,
        *,
        groups: list[str],
        ranks: dict[str, int],
        sizes: dict[str, tuple[float, float]],
        group_edges: dict[tuple[str, str], float],
        orientation: Orientation,
        initial_group: str,
    ) -> dict[str, QPointF]:
        by_rank: dict[int, list[str]] = defaultdict(list)
        for group in groups:
            by_rank[ranks[group]].append(group)
        for values in by_rank.values():
            values.sort(key=lambda group: (0 if group == initial_group else 1, group.casefold()))

        if orientation == "vertical":
            return self._place_vertical(by_rank, sizes, group_edges)
        if orientation == "folded":
            return self._place_folded(by_rank, sizes, group_edges)
        return self._place_horizontal(by_rank, sizes, group_edges)

    def _place_horizontal(
        self,
        by_rank: dict[int, list[str]],
        sizes: dict[str, tuple[float, float]],
        group_edges: dict[tuple[str, str], float],
    ) -> dict[str, QPointF]:
        rank_width = {
            rank: max(sizes[group][0] for group in groups)
            for rank, groups in by_rank.items()
        }
        x_by_rank: dict[int, float] = {}
        cursor = 0.0
        for rank in sorted(by_rank):
            x_by_rank[rank] = cursor
            cursor += rank_width[rank] + self.rank_gap

        origins: dict[str, QPointF] = {}
        for rank in sorted(by_rank):
            desired = self._desired_group_y(by_rank[rank], origins, sizes, group_edges)
            packed = self._pack_axis(by_rank[rank], desired, sizes, vertical=True)
            for group, y in packed.items():
                origins[group] = QPointF(x_by_rank[rank], y)
        self._center_axis(origins, sizes, vertical=True)
        return origins

    def _place_vertical(
        self,
        by_rank: dict[int, list[str]],
        sizes: dict[str, tuple[float, float]],
        group_edges: dict[tuple[str, str], float],
    ) -> dict[str, QPointF]:
        rank_height = {
            rank: max(sizes[group][1] for group in groups)
            for rank, groups in by_rank.items()
        }
        y_by_rank: dict[int, float] = {}
        cursor = 0.0
        for rank in sorted(by_rank):
            y_by_rank[rank] = cursor
            cursor += rank_height[rank] + self.rank_gap

        origins: dict[str, QPointF] = {}
        for rank in sorted(by_rank):
            desired = self._desired_group_x(by_rank[rank], origins, sizes, group_edges)
            packed = self._pack_axis(by_rank[rank], desired, sizes, vertical=False)
            for group, x in packed.items():
                origins[group] = QPointF(x, y_by_rank[rank])
        self._center_axis(origins, sizes, vertical=False)
        return origins

    def _place_folded(
        self,
        by_rank: dict[int, list[str]],
        sizes: dict[str, tuple[float, float]],
        group_edges: dict[tuple[str, str], float],
    ) -> dict[str, QPointF]:
        ranks = sorted(by_rank)
        if len(ranks) <= 4:
            return self._place_horizontal(by_rank, sizes, group_edges)
        per_band = max(3, math.ceil(math.sqrt(len(ranks) * 1.7)))
        rank_width = {
            rank: max(sizes[group][0] for group in by_rank[rank])
            for rank in ranks
        }
        rank_height = {
            rank: sum(sizes[group][1] for group in by_rank[rank])
            + self.group_gap * max(0, len(by_rank[rank]) - 1)
            for rank in ranks
        }
        max_column_width = max(rank_width.values(), default=240.0)
        origins: dict[str, QPointF] = {}
        band_y = 0.0
        for band_index in range(math.ceil(len(ranks) / per_band)):
            band = ranks[band_index * per_band : (band_index + 1) * per_band]
            display = list(reversed(band)) if band_index % 2 else band
            band_height = max((rank_height[rank] for rank in band), default=0.0)
            for column, rank in enumerate(display):
                x = column * (max_column_width + self.rank_gap)
                groups = by_rank[rank]
                desired = self._desired_group_y(groups, origins, sizes, group_edges)
                packed = self._pack_axis(groups, desired, sizes, vertical=True)
                packed_height = max(
                    (packed[group] + sizes[group][1] for group in groups),
                    default=0.0,
                ) - min((packed[group] for group in groups), default=0.0)
                offset_y = band_y + max(0.0, (band_height - packed_height) / 2.0)
                local_min = min((packed[group] for group in groups), default=0.0)
                for group in groups:
                    origins[group] = QPointF(x, offset_y + packed[group] - local_min)
            band_y += band_height + self.rank_gap * 1.25
        return origins

    def _desired_group_y(
        self,
        groups: list[str],
        origins: dict[str, QPointF],
        sizes: dict[str, tuple[float, float]],
        group_edges: dict[tuple[str, str], float],
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for index, group in enumerate(groups):
            values: list[tuple[float, float]] = []
            for (source, target), weight in group_edges.items():
                other = source if target == group else target if source == group else None
                if other is not None and other in origins:
                    values.append((origins[other].y() + sizes[other][1] / 2.0, weight))
            result[group] = (
                sum(value * weight for value, weight in values)
                / sum(weight for _, weight in values)
                if values
                else index * (sizes[group][1] + self.group_gap)
            )
        return result

    def _desired_group_x(
        self,
        groups: list[str],
        origins: dict[str, QPointF],
        sizes: dict[str, tuple[float, float]],
        group_edges: dict[tuple[str, str], float],
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for index, group in enumerate(groups):
            values: list[tuple[float, float]] = []
            for (source, target), weight in group_edges.items():
                other = source if target == group else target if source == group else None
                if other is not None and other in origins:
                    values.append((origins[other].x() + sizes[other][0] / 2.0, weight))
            result[group] = (
                sum(value * weight for value, weight in values)
                / sum(weight for _, weight in values)
                if values
                else index * (sizes[group][0] + self.group_gap)
            )
        return result

    def _pack_axis(
        self,
        groups: list[str],
        desired: dict[str, float],
        sizes: dict[str, tuple[float, float]],
        *,
        vertical: bool,
    ) -> dict[str, float]:
        ordered = sorted(groups, key=lambda group: (desired[group], group.casefold()))
        result: dict[str, float] = {}
        cursor = -math.inf
        for group in ordered:
            size = sizes[group][1 if vertical else 0]
            start = desired[group] - size / 2.0
            if cursor != -math.inf:
                start = max(start, cursor + self.group_gap)
            result[group] = start
            cursor = start + size
        # Center the packed set around its desired weighted center.
        if result:
            current_min = min(result.values())
            current_max = max(
                result[group] + sizes[group][1 if vertical else 0]
                for group in result
            )
            desired_center = sum(desired.values()) / len(desired)
            shift = desired_center - (current_min + current_max) / 2.0
            for group in result:
                result[group] += shift
        return result

    @staticmethod
    def _center_axis(
        origins: dict[str, QPointF],
        sizes: dict[str, tuple[float, float]],
        *,
        vertical: bool,
    ) -> None:
        if not origins:
            return
        if vertical:
            minimum = min(point.y() for point in origins.values())
            if minimum < 0:
                for group, point in list(origins.items()):
                    origins[group] = QPointF(point.x(), point.y() - minimum)
        else:
            minimum = min(point.x() for point in origins.values())
            if minimum < 0:
                for group, point in list(origins.items()):
                    origins[group] = QPointF(point.x() - minimum, point.y())

    def _materialize(
        self,
        ordered_nodes: dict[str, list[LayoutNode]],
        group_sizes: dict[str, tuple[float, float]],
        origins: dict[str, QPointF],
        ranks: dict[str, int],
        offset: QPointF,
    ) -> tuple[dict[str, NodePlacement], dict[str, QPointF]]:
        placements: dict[str, NodePlacement] = {}
        headers: dict[str, QPointF] = {}
        group_order = 0
        for group, nodes in sorted(origins.items(), key=lambda item: (item[1].y(), item[1].x())):
            origin = nodes + offset
            headers[group] = QPointF(origin.x(), origin.y())
            y = origin.y() + self.header_height
            for node in ordered_nodes[group]:
                rect = QRectF(origin.x(), y, node.width, node.height)
                placements[node.node_id] = NodePlacement(
                    node=node,
                    rect=rect,
                    rank=ranks[group],
                    group_order=group_order,
                )
                y += node.height + self.node_gap
            group_order += 1
        return placements, headers

    def _score(
        self,
        placements: dict[str, NodePlacement],
        edges: list[LayoutEdge],
        orientation: Orientation,
    ) -> float:
        if not placements:
            return float("inf")
        bounds = _rect_union([item.rect for item in placements.values()])
        width = max(bounds.width(), 1.0)
        height = max(bounds.height(), 1.0)
        aspect = width / height
        aspect_penalty = abs(math.log(max(aspect, 0.01) / self.target_aspect)) * 900.0
        area_penalty = math.sqrt(width * height) * 0.12
        edge_length = 0.0
        straight_segments: list[tuple[QPointF, QPointF]] = []
        for edge in edges:
            source = placements.get(edge.source_id)
            target = placements.get(edge.target_id)
            if source is None or target is None:
                continue
            a, b = source.rect.center(), target.rect.center()
            edge_length += (abs(a.x() - b.x()) + abs(a.y() - b.y())) * edge.weight
            straight_segments.append((a, b))
        crossings = 0
        for index, first in enumerate(straight_segments):
            for second in straight_segments[index + 1 :]:
                if _proper_segment_intersection(*first, *second):
                    crossings += 1
        orientation_bias = 0.0 if orientation == "folded" else 70.0
        return aspect_penalty + area_penalty + edge_length * 0.045 + crossings * 45.0 + orientation_bias


class OrthogonalRouter:
    """Route edges on a compressed visibility grid without crossing nodes.

    Every edge receives individual source/target ports and its own route.  A* uses
    Manhattan length, bend penalties, crossing penalties, and strong congestion
    penalties for already occupied segments.  This makes parallel edges visually
    traceable rather than drawing them on the same pixels.
    """

    def __init__(
        self,
        *,
        obstacle_margin: float = 10.0,
        port_stub: float = 14.0,
        lane_spacing: float = 16.0,
        bend_penalty: float = 42.0,
        crossing_penalty: float = 180.0,
        overlap_penalty: float = 1400.0,
    ) -> None:
        self.obstacle_margin = obstacle_margin
        self.port_stub = port_stub
        self.lane_spacing = lane_spacing
        self.bend_penalty = bend_penalty
        self.crossing_penalty = crossing_penalty
        self.overlap_penalty = overlap_penalty
        self._segment_use: dict[tuple[float, float, float, float], int] = defaultdict(int)
        self._routed_segments: list[tuple[QPointF, QPointF, str]] = []

    def route(
        self,
        placements: dict[str, NodePlacement],
        edges: Iterable[LayoutEdge],
    ) -> dict[str, RoutedEdge]:
        edge_list = list(edges)
        if not edge_list:
            return {}
        ports = self._allocate_ports(placements, edge_list)
        obstacles = [
            placement.rect.adjusted(
                -self.obstacle_margin,
                -self.obstacle_margin,
                self.obstacle_margin,
                self.obstacle_margin,
            )
            for placement in placements.values()
        ]
        coordinate_x, coordinate_y = self._routing_coordinates(obstacles, ports)
        valid_points, neighbours = self._visibility_grid(
            coordinate_x, coordinate_y, obstacles
        )

        routes: dict[str, RoutedEdge] = {}
        # Harder/backward and long edges route first so short local edges can use
        # the remaining near-node channels.
        sorted_edges = sorted(
            edge_list,
            key=lambda edge: self._routing_priority(edge, placements),
            reverse=True,
        )
        for edge in sorted_edges:
            source_port = ports[(edge.edge_id, True)]
            target_port = ports[(edge.edge_id, False)]
            points = self._a_star_route(
                source_port.external,
                target_port.external,
                valid_points,
                neighbours,
            )
            if not points:
                points = self._fallback_route(
                    source_port.external,
                    target_port.external,
                    obstacles,
                )
            if not points:
                raise RuntimeError(
                    f"No obstacle-free orthogonal route for edge {edge.edge_id}."
                )
            full = [source_port.border, source_port.external]
            full.extend(points[1:-1] if len(points) > 2 else [])
            full.extend([target_port.external, target_port.border])
            full = _simplify_orthogonal(full)
            self._reserve(edge.edge_id, full)
            routes[edge.edge_id] = RoutedEdge(
                edge=edge,
                points=full,
                source_side=source_port.side,
                target_side=target_port.side,
                label_position=self._label_position(full),
            )
        return routes

    def _allocate_ports(
        self,
        placements: dict[str, NodePlacement],
        edges: list[LayoutEdge],
    ) -> dict[tuple[str, bool], _Port]:
        requests: dict[tuple[str, Side], list[_PortRequest]] = defaultdict(list)
        for edge in edges:
            source_placement = placements[edge.source_id]
            target_placement = placements[edge.target_id]
            source = source_placement.rect
            target = target_placement.rect
            source_side, target_side = self._edge_sides(
                edge,
                source_placement,
                target_placement,
                placements,
            )
            requests[(edge.source_id, source_side)].append(
                _PortRequest(edge.edge_id, edge.source_id, edge.target_id, source_side, True)
            )
            requests[(edge.target_id, target_side)].append(
                _PortRequest(edge.edge_id, edge.target_id, edge.source_id, target_side, False)
            )

        result: dict[tuple[str, bool], _Port] = {}
        for (node_id, side), values in requests.items():
            rect = placements[node_id].rect
            values.sort(
                key=lambda request: _opposite_coordinate(
                    placements[request.other_id].rect.center(), side
                )
            )
            count = len(values)
            for index, request in enumerate(values):
                fraction = (index + 1) / (count + 1)
                border = _point_on_side(rect, side, fraction)
                external = _move_from_side(border, side, self.port_stub)
                result[(request.edge_id, request.is_source)] = _Port(
                    border=border,
                    external=external,
                    side=side,
                )
        return result

    def _edge_sides(
        self,
        edge: LayoutEdge,
        source: NodePlacement,
        target: NodePlacement,
        placements: dict[str, NodePlacement],
    ) -> tuple[Side, Side]:
        """Choose ports that do not force an edge through its own group stack."""
        if source.node.group == target.node.group:
            source_center = source.rect.center()
            target_center = target.rect.center()
            vertically_aligned = abs(source_center.x() - target_center.x()) < max(
                source.rect.width(), target.rect.width()
            ) / 2.0
            if vertically_aligned:
                top, bottom = sorted((source_center.y(), target_center.y()))
                intervening = any(
                    placement.node.group == source.node.group
                    and placement.node.node_id not in {edge.source_id, edge.target_id}
                    and top < placement.rect.center().y() < bottom
                    for placement in placements.values()
                )
                if intervening:
                    # Alternate outer sides deterministically; every edge still
                    # receives a separate lane from the congestion-aware router.
                    checksum = sum(edge.edge_id.encode("utf-8"))
                    side: Side = "right" if checksum % 2 == 0 else "left"
                    return side, side
            horizontally_aligned = abs(source_center.y() - target_center.y()) < max(
                source.rect.height(), target.rect.height()
            ) / 2.0
            if horizontally_aligned:
                left, right = sorted((source_center.x(), target_center.x()))
                intervening = any(
                    placement.node.group == source.node.group
                    and placement.node.node_id not in {edge.source_id, edge.target_id}
                    and left < placement.rect.center().x() < right
                    for placement in placements.values()
                )
                if intervening:
                    checksum = sum(edge.edge_id.encode("utf-8"))
                    side = "bottom" if checksum % 2 == 0 else "top"
                    return side, side
        return _preferred_sides(source.rect, target.rect)

    def _routing_coordinates(
        self,
        obstacles: list[QRectF],
        ports: dict[tuple[str, bool], _Port],
    ) -> tuple[list[float], list[float]]:
        xs = {port.external.x() for port in ports.values()}
        ys = {port.external.y() for port in ports.values()}
        if obstacles:
            bounds = _rect_union(obstacles)
            xs.update({bounds.left() - 80.0, bounds.right() + 80.0})
            ys.update({bounds.top() - 80.0, bounds.bottom() + 80.0})
        for rect in obstacles:
            for multiplier in (0.0, 1.0, 2.0):
                offset = self.lane_spacing * multiplier
                xs.update({rect.left() - offset, rect.right() + offset})
                ys.update({rect.top() - offset, rect.bottom() + offset})
        # Midpoints between obstacle corridors create useful central lanes.
        sorted_x_bounds = sorted({value for rect in obstacles for value in (rect.left(), rect.right())})
        sorted_y_bounds = sorted({value for rect in obstacles for value in (rect.top(), rect.bottom())})
        xs.update((a + b) / 2.0 for a, b in zip(sorted_x_bounds, sorted_x_bounds[1:]) if b - a > 24.0)
        ys.update((a + b) / 2.0 for a, b in zip(sorted_y_bounds, sorted_y_bounds[1:]) if b - a > 24.0)
        return sorted(xs), sorted(ys)

    def _visibility_grid(
        self,
        xs: list[float],
        ys: list[float],
        obstacles: list[QRectF],
    ) -> tuple[set[tuple[float, float]], dict[tuple[float, float], list[tuple[float, float]]]]:
        valid = {
            (x, y)
            for x in xs
            for y in ys
            if not any(_point_strictly_inside(QPointF(x, y), rect) for rect in obstacles)
        }
        neighbours: dict[tuple[float, float], list[tuple[float, float]]] = defaultdict(list)
        by_y: dict[float, list[float]] = defaultdict(list)
        by_x: dict[float, list[float]] = defaultdict(list)
        for x, y in valid:
            by_y[y].append(x)
            by_x[x].append(y)
        for y, values in by_y.items():
            values.sort()
            for a, b in zip(values, values[1:]):
                first, second = (a, y), (b, y)
                if _segment_clear(QPointF(*first), QPointF(*second), obstacles):
                    neighbours[first].append(second)
                    neighbours[second].append(first)
        for x, values in by_x.items():
            values.sort()
            for a, b in zip(values, values[1:]):
                first, second = (x, a), (x, b)
                if _segment_clear(QPointF(*first), QPointF(*second), obstacles):
                    neighbours[first].append(second)
                    neighbours[second].append(first)
        return valid, neighbours

    def _a_star_route(
        self,
        start: QPointF,
        goal: QPointF,
        valid: set[tuple[float, float]],
        neighbours: dict[tuple[float, float], list[tuple[float, float]]],
    ) -> list[QPointF]:
        start_key = (_round(start.x()), _round(start.y()))
        goal_key = (_round(goal.x()), _round(goal.y()))
        if start_key not in valid or goal_key not in valid:
            return []
        counter = itertools.count()
        queue: list[tuple[float, int, tuple[float, float], Direction]] = []
        heapq.heappush(queue, (0.0, next(counter), start_key, "N"))
        costs: dict[tuple[tuple[float, float], Direction], float] = {(start_key, "N"): 0.0}
        previous: dict[
            tuple[tuple[float, float], Direction],
            tuple[tuple[float, float], Direction] | None,
        ] = {(start_key, "N"): None}
        end_state: tuple[tuple[float, float], Direction] | None = None

        while queue:
            _, _, current, direction = heapq.heappop(queue)
            state = (current, direction)
            current_cost = costs.get(state)
            if current_cost is None:
                continue
            if current == goal_key:
                end_state = state
                break
            current_point = QPointF(*current)
            for nxt in neighbours.get(current, []):
                next_point = QPointF(*nxt)
                next_direction: Direction = "H" if current[1] == nxt[1] else "V"
                segment_length = abs(current[0] - nxt[0]) + abs(current[1] - nxt[1])
                bend = self.bend_penalty if direction not in {"N", next_direction} else 0.0
                congestion = self._segment_use.get(_segment_key(current_point, next_point), 0)
                overlaps = sum(
                    1
                    for first, second, _ in self._routed_segments
                    if _collinear_overlap(current_point, next_point, first, second)
                )
                crossings = sum(
                    1
                    for first, second, _ in self._routed_segments
                    if _orthogonal_crossing(current_point, next_point, first, second)
                )
                step_cost = (
                    segment_length
                    + bend
                    + (congestion + overlaps) * self.overlap_penalty
                    + crossings * self.crossing_penalty
                )
                next_state = (nxt, next_direction)
                candidate = current_cost + step_cost
                if candidate >= costs.get(next_state, float("inf")):
                    continue
                costs[next_state] = candidate
                previous[next_state] = state
                heuristic = abs(nxt[0] - goal_key[0]) + abs(nxt[1] - goal_key[1])
                heapq.heappush(
                    queue,
                    (candidate + heuristic, next(counter), nxt, next_direction),
                )

        if end_state is None:
            return []
        path: list[QPointF] = []
        state: tuple[tuple[float, float], Direction] | None = end_state
        while state is not None:
            path.append(QPointF(*state[0]))
            state = previous[state]
        path.reverse()
        return _simplify_orthogonal(path)

    def _fallback_route(
        self,
        start: QPointF,
        goal: QPointF,
        obstacles: list[QRectF],
    ) -> list[QPointF]:
        bounds = _rect_union(obstacles) if obstacles else QRectF(start, goal)
        candidates: list[list[QPointF]] = []
        for distance in (60.0, 100.0, 160.0, 240.0):
            left = bounds.left() - distance
            right = bounds.right() + distance
            top = bounds.top() - distance
            bottom = bounds.bottom() + distance
            candidates.extend(
                [
                    [start, QPointF(left, start.y()), QPointF(left, goal.y()), goal],
                    [start, QPointF(right, start.y()), QPointF(right, goal.y()), goal],
                    [start, QPointF(start.x(), top), QPointF(goal.x(), top), goal],
                    [start, QPointF(start.x(), bottom), QPointF(goal.x(), bottom), goal],
                ]
            )
        valid = [
            candidate
            for candidate in candidates
            if all(
                _segment_clear(first, second, obstacles)
                for first, second in zip(candidate, candidate[1:])
            )
        ]
        return min(valid, key=_polyline_length) if valid else []

    def _reserve(self, edge_id: str, points: list[QPointF]) -> None:
        for first, second in zip(points, points[1:]):
            if _same_point(first, second):
                continue
            self._segment_use[_segment_key(first, second)] += 1
            self._routed_segments.append((first, second, edge_id))

    @staticmethod
    def _routing_priority(
        edge: LayoutEdge, placements: dict[str, NodePlacement]
    ) -> tuple[int, float, float]:
        source = placements[edge.source_id]
        target = placements[edge.target_id]
        backward = int(target.rank <= source.rank)
        distance = abs(source.rect.center().x() - target.rect.center().x()) + abs(
            source.rect.center().y() - target.rect.center().y()
        )
        return backward, distance, edge.weight

    @staticmethod
    def _label_position(points: list[QPointF]) -> QPointF:
        segments = [
            (first, second, abs(first.x() - second.x()) + abs(first.y() - second.y()))
            for first, second in zip(points, points[1:])
        ]
        first, second, _ = max(segments, key=lambda item: item[2])
        midpoint = QPointF((first.x() + second.x()) / 2.0, (first.y() + second.y()) / 2.0)
        return midpoint + (QPointF(0, -18) if first.y() == second.y() else QPointF(8, 0))


def layout_and_route(
    nodes: Iterable[LayoutNode],
    edges: Iterable[LayoutEdge],
    *,
    initial_node_id: str,
    offset: QPointF = QPointF(40.0, 150.0),
) -> LayoutResult:
    node_list = list(nodes)
    edge_list = list(edges)
    placer = CompactLayeredLayout()
    placements, headers, orientation = placer.place(
        node_list,
        edge_list,
        initial_node_id=initial_node_id,
        offset=offset,
    )
    router = OrthogonalRouter()
    routes = router.route(placements, edge_list)
    all_rects = [placement.rect for placement in placements.values()]
    for route in routes.values():
        all_rects.append(_points_bounds(route.points))
    bounds = _rect_union(all_rects) if all_rects else QRectF()
    return LayoutResult(
        placements=placements,
        routes=routes,
        orientation=orientation,
        bounds=bounds,
        group_headers=headers,
    )


def route_fixed_nodes(
    node_rects: dict[str, QRectF],
    edges: Iterable[LayoutEdge],
) -> dict[str, RoutedEdge]:
    placements = {
        node_id: NodePlacement(
            node=LayoutNode(node_id, node_id, "fixed", rect.width(), rect.height()),
            rect=rect,
            rank=0,
            group_order=0,
        )
        for node_id, rect in node_rects.items()
    }
    return OrthogonalRouter().route(placements, list(edges))


def _preferred_sides(source: QRectF, target: QRectF) -> tuple[Side, Side]:
    dx = target.center().x() - source.center().x()
    dy = target.center().y() - source.center().y()
    if abs(dx) >= abs(dy):
        return ("right", "left") if dx >= 0 else ("left", "right")
    return ("bottom", "top") if dy >= 0 else ("top", "bottom")


def _point_on_side(rect: QRectF, side: Side, fraction: float) -> QPointF:
    if side == "left":
        return QPointF(rect.left(), rect.top() + rect.height() * fraction)
    if side == "right":
        return QPointF(rect.right(), rect.top() + rect.height() * fraction)
    if side == "top":
        return QPointF(rect.left() + rect.width() * fraction, rect.top())
    return QPointF(rect.left() + rect.width() * fraction, rect.bottom())


def _move_from_side(point: QPointF, side: Side, distance: float) -> QPointF:
    if side == "left":
        return point + QPointF(-distance, 0)
    if side == "right":
        return point + QPointF(distance, 0)
    if side == "top":
        return point + QPointF(0, -distance)
    return point + QPointF(0, distance)


def _opposite_coordinate(point: QPointF, side: Side) -> float:
    return point.y() if side in {"left", "right"} else point.x()


def _point_strictly_inside(point: QPointF, rect: QRectF) -> bool:
    epsilon = 0.01
    return (
        rect.left() + epsilon < point.x() < rect.right() - epsilon
        and rect.top() + epsilon < point.y() < rect.bottom() - epsilon
    )


def _segment_clear(start: QPointF, end: QPointF, obstacles: list[QRectF]) -> bool:
    if abs(start.y() - end.y()) < 0.01:
        y = start.y()
        left, right = sorted((start.x(), end.x()))
        return not any(
            rect.top() < y < rect.bottom()
            and max(left, rect.left()) < min(right, rect.right())
            for rect in obstacles
        )
    if abs(start.x() - end.x()) < 0.01:
        x = start.x()
        top, bottom = sorted((start.y(), end.y()))
        return not any(
            rect.left() < x < rect.right()
            and max(top, rect.top()) < min(bottom, rect.bottom())
            for rect in obstacles
        )
    return False


def _segment_key(first: QPointF, second: QPointF) -> tuple[float, float, float, float]:
    a = (_round(first.x()), _round(first.y()))
    b = (_round(second.x()), _round(second.y()))
    return (*a, *b) if a <= b else (*b, *a)


def _collinear_overlap(
    a1: QPointF, a2: QPointF, b1: QPointF, b2: QPointF
) -> bool:
    a_horizontal = abs(a1.y() - a2.y()) < 0.01
    b_horizontal = abs(b1.y() - b2.y()) < 0.01
    if a_horizontal != b_horizontal:
        return False
    if a_horizontal:
        if abs(a1.y() - b1.y()) >= 0.01:
            return False
        a_left, a_right = sorted((a1.x(), a2.x()))
        b_left, b_right = sorted((b1.x(), b2.x()))
        return max(a_left, b_left) < min(a_right, b_right) - 0.01
    if abs(a1.x() - b1.x()) >= 0.01:
        return False
    a_top, a_bottom = sorted((a1.y(), a2.y()))
    b_top, b_bottom = sorted((b1.y(), b2.y()))
    return max(a_top, b_top) < min(a_bottom, b_bottom) - 0.01


def _orthogonal_crossing(
    a1: QPointF, a2: QPointF, b1: QPointF, b2: QPointF
) -> bool:
    a_horizontal = abs(a1.y() - a2.y()) < 0.01
    b_horizontal = abs(b1.y() - b2.y()) < 0.01
    if a_horizontal == b_horizontal:
        return False
    horizontal = (a1, a2) if a_horizontal else (b1, b2)
    vertical = (b1, b2) if a_horizontal else (a1, a2)
    hx1, hx2 = sorted((horizontal[0].x(), horizontal[1].x()))
    vy1, vy2 = sorted((vertical[0].y(), vertical[1].y()))
    x, y = vertical[0].x(), horizontal[0].y()
    if not (hx1 < x < hx2 and vy1 < y < vy2):
        return False
    crossing = QPointF(x, y)
    return not any(
        _same_point(crossing, point)
        for point in (a1, a2, b1, b2)
    )


def _proper_segment_intersection(
    a1: QPointF, a2: QPointF, b1: QPointF, b2: QPointF
) -> bool:
    def orientation(p: QPointF, q: QPointF, r: QPointF) -> float:
        return (q.y() - p.y()) * (r.x() - q.x()) - (q.x() - p.x()) * (r.y() - q.y())

    return orientation(a1, a2, b1) * orientation(a1, a2, b2) < 0 and orientation(b1, b2, a1) * orientation(b1, b2, a2) < 0


def _simplify_orthogonal(points: list[QPointF]) -> list[QPointF]:
    result: list[QPointF] = []
    for point in points:
        if result and _same_point(result[-1], point):
            continue
        result.append(point)
        while len(result) >= 3:
            a, b, c = result[-3:]
            if (
                abs(a.x() - b.x()) < 0.01 and abs(b.x() - c.x()) < 0.01
            ) or (
                abs(a.y() - b.y()) < 0.01 and abs(b.y() - c.y()) < 0.01
            ):
                result.pop(-2)
            else:
                break
    return result


def _points_bounds(points: list[QPointF]) -> QRectF:
    xs = [point.x() for point in points]
    ys = [point.y() for point in points]
    return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _rect_union(rects: Iterable[QRectF]) -> QRectF:
    values = list(rects)
    if not values:
        return QRectF()
    result = QRectF(values[0])
    for rect in values[1:]:
        result = result.united(rect)
    return result


def _polyline_length(points: list[QPointF]) -> float:
    return sum(
        abs(first.x() - second.x()) + abs(first.y() - second.y())
        for first, second in zip(points, points[1:])
    )


def _same_point(first: QPointF, second: QPointF) -> bool:
    return abs(first.x() - second.x()) < 0.01 and abs(first.y() - second.y()) < 0.01


def _round(value: float) -> float:
    return round(float(value), 3)


def assign_label_positions(
    routes: Iterable[RoutedEdge],
    obstacles: Iterable[QRectF],
    *,
    label_width: float = 34.0,
    label_height: float = 22.0,
) -> dict[str, QPointF]:
    """Choose non-overlapping edge-label positions on routed segments.

    Candidates are sampled on every sufficiently long orthogonal segment.  Node
    intersections are forbidden whenever possible; already placed labels receive
    a strong penalty.  The function is deterministic and does not mutate routes.
    """
    obstacle_rects = [rect.adjusted(-5.0, -5.0, 5.0, 5.0) for rect in obstacles]
    occupied: list[QRectF] = []
    result: dict[str, QPointF] = {}
    route_list = list(routes)
    all_segments = [
        (route.edge.edge_id, first, second)
        for route in route_list
        for first, second in zip(route.points, route.points[1:])
    ]
    ordered = sorted(
        route_list,
        key=lambda route: _polyline_length(route.points),
        reverse=True,
    )
    for route in ordered:
        candidates: list[tuple[float, QPointF, QRectF]] = []
        for first, second in zip(route.points, route.points[1:]):
            length = abs(first.x() - second.x()) + abs(first.y() - second.y())
            if length < 32.0:
                continue
            for fraction in (0.5, 0.28, 0.72):
                x = first.x() + (second.x() - first.x()) * fraction
                y = first.y() + (second.y() - first.y()) * fraction
                offsets = (
                    (QPointF(-label_width / 2.0, -label_height - 5.0),
                     QPointF(-label_width / 2.0, 5.0))
                    if abs(first.y() - second.y()) < 0.01
                    else (QPointF(7.0, -label_height / 2.0),
                          QPointF(-label_width - 7.0, -label_height / 2.0))
                )
                for offset in offsets:
                    position = QPointF(x, y) + offset
                    rect = QRectF(position.x(), position.y(), label_width, label_height)
                    node_hits = sum(rect.intersects(obstacle) for obstacle in obstacle_rects)
                    label_hits = sum(rect.intersects(other) for other in occupied)
                    # Prefer long segments and central candidates, then distance from nodes.
                    shared = sum(
                        1
                        for other_edge_id, other_first, other_second in all_segments
                        if other_edge_id != route.edge.edge_id
                        and _collinear_overlap(first, second, other_first, other_second)
                    )
                    score = (
                        node_hits * 100000.0
                        + label_hits * 5000.0
                        + shared * 2500.0
                    )
                    score += abs(fraction - 0.5) * 20.0 - min(length, 500.0) * 0.02
                    candidates.append((score, position, rect))
        if candidates:
            _, position, rect = min(candidates, key=lambda item: item[0])
        else:
            position = route.label_position
            rect = QRectF(position.x(), position.y(), label_width, label_height)
        result[route.edge.edge_id] = position
        occupied.append(rect)
    return result


def separate_overlapping_routes(
    routes: Iterable[RoutedEdge],
    *,
    placements: dict[str, NodePlacement] | None = None,
    preferred_lane_gap: float = 8.0,
    max_bundle_span: float = 40.0,
) -> dict[str, RoutedEdge]:
    """Return render-only routes whose overlapping paths use parallel lanes.

    The orthogonal router deliberately allows a small number of semantic route
    buses in very dense graphs.  For presentation, exact or partially collinear
    routes are harder to follow, especially when solid production lines and
    dashed adapter lines coincide.  This post-processing step detects connected
    overlap bundles and translates each complete internal route onto a separate
    lane.  Border points remain attached to their original node ports; short
    orthogonal doglegs connect the shifted lane to the port stubs.

    The model geometry is not mutated.  The returned routes remain orthogonal.
    """
    route_list = list(routes)
    if not route_list:
        return {}

    overlap_graph: dict[str, set[str]] = defaultdict(set)
    route_by_id = {route.edge.edge_id: route for route in route_list}
    for index, first_route in enumerate(route_list):
        for second_route in route_list[index + 1 :]:
            if _routes_collinearly_overlap(first_route, second_route):
                first_id = first_route.edge.edge_id
                second_id = second_route.edge.edge_id
                overlap_graph[first_id].add(second_id)
                overlap_graph[second_id].add(first_id)

    result = {route.edge.edge_id: route for route in route_list}
    visited: set[str] = set()
    for route in route_list:
        route_id = route.edge.edge_id
        if route_id in visited or route_id not in overlap_graph:
            continue
        component: list[str] = []
        queue = [route_id]
        visited.add(route_id)
        while queue:
            current = queue.pop(0)
            component.append(current)
            for neighbour in sorted(overlap_graph.get(current, set())):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        if len(component) < 2:
            continue

        component.sort(
            key=lambda edge_id: (
                0 if route_by_id[edge_id].edge.kind == "production" else 1,
                edge_id.casefold(),
            )
        )
        lane_gap = min(
            preferred_lane_gap,
            max_bundle_span / max(1, len(component) - 1),
        )
        centre = (len(component) - 1) / 2.0
        for index, edge_id in enumerate(component):
            source = route_by_id[edge_id]
            offset = (index - centre) * lane_gap
            # Shift in both axes.  A y-only shift separates horizontal segments
            # but leaves vertical ones coincident; an x-only shift has the inverse
            # problem.  A small diagonal lane vector separates both orientations
            # while the connector helper preserves right-angle geometry.
            vectors = [
                vector
                for multiplier in (1.0, 1.5, 2.0, 2.5)
                for vector in (
                    (offset * multiplier, -offset * multiplier),
                    (-offset * multiplier, offset * multiplier),
                    (offset * multiplier, offset * multiplier),
                    (-offset * multiplier, -offset * multiplier),
                )
            ]
            result[edge_id] = _best_shifted_route(
                source,
                vectors,
                placements=placements,
                comparison_routes=[
                    other
                    for other_id, other in result.items()
                    if other_id != edge_id
                ],
            )

    # Translating one overlap component can occasionally create a new overlap
    # with a neighbouring component.  Resolve those residual cases iteratively.
    # Production routes stay fixed when they conflict with adapter routes; the
    # dashed adapter route receives the additional lane nudge.
    for pass_index in range(6):
        pairs = _overlapping_route_pairs(result)
        if not pairs:
            break
        moved: set[str] = set()
        for first_id, second_id in pairs:
            first = result[first_id]
            second = result[second_id]
            if first.edge.kind != second.edge.kind:
                move_id = (
                    first_id if first.edge.kind == "adapter" else second_id
                )
            else:
                move_id = max(first_id, second_id)
            if move_id in moved:
                continue
            moved.add(move_id)
            route = result[move_id]
            sign = -1.0 if sum(move_id.encode("utf-8")) % 2 else 1.0
            amount = sign * (3.0 + pass_index * 2.0)
            result[move_id] = _best_shifted_route(
                route,
                [
                    vector
                    for multiplier in (1.0, 1.5, 2.0, 2.5, 3.0)
                    for vector in (
                        (amount * multiplier, -amount * multiplier),
                        (-amount * multiplier, amount * multiplier),
                        (amount * multiplier, amount * multiplier),
                        (-amount * multiplier, -amount * multiplier),
                    )
                ],
                placements=placements,
                comparison_routes=[
                    other
                    for other_id, other in result.items()
                    if other_id != move_id
                ],
            )
    return result


def _best_shifted_route(
    source: RoutedEdge,
    vectors: Iterable[tuple[float, float]],
    *,
    placements: dict[str, NodePlacement] | None,
    comparison_routes: Iterable[RoutedEdge],
) -> RoutedEdge:
    comparisons = list(comparison_routes)
    candidates: list[tuple[tuple[int, int, float], list[QPointF]]] = []
    original_points = list(source.points)
    original_node_hits = _route_node_hit_count(
        source.edge, original_points, placements
    )
    original_overlap_hits = sum(
        _points_routes_collinearly_overlap(original_points, other.points)
        for other in comparisons
    )
    candidates.append(
        ((original_node_hits, original_overlap_hits, 0.0), original_points)
    )
    for dx, dy in vectors:
        shifted = _offset_route_interior(source.points, dx, dy)
        node_hits = _route_node_hit_count(source.edge, shifted, placements)
        overlap_hits = sum(
            _points_routes_collinearly_overlap(shifted, other.points)
            for other in comparisons
        )
        movement = abs(dx) + abs(dy)
        candidates.append(((node_hits, overlap_hits, movement), shifted))
    if not candidates:
        shifted = list(source.points)
    else:
        _, shifted = min(candidates, key=lambda item: item[0])
    return RoutedEdge(
        edge=source.edge,
        points=shifted,
        source_side=source.source_side,
        target_side=source.target_side,
        label_position=_route_label_position(shifted),
    )


def _route_node_hit_count(
    edge: LayoutEdge,
    points: list[QPointF],
    placements: dict[str, NodePlacement] | None,
) -> int:
    if not placements:
        return 0
    excluded = {edge.source_id, edge.target_id}
    hits = 0
    for first, second in zip(points, points[1:]):
        for node_id, placement in placements.items():
            if node_id in excluded:
                continue
            if _segment_hits_rect_interior(first, second, placement.rect):
                hits += 1
    return hits


def _segment_hits_rect_interior(
    first: QPointF, second: QPointF, rect: QRectF
) -> bool:
    inner = rect.adjusted(1.0, 1.0, -1.0, -1.0)
    if abs(first.y() - second.y()) < 0.01:
        y = first.y()
        left, right = sorted((first.x(), second.x()))
        return (
            inner.top() < y < inner.bottom()
            and max(left, inner.left()) < min(right, inner.right())
        )
    x = first.x()
    top, bottom = sorted((first.y(), second.y()))
    return (
        inner.left() < x < inner.right()
        and max(top, inner.top()) < min(bottom, inner.bottom())
    )


def _points_routes_collinearly_overlap(
    first_points: list[QPointF], second_points: list[QPointF]
) -> int:
    return int(
        any(
            _collinear_overlap(first_a, first_b, second_a, second_b)
            for first_a, first_b in zip(first_points, first_points[1:])
            for second_a, second_b in zip(second_points, second_points[1:])
        )
    )


def _overlapping_route_pairs(
    routes: dict[str, RoutedEdge],
) -> list[tuple[str, str]]:
    values = list(routes.values())
    pairs: list[tuple[str, str]] = []
    for index, first in enumerate(values):
        for second in values[index + 1 :]:
            if _routes_collinearly_overlap(first, second):
                pairs.append((first.edge.edge_id, second.edge.edge_id))
    return pairs


def _routes_collinearly_overlap(first: RoutedEdge, second: RoutedEdge) -> bool:
    for first_a, first_b in zip(first.points, first.points[1:]):
        for second_a, second_b in zip(second.points, second.points[1:]):
            if _collinear_overlap(first_a, first_b, second_a, second_b):
                return True
    return False


def _route_orientation_lengths(points: list[QPointF]) -> tuple[float, float]:
    horizontal = 0.0
    vertical = 0.0
    for first, second in zip(points, points[1:]):
        if abs(first.y() - second.y()) < 0.01:
            horizontal += abs(first.x() - second.x())
        else:
            vertical += abs(first.y() - second.y())
    return horizontal, vertical


def _offset_route_interior(
    points: list[QPointF], dx: float, dy: float
) -> list[QPointF]:
    if len(points) < 3 or (abs(dx) < 0.01 and abs(dy) < 0.01):
        return list(points)
    shifted = [QPointF(point.x() + dx, point.y() + dy) for point in points[1:-1]]
    result: list[QPointF] = [QPointF(points[0])]

    # Leave the source port directly onto the shifted lane.  Keeping the old
    # unshifted stub would make multiple routes coincide for their first segment.
    first_horizontal = abs(points[0].y() - points[1].y()) < 0.01
    first_shifted = shifted[0]
    if first_horizontal:
        elbow = QPointF(first_shifted.x(), points[0].y())
    else:
        elbow = QPointF(points[0].x(), first_shifted.y())
    if not _same_point(result[-1], elbow):
        result.append(elbow)
    if not _same_point(result[-1], first_shifted):
        result.append(first_shifted)

    for point in shifted[1:]:
        if not _same_point(result[-1], point):
            result.append(point)

    # Symmetric target connector.  It preserves the original target-side
    # orientation while avoiding a shared final stub.
    target = QPointF(points[-1])
    last_horizontal = abs(points[-2].y() - points[-1].y()) < 0.01
    current = result[-1]
    if last_horizontal:
        elbow = QPointF(target.x(), current.y())
    else:
        elbow = QPointF(current.x(), target.y())
    if not _same_point(current, elbow):
        result.append(elbow)
    if not _same_point(result[-1], target):
        result.append(target)
    return _simplify_orthogonal(result)


def _append_orthogonal_connection(
    result: list[QPointF], first: QPointF, second: QPointF
) -> None:
    """Append two points while preserving an orthogonal polyline."""
    current = result[-1]
    if not _same_point(current, first):
        if abs(current.x() - first.x()) > 0.01 and abs(current.y() - first.y()) > 0.01:
            result.append(QPointF(first.x(), current.y()))
        result.append(first)
    if abs(first.x() - second.x()) > 0.01 and abs(first.y() - second.y()) > 0.01:
        result.append(QPointF(second.x(), first.y()))
    result.append(second)


def _route_label_position(points: list[QPointF]) -> QPointF:
    segments = [
        (first, second, abs(first.x() - second.x()) + abs(first.y() - second.y()))
        for first, second in zip(points, points[1:])
        if not _same_point(first, second)
    ]
    if not segments:
        return points[0] if points else QPointF()
    first, second, _ = max(segments, key=lambda item: item[2])
    midpoint = QPointF(
        (first.x() + second.x()) / 2.0,
        (first.y() + second.y()) / 2.0,
    )
    return midpoint + (
        QPointF(0.0, -18.0)
        if abs(first.y() - second.y()) < 0.01
        else QPointF(8.0, 0.0)
    )
