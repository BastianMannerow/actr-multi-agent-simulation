"""Deterministic UX-oriented orthogonal graph layout.

The state layout follows the layered/Sugiyama family of algorithms:

* breadth-first rank assignment from the initial control state,
* median/barycentric crossing reduction inside every rank,
* compact vertical coordinate assignment,
* local inter-rank channels for ordinary forward transitions,
* interval-coloured side channels only for feedback and rank-skipping edges,
* independent N/E/S/W ports and orthogonal endpoint stubs.

Unlike the former visibility-grid/A* pipeline, routing is bounded and deterministic.
Its runtime is dominated by the crossing-reduction sweeps and interval colouring;
there are no global route-repair loops.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from statistics import median
from typing import Iterable, Literal

from PyQt6.QtCore import QPointF, QRectF

from gui.graph_layout import (
    LayoutEdge,
    LayoutNode,
    LayoutResult,
    NodePlacement,
    RoutedEdge,
    _points_bounds,
    _polyline_length,
    _rect_union,
    _route_label_position,
    _simplify_orthogonal,
)

Side = Literal["left", "right", "top", "bottom"]
RouteKind = Literal[
    "self",
    "adjacent",
    "same-rank-local",
    "same-rank-channel",
    "side-channel",
]


@dataclass(slots=True)
class _Plan:
    edge: LayoutEdge
    kind: RouteKind
    source_side: Side
    target_side: Side
    source_rank: int
    target_rank: int
    side: Side | None = None
    lane: int = 0
    track: int = 0


@dataclass(slots=True)
class _Port:
    border: QPointF
    external: QPointF
    side: Side


class StateGraphUXLayout:
    """Layered state-graph layout optimised for cyclic ACT-R control graphs."""

    def __init__(
        self,
        *,
        node_gap: float = 72.0,
        base_rank_gap: float = 108.0,
        lane_gap: float = 22.0,
        outer_margin: float = 58.0,
        port_stub: float = 18.0,
        crossing_sweeps: int = 8,
    ) -> None:
        self.node_gap = node_gap
        self.base_rank_gap = base_rank_gap
        self.lane_gap = lane_gap
        self.outer_margin = outer_margin
        self.port_stub = port_stub
        self.crossing_sweeps = crossing_sweeps

    def layout(
        self,
        nodes: Iterable[LayoutNode],
        edges: Iterable[LayoutEdge],
        *,
        initial_node_id: str,
        offset: QPointF = QPointF(42.0, 150.0),
    ) -> LayoutResult:
        node_list = list(nodes)
        edge_list = list(edges)
        if not node_list:
            return LayoutResult({}, {}, "vertical", QRectF(), {})
        node_by_id = {node.node_id: node for node in node_list}
        if initial_node_id not in node_by_id:
            initial_node_id = node_list[0].node_id

        ranks = self._assign_ranks(node_list, edge_list, initial_node_id)
        rank_nodes = self._ordered_ranks(node_list, edge_list, ranks, initial_node_id)
        rank_gaps = self._rank_gaps(rank_nodes, edge_list, ranks)
        placements = self._place_nodes(rank_nodes, ranks, rank_gaps, offset)
        plans = self._plan_routes(edge_list, placements)
        expanded_gaps = self._expand_rank_gaps_for_ports(rank_gaps, plans)
        if expanded_gaps != rank_gaps:
            rank_gaps = expanded_gaps
            placements = self._place_nodes(rank_nodes, ranks, rank_gaps, offset)
            plans = self._plan_routes(edge_list, placements)
        ports = self._allocate_ports(plans, placements)
        routes = self._materialize_routes(plans, ports, placements, rank_gaps)

        # Outer feedback lanes can extend left of the requested offset. Move the
        # complete graph right once, preserving the protected legend margin.
        all_rects = [placement.rect for placement in placements.values()]
        all_rects.extend(_points_bounds(route.points) for route in routes.values())
        bounds = _rect_union(all_rects)
        minimum_left = offset.x()
        if bounds.left() < minimum_left:
            delta = minimum_left - bounds.left()
            for placement in placements.values():
                placement.rect.translate(delta, 0.0)
            for route in routes.values():
                route.points = [QPointF(p.x() + delta, p.y()) for p in route.points]
                route.label_position = QPointF(
                    route.label_position.x() + delta,
                    route.label_position.y(),
                )
            all_rects = [placement.rect for placement in placements.values()]
            all_rects.extend(_points_bounds(route.points) for route in routes.values())
            bounds = _rect_union(all_rects)

        return LayoutResult(
            placements=placements,
            routes=routes,
            orientation="vertical",
            bounds=bounds,
            group_headers={},
        )

    @staticmethod
    def _assign_ranks(
        nodes: list[LayoutNode],
        edges: list[LayoutEdge],
        initial_node_id: str,
    ) -> dict[str, int]:
        outgoing: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            if edge.source_id != edge.target_id:
                outgoing[edge.source_id].append(edge.target_id)
        for values in outgoing.values():
            values.sort()

        ranks = {initial_node_id: 0}
        queue = deque([initial_node_id])
        while queue:
            source = queue.popleft()
            next_rank = ranks[source] + 1
            for target in outgoing.get(source, []):
                if target not in ranks:
                    ranks[target] = next_rank
                    queue.append(target)
        fallback = max(ranks.values(), default=0) + 1
        for node in sorted(nodes, key=lambda item: (item.group, item.label, item.node_id)):
            if node.node_id not in ranks:
                ranks[node.node_id] = fallback
                fallback += 1
        return ranks

    def _ordered_ranks(
        self,
        nodes: list[LayoutNode],
        edges: list[LayoutEdge],
        ranks: dict[str, int],
        initial_node_id: str,
    ) -> dict[int, list[str]]:
        rank_nodes: dict[int, list[str]] = defaultdict(list)
        node_by_id = {node.node_id: node for node in nodes}
        for node in nodes:
            rank_nodes[ranks[node.node_id]].append(node.node_id)

        # Stable semantic seed order. The initial node is always first.
        for rank, values in rank_nodes.items():
            values.sort(
                key=lambda node_id: (
                    0 if node_id == initial_node_id else 1,
                    node_by_id[node_id].group,
                    -node_by_id[node_id].priority,
                    node_by_id[node_id].label,
                    node_id,
                )
            )

        incoming: dict[str, list[str]] = defaultdict(list)
        outgoing: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            if edge.source_id == edge.target_id:
                continue
            incoming[edge.target_id].append(edge.source_id)
            outgoing[edge.source_id].append(edge.target_id)

        max_rank = max(rank_nodes, default=0)
        for _ in range(self.crossing_sweeps):
            order = {
                node_id: index
                for rank in sorted(rank_nodes)
                for index, node_id in enumerate(rank_nodes[rank])
            }
            for rank in range(1, max_rank + 1):
                values = rank_nodes.get(rank, [])
                values.sort(
                    key=lambda node_id: (
                        self._neighbour_median(
                            incoming.get(node_id, []), order, ranks, rank, before=True
                        ),
                        order.get(node_id, 0),
                    )
                )
            order = {
                node_id: index
                for rank in sorted(rank_nodes)
                for index, node_id in enumerate(rank_nodes[rank])
            }
            for rank in range(max_rank - 1, -1, -1):
                values = rank_nodes.get(rank, [])
                values.sort(
                    key=lambda node_id: (
                        self._neighbour_median(
                            outgoing.get(node_id, []), order, ranks, rank, before=False
                        ),
                        order.get(node_id, 0),
                    )
                )
        return dict(rank_nodes)

    @staticmethod
    def _neighbour_median(
        neighbours: list[str],
        order: dict[str, int],
        ranks: dict[str, int],
        rank: int,
        *,
        before: bool,
    ) -> float:
        values = [
            order[node_id]
            for node_id in neighbours
            if (ranks.get(node_id, rank) < rank if before else ranks.get(node_id, rank) > rank)
        ]
        return float(median(values)) if values else float("inf")

    def _rank_gaps(
        self,
        rank_nodes: dict[int, list[str]],
        edges: list[LayoutEdge],
        ranks: dict[str, int],
    ) -> dict[int, float]:
        adjacent_counts: dict[int, int] = defaultdict(int)
        for edge in edges:
            source_rank = ranks[edge.source_id]
            target_rank = ranks[edge.target_id]
            if target_rank == source_rank + 1:
                adjacent_counts[source_rank] += 1
        result: dict[int, float] = {}
        for rank in range(max(rank_nodes, default=0)):
            count = adjacent_counts.get(rank, 0)
            result[rank] = max(
                self.base_rank_gap,
                42.0 + max(1, count) * self.lane_gap,
            )
        return result

    def _expand_rank_gaps_for_ports(
        self,
        rank_gaps: dict[int, float],
        plans: list[_Plan],
    ) -> dict[int, float]:
        bottom: dict[int, int] = defaultdict(int)
        top: dict[int, int] = defaultdict(int)
        for plan in plans:
            if plan.source_side == "bottom":
                bottom[plan.source_rank] += 1
            elif plan.source_side == "top":
                top[plan.source_rank] += 1
            if plan.target_side == "bottom":
                bottom[plan.target_rank] += 1
            elif plan.target_side == "top":
                top[plan.target_rank] += 1
        result = dict(rank_gaps)
        for rank in result:
            required = 56.0 + (bottom.get(rank, 0) + top.get(rank + 1, 0)) * self.lane_gap
            result[rank] = max(result[rank], required)
        return result

    def _place_nodes(
        self,
        rank_nodes: dict[int, list[str]],
        ranks: dict[str, int],
        rank_gaps: dict[int, float],
        offset: QPointF,
    ) -> dict[str, NodePlacement]:
        # The LayoutNode objects are recovered from a temporary map set by caller.
        # This method is filled by replacing ids after width calculations below.
        raise NotImplementedError

    def _plan_routes(
        self,
        edges: list[LayoutEdge],
        placements: dict[str, NodePlacement],
    ) -> list[_Plan]:
        by_rank: dict[int, list[str]] = defaultdict(list)
        for node_id, placement in placements.items():
            by_rank[placement.rank].append(node_id)
        for values in by_rank.values():
            values.sort(key=lambda node_id: placements[node_id].rect.center().x())
        position = {
            node_id: index
            for rank, values in by_rank.items()
            for index, node_id in enumerate(values)
        }

        plans: list[_Plan] = []
        side_balance = {"left": 0, "right": 0}
        graph_center = _rect_union(
            placement.rect for placement in placements.values()
        ).center().x()
        for edge in sorted(edges, key=lambda item: (item.source_id, item.target_id, item.kind, item.edge_id)):
            source = placements[edge.source_id]
            target = placements[edge.target_id]
            source_rank = source.rank
            target_rank = target.rank
            if edge.source_id == edge.target_id:
                plans.append(_Plan(edge, "self", "right", "right", source_rank, target_rank))
                continue
            if target_rank == source_rank + 1:
                plans.append(_Plan(edge, "adjacent", "bottom", "top", source_rank, target_rank))
                continue
            if target_rank == source_rank:
                if abs(position[edge.source_id] - position[edge.target_id]) == 1:
                    if source.rect.center().x() < target.rect.center().x():
                        sides: tuple[Side, Side] = ("right", "left")
                    else:
                        sides = ("left", "right")
                    plans.append(_Plan(edge, "same-rank-local", *sides, source_rank, target_rank))
                else:
                    # Non-local edges within one rank use a dedicated horizontal
                    # channel above or below the complete row.
                    side: Side = "top" if source_rank % 2 == 0 else "bottom"
                    plans.append(
                        _Plan(
                            edge, "same-rank-channel", side, side,
                            source_rank, target_rank, side=side
                        )
                    )
                continue

            midpoint = (source.rect.center().x() + target.rect.center().x()) / 2.0
            preferred: Side = "left" if midpoint < graph_center else "right"
            alternate: Side = "right" if preferred == "left" else "left"
            side = preferred if side_balance[preferred] <= side_balance[alternate] + 2 else alternate
            side_balance[side] += 1
            preferred_source: Side = "bottom" if target_rank > source_rank else "top"
            preferred_target: Side = preferred_source
            source_port_side, target_port_side = self._best_side_channel_ports(
                edge, placements, side, preferred_source, preferred_target
            )
            plans.append(
                _Plan(
                    edge, "side-channel", source_port_side, target_port_side,
                    source_rank, target_rank, side=side
                )
            )

        self._rebalance_side_channel_ports(plans, placements)
        self._assign_adjacent_tracks(plans, placements)
        self._assign_side_lanes(plans)
        self._assign_same_rank_tracks(plans, placements)
        return plans

    def _best_side_channel_ports(
        self,
        edge: LayoutEdge,
        placements: dict[str, NodePlacement],
        lane_side: Side,
        preferred_source: Side,
        preferred_target: Side,
    ) -> tuple[Side, Side]:
        """Choose endpoint sides by bounded obstacle scoring.

        This replaces the old global repair loop. Sixteen short candidates are
        evaluated before routing; the selected candidate is the shortest one
        that does not cross another node.
        """
        graph = _rect_union(p.rect for p in placements.values())
        lane_x = (
            graph.left() - self.outer_margin
            if lane_side == "left"
            else graph.right() + self.outer_margin
        )
        source_rect = placements[edge.source_id].rect
        target_rect = placements[edge.target_id].rect
        sides: tuple[Side, ...] = ("top", "bottom", "left", "right")
        best: tuple[tuple[float, ...], tuple[Side, Side]] | None = None
        for source_side in sides:
            for target_side in sides:
                source_border = self._point_on_side(source_rect, source_side, 0.5)
                target_border = self._point_on_side(target_rect, target_side, 0.5)
                source_external = self._move_from_side(source_border, source_side, self.port_stub)
                target_external = self._move_from_side(target_border, target_side, self.port_stub)
                points = _simplify_orthogonal([
                    source_border,
                    source_external,
                    QPointF(lane_x, source_external.y()),
                    QPointF(lane_x, target_external.y()),
                    target_external,
                    target_border,
                ])
                hits = 0
                for node_id, placement in placements.items():
                    if node_id in {edge.source_id, edge.target_id}:
                        continue
                    for first, second in zip(points, points[1:]):
                        if _segment_hits_rectangle(first, second, placement.rect):
                            hits += 1
                            break
                preference = (
                    int(source_side != preferred_source)
                    + int(target_side != preferred_target)
                    + 0.35 * int(source_side != lane_side)
                    + 0.35 * int(target_side != lane_side)
                )
                score = (
                    float(hits),
                    _polyline_length(points),
                    float(max(0, len(points) - 2)),
                    preference,
                )
                if best is None or score < best[0]:
                    best = (score, (source_side, target_side))
        assert best is not None
        return best[1]

    def _rebalance_side_channel_ports(
        self, plans: list[_Plan], placements: dict[str, NodePlacement]
    ) -> None:
        """Distribute high-degree endpoints over the complete node perimeter."""
        incident: dict[str, list[tuple[_Plan, bool]]] = defaultdict(list)
        for plan in plans:
            incident[plan.edge.source_id].append((plan, True))
            incident[plan.edge.target_id].append((plan, False))

        for node_id, endpoints in incident.items():
            rect = placements[node_id].rect
            capacities = {
                "top": max(1, int((rect.width() - 24.0) // self.lane_gap)),
                "bottom": max(1, int((rect.width() - 24.0) // self.lane_gap)),
                "left": max(1, int((rect.height() - 24.0) // self.lane_gap)),
                "right": max(1, int((rect.height() - 24.0) // self.lane_gap)),
            }
            occupancy: dict[Side, int] = defaultdict(int)
            movable: list[tuple[_Plan, bool]] = []
            for plan, is_source in endpoints:
                side = plan.source_side if is_source else plan.target_side
                if plan.kind == "side-channel":
                    movable.append((plan, is_source))
                else:
                    occupancy[side] += 1

            movable.sort(
                key=lambda item: (
                    item[0].source_rank if item[1] else item[0].target_rank,
                    item[0].edge.kind, item[0].edge.edge_id, int(item[1]),
                )
            )
            for plan, is_source in movable:
                current = plan.source_side if is_source else plan.target_side
                other = plan.target_side if is_source else plan.source_side
                ranked: list[tuple[tuple[float, ...], Side]] = []
                for side in ("top", "bottom", "left", "right"):
                    if is_source:
                        source_side, target_side = side, other
                    else:
                        source_side, target_side = other, side
                    hits, length, bends = self._side_channel_geometry_score(
                        plan.edge, placements, plan.side or "right",
                        source_side, target_side
                    )
                    overflow = max(0, occupancy[side] + 1 - capacities[side])
                    fill = occupancy[side] / max(1, capacities[side])
                    ranked.append(((float(hits), float(overflow), fill,
                                    0.0 if side == current else 1.0,
                                    length, float(bends)), side))
                chosen = min(ranked, key=lambda item: item[0])[1]
                if is_source:
                    plan.source_side = chosen
                else:
                    plan.target_side = chosen
                occupancy[chosen] += 1

    def _side_channel_geometry_score(
        self,
        edge: LayoutEdge,
        placements: dict[str, NodePlacement],
        lane_side: Side,
        source_side: Side,
        target_side: Side,
    ) -> tuple[int, float, int]:
        graph = _rect_union(p.rect for p in placements.values())
        lane_x = graph.left() - self.outer_margin if lane_side == "left" else graph.right() + self.outer_margin
        source_rect = placements[edge.source_id].rect
        target_rect = placements[edge.target_id].rect
        source_border = self._point_on_side(source_rect, source_side, 0.5)
        target_border = self._point_on_side(target_rect, target_side, 0.5)
        source_external = self._move_from_side(source_border, source_side, self.port_stub)
        target_external = self._move_from_side(target_border, target_side, self.port_stub)
        points = _simplify_orthogonal([
            source_border, source_external,
            QPointF(lane_x, source_external.y()),
            QPointF(lane_x, target_external.y()),
            target_external, target_border,
        ])
        hits = 0
        for node_id, placement in placements.items():
            if node_id in {edge.source_id, edge.target_id}:
                continue
            if any(_segment_hits_rectangle(a, b, placement.rect) for a, b in zip(points, points[1:])):
                hits += 1
        return hits, _polyline_length(points), max(0, len(points) - 2)

    @staticmethod
    def _assign_adjacent_tracks(plans: list[_Plan], placements: dict[str, NodePlacement]) -> None:
        groups: dict[tuple[int, int], list[_Plan]] = defaultdict(list)
        for plan in plans:
            if plan.kind == "adjacent":
                groups[(plan.source_rank, plan.target_rank)].append(plan)
        for values in groups.values():
            values.sort(
                key=lambda plan: (
                    placements[plan.edge.source_id].rect.center().x()
                    + placements[plan.edge.target_id].rect.center().x(),
                    plan.edge.kind,
                    plan.edge.edge_id,
                )
            )
            for index, plan in enumerate(values):
                plan.track = index

    @staticmethod
    def _assign_side_lanes(plans: list[_Plan]) -> None:
        for side in ("left", "right", "top", "bottom"):
            values = [
                plan for plan in plans
                if plan.side == side and plan.kind in {"side-channel", "same-rank-channel"}
            ]
            values.sort(
                key=lambda plan: (
                    min(plan.source_rank, plan.target_rank),
                    max(plan.source_rank, plan.target_rank),
                    plan.edge.kind,
                    plan.edge.edge_id,
                )
            )
            lane_ends: list[int] = []
            for plan in values:
                start = min(plan.source_rank, plan.target_rank)
                end = max(plan.source_rank, plan.target_rank)
                lane = next(
                    (index for index, lane_end in enumerate(lane_ends) if lane_end < start),
                    len(lane_ends),
                )
                if lane == len(lane_ends):
                    lane_ends.append(end)
                else:
                    lane_ends[lane] = end
                plan.lane = lane

    @staticmethod
    def _assign_same_rank_tracks(plans: list[_Plan], placements: dict[str, NodePlacement]) -> None:
        groups: dict[int, list[_Plan]] = defaultdict(list)
        for plan in plans:
            if plan.kind == "same-rank-local":
                groups[plan.source_rank].append(plan)
        for values in groups.values():
            values.sort(
                key=lambda plan: (
                    min(
                        placements[plan.edge.source_id].rect.center().x(),
                        placements[plan.edge.target_id].rect.center().x(),
                    ),
                    plan.edge.edge_id,
                )
            )
            for index, plan in enumerate(values):
                plan.track = index

    def _allocate_ports(
        self,
        plans: list[_Plan],
        placements: dict[str, NodePlacement],
    ) -> dict[tuple[str, bool], _Port]:
        requests: dict[tuple[str, Side], list[tuple[_Plan, bool, float]]] = defaultdict(list)
        for plan in plans:
            source_other = placements[plan.edge.target_id].rect.center()
            target_other = placements[plan.edge.source_id].rect.center()
            requests[(plan.edge.source_id, plan.source_side)].append(
                (plan, True, self._port_sort_coordinate(source_other, plan.source_side))
            )
            requests[(plan.edge.target_id, plan.target_side)].append(
                (plan, False, self._port_sort_coordinate(target_other, plan.target_side))
            )

        global_tracks: dict[tuple[str, bool], int] = {}
        global_track_counts: dict[tuple[int, Side], int] = {}
        rank_side_requests: dict[tuple[int, Side], list[tuple[_Plan, bool, float]]] = defaultdict(list)
        for (node_id, side), values in requests.items():
            if side not in {"top", "bottom"}:
                continue
            rank = placements[node_id].rank
            rank_side_requests[(rank, side)].extend(values)
        for rank_side, values in rank_side_requests.items():
            values.sort(
                key=lambda item: (
                    placements[item[0].edge.source_id if item[1] else item[0].edge.target_id].rect.center().x(),
                    item[2], item[0].edge.kind, item[0].edge.edge_id,
                )
            )
            global_track_counts[rank_side] = len(values)
            for track, (plan, is_source, _coordinate) in enumerate(values):
                global_tracks[(plan.edge.edge_id, is_source)] = track

        result: dict[tuple[str, bool], _Port] = {}
        for (node_id, side), values in requests.items():
            values.sort(key=lambda item: (item[2], item[0].edge.kind, item[0].edge.edge_id))
            rect = placements[node_id].rect
            count = len(values)
            for index, (plan, is_source, _coordinate) in enumerate(values):
                # A single incident edge uses the geometric centre. Multiple
                # edges are distributed uniformly. Hash-based jitter made linear
                # chains needlessly zig-zag and is intentionally avoided.
                fraction = (index + 1) / (count + 1)
                border = self._point_on_side(rect, side, fraction)
                distance = self.port_stub
                if side in {"top", "bottom"}:
                    distance += global_tracks.get((plan.edge.edge_id, is_source), 0) * self.lane_gap
                external = self._move_from_side(border, side, distance)
                result[(plan.edge.edge_id, is_source)] = _Port(border, external, side)
        return result

    @staticmethod
    def _port_sort_coordinate(point: QPointF, side: Side) -> float:
        return point.y() if side in {"left", "right"} else point.x()

    @staticmethod
    def _point_on_side(rect: QRectF, side: Side, fraction: float) -> QPointF:
        padding = 11.0
        if side in {"left", "right"}:
            usable = max(1.0, rect.height() - 2 * padding)
            y = rect.top() + padding + usable * fraction
            return QPointF(rect.left() if side == "left" else rect.right(), y)
        usable = max(1.0, rect.width() - 2 * padding)
        x = rect.left() + padding + usable * fraction
        return QPointF(x, rect.top() if side == "top" else rect.bottom())

    @staticmethod
    def _move_from_side(point: QPointF, side: Side, distance: float) -> QPointF:
        if side == "left":
            return QPointF(point.x() - distance, point.y())
        if side == "right":
            return QPointF(point.x() + distance, point.y())
        if side == "top":
            return QPointF(point.x(), point.y() - distance)
        return QPointF(point.x(), point.y() + distance)

    def _materialize_routes(
        self,
        plans: list[_Plan],
        ports: dict[tuple[str, bool], _Port],
        placements: dict[str, NodePlacement],
        rank_gaps: dict[int, float],
    ) -> dict[str, RoutedEdge]:
        graph_bounds = _rect_union(placement.rect for placement in placements.values())
        left_base = graph_bounds.left() - self.outer_margin
        right_base = graph_bounds.right() + self.outer_margin
        rank_bounds: dict[int, QRectF] = {}
        for placement in placements.values():
            current = rank_bounds.get(placement.rank)
            rank_bounds[placement.rank] = (
                placement.rect if current is None else current.united(placement.rect)
            )

        adjacent_groups: dict[tuple[int, int], list[_Plan]] = defaultdict(list)
        for plan in plans:
            if plan.kind == "adjacent":
                adjacent_groups[(plan.source_rank, plan.target_rank)].append(plan)

        routes: dict[str, RoutedEdge] = {}
        for plan in plans:
            source = ports[(plan.edge.edge_id, True)]
            target = ports[(plan.edge.edge_id, False)]
            points: list[QPointF]
            if plan.kind == "self":
                rect = placements[plan.edge.source_id].rect
                x = rect.right() + self.outer_margin + plan.lane * self.lane_gap
                top = min(source.external.y(), target.external.y()) - 28.0 - plan.track * self.lane_gap
                points = [
                    source.border,
                    source.external,
                    QPointF(x, source.external.y()),
                    QPointF(x, top),
                    QPointF(target.external.x(), top),
                    target.external,
                    target.border,
                ]
            elif plan.kind == "adjacent":
                values = adjacent_groups[(plan.source_rank, plan.target_rank)]
                source_rank_rect = rank_bounds[plan.source_rank]
                target_rank_rect = rank_bounds[plan.target_rank]
                available_top = source_rank_rect.bottom() + self.port_stub + 14.0
                available_bottom = target_rank_rect.top() - self.port_stub - 14.0
                count = max(1, len(values))
                if count == 1:
                    y = (available_top + available_bottom) / 2.0
                else:
                    step = max(
                        self.lane_gap,
                        (available_bottom - available_top) / max(1, count - 1),
                    )
                    y = available_top + plan.track * step
                    y = min(y, available_bottom)
                centered_track = plan.track - (count - 1) / 2.0
                track_x = (source.external.x() + target.external.x()) / 2.0 + centered_track * self.lane_gap
                points = [
                    source.border,
                    source.external,
                    QPointF(track_x, source.external.y()),
                    QPointF(track_x, y),
                    QPointF(target.external.x(), y),
                    target.external,
                    target.border,
                ]
            elif plan.kind == "same-rank-local":
                midpoint_x = (source.external.x() + target.external.x()) / 2.0
                offset = (plan.track - 0.5) * self.lane_gap
                midpoint_x += offset
                points = [
                    source.border,
                    source.external,
                    QPointF(midpoint_x, source.external.y()),
                    QPointF(midpoint_x, target.external.y()),
                    target.external,
                    target.border,
                ]
            elif plan.kind == "same-rank-channel":
                row = rank_bounds[plan.source_rank]
                side = plan.side or "top"
                lane_y = (
                    row.top() - self.outer_margin - plan.lane * self.lane_gap
                    if side == "top"
                    else row.bottom() + self.outer_margin + plan.lane * self.lane_gap
                )
                points = [
                    source.border,
                    source.external,
                    QPointF(source.external.x(), lane_y),
                    QPointF(target.external.x(), lane_y),
                    target.external,
                    target.border,
                ]
            else:
                side = plan.side or "right"
                lane_x = (
                    left_base - plan.lane * self.lane_gap
                    if side == "left"
                    else right_base + plan.lane * self.lane_gap
                )
                points = [
                    source.border,
                    source.external,
                    QPointF(lane_x, source.external.y()),
                    QPointF(lane_x, target.external.y()),
                    target.external,
                    target.border,
                ]
            points = _simplify_orthogonal(points)
            routes[plan.edge.edge_id] = RoutedEdge(
                edge=plan.edge,
                points=points,
                source_side=source.side,
                target_side=target.side,
                label_position=_route_label_position(points),
            )
        return routes


def _segment_hits_rectangle(first: QPointF, second: QPointF, rect: QRectF) -> bool:
    """Return True only when an orthogonal segment crosses rectangle interior."""
    epsilon = 0.5
    inner = rect.adjusted(epsilon, epsilon, -epsilon, -epsilon)
    if abs(first.x() - second.x()) < 0.01:
        x = first.x()
        low, high = sorted((first.y(), second.y()))
        return inner.left() < x < inner.right() and high > inner.top() and low < inner.bottom()
    y = first.y()
    low, high = sorted((first.x(), second.x()))
    return inner.top() < y < inner.bottom() and high > inner.left() and low < inner.right()


class _ConcreteStateGraphUXLayout(StateGraphUXLayout):
    """Implementation subclass carrying the node map during placement."""

    def __init__(self, node_by_id: dict[str, LayoutNode], **kwargs) -> None:
        super().__init__(**kwargs)
        self.node_by_id = node_by_id

    def _place_nodes(
        self,
        rank_nodes: dict[int, list[str]],
        ranks: dict[str, int],
        rank_gaps: dict[int, float],
        offset: QPointF,
    ) -> dict[str, NodePlacement]:
        rank_widths: dict[int, float] = {}
        rank_heights: dict[int, float] = {}
        for rank, values in rank_nodes.items():
            widths = [self.node_by_id[node_id].width for node_id in values]
            rank_widths[rank] = sum(widths) + max(0, len(values) - 1) * self.node_gap
            rank_heights[rank] = max(self.node_by_id[node_id].height for node_id in values)
        graph_width = max(rank_widths.values(), default=1.0)

        placements: dict[str, NodePlacement] = {}
        y = offset.y()
        for rank in sorted(rank_nodes):
            values = rank_nodes[rank]
            x = offset.x() + (graph_width - rank_widths[rank]) / 2.0
            height = rank_heights[rank]
            for order, node_id in enumerate(values):
                node = self.node_by_id[node_id]
                rect = QRectF(x, y + (height - node.height) / 2.0, node.width, node.height)
                placements[node_id] = NodePlacement(node, rect, rank, order)
                x += node.width + self.node_gap
            y += height + rank_gaps.get(rank, self.base_rank_gap)
        return placements


def layout_state_graph(
    nodes: Iterable[LayoutNode],
    edges: Iterable[LayoutEdge],
    *,
    initial_node_id: str,
    offset: QPointF = QPointF(42.0, 150.0),
) -> LayoutResult:
    node_list = list(nodes)
    node_by_id = {node.node_id: node for node in node_list}
    result = _ConcreteStateGraphUXLayout(node_by_id).layout(
        node_list,
        list(edges),
        initial_node_id=initial_node_id,
        offset=offset,
    )
    return normalize_route_lanes(result, minimum_gap=20.0)


def route_quality(result: LayoutResult) -> dict[str, float | int]:
    """Return deterministic layout diagnostics for tests and documentation."""
    total_length = sum(_polyline_length(route.points) for route in result.routes.values())
    bends = sum(max(0, len(route.points) - 2) for route in result.routes.values())
    maximum_bends = max((max(0, len(route.points) - 2) for route in result.routes.values()), default=0)
    return {
        "nodes": len(result.placements),
        "edges": len(result.routes),
        "total_manhattan_length": round(total_length, 3),
        "total_bends": bends,
        "maximum_bends_per_edge": maximum_bends,
        "width": round(result.bounds.width(), 3),
        "height": round(result.bounds.height(), 3),
    }


def normalize_route_lanes(
    result: LayoutResult,
    *,
    minimum_gap: float = 20.0,
    max_repairs: int = 160,
) -> LayoutResult:
    """Resolve residual lane conflicts with incremental local scoring.

    The previous normalizer copied and rescored the complete route set for every
    candidate dogleg. That made dense cyclic graphs effectively quadratic inside
    another repair loop. This implementation evaluates only the route that would
    change. The global conflict count changes exactly by the difference between
    that route's old and new local conflict counts, so no full-graph trial copy is
    required.
    """
    routes = {
        edge_id: RoutedEdge(
            edge=route.edge,
            points=[QPointF(point) for point in route.points],
            source_side=route.source_side,
            target_side=route.target_side,
            label_position=QPointF(route.label_position),
        )
        for edge_id, route in result.routes.items()
    }

    if _first_lane_conflict(routes, minimum_gap) is None:
        return result

    for _ in range(max_repairs):
        conflict = _first_lane_conflict(routes, minimum_gap)
        if conflict is None:
            break
        first_id, first_index, second_id, second_index = conflict
        candidates: list[tuple[tuple[float, ...], str, list[QPointF]]] = []

        for route_id, segment_index, other_id, other_index in (
            (first_id, first_index, second_id, second_index),
            (second_id, second_index, first_id, first_index),
        ):
            route = routes[route_id]
            other = routes[other_id]
            a, b = route.points[segment_index], route.points[segment_index + 1]
            c, d = other.points[other_index], other.points[other_index + 1]
            horizontal = abs(a.y() - b.y()) < 0.01
            other_coordinate = c.y() if horizontal else c.x()
            current_coordinate = a.y() if horizontal else a.x()
            old_conflicts, old_overlaps = _route_lane_conflict_score(
                route_id, route.points, routes, minimum_gap
            )
            old_metric = (old_overlaps, old_conflicts)

            # Nearby lanes first. The coordinate set is deliberately bounded: a
            # local repair must never turn into a perimeter-spanning detour.
            coordinates: list[float] = []
            for step in (1, 2, 3):
                delta = minimum_gap * step
                coordinates.extend(
                    (
                        other_coordinate - delta,
                        other_coordinate + delta,
                        current_coordinate - delta,
                        current_coordinate + delta,
                    )
                )
            seen: set[float] = set()
            for coordinate in coordinates:
                rounded = round(coordinate, 4)
                if rounded in seen or abs(coordinate - current_coordinate) < 0.01:
                    continue
                seen.add(rounded)
                points = _detour_segment(route.points, segment_index, coordinate)
                if points is None:
                    continue
                if _route_hits_foreign_node(route.edge, points, result.placements):
                    continue
                conflicts, overlaps = _route_lane_conflict_score(
                    route_id, points, routes, minimum_gap
                )
                metric = (overlaps, conflicts)
                if metric >= old_metric:
                    continue
                score = (
                    float(overlaps),
                    float(conflicts),
                    float(max(0, len(points) - 2)),
                    _polyline_length(points),
                    abs(coordinate - current_coordinate),
                )
                candidates.append((score, route_id, points))

        if not candidates:
            # No bounded local move improves the graph. Keeping the current route
            # is preferable to the long loops produced by global escape routing.
            break

        _score, route_id, points = min(candidates, key=lambda item: item[0])
        route = routes[route_id]
        routes[route_id] = RoutedEdge(
            edge=route.edge,
            points=points,
            source_side=route.source_side,
            target_side=route.target_side,
            label_position=_route_label_position(points),
        )

    all_rects = [placement.rect for placement in result.placements.values()]
    all_rects.extend(_points_bounds(route.points) for route in routes.values())
    return LayoutResult(
        placements=result.placements,
        routes=routes,
        orientation=result.orientation,
        bounds=_rect_union(all_rects),
        group_headers=result.group_headers,
    )


def _first_lane_conflict(
    routes: dict[str, RoutedEdge], minimum_gap: float
) -> tuple[str, int, str, int] | None:
    values = sorted(routes.items())
    for first_pos, (first_id, first) in enumerate(values):
        for second_id, second in values[first_pos + 1 :]:
            for first_index, (a, b) in enumerate(zip(first.points, first.points[1:])):
                for second_index, (c, d) in enumerate(zip(second.points, second.points[1:])):
                    distance, overlap = _parallel_distance_and_overlap(a, b, c, d)
                    if overlap > 1.0 and distance < minimum_gap - 0.01:
                        return first_id, first_index, second_id, second_index
    return None


def _lane_conflict_score(
    routes: dict[str, RoutedEdge], minimum_gap: float
) -> tuple[int, int]:
    conflicts = 0
    overlaps = 0
    values = list(routes.values())
    for first_pos, first in enumerate(values):
        for second in values[first_pos + 1 :]:
            for a, b in zip(first.points, first.points[1:]):
                for c, d in zip(second.points, second.points[1:]):
                    distance, overlap = _parallel_distance_and_overlap(a, b, c, d)
                    if overlap <= 1.0 or distance >= minimum_gap - 0.01:
                        continue
                    conflicts += 1
                    if distance < 0.01:
                        overlaps += 1
    return conflicts, overlaps


def _route_lane_conflict_score(
    route_id: str,
    points: list[QPointF],
    routes: dict[str, RoutedEdge],
    minimum_gap: float,
) -> tuple[int, int]:
    """Count only conflicts involving one candidate route."""
    conflicts = 0
    overlaps = 0
    for other_id, other in routes.items():
        if other_id == route_id:
            continue
        for a, b in zip(points, points[1:]):
            for c, d in zip(other.points, other.points[1:]):
                distance, overlap = _parallel_distance_and_overlap(a, b, c, d)
                if overlap <= 1.0 or distance >= minimum_gap - 0.01:
                    continue
                conflicts += 1
                if distance < 0.01:
                    overlaps += 1
    return conflicts, overlaps

def _parallel_distance_and_overlap(
    a: QPointF, b: QPointF, c: QPointF, d: QPointF
) -> tuple[float, float]:
    first_horizontal = abs(a.y() - b.y()) < 0.01
    second_horizontal = abs(c.y() - d.y()) < 0.01
    if first_horizontal != second_horizontal:
        return float("inf"), 0.0
    if first_horizontal:
        overlap = min(max(a.x(), b.x()), max(c.x(), d.x())) - max(
            min(a.x(), b.x()), min(c.x(), d.x())
        )
        return abs(a.y() - c.y()), overlap
    overlap = min(max(a.y(), b.y()), max(c.y(), d.y())) - max(
        min(a.y(), b.y()), min(c.y(), d.y())
    )
    return abs(a.x() - c.x()), overlap


def _detour_segment(
    points: list[QPointF], segment_index: int, coordinate: float
) -> list[QPointF] | None:
    if not (0 <= segment_index < len(points) - 1):
        return None
    first = points[segment_index]
    second = points[segment_index + 1]
    horizontal = abs(first.y() - second.y()) < 0.01
    if horizontal:
        replacement = [
            QPointF(first.x(), first.y()),
            QPointF(first.x(), coordinate),
            QPointF(second.x(), coordinate),
            QPointF(second.x(), second.y()),
        ]
    elif abs(first.x() - second.x()) < 0.01:
        replacement = [
            QPointF(first.x(), first.y()),
            QPointF(coordinate, first.y()),
            QPointF(coordinate, second.y()),
            QPointF(second.x(), second.y()),
        ]
    else:
        return None
    rebuilt = [QPointF(p) for p in points[:segment_index]]
    rebuilt.extend(replacement)
    rebuilt.extend(QPointF(p) for p in points[segment_index + 2 :])
    # Preserve endpoint stubs. Simplify only duplicate/collinear interior points.
    return _simplify_orthogonal(rebuilt)


def _route_hits_foreign_node(
    edge: LayoutEdge,
    points: list[QPointF],
    placements: dict[str, NodePlacement],
) -> bool:
    for node_id, placement in placements.items():
        if node_id in {edge.source_id, edge.target_id}:
            continue
        if any(
            _segment_hits_rectangle(first, second, placement.rect)
            for first, second in zip(points, points[1:])
        ):
            return True
    return False


def route_semantic_columns(
    node_rects: dict[str, QRectF],
    edges: Iterable[LayoutEdge],
    *,
    column_by_node: dict[str, int],
    section_top: float,
    lane_gap: float = 22.0,
) -> dict[str, RoutedEdge]:
    """Route semantic reference edges through local inter-column channels.

    Adjacent categories use the gap between their columns, same-column links use
    a narrow side rail, and the few non-adjacent links use compact lanes directly
    above the semantic sections. No route is allowed to escape around the full
    scene perimeter.
    """
    edge_list = sorted(edges, key=lambda edge: (edge.source_id, edge.target_id, edge.edge_id))
    plans: dict[str, tuple[str, Side, Side, int]] = {}
    adjacent_groups: dict[tuple[int, int], list[LayoutEdge]] = defaultdict(list)
    same_groups: dict[int, list[LayoutEdge]] = defaultdict(list)
    long_edges: list[LayoutEdge] = []
    for edge in edge_list:
        source_col = column_by_node.get(edge.source_id, 0)
        target_col = column_by_node.get(edge.target_id, 0)
        delta = target_col - source_col
        if abs(delta) == 1:
            source_side: Side = "right" if delta > 0 else "left"
            target_side: Side = "left" if delta > 0 else "right"
            adjacent_groups[(min(source_col, target_col), max(source_col, target_col))].append(edge)
            plans[edge.edge_id] = ("adjacent", source_side, target_side, 0)
        elif delta == 0:
            same_groups[source_col].append(edge)
            plans[edge.edge_id] = ("same", "right", "right", 0)
        else:
            long_edges.append(edge)
            plans[edge.edge_id] = ("long", "top", "top", 0)

    for values in adjacent_groups.values():
        values.sort(key=lambda edge: (
            (node_rects[edge.source_id].center().y() + node_rects[edge.target_id].center().y()) / 2.0,
            edge.edge_id,
        ))
        for index, edge in enumerate(values):
            kind, source_side, target_side, _ = plans[edge.edge_id]
            plans[edge.edge_id] = (kind, source_side, target_side, index)
    for values in same_groups.values():
        values.sort(key=lambda edge: (
            min(node_rects[edge.source_id].center().y(), node_rects[edge.target_id].center().y()),
            max(node_rects[edge.source_id].center().y(), node_rects[edge.target_id].center().y()),
            edge.edge_id,
        ))
        lane_ends: list[float] = []
        for edge in values:
            start = min(node_rects[edge.source_id].center().y(), node_rects[edge.target_id].center().y())
            end = max(node_rects[edge.source_id].center().y(), node_rects[edge.target_id].center().y())
            lane = next((i for i, value in enumerate(lane_ends) if value < start), len(lane_ends))
            if lane == len(lane_ends):
                lane_ends.append(end)
            else:
                lane_ends[lane] = end
            kind, source_side, target_side, _ = plans[edge.edge_id]
            plans[edge.edge_id] = (kind, source_side, target_side, lane)
    for index, edge in enumerate(long_edges):
        kind, source_side, target_side, _ = plans[edge.edge_id]
        plans[edge.edge_id] = (kind, source_side, target_side, index)

    # Allocate stable independent ports on each requested side.
    requests: dict[tuple[str, Side], list[tuple[LayoutEdge, bool, float]]] = defaultdict(list)
    for edge in edge_list:
        _kind, source_side, target_side, _lane = plans[edge.edge_id]
        requests[(edge.source_id, source_side)].append((edge, True, node_rects[edge.target_id].center().y()))
        requests[(edge.target_id, target_side)].append((edge, False, node_rects[edge.source_id].center().y()))
    ports: dict[tuple[str, bool], _Port] = {}
    for (node_id, side), values in requests.items():
        values.sort(key=lambda item: (item[2], item[0].edge_id))
        rect = node_rects[node_id]
        for index, (edge, is_source, _value) in enumerate(values):
            fraction = (index + 1) / (len(values) + 1)
            border = StateGraphUXLayout._point_on_side(rect, side, fraction)
            external = StateGraphUXLayout._move_from_side(border, side, 18.0)
            ports[(edge.edge_id, is_source)] = _Port(border, external, side)

    column_bounds: dict[int, QRectF] = {}
    for node_id, column in column_by_node.items():
        rect = node_rects[node_id]
        column_bounds[column] = rect if column not in column_bounds else column_bounds[column].united(rect)

    routes: dict[str, RoutedEdge] = {}
    for edge in edge_list:
        kind, source_side, target_side, lane = plans[edge.edge_id]
        source = ports[(edge.edge_id, True)]
        target = ports[(edge.edge_id, False)]
        source_col = column_by_node.get(edge.source_id, 0)
        target_col = column_by_node.get(edge.target_id, 0)
        if kind == "adjacent":
            left_col, right_col = sorted((source_col, target_col))
            left_bound = column_bounds[left_col].right()
            right_bound = column_bounds[right_col].left()
            count = len(adjacent_groups[(left_col, right_col)])
            center_x = (left_bound + right_bound) / 2.0
            track_x = center_x + (lane - (count - 1) / 2.0) * lane_gap
            points = [
                source.border, source.external,
                QPointF(track_x, source.external.y()),
                QPointF(track_x, target.external.y()),
                target.external, target.border,
            ]
        elif kind == "same":
            bound = column_bounds[source_col]
            track_x = bound.right() + 34.0 + lane * lane_gap
            points = [
                source.border, source.external,
                QPointF(track_x, source.external.y()),
                QPointF(track_x, target.external.y()),
                target.external, target.border,
            ]
        else:
            lane_y = section_top - 34.0 - lane * lane_gap
            points = [
                source.border, source.external,
                QPointF(source.external.x(), lane_y),
                QPointF(target.external.x(), lane_y),
                target.external, target.border,
            ]
        points = _simplify_orthogonal(points)
        routes[edge.edge_id] = RoutedEdge(
            edge=edge,
            points=points,
            source_side=source.side,
            target_side=target.side,
            label_position=_route_label_position(points),
        )

    placements = {
        node_id: NodePlacement(
            LayoutNode(node_id, node_id, "semantic", rect.width(), rect.height()),
            rect, 0, index,
        )
        for index, (node_id, rect) in enumerate(node_rects.items())
    }
    all_rects = list(node_rects.values()) + [_points_bounds(route.points) for route in routes.values()]
    normalized = normalize_route_lanes(
        LayoutResult(placements, routes, "vertical", _rect_union(all_rects), {}),
        minimum_gap=20.0,
        max_repairs=80,
    )
    return normalized.routes
