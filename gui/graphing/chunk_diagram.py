"""ACT-R declarative-memory diagrams using UML-like chunk cards."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from PyQt6.QtCore import QPointF, QRectF
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import QGraphicsSimpleTextItem, QGraphicsScene

from gui.graph_layout import assign_label_positions
from gui.graphing.layout_engine import ReusableLayeredGraphLayout
from gui.graphing.models import DiagramEdge, DiagramNode
from gui.graphing.renderers import EdgeRenderer, NodeRenderer, SceneChrome
from simulation.inspection.declarative_memory import (
    DeclarativeMemoryInspector,
    DeclarativeMemorySnapshot,
    MemoryChunk,
)


_TYPE_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "semantic_concept",
            "strategy_schema",
            "knowledge_relation",
            "target_memory",
            "cell_memory",
            "spatial_relation",
            "episode_memory",
        )
    )
}


class DeclarativeMemoryDiagramBuilder:
    """Build a compact, retrieval-centred ACT-R memory diagram.

    Chunks are typed slot-value records in ACT-R. They are rendered as UML-like
    class cards: stereotype, chunk type/identity, then slot-value attributes.
    A directed edge exists only when a slot explicitly names another chunk.
    """

    def __init__(self) -> None:
        self.layout_engine = ReusableLayeredGraphLayout(
            minimum_lane_gap=22.0,
            node_gap=58.0,
            rank_gap=128.0,
            port_stub=20.0,
        )

    def build(
        self,
        snapshot: DeclarativeMemorySnapshot,
        *,
        title: str,
    ) -> QGraphicsScene:
        snapshot = DeclarativeMemoryInspector.normalize_snapshot(snapshot)
        scene = SceneChrome.new_scene()
        SceneChrome.add_title(scene, title)
        legend = SceneChrome.add_legend(
            scene,
            [
                ("Retrieval buffer", QColor("#155e75"), "box"),
                ("Declarative memory", QColor("#1d4ed8"), "box"),
                ("Runtime retrieval candidate", QColor("#0f766e"), "box"),
                ("Declared chunk", QColor("#6d28d9"), "box"),
                ("slot → chunk", QColor("#38bdf8"), "line"),
                ("retrieval request/result", QColor("#f59e0b"), "dash"),
            ],
            y=52.0,
            max_width=1420.0,
        )

        chunks = self._eligible_chunks(snapshot)
        retrieval_buffers = list(snapshot.retrieval_buffers)
        memory_names = list(snapshot.retrieval_memory_names or snapshot.memories)
        if not retrieval_buffers:
            retrieval_buffers = self._retrieval_buffers_from_operations(snapshot)
        if not memory_names:
            memory_names = sorted({chunk.memory_name for chunk in chunks}, key=str.casefold)

        if not retrieval_buffers or not memory_names:
            SceneChrome.add_message(
                scene,
                "No declarative memory connected to a pyactr retrieval buffer was detected.",
                y=legend.bottom() + 34.0,
            )
            scene.setSceneRect(scene.itemsBoundingRect().adjusted(-24, -24, 24, 24))
            return scene
        if not chunks:
            SceneChrome.add_message(
                scene,
                "The retrieval buffer is connected, but no stored chunk matches a retrieval request declared by the agent.",
                y=legend.bottom() + 34.0,
            )
            scene.setSceneRect(scene.itemsBoundingRect().adjusted(-24, -24, 24, 24))
            return scene

        nodes: list[DiagramNode] = []
        edges: list[DiagramEdge] = []
        rank_hints: dict[str, int] = {}

        memory_node_ids: dict[str, str] = {}
        for memory_name in memory_names:
            node_id = f"memory:{memory_name}"
            memory_node_ids[memory_name] = node_id
            nodes.append(
                DiagramNode(
                    node_id=node_id,
                    title=f"Declarative memory\n{memory_name}",
                    group="Declarative memory",
                    width=310.0,
                    height=78.0,
                    priority=90,
                    fill="#1d4ed8",
                    border="#bfdbfe",
                    tooltip=(
                        "Contains chunks searched by the linked pyactr retrieval "
                        "buffer. The diagram shows only chunks matching an actual "
                        "+retrieval production request."
                    ),
                )
            )
            rank_hints[node_id] = 1

        first_retrieval_id: str | None = None
        for index, buffer_name in enumerate(sorted(retrieval_buffers, key=str.casefold)):
            node_id = f"retrieval:{buffer_name}"
            if first_retrieval_id is None:
                first_retrieval_id = node_id
            nodes.append(
                DiagramNode(
                    node_id=node_id,
                    title=f"Retrieval buffer\n{buffer_name}",
                    group="Retrieval interface",
                    width=310.0,
                    height=78.0,
                    priority=100,
                    fill="#155e75",
                    border="#a5f3fc",
                    tooltip=(
                        "A production places a pattern request here. pyactr "
                        "matches the request against the linked declarative memory "
                        "and returns at most one chunk to this buffer."
                    ),
                )
            )
            rank_hints[node_id] = 0
            for memory_name in memory_names:
                edge_id = f"retrieval-link:{buffer_name}:{memory_name}"
                edges.append(
                    DiagramEdge(
                        edge_id=edge_id,
                        source_id=node_id,
                        target_id=memory_node_ids[memory_name],
                        label="request / result",
                        kind="retrieval",
                        color="#f59e0b",
                        dashed=True,
                        weight=1.4,
                        tooltip=(
                            "pyactr DecMemBuffer.retrieve iterates the chunks in "
                            f"{memory_name}."
                        ),
                    )
                )

        chunks_by_type: dict[str, list[MemoryChunk]] = defaultdict(list)
        for chunk in chunks:
            chunks_by_type[chunk.chunk_type].append(chunk)
        ordered_types = sorted(
            chunks_by_type,
            key=lambda name: (_TYPE_ORDER.get(name, 10_000), name.casefold()),
        )
        chunk_node_ids: dict[str, str] = {}
        for chunk_type in ordered_types:
            for chunk in sorted(
                chunks_by_type[chunk_type],
                key=lambda value: (value.label.casefold(), value.chunk_id),
            ):
                node_id = f"chunk:{chunk.chunk_id}"
                chunk_node_ids[chunk.chunk_id] = node_id
                attributes, omitted = self._visible_attributes(chunk)
                if omitted:
                    attributes.append(("…", f"{omitted} additional slot(s)"))
                title_text = f"{chunk.chunk_type}\n{chunk.label}"
                tooltip = self._chunk_tooltip(chunk)
                height = 96.0 + max(1, len(attributes)) * 20.0
                nodes.append(
                    DiagramNode(
                        node_id=node_id,
                        title=title_text,
                        group=chunk.chunk_type,
                        stereotype="«chunk»",
                        attributes=tuple(attributes),
                        width=340.0,
                        height=height,
                        priority=max(0, 100 - _TYPE_ORDER.get(chunk.chunk_type, 90)),
                        fill="#0f766e" if str(chunk.source).startswith("runtime") else "#6d28d9",
                        border="#dbeafe",
                        kind="uml_chunk",
                        tooltip=tooltip,
                    )
                )

        for index, memory_edge in enumerate(snapshot.edges):
            if memory_edge.relation not in {"reference", "slot_reference"}:
                continue
            source = chunk_node_ids.get(memory_edge.source_id)
            target = chunk_node_ids.get(memory_edge.target_id)
            if source is None or target is None:
                continue
            edges.append(
                DiagramEdge(
                    edge_id=f"slot-reference:{index}:{source}:{target}",
                    source_id=source,
                    target_id=target,
                    label=memory_edge.label,
                    kind="slot_reference",
                    color="#38bdf8",
                    tooltip=(
                        f"The slot '{memory_edge.label}' of the source chunk "
                        "contains an identity of the target chunk."
                    ),
                )
            )

        rank_hints.update(
            self._semantic_rank_hints(
                list(chunk_node_ids.values()),
                [edge for edge in edges if edge.kind == "slot_reference"],
                base_rank=2,
            )
        )

        model_nodes = {node.node_id: node for node in nodes}
        model_edges = {edge.edge_id: edge for edge in edges}
        geometry = self.layout_engine.layout(
            nodes,
            edges,
            initial_node_id=first_retrieval_id,
            rank_hints=rank_hints,
            offset=QPointF(44.0, legend.bottom() + 58.0),
        )

        for node_id, placement in geometry.placements.items():
            node = model_nodes[node_id]
            if node.kind == "uml_chunk":
                NodeRenderer.draw_uml_chunk(scene, placement.rect, node)
            else:
                NodeRenderer.draw_card(scene, placement.rect, node)

        label_sizes = {
            edge_id: (max(72.0, 8.0 * len(edge.label) + 18.0), 25.0)
            for edge_id, edge in model_edges.items()
            if edge.label
        }
        label_positions = assign_label_positions(
            geometry.routes.values(),
            [placement.rect for placement in geometry.placements.values()],
            label_width=72.0,
            label_height=25.0,
            label_sizes=label_sizes,
        )
        for edge_id, route in geometry.routes.items():
            EdgeRenderer.draw(
                scene,
                route,
                model_edges[edge_id],
                label_position=label_positions.get(edge_id),
            )

        note = QGraphicsSimpleTextItem(
            "ACT-R view: typed slot-value chunks are stored in declarative memory; "
            "a retrieval request returns one matching chunk to the retrieval buffer."
        )
        note.setBrush(QBrush(QColor("#94a3b8")))
        note.setPos(44.0, geometry.bounds.bottom() + 50.0)
        scene.addItem(note)
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-30.0, -30.0, 30.0, 30.0))
        return scene

    @staticmethod
    def _semantic_rank_hints(
        node_ids: list[str],
        edges: list[DiagramEdge],
        *,
        base_rank: int,
    ) -> dict[str, int]:
        """Rank chunk nodes by semantic dependency rather than by chunk type.

        A relation or strategy chunk that points to a concept is placed before
        that concept. This turns slot references into short forward edges and
        avoids the large perimeter loops produced by rigid type rows.
        """
        node_set = set(node_ids)
        outgoing: dict[str, list[str]] = defaultdict(list)
        incoming: dict[str, int] = {node_id: 0 for node_id in node_ids}
        undirected: dict[str, set[str]] = defaultdict(set)
        for edge in edges:
            if edge.source_id not in node_set or edge.target_id not in node_set:
                continue
            outgoing[edge.source_id].append(edge.target_id)
            incoming[edge.target_id] += 1
            undirected[edge.source_id].add(edge.target_id)
            undirected[edge.target_id].add(edge.source_id)

        ranks: dict[str, int] = {}
        remaining = set(node_ids)
        while remaining:
            component_start = min(remaining)
            component: set[str] = set()
            stack = [component_start]
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(undirected.get(current, set()) - component)
            remaining -= component
            roots = sorted(node_id for node_id in component if incoming.get(node_id, 0) == 0)
            if not roots:
                roots = [min(component)]
            queue = [(root, base_rank) for root in roots]
            for root in roots:
                ranks[root] = min(ranks.get(root, base_rank), base_rank)
            position = 0
            while position < len(queue):
                current, rank = queue[position]
                position += 1
                for target in sorted(outgoing.get(current, [])):
                    if target not in component:
                        continue
                    proposed = rank + 1
                    if target not in ranks or proposed < ranks[target]:
                        ranks[target] = proposed
                        queue.append((target, proposed))
            for node_id in component:
                ranks.setdefault(node_id, base_rank)
        return ranks

    @staticmethod
    def _eligible_chunks(snapshot: DeclarativeMemorySnapshot) -> list[MemoryChunk]:
        eligible_names = set(snapshot.retrieval_memory_names or snapshot.memories)
        return [
            chunk
            for chunk in snapshot.chunks
            if not eligible_names or chunk.memory_name in eligible_names
        ]

    @staticmethod
    def _retrieval_buffers_from_operations(
        snapshot: DeclarativeMemorySnapshot,
    ) -> list[str]:
        values = []
        for operation in snapshot.operations:
            if str(operation.get("mode")) not in {"retrieval_link", "linked"}:
                continue
            actor = str(operation.get("actor", "retrieval"))
            values.append(actor.removeprefix("buffer:"))
        return sorted(set(values), key=str.casefold)

    @staticmethod
    def _visible_attributes(chunk: MemoryChunk) -> tuple[list[tuple[str, str]], int]:
        items = [(str(name), DeclarativeMemoryDiagramBuilder._value(value)) for name, value in chunk.slots.items()]
        identity_names = {
            "entity_id", "relation_id", "strategy_id", "cell_id",
            "target_id", "episode_id", "name", "id", "key",
        }
        ordered = sorted(
            items,
            key=lambda item: (0 if item[0] in identity_names else 1, item[0].casefold()),
        )
        limit = 9
        return ordered[:limit], max(0, len(ordered) - limit)

    @staticmethod
    def _value(value: Any) -> str:
        if value is None:
            return "∅"
        text = str(value)
        return text if len(text) <= 42 else text[:39] + "…"

    @staticmethod
    def _chunk_tooltip(chunk: MemoryChunk) -> str:
        body = {
            "chunk_type": chunk.chunk_type,
            "identity": chunk.label,
            "memory": chunk.memory_name,
            "retrieval_buffers": list(chunk.retrieval_buffers),
            "matched_retrieval_queries": list(chunk.matched_queries),
            "slots": chunk.slots,
            "activation": chunk.activation,
            "traces": chunk.traces,
        }
        return json.dumps(body, ensure_ascii=False, indent=2, default=str)


def build_declarative_memory_scene(
    snapshot: DeclarativeMemorySnapshot,
    *,
    title: str,
) -> QGraphicsScene:
    """Compatibility facade used by runtime and static analysis views."""
    return DeclarativeMemoryDiagramBuilder().build(snapshot, title=title)
