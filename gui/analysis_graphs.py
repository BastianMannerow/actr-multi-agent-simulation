"""Zoomable, exportable graph views for ACT-R explainability."""

from __future__ import annotations

import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtSvg import QSvgGenerator
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsTextItem,
    QGraphicsView,
    QMenu,
)

from simulation.inspection.declarative_memory import DeclarativeMemorySnapshot
from simulation.inspection.source_analysis import AgentStaticAnalysis, MethodBufferInteraction


SCENE_BACKGROUND = QColor("#0f172a")
TEXT_COLOR = QColor("#f8fafc")
MUTED_TEXT = QColor("#cbd5e1")
LABEL_BACKGROUND = QColor(15, 23, 42, 225)


class ZoomableGraphicsView(QGraphicsView):
    """Graphics view with wheel zoom, panning, and transparent export."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(SCENE_BACKGROUND))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._fit_pending = False

    def wheelEvent(self, event):  # noqa: N802
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def reset_zoom(self) -> None:
        self.resetTransform()
        if self.scene() is None:
            return
        if not self.isVisible() or self.viewport().width() < 50:
            self._fit_pending = True
            return
        bounds = self.scene().itemsBoundingRect().adjusted(-24, -24, 24, 24)
        self.fitInView(bounds, Qt.AspectRatioMode.KeepAspectRatio)
        self._fit_pending = False

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if self._fit_pending:
            self.reset_zoom()

    def contextMenuEvent(self, event):  # noqa: N802
        menu = QMenu(self)
        fit_action = QAction("Fit to view", self)
        fit_action.triggered.connect(self.reset_zoom)
        menu.addAction(fit_action)
        png_action = QAction("Export PNG", self)
        png_action.triggered.connect(lambda: self.export_dialog("png"))
        menu.addAction(png_action)
        svg_action = QAction("Export SVG", self)
        svg_action.triggered.connect(lambda: self.export_dialog("svg"))
        menu.addAction(svg_action)
        menu.exec(event.globalPos())

    def export_dialog(self, kind: str) -> Path | None:
        if self.scene() is None:
            return None
        suffix = ".svg" if kind == "svg" else ".png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {kind.upper()}",
            str(Path.home() / f"agent_analysis{suffix}"),
            f"{kind.upper()} file (*{suffix})",
        )
        if not path:
            return None
        return self.export_to(path)

    def export_to(self, path: str | Path) -> Path:
        if self.scene() is None:
            raise RuntimeError("There is no scene to export.")
        destination = Path(path)
        rect = self.scene().itemsBoundingRect().adjusted(-36, -36, 36, 36)
        scene = self.scene()
        original_background = scene.backgroundBrush()
        scene.setBackgroundBrush(QBrush(Qt.BrushStyle.NoBrush))
        try:
            if destination.suffix.lower() == ".svg":
                generator = QSvgGenerator()
                generator.setFileName(str(destination))
                generator.setSize(rect.size().toSize())
                generator.setViewBox(rect.toRect())
                painter = QPainter(generator)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                scene.render(painter, QRectF(), rect)
                painter.end()
            else:
                if destination.suffix.lower() != ".png":
                    destination = destination.with_suffix(".png")
                image = QImage(rect.size().toSize(), QImage.Format.Format_ARGB32)
                image.fill(Qt.GlobalColor.transparent)
                painter = QPainter(image)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                scene.render(painter, QRectF(image.rect()), rect)
                painter.end()
                image.save(str(destination))
        finally:
            scene.setBackgroundBrush(original_background)
        return destination


def build_state_transition_scene(analysis: AgentStaticAnalysis) -> QGraphicsScene:
    """Build a collision-aware state graph with routed, labelled edges."""
    scene = _new_scene()
    _add_scene_title(scene, f"State transitions — {analysis.agent_type}")
    _add_legend(
        scene,
        [
            ("Initial", QColor("#2563eb"), "box"),
            ("Reachable", QColor("#0f766e"), "box"),
            ("Loop", QColor("#7c3aed"), "box"),
            ("Dead end", QColor("#be123c"), "box"),
            ("Unreachable", QColor("#475569"), "box"),
        ],
        y=52,
    )

    initial = analysis.initial_state_label
    graph: dict[str, set[str]] = defaultdict(set)
    nodes: list[str] = []
    for state in [initial, *[p.source_label for p in analysis.productions], *[p.target_label for p in analysis.productions]]:
        if state not in nodes:
            nodes.append(state)
    for production in analysis.productions:
        graph[production.source_label].add(production.target_label)
        graph.setdefault(production.target_label, set())
    for state in nodes:
        graph.setdefault(state, set())

    depths = _bfs_depths(graph, initial)
    fallback_depth = max(depths.values(), default=0) + 1
    columns: dict[int, list[str]] = defaultdict(list)
    for state in nodes:
        columns[depths.get(state, fallback_depth)].append(state)

    font = QFont("Sans Serif", 9)
    node_width = 310.0
    column_gap = 220.0
    row_gap = 110.0
    top = 130.0
    item_rects: dict[str, QRectF] = {}
    for column, states in sorted(columns.items()):
        y = top
        for state in sorted(states):
            wrapped = _wrap_label(state, 42)
            size = _label_rect(wrapped, font, width=int(node_width), padding=24)
            rect = QRectF(
                40 + column * (node_width + column_gap),
                y,
                node_width,
                max(74.0, size.height()),
            )
            item_rects[state] = rect
            y += rect.height() + row_gap

    dead_ends = set(analysis.dead_end_states)
    loops = set(analysis.loop_states)
    for state, rect in item_rects.items():
        reachable = state in depths or state == initial
        color = (
            QColor("#2563eb")
            if state == initial
            else QColor("#be123c")
            if state in dead_ends
            else QColor("#7c3aed")
            if state in loops
            else QColor("#0f766e")
            if reachable
            else QColor("#475569")
        )
        _add_node(scene, rect, state, color, wrap_width=42)

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for production in analysis.productions:
        pair_counts[(production.source_label, production.target_label)] += 1
    pair_index: dict[tuple[str, str], int] = defaultdict(int)

    for production in analysis.productions:
        source = item_rects[production.source_label]
        target = item_rects[production.target_label]
        key = (production.source_label, production.target_label)
        index = pair_index[key]
        pair_index[key] += 1
        color = QColor("#e2e8f0") if production.reachable else QColor("#64748b")
        pen = QPen(color, 2.2 if production.reachable else 1.6)
        if not production.reachable:
            pen.setStyle(Qt.PenStyle.DashLine)

        if source == target:
            anchor = QPointF(source.right(), source.center().y())
            loop_extent = 90 + index * 34
            path = QPainterPath(anchor)
            path.cubicTo(
                anchor + QPointF(loop_extent, loop_extent),
                anchor + QPointF(loop_extent * 1.8, loop_extent),
                anchor + QPointF(loop_extent * 1.8, 0),
            )
            path.cubicTo(
                anchor + QPointF(loop_extent * 1.8, -loop_extent * 0.35),
                anchor + QPointF(loop_extent, -loop_extent * 0.35),
                anchor,
            )
            _add_path(scene, path, pen)
            _draw_arrow(scene, anchor + QPointF(14, 5), anchor, color)
            label_pos = anchor + QPointF(loop_extent * 0.9, loop_extent + 10)
        elif target.left() >= source.right():
            start = QPointF(source.right(), source.center().y())
            end = QPointF(target.left(), target.center().y())
            duplicate_offset = (index - (pair_counts[key] - 1) / 2) * 28
            mid_x = (start.x() + end.x()) / 2 + duplicate_offset
            path = QPainterPath(start)
            path.cubicTo(
                QPointF(mid_x, start.y()),
                QPointF(mid_x, end.y()),
                end,
            )
            _add_path(scene, path, pen)
            _draw_arrow(scene, QPointF(mid_x, end.y()), end, color)
            label_pos = QPointF(mid_x - 70, (start.y() + end.y()) / 2 - 14)
        else:
            start = QPointF(source.left(), source.center().y())
            end = QPointF(target.right(), target.center().y())
            route_y = min(source.top(), target.top()) - 70 - index * 34
            path = QPainterPath(start)
            path.lineTo(start.x() - 46, start.y())
            path.lineTo(start.x() - 46, route_y)
            path.lineTo(end.x() + 46, route_y)
            path.lineTo(end.x() + 46, end.y())
            path.lineTo(end)
            _add_path(scene, path, pen)
            _draw_arrow(scene, QPointF(end.x() + 46, end.y()), end, color)
            label_pos = QPointF((start.x() + end.x()) / 2 - 70, route_y - 24)
        _add_edge_label(scene, production.name, label_pos, color)
    return scene


def build_interaction_scene(
    title: str,
    interactions: Iterable[MethodBufferInteraction],
) -> QGraphicsScene:
    """Build a routed bipartite graph without overlapping connection lines."""
    scene = _new_scene()
    _add_scene_title(scene, title)
    _add_legend(
        scene,
        [
            ("Read", QColor("#60a5fa"), "line"),
            ("Write", QColor("#f59e0b"), "line"),
            ("Delete / clear", QColor("#fb7185"), "line"),
        ],
        y=50,
    )
    rows = list(interactions)
    if not rows:
        _add_empty_message(scene, "No buffer interactions were detected.")
        return scene

    actors = sorted({row.method_name for row in rows}, key=str.lower)
    buffers = sorted({row.buffer_name for row in rows}, key=str.lower)
    actor_height = 60.0
    buffer_height = 60.0
    spacing = 92.0
    left_x, left_width = 40.0, 310.0
    right_x, right_width = 920.0, 260.0
    top = 125.0
    actor_rects: dict[str, QRectF] = {}
    buffer_rects: dict[str, QRectF] = {}
    for index, actor in enumerate(actors):
        rect = QRectF(left_x, top + index * spacing, left_width, actor_height)
        actor_rects[actor] = rect
        _add_node(scene, rect, actor, QColor("#1e3a8a"), wrap_width=38)
    for index, buffer_name in enumerate(buffers):
        rect = QRectF(right_x, top + index * spacing, right_width, buffer_height)
        buffer_rects[buffer_name] = rect
        _add_node(scene, rect, buffer_name, QColor("#166534"), wrap_width=30)

    actor_edges: dict[str, list[MethodBufferInteraction]] = defaultdict(list)
    buffer_edges: dict[str, list[MethodBufferInteraction]] = defaultdict(list)
    for row in rows:
        actor_edges[row.method_name].append(row)
        buffer_edges[row.buffer_name].append(row)

    for lane, row in enumerate(rows):
        source = actor_rects[row.method_name]
        target = buffer_rects[row.buffer_name]
        source_rows = actor_edges[row.method_name]
        target_rows = buffer_edges[row.buffer_name]
        source_index = source_rows.index(row)
        target_index = target_rows.index(row)
        start = QPointF(
            source.right(),
            source.top() + (source_index + 1) * source.height() / (len(source_rows) + 1),
        )
        end = QPointF(
            target.left(),
            target.top() + (target_index + 1) * target.height() / (len(target_rows) + 1),
        )
        lane_x = 430.0 + lane * 24.0
        mode = row.mode.lower()
        color = (
            QColor("#f59e0b")
            if mode in {"write", "request"}
            else QColor("#fb7185")
            if mode in {"delete", "clear"}
            else QColor("#60a5fa")
        )
        pen = QPen(color, 2.0)
        path = QPainterPath(start)
        path.lineTo(lane_x, start.y())
        path.lineTo(lane_x, end.y())
        path.lineTo(end)
        _add_path(scene, path, pen)
        _draw_arrow(scene, QPointF(lane_x, end.y()), end, color)
        _add_edge_label(
            scene,
            row.mode,
            QPointF(lane_x + 5, (start.y() + end.y()) / 2 - 12),
            color,
        )
    return scene


def build_buffer_history_scene(
    agent_name: str,
    history: dict[str, list[dict[str, Any]]],
) -> QGraphicsScene:
    scene = _new_scene()
    _add_scene_title(scene, f"Buffer history — {agent_name}")
    if not history:
        _add_empty_message(scene, "No buffer history is available yet.")
        return scene

    max_time = max(
        [
            float(entry.get("timestamp", 0.0))
            for entries in history.values()
            for entry in entries
        ]
        or [1.0]
    )
    max_time = max(max_time, 1.0)
    left = 210.0
    row_height = 84.0
    width = 1040.0
    for index, (buffer_name, entries) in enumerate(sorted(history.items())):
        y = 90.0 + index * row_height
        label = QGraphicsSimpleTextItem(buffer_name)
        label.setBrush(QBrush(TEXT_COLOR))
        label.setPos(24, y - 10)
        scene.addItem(label)
        baseline = QPainterPath(QPointF(left, y + 10))
        baseline.lineTo(left + width, y + 10)
        _add_path(scene, baseline, QPen(QColor("#334155"), 1.3))
        previous: QPointF | None = None
        for entry in entries:
            timestamp = float(entry.get("timestamp", 0.0))
            point = QPointF(left + (timestamp / max_time) * width, y + 10)
            if previous is not None:
                path = QPainterPath(previous)
                path.lineTo(point)
                _add_path(scene, path, QPen(QColor("#64748b"), 1.5))
            change = str(entry.get("change", "content_changed"))
            color = {
                "initial": QColor("#38bdf8"),
                "filled": QColor("#22c55e"),
                "cleared": QColor("#ef4444"),
                "state_changed": QColor("#f59e0b"),
                "content_changed": QColor("#a78bfa"),
                "module_changed": QColor("#e879f9"),
            }.get(change, QColor("#94a3b8"))
            marker = QGraphicsEllipseItem(point.x() - 7, point.y() - 7, 14, 14)
            marker.setPen(QPen(QColor("#e2e8f0"), 1.0))
            marker.setBrush(QBrush(color))
            marker.setToolTip(
                f"t={timestamp:.3f}\nchange={change}\n"
                f"state={entry.get('snapshot', {}).get('state')}"
            )
            scene.addItem(marker)
            previous = point
    return scene


def build_jump_progress_scene(
    analysis: AgentStaticAnalysis,
    target_production: str,
    fired_productions: list[str],
) -> QGraphicsScene:
    """Render a target path and highlight productions fired in order."""
    scene = _new_scene()
    _add_scene_title(scene, f"Jump path to production: {target_production}")
    path = analysis.path_to_production(target_production)
    if not path:
        warning = QGraphicsTextItem(
            "No statically reachable path could be derived. Adapter side effects "
            "or dynamic buffer changes may still make the target reachable."
        )
        warning.setDefaultTextColor(QColor("#fecaca"))
        warning.setTextWidth(760)
        warning.setPos(24, 72)
        scene.addItem(warning)
        target = analysis.production(target_production)
        if target is not None:
            _add_node(scene, QRectF(30, 170, 330, 96), target.source_label, QColor("#7f1d1d"))
            _add_edge_label(scene, target.name, QPointF(420, 202), QColor("#fecaca"))
            _add_node(scene, QRectF(610, 170, 330, 96), target.target_label, QColor("#7f1d1d"))
        return scene

    progress = _ordered_path_progress(path, fired_productions)
    node_width, node_height, spacing = 300.0, 96.0, 190.0
    y = 150.0
    states = analysis.state_sequence_for_path(path)
    for index, state in enumerate(states):
        x = 30 + index * (node_width + spacing)
        color = (
            QColor("#6b21a8")
            if index == len(states) - 1 and progress >= len(path)
            else QColor("#166534")
            if index <= progress
            else QColor("#075985")
            if index == progress + 1
            else QColor("#334155")
        )
        _add_node(scene, QRectF(x, y, node_width, node_height), state, color, wrap_width=40)

    for index, production in enumerate(path):
        source_x = 30 + index * (node_width + spacing) + node_width
        target_x = 30 + (index + 1) * (node_width + spacing)
        start = QPointF(source_x, y + node_height / 2)
        end = QPointF(target_x, y + node_height / 2)
        completed = index < progress
        active = index == progress and progress < len(path)
        color = QColor("#22c55e") if completed else QColor("#38bdf8") if active else QColor("#64748b")
        route_y = y + node_height / 2 + (index % 2) * 30 - 15
        graph_path = QPainterPath(start)
        graph_path.cubicTo(
            QPointF((start.x() + end.x()) / 2, route_y),
            QPointF((start.x() + end.x()) / 2, route_y),
            end,
        )
        _add_path(scene, graph_path, QPen(color, 3.0 if completed or active else 2.0))
        _draw_arrow(scene, QPointF((start.x() + end.x()) / 2, route_y), end, color)
        _add_edge_label(scene, production.name, QPointF(source_x + 40, y - 44), color)
        condition = QGraphicsTextItem("Requires:\n" + _wrap_label(production.source_label, 32))
        condition.setDefaultTextColor(MUTED_TEXT)
        condition.setTextWidth(spacing - 20)
        condition.setPos(source_x + 8, y + node_height + 20)
        scene.addItem(condition)

    status = QGraphicsSimpleTextItem(
        "Target production fired."
        if progress >= len(path)
        else f"Reached {progress} of {len(path)} required production steps."
    )
    status.setBrush(QBrush(QColor("#86efac") if progress >= len(path) else QColor("#bae6fd")))
    status.setPos(24, 78)
    scene.addItem(status)
    return scene


def build_declarative_memory_scene(
    snapshot: DeclarativeMemorySnapshot,
    *,
    title: str,
) -> QGraphicsScene:
    """Render memories, chunks, operations, and inferred links without overlap."""
    scene = _new_scene()
    _add_scene_title(scene, title)
    _add_legend(
        scene,
        [
            ("Memory", QColor("#2563eb"), "box"),
            ("Runtime chunk", QColor("#0f766e"), "box"),
            ("Static chunk", QColor("#7c3aed"), "box"),
            ("Explicit reference", QColor("#38bdf8"), "line"),
            ("Shared value", QColor("#94a3b8"), "dash"),
            ("Read", QColor("#60a5fa"), "line"),
            ("Write", QColor("#f59e0b"), "line"),
            ("Delete", QColor("#fb7185"), "line"),
        ],
        y=52,
        max_width=1180,
    )

    if not snapshot.memories and not snapshot.chunks:
        _add_empty_message(
            scene,
            "No declarative-memory chunks were detected. The memory may be populated later during simulation.",
            y=118,
        )
        return scene

    memory_names = snapshot.memories or sorted({chunk.memory_name for chunk in snapshot.chunks})
    chunks_by_memory: dict[str, list[Any]] = {name: [] for name in memory_names}
    for chunk in snapshot.chunks:
        chunks_by_memory.setdefault(chunk.memory_name, []).append(chunk)

    column_width = 330.0
    column_gap = 170.0
    chunk_gap = 52.0
    header_y = 125.0
    chunk_y = 235.0
    memory_rects: dict[str, QRectF] = {}
    chunk_rects: dict[str, QRectF] = {}
    memory_index: dict[str, int] = {}
    max_bottom = chunk_y

    for index, memory_name in enumerate(memory_names):
        x = 40 + index * (column_width + column_gap)
        memory_index[memory_name] = index
        header = QRectF(x, header_y, column_width, 62)
        memory_rects[memory_name] = header
        _add_node(scene, header, f"Memory: {memory_name}", QColor("#2563eb"), wrap_width=38)
        y = chunk_y
        for chunk in chunks_by_memory.get(memory_name, []):
            activation = f"activation={chunk.activation:.3f}" if chunk.activation is not None else ""
            traces = (
                "traces=" + ", ".join(f"{value:.3f}" for value in chunk.traces[-4:])
                if chunk.traces
                else ""
            )
            detail = " · ".join(value for value in (activation, traces) if value)
            label = chunk.label + (f"\n{detail}" if detail else "")
            wrapped = _wrap_label(label, 42)
            height = max(98.0, _label_rect(wrapped, QFont("Sans Serif", 9), int(column_width), 26).height())
            rect = QRectF(x, y, column_width, height)
            chunk_rects[chunk.chunk_id] = rect
            color = QColor("#0f766e") if chunk.source == "runtime" else QColor("#7c3aed")
            _add_node(scene, rect, label, color, wrap_width=42)
            link_path = QPainterPath(QPointF(header.center().x(), header.bottom()))
            link_path.lineTo(QPointF(header.center().x(), rect.top()))
            _add_path(scene, link_path, QPen(QColor("#64748b"), 1.4))
            y += rect.height() + chunk_gap
            max_bottom = max(max_bottom, rect.bottom())

    same_memory_lane: dict[str, int] = defaultdict(int)
    cross_lane = 0
    for edge in snapshot.edges:
        source = chunk_rects.get(edge.source_id)
        target = chunk_rects.get(edge.target_id)
        if source is None or target is None:
            continue
        color = QColor("#38bdf8") if edge.relation == "reference" else QColor("#94a3b8")
        pen = QPen(color, 2.2 if edge.relation == "reference" else 1.4)
        if edge.relation != "reference":
            pen.setStyle(Qt.PenStyle.DashLine)

        source_chunk = next((c for c in snapshot.chunks if c.chunk_id == edge.source_id), None)
        target_chunk = next((c for c in snapshot.chunks if c.chunk_id == edge.target_id), None)
        same_memory = source_chunk is not None and target_chunk is not None and source_chunk.memory_name == target_chunk.memory_name
        if same_memory and source_chunk is not None:
            lane = same_memory_lane[source_chunk.memory_name]
            same_memory_lane[source_chunk.memory_name] += 1
            side_x = source.right() + 46 + lane * 24
            start = QPointF(source.right(), source.center().y())
            end = QPointF(target.right(), target.center().y())
            path = QPainterPath(start)
            path.lineTo(side_x, start.y())
            path.lineTo(side_x, end.y())
            path.lineTo(end)
            label_pos = QPointF(side_x + 6, (start.y() + end.y()) / 2 - 12)
            arrow_from = QPointF(side_x, end.y())
        else:
            lane = cross_lane
            cross_lane += 1
            start = QPointF(source.center().x(), source.bottom())
            end = QPointF(target.center().x(), target.bottom())
            route_y = max_bottom + 70 + lane * 30
            path = QPainterPath(start)
            path.lineTo(start.x(), route_y)
            path.lineTo(end.x(), route_y)
            path.lineTo(end)
            label_pos = QPointF((start.x() + end.x()) / 2 - 65, route_y - 24)
            arrow_from = QPointF(end.x(), route_y)
        _add_path(scene, path, pen)
        _draw_arrow(scene, arrow_from, end, color)
        _add_edge_label(scene, edge.label, label_pos, color)

    operations_top = max_bottom + 120 + cross_lane * 30
    if snapshot.operations:
        heading = QGraphicsSimpleTextItem("Memory operations")
        heading.setBrush(QBrush(TEXT_COLOR))
        heading.setPos(24, operations_top - 42)
        scene.addItem(heading)
        for index, operation in enumerate(snapshot.operations):
            column = index % max(1, min(3, len(memory_names) or 3))
            row = index // max(1, min(3, len(memory_names) or 3))
            x = 40 + column * 390
            y = operations_top + row * 112
            actor = str(operation.get("actor", "code"))
            mode = str(operation.get("mode", "access")).lower()
            memory_name = str(operation.get("memory_name", "decmem"))
            detail = str(operation.get("detail", ""))
            label = f"{actor}\n{mode} → {memory_name}" + (f"\n{detail}" if detail else "")
            color = QColor("#fb7185") if mode in {"delete", "clear"} else QColor("#f59e0b") if mode in {"write", "add"} else QColor("#60a5fa")
            rect = QRectF(x, y, 340, 78)
            _add_node(scene, rect, label, color, wrap_width=42)
            target = memory_rects.get(memory_name)
            if target is not None:
                start = QPointF(rect.center().x(), rect.top())
                end = QPointF(target.center().x(), target.bottom())
                route_x = rect.right() + 24 + index * 12
                path = QPainterPath(start)
                path.lineTo(route_x, start.y())
                path.lineTo(route_x, header_y + 92)
                path.lineTo(end.x(), header_y + 92)
                path.lineTo(end)
                pen = QPen(color, 1.8, Qt.PenStyle.DashLine)
                _add_path(scene, path, pen)
                _draw_arrow(scene, QPointF(end.x(), header_y + 92), end, color)
    return scene


def _ordered_path_progress(path: list[Any], fired_productions: list[str]) -> int:
    index = 0
    for fired in fired_productions:
        if index >= len(path):
            break
        if str(fired).casefold() == str(path[index].name).casefold():
            index += 1
    return index


def _new_scene() -> QGraphicsScene:
    scene = QGraphicsScene()
    scene.setBackgroundBrush(QBrush(SCENE_BACKGROUND))
    return scene


def _add_scene_title(scene: QGraphicsScene, text: str) -> None:
    title = QGraphicsSimpleTextItem(text)
    font = QFont("Sans Serif", 11)
    font.setBold(True)
    title.setFont(font)
    title.setBrush(QBrush(TEXT_COLOR))
    title.setPos(20, 12)
    scene.addItem(title)


def _add_empty_message(scene: QGraphicsScene, text: str, *, y: float = 82.0) -> None:
    item = QGraphicsTextItem(text)
    item.setDefaultTextColor(MUTED_TEXT)
    item.setTextWidth(760)
    item.setPos(24, y)
    scene.addItem(item)


def _add_legend(
    scene: QGraphicsScene,
    items: list[tuple[str, QColor, str]],
    *,
    y: float,
    max_width: float = 1050,
) -> None:
    x = 24.0
    row = 0
    for label_text, color, kind in items:
        estimated = 44 + len(label_text) * 7
        if x + estimated > max_width and x > 24:
            row += 1
            x = 24.0
        current_y = y + row * 34
        if kind == "box":
            swatch = QGraphicsRectItem(QRectF(x, current_y, 20, 20))
            swatch.setPen(QPen(QColor("#cbd5e1"), 1.0))
            swatch.setBrush(QBrush(color))
            scene.addItem(swatch)
        else:
            path = QPainterPath(QPointF(x, current_y + 10))
            path.lineTo(x + 26, current_y + 10)
            pen = QPen(color, 2.2)
            if kind == "dash":
                pen.setStyle(Qt.PenStyle.DashLine)
            _add_path(scene, path, pen)
        label = QGraphicsSimpleTextItem(label_text)
        label.setBrush(QBrush(TEXT_COLOR))
        label.setPos(x + 32, current_y - 2)
        scene.addItem(label)
        x += estimated


def _add_node(
    scene: QGraphicsScene,
    rect: QRectF,
    label: str,
    color: QColor,
    *,
    wrap_width: int = 38,
) -> None:
    node = QGraphicsRectItem(rect)
    node.setPen(QPen(QColor("#dbe4f0"), 1.5))
    node.setBrush(QBrush(color))
    scene.addItem(node)
    text = QGraphicsTextItem(_wrap_label(label, wrap_width))
    text.setDefaultTextColor(TEXT_COLOR)
    text.setTextWidth(rect.width() - 18)
    text.setPos(rect.x() + 9, rect.y() + 8)
    scene.addItem(text)


def _add_path(scene: QGraphicsScene, path: QPainterPath, pen: QPen) -> None:
    item = QGraphicsPathItem(path)
    item.setPen(pen)
    item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
    scene.addItem(item)


def _add_edge_label(
    scene: QGraphicsScene,
    text: str,
    position: QPointF,
    color: QColor,
) -> None:
    label = QGraphicsSimpleTextItem(text)
    label.setBrush(QBrush(color))
    label.setPos(position)
    bounds = label.boundingRect().adjusted(-6, -3, 6, 3)
    background = QGraphicsRectItem(
        QRectF(position.x() + bounds.x(), position.y() + bounds.y(), bounds.width(), bounds.height())
    )
    background.setPen(QPen(QColor("#334155"), 0.8))
    background.setBrush(QBrush(LABEL_BACKGROUND))
    scene.addItem(background)
    scene.addItem(label)


def _draw_arrow(scene: QGraphicsScene, start: QPointF, end: QPointF, color: QColor) -> None:
    angle = math.atan2(end.y() - start.y(), end.x() - start.x())
    arrow_size = 11
    p1 = end - QPointF(
        math.cos(angle - math.pi / 6) * arrow_size,
        math.sin(angle - math.pi / 6) * arrow_size,
    )
    p2 = end - QPointF(
        math.cos(angle + math.pi / 6) * arrow_size,
        math.sin(angle + math.pi / 6) * arrow_size,
    )
    scene.addLine(end.x(), end.y(), p1.x(), p1.y(), QPen(color, 2.0))
    scene.addLine(end.x(), end.y(), p2.x(), p2.y(), QPen(color, 2.0))


def _wrap_label(text: str, width: int) -> str:
    lines: list[str] = []
    for raw in text.splitlines() or [text]:
        lines.extend(textwrap.wrap(raw, width=width) or [raw])
    return "\n".join(lines)


def _label_rect(text: str, font: QFont, width: int, padding: int) -> QRectF:
    metrics = QFontMetrics(font)
    height = 0
    for line in text.splitlines() or [text]:
        height += metrics.boundingRect(line).height() + 3
    return QRectF(0, 0, width, max(60, height + padding))


def _bfs_depths(graph: dict[str, set[str]], start: str) -> dict[str, int]:
    depths = {start: 0}
    queue = [start]
    while queue:
        current = queue.pop(0)
        for nxt in graph.get(current, set()):
            if nxt not in depths:
                depths[nxt] = depths[current] + 1
                queue.append(nxt)
    return depths
