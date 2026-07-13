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
from gui.graph_layout import (
    LayoutEdge,
    LayoutNode,
    assign_label_positions,
    layout_and_route,
    route_fixed_nodes,
    separate_overlapping_routes,
)

from simulation.inspection.source_analysis import (
    AgentStaticAnalysis,
    MethodBufferInteraction,
    StateTransitionAnalysis,
)


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
    """Render the reachable control graph with compact placement and orthogonal routing."""
    scene = _new_scene()
    _add_scene_title(scene, f"State transitions — {analysis.agent_type}")
    _add_legend(
        scene,
        [
            ("Initial", QColor("#1d4ed8"), "box"),
            ("Reachable", QColor("#047857"), "box"),
            ("Adapter handoff", QColor("#b45309"), "box"),
            ("Terminal", QColor("#0e7490"), "box"),
            ("Dead end", QColor("#be123c"), "box"),
            ("Loop outline", QColor("#a855f7"), "line"),
            ("Production", QColor("#7dd3fc"), "line"),
            ("Adapter override", QColor("#f0abfc"), "dash"),
        ],
        y=52,
        max_width=1500,
    )

    reachable_states = {
        state_id: state
        for state_id, state in analysis.states.items()
        if state.reachable
    }
    transitions = [
        transition
        for transition in analysis.transitions
        if transition.reachable
        and transition.source_state_id in reachable_states
        and transition.target_state_id in reachable_states
    ]
    if not reachable_states:
        _add_empty_message(scene, "No reachable control states were detected.", y=118)
        return scene

    layout_nodes = [
        LayoutNode(
            node_id=state_id,
            label=state.label,
            group=state.phase,
            width=238.0,
            height=76.0,
            priority=(
                100 if state_id == analysis.initial_state_id else
                80 if state.terminal else
                60 if state.adapter_handoff else
                0
            ),
        )
        for state_id, state in reachable_states.items()
    ]
    layout_edges = [
        LayoutEdge(
            edge_id=transition.transition_id,
            source_id=transition.source_state_id,
            target_id=transition.target_state_id,
            kind=transition.kind,
            weight=1.25 if transition.kind == "adapter" else 1.0,
        )
        for transition in transitions
    ]
    geometry = layout_and_route(
        layout_nodes,
        layout_edges,
        initial_node_id=analysis.initial_state_id,
        offset=QPointF(42.0, 150.0),
    )

    header_font = QFont("Sans Serif", 10)
    header_font.setBold(True)
    for phase, point in geometry.group_headers.items():
        header = QGraphicsSimpleTextItem(phase.upper())
        header.setFont(header_font)
        header.setBrush(QBrush(QColor("#cbd5e1")))
        header.setPos(point.x(), point.y())
        scene.addItem(header)

    for state_id, placement in geometry.placements.items():
        state = reachable_states[state_id]
        color = (
            QColor("#1d4ed8")
            if state_id == analysis.initial_state_id
            else QColor("#be123c")
            if state.dead_end
            else QColor("#0e7490")
            if state.terminal
            else QColor("#b45309")
            if state.adapter_handoff
            else QColor("#047857")
        )
        border = QColor("#a855f7") if state.loop_member else QColor("#dbe4f0")
        _add_node(
            scene,
            placement.rect,
            state.label,
            color,
            wrap_width=26,
            border_color=border,
            border_width=2.6 if state.loop_member else 1.5,
        )

    display_routes = separate_overlapping_routes(
        geometry.routes.values(),
        placements=geometry.placements,
    )
    label_positions = assign_label_positions(
        display_routes.values(),
        [placement.rect for placement in geometry.placements.values()],
    )

    transition_by_id = {item.transition_id: item for item in transitions}

    # Number production and adapter transitions independently in deterministic
    # graph order.  The same order is reused by the detail catalogue below, so
    # P1..Pn and A1..An can be found without scanning a mixed list.
    ordered_transitions = _ordered_transitions_for_codes(transitions, geometry)
    code_by_id: dict[str, str] = {}
    counters = {"production": 0, "adapter": 0}
    for transition in ordered_transitions:
        counters[transition.kind] += 1
        prefix = "A" if transition.kind == "adapter" else "P"
        code_by_id[transition.transition_id] = f"{prefix}{counters[transition.kind]}"

    details_by_kind: dict[str, list[tuple[str, str, QColor]]] = {
        "production": [],
        "adapter": [],
    }
    for edge_id, route in display_routes.items():
        transition = transition_by_id[edge_id]
        code = code_by_id[edge_id]
        color = QColor("#f0abfc") if transition.kind == "adapter" else QColor("#7dd3fc")

        # A wide background-coloured underlay creates a visual bridge at every
        # crossing.  Dashed adapter edges therefore never disappear into solid
        # production edges, even when they cross at the same coordinate.
        halo_pen = QPen(SCENE_BACKGROUND, 7.0)
        halo_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        halo_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        halo_item = _add_polyline_path(scene, route.points, halo_pen)
        halo_item.setZValue(0.0)

        pen = QPen(color, 2.4 if transition.kind == "adapter" else 2.25)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if transition.kind == "adapter":
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([7.0, 4.0])
            pen.setDashOffset((_numeric_code(code) % 4) * 2.0)
        path_item = _add_polyline_path(scene, route.points, pen)
        path_item.setZValue(1.0)
        tooltip = (
            f"{code} · {transition.kind}: {transition.label}\n"
            f"Guard: {transition.guard_label or 'none'}\n"
            f"Actions: {transition.action_label or 'control-state update'}"
        )
        path_item.setToolTip(tooltip)
        if len(route.points) >= 2:
            _draw_arrow(scene, route.points[-2], route.points[-1], color)
        label = _add_edge_label(
            scene, code, label_positions[edge_id], color
        )
        label.setToolTip(tooltip)
        source = reachable_states[transition.source_state_id]
        target = reachable_states[transition.target_state_id]
        detail = transition.label
        if transition.guard_label:
            detail += f" | guard: {transition.guard_label.replace(chr(10), '; ')}"
        if transition.action_label:
            detail += f" | action: {transition.action_label.replace(chr(10), '; ')}"
        details_by_kind[transition.kind].append(
            (
                code,
                f"{source.label} → {target.label}\n{detail}",
                color,
            )
        )

    _add_route_bundle_markers(scene, list(display_routes.values()))

    details_y = geometry.bounds.bottom() + 100.0
    details_font = QFont("Sans Serif", 10)
    details_font.setBold(True)
    detail_columns = 3
    detail_width = 600.0
    detail_height = 104.0

    current_y = details_y
    for kind, heading_text in (
        ("production", "Production transitions"),
        ("adapter", "Adapter overrides"),
    ):
        values = sorted(
            details_by_kind[kind],
            key=lambda item: _numeric_code(item[0]),
        )
        if not values:
            continue
        first_code = values[0][0]
        last_code = values[-1][0]
        heading = QGraphicsSimpleTextItem(
            f"{heading_text} · {first_code}–{last_code} · layout: {geometry.orientation}"
        )
        heading.setFont(details_font)
        heading.setBrush(QBrush(QColor("#f8fafc")))
        heading.setPos(42.0, current_y)
        scene.addItem(heading)

        for index, (code, detail, color) in enumerate(values):
            column = index % detail_columns
            row = index // detail_columns
            x = 42.0 + column * (detail_width + 24.0)
            y = current_y + 38.0 + row * (detail_height + 18.0)
            _add_node(
                scene,
                QRectF(x, y, 58.0, detail_height),
                code,
                QColor("#312e81") if code.startswith("P") else QColor("#86198f"),
                wrap_width=6,
                border_color=color,
                border_width=2.0,
            )
            text_item = QGraphicsTextItem(_wrap_label(detail, 70))
            text_item.setDefaultTextColor(QColor("#e2e8f0"))
            text_item.setTextWidth(detail_width - 72.0)
            text_item.setPos(x + 70.0, y + 4.0)
            scene.addItem(text_item)
        rows = (len(values) + detail_columns - 1) // detail_columns
        current_y += 38.0 + rows * (detail_height + 18.0) + 54.0

    unreachable_y = current_y
    if analysis.unreachable_productions:
        unreachable_heading = QGraphicsSimpleTextItem("Statically unreachable productions")
        unreachable_heading.setFont(details_font)
        unreachable_heading.setBrush(QBrush(QColor("#f8fafc")))
        unreachable_heading.setPos(42.0, unreachable_y)
        scene.addItem(unreachable_heading)
        production_by_name = {item.name: item for item in analysis.productions}
        columns = 3
        cell_width = 430.0
        for index, name in enumerate(analysis.unreachable_productions):
            production = production_by_name[name]
            source_state = analysis.states.get(production.source_state_id)
            target_state = analysis.states.get(production.target_state_id)
            value = (
                f"{name}\n"
                f"{source_state.label if source_state else '?'} → "
                f"{target_state.label if target_state else '?'}"
            )
            column = index % columns
            row = index // columns
            _add_node(
                scene,
                QRectF(
                    42.0 + column * (cell_width + 26.0),
                    unreachable_y + 42.0 + row * 92.0,
                    cell_width,
                    70.0,
                ),
                value,
                QColor("#334155"),
                wrap_width=52,
                border_color=QColor("#64748b"),
            )
    return scene

def build_interaction_scene(
    title: str,
    interactions: Iterable[MethodBufferInteraction],
) -> QGraphicsScene:
    """Render interactions as a matrix, eliminating ambiguous crossing edges."""
    scene = _new_scene()
    _add_scene_title(scene, title)
    _add_legend(
        scene,
        [
            ("Read", QColor("#1d4ed8"), "box"),
            ("Write / request", QColor("#b45309"), "box"),
            ("Read + write", QColor("#6d28d9"), "box"),
            ("Delete / clear", QColor("#be123c"), "box"),
        ],
        y=50,
        max_width=1200,
    )
    rows = list(interactions)
    if not rows:
        _add_empty_message(scene, "No buffer interactions were detected.", y=104)
        return scene

    actors = sorted({row.method_name for row in rows}, key=str.lower)
    buffers = sorted({row.buffer_name for row in rows}, key=str.lower)
    actor_width = 330.0
    cell_width = 132.0
    row_height = 72.0
    header_y = 126.0
    body_y = 212.0
    left = 36.0

    actor_map: dict[str, list[MethodBufferInteraction]] = defaultdict(list)
    cell_map: dict[tuple[str, str], list[MethodBufferInteraction]] = defaultdict(list)
    for interaction in rows:
        actor_map[interaction.method_name].append(interaction)
        cell_map[(interaction.method_name, interaction.buffer_name)].append(interaction)

    actor_header = QGraphicsSimpleTextItem("Production / adapter handler")
    actor_header.setBrush(QBrush(QColor("#cbd5e1")))
    actor_header.setPos(left, header_y + 22)
    scene.addItem(actor_header)
    for column, buffer_name in enumerate(buffers):
        rect = QRectF(
            left + actor_width + 18 + column * cell_width,
            header_y,
            cell_width - 8,
            64,
        )
        _add_node(
            scene,
            rect,
            buffer_name,
            QColor("#14532d"),
            wrap_width=18,
            border_color=QColor("#86efac"),
        )

    trigger_x = left + actor_width + 18 + len(buffers) * cell_width + 22
    has_triggers = any(item.triggered_by for item in rows)
    if has_triggers:
        trigger_header = QGraphicsSimpleTextItem("Triggered after production")
        trigger_header.setBrush(QBrush(QColor("#cbd5e1")))
        trigger_header.setPos(trigger_x, header_y + 22)
        scene.addItem(trigger_header)

    for row_index, actor in enumerate(actors):
        y = body_y + row_index * row_height
        interactions_for_actor = actor_map[actor]
        actor_label = actor
        actor_rect = QRectF(left, y, actor_width, 54)
        _add_node(
            scene,
            actor_rect,
            actor_label,
            QColor("#1e3a8a"),
            wrap_width=38,
        )
        for column, buffer_name in enumerate(buffers):
            cell = cell_map.get((actor, buffer_name), [])
            rect = QRectF(
                left + actor_width + 18 + column * cell_width,
                y,
                cell_width - 8,
                54,
            )
            if not cell:
                empty = QGraphicsRectItem(rect)
                empty.setPen(QPen(QColor("#334155"), 0.8))
                empty.setBrush(QBrush(QColor("#111827")))
                scene.addItem(empty)
                continue
            modes = {item.mode.lower() for item in cell}
            has_read = bool(modes & {"read", "query"})
            has_write = bool(modes & {"write", "request"})
            has_delete = bool(modes & {"delete", "clear"})
            if has_delete:
                color = QColor("#be123c")
                code = "D"
            elif has_read and has_write:
                color = QColor("#6d28d9")
                code = "R/W"
            elif has_write:
                color = QColor("#b45309")
                code = "W"
            else:
                color = QColor("#1d4ed8")
                code = "R"
            _add_node(
                scene,
                rect,
                code,
                color,
                wrap_width=8,
                border_color=QColor("#e2e8f0"),
            )
            tooltip = "\n\n".join(
                f"{item.mode}: {item.function_name}\n{item.detail or ''}"
                for item in cell
            )
            for graphics_item in scene.items(rect):
                graphics_item.setToolTip(tooltip)
        if has_triggers:
            triggers = sorted(
                {
                    trigger
                    for item in interactions_for_actor
                    for trigger in item.triggered_by
                },
                key=str.lower,
            )
            trigger_text = ", ".join(triggers) if triggers else "—"
            trigger_item = QGraphicsTextItem(_wrap_label(trigger_text, 42))
            trigger_item.setDefaultTextColor(QColor("#e2e8f0"))
            trigger_item.setTextWidth(310)
            trigger_item.setPos(trigger_x, y + 4)
            scene.addItem(trigger_item)
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
    """Render a jump path including adapter overrides between productions."""
    scene = _new_scene()
    _add_scene_title(scene, f"Jump path to production: {target_production}")
    path = analysis.transition_path_to_production(target_production)
    if not path:
        warning = QGraphicsTextItem(
            "No statically reachable path could be derived. The target is shown "
            "without claiming that it is reachable."
        )
        warning.setDefaultTextColor(QColor("#fecaca"))
        warning.setTextWidth(760)
        warning.setPos(24, 72)
        scene.addItem(warning)
        target = analysis.production(target_production)
        if target is not None:
            source = analysis.states.get(target.source_state_id)
            destination = analysis.states.get(target.target_state_id)
            _add_node(
                scene,
                QRectF(30, 170, 280, 88),
                source.label if source else target.source_label,
                QColor("#7f1d1d"),
            )
            _add_edge_label(scene, target.name, QPointF(350, 196), QColor("#fecaca"))
            _add_node(
                scene,
                QRectF(590, 170, 280, 88),
                destination.label if destination else target.target_label,
                QColor("#7f1d1d"),
            )
        return scene

    progress = _ordered_transition_progress(path, fired_productions)
    node_width, node_height, spacing = 245.0, 82.0, 150.0
    y = 150.0
    states = analysis.state_sequence_for_transition_path(path)
    for index, state in enumerate(states):
        x = 30 + index * (node_width + spacing)
        color = (
            QColor("#0e7490")
            if index == len(states) - 1 and progress >= len(path)
            else QColor("#047857")
            if index <= progress
            else QColor("#334155")
        )
        _add_node(
            scene,
            QRectF(x, y, node_width, node_height),
            state,
            color,
            wrap_width=28,
        )

    for index, transition in enumerate(path):
        source_x = 30 + index * (node_width + spacing) + node_width
        target_x = 30 + (index + 1) * (node_width + spacing)
        start = QPointF(source_x, y + node_height / 2)
        end = QPointF(target_x, y + node_height / 2)
        completed = index < progress
        active = index == progress and progress < len(path)
        base = QColor("#f0abfc") if transition.kind == "adapter" else QColor("#7dd3fc")
        color = QColor("#22c55e") if completed else base if active else QColor("#64748b")
        pen = QPen(color, 3.0 if completed or active else 2.0)
        if transition.kind == "adapter":
            pen.setStyle(Qt.PenStyle.DashLine)
        graph_path = QPainterPath(start)
        graph_path.cubicTo(
            QPointF((start.x() + end.x()) / 2, y + 18),
            QPointF((start.x() + end.x()) / 2, y + node_height - 18),
            end,
        )
        _add_path(scene, graph_path, pen)
        _draw_arrow(scene, QPointF((start.x() + end.x()) / 2, y + node_height / 2), end, color)
        label = transition.label
        if transition.guard_label:
            label += "\n[" + transition.guard_label.replace("\n", "; ") + "]"
        _add_edge_label(scene, label, QPointF(source_x + 18, y - 55), color)

    status = QGraphicsSimpleTextItem(
        "Target production fired."
        if progress >= len(path)
        else f"Reached {progress} of {len(path)} control-flow transitions."
    )
    status.setBrush(
        QBrush(QColor("#86efac") if progress >= len(path) else QColor("#bae6fd"))
    )
    status.setPos(24, 78)
    scene.addItem(status)
    return scene

def build_declarative_memory_scene(
    snapshot: DeclarativeMemorySnapshot,
    *,
    title: str,
) -> QGraphicsScene:
    """Render memory contents and operations with obstacle-aware orthogonal routes."""
    scene = _new_scene()
    _add_scene_title(scene, title)
    _add_legend(
        scene,
        [
            ("Memory", QColor("#1d4ed8"), "box"),
            ("Runtime chunk", QColor("#0f766e"), "box"),
            ("Explicit static DM chunk", QColor("#6d28d9"), "box"),
            ("Buffer harvest/retrieval link", QColor("#22d3ee"), "dash"),
            ("Explicit memory write", QColor("#f59e0b"), "line"),
            ("Chunk reference", QColor("#38bdf8"), "line"),
            ("Shared value", QColor("#94a3b8"), "dash"),
        ],
        y=52,
        max_width=1500,
    )

    memory_names = snapshot.memories or sorted(
        {chunk.memory_name for chunk in snapshot.chunks}
    )
    if not memory_names:
        _add_empty_message(scene, "No declarative memory was detected.", y=118)
        return scene

    buffer_links = [
        operation
        for operation in snapshot.operations
        if str(operation.get("mode")) == "buffer_link"
    ]
    memory_ops = [
        operation
        for operation in snapshot.operations
        if str(operation.get("mode")) != "buffer_link"
    ]
    chunks_by_memory: dict[str, list[Any]] = defaultdict(list)
    for chunk in snapshot.chunks:
        chunks_by_memory[chunk.memory_name].append(chunk)

    node_rects: dict[str, QRectF] = {}
    node_specs: dict[str, tuple[str, QColor, int, QColor | None]] = {}
    edge_specs: list[tuple[LayoutEdge, QColor, Qt.PenStyle, str]] = []

    top = 150.0
    left_x = 36.0
    buffer_width = 320.0
    center_x = 520.0
    memory_width = 330.0
    memory_column_gap = 470.0

    if buffer_links:
        heading = QGraphicsSimpleTextItem("Buffers linked by pyactr simulation()")
        heading.setBrush(QBrush(QColor("#f8fafc")))
        heading.setPos(left_x, top - 38.0)
        scene.addItem(heading)
    for index, operation in enumerate(buffer_links):
        node_id = f"buffer:{index}"
        rect = QRectF(left_x, top + index * 88.0, buffer_width, 58.0)
        node_rects[node_id] = rect
        node_specs[node_id] = (
            str(operation.get("actor", "buffer")),
            QColor("#155e75"),
            38,
            QColor("#a5f3fc"),
        )

    memory_rects: dict[str, str] = {}
    chunk_ids: dict[str, str] = {}
    max_memory_bottom = top
    for memory_index, memory_name in enumerate(memory_names):
        x = center_x + memory_index * memory_column_gap
        memory_id = f"memory:{memory_name}"
        memory_rects[memory_name] = memory_id
        memory_rect = QRectF(x, top, memory_width, 70.0)
        node_rects[memory_id] = memory_rect
        node_specs[memory_id] = (
            f"Memory: {memory_name}", QColor("#1d4ed8"), 36, QColor("#bfdbfe")
        )
        y = memory_rect.bottom() + 58.0
        for chunk_index, chunk in enumerate(chunks_by_memory.get(memory_name, [])):
            detail = chunk.label
            if chunk.traces:
                detail += "\ntraces=" + ", ".join(
                    f"{value:.3f}" for value in chunk.traces[-4:]
                )
            if chunk.activation is not None:
                detail += f"\nactivation={chunk.activation:.3f}"
            height = max(
                90.0,
                _label_rect(
                    _wrap_label(detail, 40), QFont("Sans Serif", 9), memory_width, 28
                ).height(),
            )
            chunk_node_id = f"chunk:{chunk.chunk_id}"
            chunk_ids[chunk.chunk_id] = chunk_node_id
            rect = QRectF(x, y, memory_width, height)
            node_rects[chunk_node_id] = rect
            node_specs[chunk_node_id] = (
                detail,
                QColor("#0f766e") if chunk.source == "runtime" else QColor("#6d28d9"),
                40,
                QColor("#dbeafe"),
            )
            edge_specs.append(
                (
                    LayoutEdge(
                        f"contains:{memory_name}:{chunk_index}",
                        memory_id,
                        chunk_node_id,
                        "contains",
                    ),
                    QColor("#64748b"),
                    Qt.PenStyle.SolidLine,
                    "",
                )
            )
            y += height + 52.0
        max_memory_bottom = max(max_memory_bottom, y)

    # Place explicit operations below the tallest memory column so their routes
    # do not pass through chunks or buffer nodes.
    operations_y = max(
        max_memory_bottom + 80.0,
        top + max(1, len(buffer_links)) * 88.0 + 90.0,
    )
    if memory_ops:
        heading = QGraphicsSimpleTextItem("Explicit memory operations")
        heading.setBrush(QBrush(QColor("#f8fafc")))
        heading.setPos(left_x, operations_y - 42.0)
        scene.addItem(heading)
    for index, operation in enumerate(memory_ops):
        column = index % 2
        row = index // 2
        node_id = f"operation:{index}"
        x = left_x + column * 390.0
        y = operations_y + row * 112.0
        rect = QRectF(x, y, 350.0, 78.0)
        mode = str(operation.get("mode", "access"))
        color = (
            QColor("#be123c")
            if mode in {"delete", "clear"}
            else QColor("#f59e0b")
            if mode in {"write", "add"}
            else QColor("#1d4ed8")
        )
        label = (
            f"{operation.get('actor', 'code')}\n"
            f"{mode} → {operation.get('memory_name', 'decmem')}\n"
            f"{operation.get('detail', '')}"
        )
        node_rects[node_id] = rect
        node_specs[node_id] = (label, color, 44, QColor("#e2e8f0"))

    # Add all semantic edges after every obstacle is known.
    for index, operation in enumerate(buffer_links):
        target = memory_rects.get(str(operation.get("memory_name", "decmem")))
        if target:
            edge_specs.append(
                (
                    LayoutEdge(f"buffer-link:{index}", f"buffer:{index}", target, "buffer_link"),
                    QColor("#22d3ee"),
                    Qt.PenStyle.DashLine,
                    "",
                )
            )
    for index, operation in enumerate(memory_ops):
        target = memory_rects.get(str(operation.get("memory_name", "decmem")))
        if target:
            mode = str(operation.get("mode", "access"))
            color = (
                QColor("#be123c")
                if mode in {"delete", "clear"}
                else QColor("#f59e0b")
                if mode in {"write", "add"}
                else QColor("#1d4ed8")
            )
            edge_specs.append(
                (
                    LayoutEdge(f"memory-op:{index}", f"operation:{index}", target, mode),
                    color,
                    Qt.PenStyle.SolidLine,
                    mode,
                )
            )
    for index, edge in enumerate(snapshot.edges):
        source = chunk_ids.get(edge.source_id)
        target = chunk_ids.get(edge.target_id)
        if source is None or target is None:
            continue
        color = QColor("#38bdf8") if edge.relation == "reference" else QColor("#94a3b8")
        style = Qt.PenStyle.SolidLine if edge.relation == "reference" else Qt.PenStyle.DashLine
        edge_specs.append(
            (
                LayoutEdge(f"chunk-edge:{index}", source, target, edge.relation),
                color,
                style,
                edge.label,
            )
        )

    for node_id, rect in node_rects.items():
        label, color, wrap, border = node_specs[node_id]
        _add_node(
            scene, rect, label, color, wrap_width=wrap, border_color=border
        )

    routes = route_fixed_nodes(
        node_rects, [spec[0] for spec in edge_specs]
    )
    edge_render = {spec[0].edge_id: spec[1:] for spec in edge_specs}
    for edge_id, route in routes.items():
        color, style, label_text = edge_render[edge_id]
        pen = QPen(color, 2.0)
        pen.setStyle(style)
        _add_polyline_path(scene, route.points, pen)
        if len(route.points) >= 2:
            _draw_arrow(scene, route.points[-2], route.points[-1], color)
        if label_text:
            _add_edge_label(scene, label_text, route.label_position, color)
    return scene

def _ordered_transition_progress(
    path: list[StateTransitionAnalysis], fired_productions: list[str]
) -> int:
    """Count transitions reached; adapter edges complete with their trigger rule."""
    fired_index = 0
    transition_index = 0
    while transition_index < len(path):
        transition = path[transition_index]
        if transition.kind == "adapter":
            trigger = transition.trigger_production
            if trigger and any(
                name.casefold() == trigger.casefold()
                for name in fired_productions[:fired_index]
            ):
                transition_index += 1
                continue
            break
        while fired_index < len(fired_productions):
            fired = fired_productions[fired_index]
            fired_index += 1
            if (
                transition.production_name
                and fired.casefold() == transition.production_name.casefold()
            ):
                transition_index += 1
                break
        else:
            break
    return transition_index

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
    border_color: QColor | None = None,
    border_width: float = 1.5,
) -> QGraphicsRectItem:
    node = QGraphicsRectItem(rect)
    node.setPen(QPen(border_color or QColor("#dbe4f0"), border_width))
    node.setBrush(QBrush(color))
    scene.addItem(node)
    text = QGraphicsTextItem(_wrap_label(label, wrap_width))
    text.setDefaultTextColor(TEXT_COLOR)
    text.setTextWidth(rect.width() - 18)
    text.setPos(rect.x() + 9, rect.y() + 8)
    scene.addItem(text)
    return node


def _add_route_bundle_markers(scene: QGraphicsScene, routes: list[Any]) -> None:
    """Mark intentional shared trunks so merges and splits remain traceable."""
    segment_routes: dict[tuple[float, float, float, float], set[str]] = defaultdict(set)
    segment_points: dict[tuple[float, float, float, float], tuple[QPointF, QPointF]] = {}
    for route in routes:
        for first, second in zip(route.points, route.points[1:]):
            a = (round(first.x(), 3), round(first.y(), 3))
            b = (round(second.x(), 3), round(second.y(), 3))
            key = (*a, *b) if a <= b else (*b, *a)
            segment_routes[key].add(route.edge.edge_id)
            segment_points[key] = (first, second)
    marked: set[tuple[float, float]] = set()
    for key, edge_ids in segment_routes.items():
        if len(edge_ids) < 2:
            continue
        first, second = segment_points[key]
        for point in (first, second):
            point_key = (round(point.x(), 3), round(point.y(), 3))
            if point_key in marked:
                continue
            marked.add(point_key)
            marker = QGraphicsEllipseItem(point.x() - 4.5, point.y() - 4.5, 9.0, 9.0)
            marker.setPen(QPen(QColor("#f8fafc"), 1.2))
            marker.setBrush(QBrush(QColor("#334155")))
            marker.setToolTip(
                "Shared route bus: " + ", ".join(sorted(edge_ids))
            )
            scene.addItem(marker)


def _add_polyline_path(
    scene: QGraphicsScene, points: list[QPointF], pen: QPen
) -> QGraphicsPathItem:
    if not points:
        return _add_path(scene, QPainterPath(), pen)
    path = QPainterPath(points[0])
    for point in points[1:]:
        path.lineTo(point)
    return _add_path(scene, path, pen)


def _add_path(
    scene: QGraphicsScene, path: QPainterPath, pen: QPen
) -> QGraphicsPathItem:
    item = QGraphicsPathItem(path)
    item.setPen(pen)
    item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
    scene.addItem(item)
    return item


def _numeric_code(code: str) -> int:
    try:
        return int(code[1:])
    except (TypeError, ValueError, IndexError):
        return 10**9


def _ordered_transitions_for_codes(
    transitions: list[StateTransitionAnalysis],
    geometry: Any,
) -> list[StateTransitionAnalysis]:
    """Return stable P/A numbering based on the rendered graph geometry.

    Production and adapter codes are independent.  Within each family, source
    nodes are ordered by rank and screen position, followed by target position
    and the semantic transition label.  The result is deterministic and mirrors
    how a reader scans the graph.
    """
    def key(transition: StateTransitionAnalysis) -> tuple[Any, ...]:
        source = geometry.placements[transition.source_state_id]
        target = geometry.placements[transition.target_state_id]
        family = 0 if transition.kind == "production" else 1
        return (
            family,
            source.rank,
            round(source.rect.top(), 3),
            round(source.rect.left(), 3),
            target.rank,
            round(target.rect.top(), 3),
            round(target.rect.left(), 3),
            transition.label.casefold(),
            transition.transition_id,
        )

    return sorted(transitions, key=key)


def _add_edge_label(
    scene: QGraphicsScene,
    text: str,
    position: QPointF,
    color: QColor,
) -> QGraphicsSimpleTextItem:
    label = QGraphicsSimpleTextItem(text)
    label.setBrush(QBrush(color))
    label.setPos(position)
    bounds = label.boundingRect().adjusted(-6, -3, 6, 3)
    background = QGraphicsRectItem(
        QRectF(
            position.x() + bounds.x(),
            position.y() + bounds.y(),
            bounds.width(),
            bounds.height(),
        )
    )
    background.setPen(QPen(QColor("#334155"), 0.8))
    background.setBrush(QBrush(LABEL_BACKGROUND))
    scene.addItem(background)
    scene.addItem(label)
    return label


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
