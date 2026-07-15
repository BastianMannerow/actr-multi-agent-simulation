"""Zoomable, exportable graph views for ACT-R explainability."""

from __future__ import annotations

import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

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

from gui.graph_layout import (
    LayoutEdge,
    LayoutNode,
    assign_label_positions,
)

from gui.graphing.chunk_diagram import build_declarative_memory_scene
from gui.graphing.layout_engine import ReusableLayeredGraphLayout
from gui.graphing.edge_items import HoverableEdgePathItem
from gui.graphing.models import DiagramEdge, DiagramNode
from gui.graphing.renderers import EdgeRenderer, NodeRenderer
from gui.graphing.transition_explainability import (
    compact_module_caption,
    transition_explanation,
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
MAX_RASTER_EXPORT_DIMENSION = 4096


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
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._fit_pending = False
        self._llm_payload: dict[str, Any] | None = None
        self._llm_default_name = "agent_analysis"
        self._item_activation_handler: Callable[[dict[str, Any]], None] | None = None
        self._activation_press_position: QPointF | None = None
        self._activation_payload: dict[str, Any] | None = None
        self._activation_item = None
        self._activation_drag_threshold = 6.0
        self._hovered_edge: HoverableEdgePathItem | None = None
        self._edge_hover_radius_px = 7.0

    def set_item_activation_handler(
        self, handler: Callable[[dict[str, Any]], None] | None
    ) -> None:
        """Register a handler for clickable explainability items in a scene."""
        self._item_activation_handler = handler

    @staticmethod
    def _payload_from_item(item) -> tuple[dict[str, Any] | None, Any | None]:
        """Return payload only from an explicitly interactive painted item."""
        while item is not None:
            payload = item.data(0)
            if (
                isinstance(payload, dict)
                and payload.get("payload_type")
                and item.data(1) in {"interactive-edge", "interactive-edge-label"}
            ):
                return payload, item
            item = item.parentItem()
        return None, None

    def mousePressEvent(self, event):  # noqa: N802
        self._activation_press_position = None
        self._activation_payload = None
        self._activation_item = None
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._item_activation_handler is not None
        ):
            payload, item = self._payload_from_item(
                self.itemAt(event.position().toPoint())
            )
            if payload is not None:
                self._activation_press_position = QPointF(event.position())
                self._activation_payload = payload
                self._activation_item = item
        # Never consume the press. ScrollHandDrag must remain able to start even
        # when the press happens on a clickable edge.
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._activation_press_position is not None:
            delta = event.position() - self._activation_press_position
            if abs(delta.x()) + abs(delta.y()) > self._activation_drag_threshold:
                self._activation_payload = None
                self._activation_item = None
        super().mouseMoveEvent(event)
        if event.buttons() == Qt.MouseButton.NoButton:
            self._update_edge_hover(event.position())
        else:
            # Panning must never leave a stale yellow edge behind or spend time
            # on proximity tests while the viewport is moving.
            self._set_hovered_edge(None)

    def mouseReleaseEvent(self, event):  # noqa: N802
        payload = self._activation_payload
        candidate = self._activation_item
        press_position = self._activation_press_position
        self._activation_press_position = None
        self._activation_payload = None
        self._activation_item = None
        super().mouseReleaseEvent(event)
        if (
            event.button() != Qt.MouseButton.LeftButton
            or payload is None
            or candidate is None
            or press_position is None
            or self._item_activation_handler is None
        ):
            return
        delta = event.position() - press_position
        if abs(delta.x()) + abs(delta.y()) > self._activation_drag_threshold:
            return
        release_payload, release_item = self._payload_from_item(
            self.itemAt(event.position().toPoint())
        )
        if release_item is candidate and release_payload == payload:
            self._item_activation_handler(payload)

    def wheelEvent(self, event):  # noqa: N802
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        # Re-evaluate in viewport coordinates after the transform changed.
        self._update_edge_hover(event.position())

    def leaveEvent(self, event):  # noqa: N802
        self._set_hovered_edge(None)
        super().leaveEvent(event)

    def setScene(self, scene) -> None:  # noqa: N802
        self._set_hovered_edge(None)
        super().setScene(scene)

    def _set_hovered_edge(
        self, edge: HoverableEdgePathItem | None
    ) -> None:
        if edge is self._hovered_edge:
            return
        previous = self._hovered_edge
        self._hovered_edge = edge
        if previous is not None and previous.scene() is not None:
            previous.set_highlighted(False)
        if edge is not None and edge.scene() is not None:
            edge.set_highlighted(True)

    def _update_edge_hover(self, viewport_position: QPointF) -> None:
        self._set_hovered_edge(self._edge_at_view_position(viewport_position))

    def _edge_at_view_position(
        self, viewport_position: QPointF
    ) -> HoverableEdgePathItem | None:
        """Return the nearest routed edge within a constant screen radius.

        Qt normally tests ``QGraphicsPathItem.shape()`` in scene coordinates.
        After a large graph is fitted, a 2-pixel scene pen can occupy only a
        fraction of a viewport pixel; at orthogonal joins its generated shape
        can also differ from the visibly antialiased stroke. This method first
        narrows candidates through the scene index and then measures the actual
        transformed polyline in viewport pixels.
        """
        scene = self.scene()
        if scene is None:
            return None

        viewport_point = viewport_position.toPoint()
        # Labels and arrowheads are registered as edge companions. Resolve them
        # first, independent of their distance from the centreline.
        for visual_item in self.items(viewport_point):
            owner = getattr(visual_item, "_hover_edge_owner", None)
            if isinstance(owner, HoverableEdgePathItem):
                return owner

        radius = self._edge_hover_radius_px
        view_rect = QRectF(
            viewport_position.x() - radius,
            viewport_position.y() - radius,
            radius * 2.0,
            radius * 2.0,
        )
        scene_rect = self.mapToScene(view_rect.toRect()).boundingRect()
        candidates = scene.items(
            scene_rect,
            Qt.ItemSelectionMode.IntersectsItemShape,
            Qt.SortOrder.DescendingOrder,
            self.viewportTransform(),
        )

        best: HoverableEdgePathItem | None = None
        best_distance = float("inf")
        transform = self.viewportTransform()
        for candidate in candidates:
            if not isinstance(candidate, HoverableEdgePathItem):
                continue
            try:
                scene_path = candidate.mapToScene(candidate.path())
                viewport_path = transform.map(scene_path)
            except RuntimeError:
                continue
            distance = self._distance_to_view_path(
                viewport_position, viewport_path
            )
            if distance <= radius and (
                distance < best_distance - 0.05
                or (
                    abs(distance - best_distance) <= 0.05
                    and best is not None
                    and candidate.zValue() > best.zValue()
                )
            ):
                best = candidate
                best_distance = distance
        return best

    @staticmethod
    def _distance_to_view_path(point: QPointF, path: QPainterPath) -> float:
        """Measure point-to-polyline distance after all view transforms."""
        best = float("inf")
        for polygon in path.toSubpathPolygons():
            if len(polygon) == 1:
                delta = polygon[0] - point
                best = min(best, math.hypot(delta.x(), delta.y()))
                continue
            for first, second in zip(polygon, polygon[1:]):
                dx = second.x() - first.x()
                dy = second.y() - first.y()
                length_squared = dx * dx + dy * dy
                if length_squared <= 1e-12:
                    distance = math.hypot(
                        point.x() - first.x(), point.y() - first.y()
                    )
                else:
                    projection = (
                        (point.x() - first.x()) * dx
                        + (point.y() - first.y()) * dy
                    ) / length_squared
                    projection = max(0.0, min(1.0, projection))
                    nearest_x = first.x() + projection * dx
                    nearest_y = first.y() + projection * dy
                    distance = math.hypot(
                        point.x() - nearest_x,
                        point.y() - nearest_y,
                    )
                best = min(best, distance)
        return best

    def reset_zoom(self) -> None:
        self._set_hovered_edge(None)
        self.resetTransform()
        if self.scene() is None:
            return
        if not self.isVisible() or self.viewport().width() < 50:
            self._fit_pending = True
            return
        bounds = self.scene().sceneRect()
        if bounds.isNull() or bounds.isEmpty():
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
        llm_action = QAction("Export for LLM", self)
        llm_action.setEnabled(self._llm_payload is not None)
        llm_action.triggered.connect(self.export_for_llm_dialog)
        menu.addAction(llm_action)
        menu.exec(event.globalPos())

    def set_llm_export_data(
        self, payload: dict[str, Any] | None, *, default_name: str = "agent_analysis"
    ) -> None:
        self._llm_payload = payload
        self._llm_default_name = default_name or "agent_analysis"

    def export_for_llm_dialog(self) -> Path | None:
        if self._llm_payload is None:
            return None
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export for LLM",
            str(Path.home() / f"{self._llm_default_name}.json"),
            "Structured JSON (*.json)",
        )
        if not path:
            return None
        return self.export_llm_to(path)

    def export_llm_to(self, path: str | Path) -> Path:
        if self._llm_payload is None:
            raise RuntimeError("There is no structured graph data to export.")
        destination = Path(path)
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")
        destination.write_text(
            json.dumps(self._llm_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

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
        rect = self.scene().sceneRect()
        if rect.isNull() or rect.isEmpty():
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
                longest_side = max(rect.width(), rect.height(), 1.0)
                raster_scale = min(
                    1.0,
                    MAX_RASTER_EXPORT_DIMENSION / longest_side,
                )
                width = max(1, int(round(rect.width() * raster_scale)))
                height = max(1, int(round(rect.height() * raster_scale)))
                image = QImage(width, height, QImage.Format.Format_ARGB32)
                if image.isNull():
                    raise RuntimeError(
                        "The PNG export buffer could not be allocated. "
                        "Use SVG or Export for LLM for this graph."
                    )
                image.fill(Qt.GlobalColor.transparent)
                painter = QPainter(image)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
                scene.render(
                    painter,
                    QRectF(0.0, 0.0, float(width), float(height)),
                    rect,
                )
                painter.end()
                image.save(str(destination))
        finally:
            scene.setBackgroundBrush(original_background)
        return destination


def _translate_layout_below(geometry: Any, minimum_y: float) -> None:
    """Move a complete routed layout below protected header/legend space."""
    values = [placement.rect.top() for placement in geometry.placements.values()]
    values.extend(
        point.y()
        for route in geometry.routes.values()
        for point in route.points
    )
    values.extend(point.y() for point in geometry.group_headers.values())
    if not values:
        return
    delta = minimum_y - min(values)
    if delta <= 0.0:
        return
    for placement in geometry.placements.values():
        placement.rect.translate(0.0, delta)
    for route in geometry.routes.values():
        route.points = [QPointF(point.x(), point.y() + delta) for point in route.points]
        route.label_position = QPointF(
            route.label_position.x(), route.label_position.y() + delta
        )
    geometry.group_headers = {
        group: QPointF(point.x(), point.y() + delta)
        for group, point in geometry.group_headers.items()
    }
    geometry.bounds.translate(0.0, delta)


def _translate_rendered_layout_below(
    geometry: Any,
    routes: dict[str, Any],
    minimum_y: float,
) -> None:
    values = [placement.rect.top() for placement in geometry.placements.values()]
    values.extend(point.y() for route in routes.values() for point in route.points)
    values.extend(point.y() for point in geometry.group_headers.values())
    if not values:
        return
    delta = minimum_y - min(values)
    if delta <= 0.0:
        return
    for placement in geometry.placements.values():
        placement.rect.translate(0.0, delta)
    for route in routes.values():
        route.points = [QPointF(point.x(), point.y() + delta) for point in route.points]
        route.label_position = QPointF(
            route.label_position.x(), route.label_position.y() + delta
        )
    for route in geometry.routes.values():
        route.points = [QPointF(point.x(), point.y() + delta) for point in route.points]
        route.label_position = QPointF(
            route.label_position.x(), route.label_position.y() + delta
        )
    geometry.group_headers = {
        group: QPointF(point.x(), point.y() + delta)
        for group, point in geometry.group_headers.items()
    }
    geometry.bounds.translate(0.0, delta)


def build_state_transition_scene(analysis: AgentStaticAnalysis) -> QGraphicsScene:
    """Render an overview graph with compact, clickable module labels.

    Conditions and actions are intentionally absent from the scene.  Each P/A
    label and its route carry a structured explanation payload that the owning
    view opens in a dedicated dialog.
    """
    scene = _new_scene()
    _add_scene_title(scene, f"State transitions — {analysis.agent_type}")
    legend_bounds = _add_legend(
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
            ("Click P/A for details", QColor("#e2e8f0"), "box"),
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

    transition_bundles: dict[
        tuple[str, str, str], list[StateTransitionAnalysis]
    ] = defaultdict(list)
    for transition in transitions:
        transition_bundles[
            (
                transition.source_state_id,
                transition.target_state_id,
                transition.kind,
            )
        ].append(transition)

    bundle_transitions: dict[str, list[StateTransitionAnalysis]] = {}
    layout_edges: list[LayoutEdge] = []
    for index, (key, values) in enumerate(sorted(transition_bundles.items())):
        source_id, target_id, kind = key
        bundle_id = f"bundle:{kind}:{index}"
        bundle_transitions[bundle_id] = sorted(
            values, key=lambda item: (item.label.casefold(), item.transition_id)
        )
        layout_edges.append(
            LayoutEdge(
                edge_id=bundle_id,
                source_id=source_id,
                target_id=target_id,
                kind=kind,
                weight=1.0 + min(0.8, 0.12 * (len(values) - 1)),
            )
        )

    geometry = ReusableLayeredGraphLayout(
        minimum_lane_gap=20.0,
        node_gap=72.0,
        rank_gap=118.0,
        port_stub=18.0,
    ).layout(
        layout_nodes,
        layout_edges,
        initial_node_id=analysis.initial_state_id,
        offset=QPointF(42.0, legend_bounds.bottom() + 58.0),
    )
    _translate_layout_below(geometry, legend_bounds.bottom() + 42.0)
    display_routes = dict(geometry.routes)
    _translate_rendered_layout_below(
        geometry, display_routes, legend_bounds.bottom() + 42.0
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

    ordered_transitions = _ordered_transitions_for_codes(transitions, geometry)
    code_by_id: dict[str, str] = {}
    counters = {"production": 0, "adapter": 0}
    for transition in ordered_transitions:
        counters[transition.kind] += 1
        prefix = "A" if transition.kind == "adapter" else "P"
        code_by_id[transition.transition_id] = f"{prefix}{counters[transition.kind]}"
    scene._transition_codes = dict(code_by_id)

    bundle_payloads: dict[str, dict[str, Any]] = {}
    bundle_label_text: dict[str, str] = {}
    label_sizes: dict[str, tuple[float, float]] = {}
    metrics = QFontMetrics(QFont("Sans Serif", 8))
    for edge_id, bundled in bundle_transitions.items():
        explanations = [
            transition_explanation(
                analysis, transition, code_by_id[transition.transition_id]
            )
            for transition in bundled
        ]
        codes = [item.code for item in explanations]
        code_caption = _compact_code_range(codes)
        module_caption = compact_module_caption(explanations)
        label_text = f"{code_caption}\n{module_caption}"
        bundle_label_text[edge_id] = label_text
        width = max(
            metrics.horizontalAdvance(code_caption),
            metrics.horizontalAdvance(module_caption),
        ) + 18.0
        label_sizes[edge_id] = (min(max(width, 68.0), 250.0), 42.0)
        bundle_payloads[edge_id] = {
            "payload_type": "state_transition_bundle",
            "caption": code_caption,
            "transitions": [item.to_payload() for item in explanations],
        }
    scene._transition_details = dict(bundle_payloads)

    label_positions = assign_label_positions(
        display_routes.values(),
        [placement.rect for placement in geometry.placements.values()],
        label_width=90.0,
        label_height=42.0,
        label_sizes=label_sizes,
    )

    for edge_id, route in display_routes.items():
        bundled = bundle_transitions[edge_id]
        kind = bundled[0].kind
        color = QColor("#f0abfc") if kind == "adapter" else QColor("#7dd3fc")
        payload = bundle_payloads[edge_id]

        halo_pen = QPen(SCENE_BACKGROUND, 7.0)
        halo_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        halo_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        halo_item = _add_polyline_path(scene, route.points, halo_pen)
        halo_item.setZValue(0.0)

        pen = QPen(color, 2.4 if kind == "adapter" else 2.25)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if kind == "adapter":
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([7.0, 4.0])
        path = QPainterPath(route.points[0])
        for point in route.points[1:]:
            path.lineTo(point)
        path_item = HoverableEdgePathItem(path, pen, payload=payload)
        path_item.set_base_z(1.0)
        path_item.setToolTip("Click to explain this transition.")
        scene.addItem(path_item)
        if len(route.points) >= 2:
            arrow_items = _draw_arrow(scene, route.points[-2], route.points[-1], color)
            for arrow_item in arrow_items:
                path_item.add_companion(arrow_item)
        label = _add_edge_label(
            scene,
            bundle_label_text[edge_id],
            label_positions[edge_id],
            color,
            payload=payload,
        )
        label.setToolTip("Click to explain this transition.")
        path_item.add_companion(label)
        background = getattr(label, "_edge_label_background", None)
        if background is not None:
            path_item.add_companion(background)

    _add_route_bundle_markers(scene, list(display_routes.values()))

    if analysis.unreachable_productions:
        y = geometry.bounds.bottom() + 82.0
        heading = QGraphicsSimpleTextItem("Statically unreachable productions")
        font = QFont("Sans Serif", 10)
        font.setBold(True)
        heading.setFont(font)
        heading.setBrush(QBrush(QColor("#f8fafc")))
        heading.setPos(42.0, y)
        scene.addItem(heading)
        value = QGraphicsTextItem(
            _wrap_label(", ".join(analysis.unreachable_productions), 100)
        )
        value.setDefaultTextColor(QColor("#cbd5e1"))
        value.setTextWidth(1000.0)
        value.setPos(42.0, y + 30.0)
        scene.addItem(value)

    scene.setSceneRect(scene.itemsBoundingRect().adjusted(-28, -28, 28, 28))
    return scene

def build_interaction_scene(
    title: str,
    interactions: Iterable[MethodBufferInteraction],
) -> QGraphicsScene:
    """Render interactions as a matrix, eliminating ambiguous crossing edges."""
    scene = _new_scene()
    _add_scene_title(scene, title)
    legend_bounds = _add_legend(
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
    header_y = legend_bounds.bottom() + 44.0
    body_y = header_y + 86.0
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
        history_path: QPainterPath | None = None
        markers: list[QGraphicsEllipseItem] = []
        for entry in entries:
            timestamp = float(entry.get("timestamp", 0.0))
            point = QPointF(left + (timestamp / max_time) * width, y + 10)
            if history_path is None:
                history_path = QPainterPath(point)
            else:
                history_path.lineTo(point)
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
            marker.setZValue(4.0)
            scene.addItem(marker)
            markers.append(marker)
        if history_path is not None and history_path.elementCount() >= 2:
            history_edge = HoverableEdgePathItem(
                history_path,
                QPen(QColor("#64748b"), 1.8),
            )
            history_edge.set_base_z(2.0)
            scene.addItem(history_edge)
            for marker in markers:
                history_edge.add_companion(marker)
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
    states = analysis.state_sequence_for_transition_path(path)
    diagram_nodes: list[DiagramNode] = []
    diagram_edges: list[DiagramEdge] = []
    rank_hints: dict[str, int] = {}
    for index, state in enumerate(states):
        node_id = f"jump-state:{index}"
        rank_hints[node_id] = index
        color = (
            "#0e7490"
            if index == len(states) - 1 and progress >= len(path)
            else "#047857"
            if index <= progress
            else "#334155"
        )
        diagram_nodes.append(
            DiagramNode(
                node_id=node_id,
                title=state,
                group="jump path",
                width=285.0,
                height=86.0,
                priority=100 - index,
                fill=color,
                border="#dbeafe",
            )
        )

    for index, transition in enumerate(path):
        completed = index < progress
        active = index == progress and progress < len(path)
        base = "#f0abfc" if transition.kind == "adapter" else "#7dd3fc"
        color = "#22c55e" if completed else base if active else "#64748b"
        label = transition.label
        if transition.guard_label:
            label += " | " + transition.guard_label.replace("\n", "; ")
        diagram_edges.append(
            DiagramEdge(
                edge_id=f"jump-edge:{index}",
                source_id=f"jump-state:{index}",
                target_id=f"jump-state:{index + 1}",
                label=label,
                kind=transition.kind,
                color=color,
                dashed=transition.kind == "adapter",
                tooltip=(
                    f"{transition.kind}: {transition.label}\n"
                    f"Guard: {transition.guard_label or 'none'}\n"
                    f"Actions: {transition.action_label or 'control-state update'}"
                ),
            )
        )

    geometry = ReusableLayeredGraphLayout(
        minimum_lane_gap=20.0,
        node_gap=54.0,
        rank_gap=126.0,
        port_stub=18.0,
    ).layout(
        diagram_nodes,
        diagram_edges,
        initial_node_id="jump-state:0",
        rank_hints=rank_hints,
        offset=QPointF(42.0, 140.0),
    )
    node_by_id = {node.node_id: node for node in diagram_nodes}
    edge_by_id = {edge.edge_id: edge for edge in diagram_edges}
    for node_id, placement in geometry.placements.items():
        NodeRenderer.draw_card(scene, placement.rect, node_by_id[node_id])
    label_positions = assign_label_positions(
        geometry.routes.values(),
        [placement.rect for placement in geometry.placements.values()],
        label_width=140.0,
        label_height=25.0,
    )
    for edge_id, route in geometry.routes.items():
        EdgeRenderer.draw(
            scene,
            route,
            edge_by_id[edge_id],
            label_position=label_positions.get(edge_id),
        )

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

def _rect_union_for_scene(rects: Iterable[QRectF], routes: Iterable[Any]) -> QRectF:
    values = [QRectF(rect) for rect in rects]
    for route in routes:
        if route.points:
            xs = [point.x() for point in route.points]
            ys = [point.y() for point in route.points]
            values.append(QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
    if not values:
        return QRectF()
    result = QRectF(values[0])
    for rect in values[1:]:
        result = result.united(rect)
    return result


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
) -> QRectF:
    """Render a protected legend panel and return its occupied bounds."""
    x = 24.0
    row = 0
    placed: list[tuple[float, float, str, QColor, str]] = []
    for label_text, color, kind in items:
        estimated = 44 + len(label_text) * 7
        if x + estimated > max_width and x > 24:
            row += 1
            x = 24.0
        current_y = y + row * 34
        placed.append((x, current_y, label_text, color, kind))
        x += estimated

    bounds = QRectF(14.0, y - 12.0, max_width + 4.0, (row + 1) * 34.0 + 18.0)
    panel = QGraphicsRectItem(bounds)
    panel.setPen(QPen(QColor("#334155"), 0.8))
    panel.setBrush(QBrush(QColor(15, 23, 42, 248)))
    panel.setZValue(40.0)
    panel.setData(0, "legend-panel")
    scene.addItem(panel)

    for x, current_y, label_text, color, kind in placed:
        if kind == "box":
            swatch = QGraphicsRectItem(QRectF(x, current_y, 20, 20))
            swatch.setPen(QPen(QColor("#cbd5e1"), 1.0))
            swatch.setBrush(QBrush(color))
            swatch.setZValue(41.0)
            scene.addItem(swatch)
        else:
            path = QPainterPath(QPointF(x, current_y + 10))
            path.lineTo(x + 26, current_y + 10)
            pen = QPen(color, 2.2)
            if kind == "dash":
                pen.setStyle(Qt.PenStyle.DashLine)
            item = _add_path(scene, path, pen)
            item.setZValue(41.0)
        label = QGraphicsSimpleTextItem(label_text)
        label.setBrush(QBrush(TEXT_COLOR))
        label.setPos(x + 32, current_y - 2)
        label.setZValue(42.0)
        scene.addItem(label)
    return bounds


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
    node.setZValue(5.0)
    scene.addItem(node)
    text = QGraphicsTextItem(_wrap_label(label, wrap_width))
    text.setDefaultTextColor(TEXT_COLOR)
    text.setTextWidth(rect.width() - 18)
    text.setPos(rect.x() + 9, rect.y() + 8)
    text.setZValue(6.0)
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


def _compact_code_range(codes: list[str]) -> str:
    """Compact a visual edge bundle without hiding its individual transitions."""
    if not codes:
        return ""
    if len(codes) == 1:
        return codes[0]
    prefix = codes[0][0]
    numbers = sorted(_numeric_code(code) for code in codes)
    if all(b == a + 1 for a, b in zip(numbers, numbers[1:])):
        return f"{prefix}{numbers[0]}–{prefix}{numbers[-1]}"
    if len(codes) <= 3:
        return ", ".join(codes)
    return f"{codes[0]} +{len(codes) - 1}"


def _add_edge_label(
    scene: QGraphicsScene,
    text: str,
    position: QPointF,
    color: QColor,
    *,
    payload: dict[str, Any] | None = None,
) -> QGraphicsSimpleTextItem:
    label = QGraphicsSimpleTextItem(text)
    font = QFont("Sans Serif", 8)
    font.setBold(True)
    label.setFont(font)
    label.setBrush(QBrush(color))
    label.setPos(position)
    bounds = label.boundingRect().adjusted(-7, -4, 7, 4)
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
    background.setZValue(8.0)
    label.setZValue(9.0)
    background.setData(2, "edge-label-panel")
    if payload is not None:
        label.setData(0, payload)
        background.setData(0, payload)
        label.setData(1, "interactive-edge-label")
        background.setData(1, "interactive-edge-label")
        label.setCursor(Qt.CursorShape.PointingHandCursor)
        background.setCursor(Qt.CursorShape.PointingHandCursor)
    scene.addItem(background)
    scene.addItem(label)
    label._edge_label_background = background
    return label


def _draw_arrow(scene: QGraphicsScene, start: QPointF, end: QPointF, color: QColor) -> list[Any]:
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
    first = scene.addLine(end.x(), end.y(), p1.x(), p1.y(), QPen(color, 2.0))
    second = scene.addLine(end.x(), end.y(), p2.x(), p2.y(), QPen(color, 2.0))
    first.setZValue(4.0)
    second.setZValue(4.0)
    return [first, second]


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
