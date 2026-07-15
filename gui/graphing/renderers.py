"""Reusable QGraphics renderers for cards, UML-like chunks and routed edges."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsTextItem,
)

from gui.graph_layout import RoutedEdge
from gui.graphing.edge_items import HoverableEdgePathItem
from gui.graphing.models import DiagramEdge, DiagramNode


SCENE_BACKGROUND = QColor("#0f172a")
TEXT_COLOR = QColor("#f8fafc")
MUTED_TEXT = QColor("#cbd5e1")


@dataclass(frozen=True, slots=True)
class EdgeStyle:
    color: QColor
    dashed: bool = False
    width: float = 2.2


class SceneChrome:
    @staticmethod
    def new_scene() -> QGraphicsScene:
        scene = QGraphicsScene()
        scene.setBackgroundBrush(QBrush(SCENE_BACKGROUND))
        return scene

    @staticmethod
    def add_title(scene: QGraphicsScene, text: str) -> None:
        item = QGraphicsSimpleTextItem(text)
        font = QFont("Sans Serif", 11)
        font.setBold(True)
        item.setFont(font)
        item.setBrush(QBrush(TEXT_COLOR))
        item.setPos(20.0, 12.0)
        scene.addItem(item)

    @staticmethod
    def add_message(scene: QGraphicsScene, text: str, *, y: float = 82.0) -> None:
        item = QGraphicsTextItem(text)
        item.setDefaultTextColor(MUTED_TEXT)
        item.setTextWidth(900.0)
        item.setPos(24.0, y)
        scene.addItem(item)

    @staticmethod
    def add_legend(
        scene: QGraphicsScene,
        entries: list[tuple[str, QColor, str]],
        *,
        y: float = 52.0,
        max_width: float = 1500.0,
    ) -> QRectF:
        x = 24.0
        row = 0
        placed = []
        for label, color, kind in entries:
            width = 48.0 + len(label) * 7.0
            if x + width > max_width and x > 24.0:
                row += 1
                x = 24.0
            placed.append((x, y + row * 34.0, label, color, kind))
            x += width
        bounds = QRectF(14.0, y - 12.0, max_width + 4.0, (row + 1) * 34.0 + 18.0)
        panel = QGraphicsRectItem(bounds)
        panel.setPen(QPen(QColor("#334155"), 0.8))
        panel.setBrush(QBrush(QColor(15, 23, 42, 248)))
        panel.setZValue(40.0)
        scene.addItem(panel)
        for x, item_y, label, color, kind in placed:
            if kind == "box":
                swatch = QGraphicsRectItem(QRectF(x, item_y, 20.0, 20.0))
                swatch.setPen(QPen(QColor("#cbd5e1"), 1.0))
                swatch.setBrush(QBrush(color))
                swatch.setZValue(41.0)
                scene.addItem(swatch)
            else:
                path = QPainterPath(QPointF(x, item_y + 10.0))
                path.lineTo(x + 26.0, item_y + 10.0)
                pen = QPen(color, 2.2)
                if kind == "dash":
                    pen.setStyle(Qt.PenStyle.DashLine)
                line = QGraphicsPathItem(path)
                line.setPen(pen)
                line.setZValue(41.0)
                scene.addItem(line)
            text = QGraphicsSimpleTextItem(label)
            text.setBrush(QBrush(TEXT_COLOR))
            text.setPos(x + 32.0, item_y - 2.0)
            text.setZValue(42.0)
            scene.addItem(text)
        return bounds


class NodeRenderer:
    @staticmethod
    def draw_card(scene: QGraphicsScene, rect: QRectF, node: DiagramNode) -> None:
        box = QGraphicsRectItem(rect)
        box.setPen(QPen(QColor(node.border), 1.6))
        box.setBrush(QBrush(QColor(node.fill)))
        box.setZValue(10.0)
        if node.tooltip:
            box.setToolTip(node.tooltip)
        scene.addItem(box)
        text = QGraphicsTextItem(node.title)
        text.setDefaultTextColor(TEXT_COLOR)
        font = QFont("Sans Serif", 9)
        font.setBold(True)
        text.setFont(font)
        text.setTextWidth(max(40.0, rect.width() - 24.0))
        text.setPos(rect.left() + 12.0, rect.top() + 10.0)
        text.setZValue(11.0)
        if node.tooltip:
            text.setToolTip(node.tooltip)
        scene.addItem(text)

    @staticmethod
    def draw_uml_chunk(scene: QGraphicsScene, rect: QRectF, node: DiagramNode) -> None:
        box = QGraphicsRectItem(rect)
        box.setPen(QPen(QColor(node.border), 1.6))
        box.setBrush(QBrush(QColor(node.fill)))
        box.setZValue(10.0)
        if node.tooltip:
            box.setToolTip(node.tooltip)
        scene.addItem(box)

        stereotype_height = 24.0
        header_height = 56.0
        first_line_y = rect.top() + stereotype_height
        second_line_y = rect.top() + stereotype_height + header_height
        for y in (first_line_y, second_line_y):
            path = QPainterPath(QPointF(rect.left(), y))
            path.lineTo(rect.right(), y)
            item = QGraphicsPathItem(path)
            item.setPen(QPen(QColor(node.border), 0.9))
            item.setZValue(11.0)
            scene.addItem(item)

        stereotype = QGraphicsSimpleTextItem(node.stereotype or "«chunk»")
        stereotype.setBrush(QBrush(QColor("#cbd5e1")))
        stereotype.setFont(QFont("Sans Serif", 8))
        stereotype.setPos(rect.left() + 10.0, rect.top() + 3.0)
        stereotype.setZValue(12.0)
        scene.addItem(stereotype)

        title = QGraphicsTextItem(node.title)
        title.setDefaultTextColor(TEXT_COLOR)
        font = QFont("Sans Serif", 9)
        font.setBold(True)
        title.setFont(font)
        title.setTextWidth(rect.width() - 20.0)
        title.setPos(rect.left() + 10.0, first_line_y + 3.0)
        title.setZValue(12.0)
        scene.addItem(title)

        y = second_line_y + 6.0
        attr_font = QFont("Monospace", 8)
        for name, value in node.attributes:
            attr = QGraphicsTextItem(f"{name} : {value}")
            attr.setDefaultTextColor(QColor("#e2e8f0"))
            attr.setFont(attr_font)
            attr.setTextWidth(rect.width() - 20.0)
            attr.setPos(rect.left() + 10.0, y)
            attr.setZValue(12.0)
            scene.addItem(attr)
            y += 20.0
        if node.tooltip:
            title.setToolTip(node.tooltip)


class EdgeRenderer:
    @staticmethod
    def draw(
        scene: QGraphicsScene,
        route: RoutedEdge,
        edge: DiagramEdge,
        *,
        label_position: QPointF | None = None,
    ) -> HoverableEdgePathItem | None:
        points = route.points
        if len(points) < 2:
            return None
        color = QColor(edge.color)
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)

        halo = QGraphicsPathItem(path)
        halo_pen = QPen(SCENE_BACKGROUND, 7.5)
        halo_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        halo_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        halo.setPen(halo_pen)
        halo.setZValue(1.0)
        scene.addItem(halo)

        pen = QPen(color, 2.25)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        if edge.dashed:
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([7.0, 4.0])
        item = HoverableEdgePathItem(path, pen)
        item.set_base_z(2.0)
        if edge.tooltip:
            item.setToolTip(edge.tooltip)
        scene.addItem(item)

        arrow = EdgeRenderer._arrow(scene, points[-2], points[-1], color)
        item.add_companion(arrow)
        if edge.label:
            panel, label = EdgeRenderer._label(
                scene,
                edge.label,
                label_position or route.label_position,
                color,
                edge.tooltip,
            )
            item.add_companion(panel)
            item.add_companion(label)
        return item

    @staticmethod
    def _label(
        scene: QGraphicsScene,
        text: str,
        position: QPointF,
        color: QColor,
        tooltip: str,
    ) -> tuple[QGraphicsRectItem, QGraphicsSimpleTextItem]:
        label = QGraphicsSimpleTextItem(text)
        font = QFont("Sans Serif", 8)
        font.setBold(True)
        label.setFont(font)
        label.setBrush(QBrush(color))
        bounds = label.boundingRect().adjusted(-6.0, -3.0, 6.0, 3.0)
        panel = QGraphicsRectItem(QRectF(position.x(), position.y(), bounds.width(), bounds.height()))
        panel.setPen(QPen(QColor("#334155"), 0.8))
        panel.setBrush(QBrush(QColor(15, 23, 42, 238)))
        panel.setData(2, "edge-label-panel")
        panel.setZValue(4.0)
        scene.addItem(panel)
        label.setPos(position.x() + 6.0, position.y() + 3.0)
        label.setZValue(5.0)
        if tooltip:
            panel.setToolTip(tooltip)
            label.setToolTip(tooltip)
        scene.addItem(label)
        return panel, label

    @staticmethod
    def _arrow(scene: QGraphicsScene, start: QPointF, end: QPointF, color: QColor) -> QGraphicsPolygonItem:
        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        size = 10.0
        left = QPointF(
            end.x() - size * math.cos(angle - math.pi / 6.0),
            end.y() - size * math.sin(angle - math.pi / 6.0),
        )
        right = QPointF(
            end.x() - size * math.cos(angle + math.pi / 6.0),
            end.y() - size * math.sin(angle + math.pi / 6.0),
        )
        polygon = QGraphicsPolygonItem()
        from PyQt6.QtGui import QPolygonF
        polygon.setPolygon(QPolygonF([end, left, right]))
        polygon.setPen(QPen(color, 1.0))
        polygon.setBrush(QBrush(color))
        polygon.setZValue(3.0)
        scene.addItem(polygon)
        return polygon

