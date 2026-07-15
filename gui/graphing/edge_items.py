"""Interactive graphics items shared by every routed graph view.

The visible stroke remains narrow and precise. Hover acquisition is performed by
the owning :class:`QGraphicsView` in viewport coordinates. This keeps the target
width constant on screen at every zoom level and avoids Qt's path-shape ambiguity
at orthogonal joins, while click hit-testing and background dragging still use the
narrow painted item.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QPainterPath, QPainterPathStroker, QPen
from PyQt6.QtWidgets import QGraphicsItem, QGraphicsPathItem


EDGE_HIGHLIGHT_COLOR = QColor("#facc15")
EDGE_HIGHLIGHT_Z = 32.0


@dataclass(slots=True)
class _CompanionStyle:
    item: QGraphicsItem
    pen: QPen | None
    brush: QBrush | None
    z_value: float


class HoverableEdgePathItem(QGraphicsPathItem):
    """A routed edge that highlights itself and its visual companions.

    Arrowheads and edge labels can be registered as companions. They are styled
    together with the path while the pointer is close to the rendered edge.
    The view performs that proximity test in screen pixels. The item deliberately
    keeps its normal Qt shape narrow, preserving reliable graph panning on empty
    canvas areas.
    """

    def __init__(
        self,
        path: QPainterPath,
        pen: QPen,
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(path)
        self._base_pen = QPen(pen)
        self._base_z = 2.0
        self._companions: list[_CompanionStyle] = []
        self.setPen(QPen(pen))
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        # Native QGraphicsItem hover detection is intentionally disabled. Its
        # hit target is expressed in scene units, so it becomes extremely small
        # after fit-to-view on large graphs and behaves inconsistently around
        # right-angle joins. ZoomableGraphicsView performs one deterministic,
        # screen-space proximity test instead.
        self.setAcceptHoverEvents(False)
        self.setData(1, "interactive-edge")
        if payload is not None:
            self.setData(0, payload)
            self.setCursor(Qt.CursorShape.PointingHandCursor)

    def shape(self) -> QPainterPath:
        """Return only the stroked centreline, never the path's fill area.

        ``QGraphicsPathItem.shape()`` includes the fill geometry of an open
        painter path. For an orthogonal edge this implicitly closes the path
        between its endpoints and can create a large invisible triangular hit
        area. That was the source of false hover and click matches far away from
        the visible line.
        """
        # Keep the interaction geometry stable while the visual pen widens
        # during highlighting. Otherwise an already-hovered edge would gain a
        # different click shape until the pointer leaves it.
        pen = self._base_pen
        stroker = QPainterPathStroker()
        stroker.setWidth(max(0.75, pen.widthF()))
        stroker.setCapStyle(pen.capStyle())
        stroker.setJoinStyle(pen.joinStyle())
        stroker.setMiterLimit(pen.miterLimit())
        return stroker.createStroke(self.path())

    def set_base_z(self, value: float) -> None:
        self._base_z = float(value)
        self.setZValue(self._base_z)

    def add_companion(self, item: QGraphicsItem) -> None:
        # A companion can be visually separated from the centreline (for
        # example a label). The view can still resolve it back to this edge.
        setattr(item, "_hover_edge_owner", self)
        pen = QPen(item.pen()) if hasattr(item, "pen") else None
        brush = QBrush(item.brush()) if hasattr(item, "brush") else None
        self._companions.append(
            _CompanionStyle(
                item=item,
                pen=pen,
                brush=brush,
                z_value=item.zValue(),
            )
        )

    @staticmethod
    def _highlight_pen(base: QPen) -> QPen:
        pen = QPen(base)
        pen.setColor(EDGE_HIGHLIGHT_COLOR)
        pen.setWidthF(max(base.widthF() + 1.4, 3.6))
        return pen

    def set_highlighted(self, highlighted: bool) -> None:
        """Apply or restore the shared edge-hover presentation."""
        if highlighted:
            self.setPen(self._highlight_pen(self._base_pen))
            self.setZValue(EDGE_HIGHLIGHT_Z)
            for companion in self._companions:
                item = companion.item
                if companion.pen is not None and hasattr(item, "setPen"):
                    item.setPen(self._highlight_pen(companion.pen))
                if companion.brush is not None and hasattr(item, "setBrush"):
                    brush = QBrush(companion.brush)
                    # Text and filled arrowheads should become yellow. Background
                    # panels keep their dark fill and only receive a yellow border.
                    if item.data(2) != "edge-label-panel":
                        brush.setColor(EDGE_HIGHLIGHT_COLOR)
                        item.setBrush(brush)
                item.setZValue(max(EDGE_HIGHLIGHT_Z + 1.0, companion.z_value))
            return

        self.setPen(QPen(self._base_pen))
        self.setZValue(self._base_z)
        for companion in self._companions:
            item = companion.item
            if companion.pen is not None and hasattr(item, "setPen"):
                item.setPen(QPen(companion.pen))
            if companion.brush is not None and hasattr(item, "setBrush"):
                item.setBrush(QBrush(companion.brush))
            item.setZValue(companion.z_value)

    def hoverEnterEvent(self, event) -> None:  # noqa: N802
        # Kept for compatibility with scenes embedding this item in another
        # view implementation. The standard application view does not enable
        # native hover events.
        self.set_highlighted(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802
        self.set_highlighted(False)
        super().hoverLeaveEvent(event)
