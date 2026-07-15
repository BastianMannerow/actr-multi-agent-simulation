"""Low-level, cached rendering of the simulation environment."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QToolTip, QWidget

from gui.environment_symbols import draw_environment_symbol, symbol_for_agent, symbol_for_terrain


class GridCanvas(QWidget):
    """Paint static terrain once and overlay only dynamic occupants."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.environment: Any | None = None
        self._grid_rect = QRectF()
        self._cell_size = 0.0
        self._static_cache: QPixmap | None = None
        self._static_cache_key: tuple[Any, ...] | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(560, 560)

    def set_environment(self, environment: Any) -> None:
        self.environment = environment
        self.invalidate_static_cache()
        consume = getattr(environment, "consume_render_changes", None)
        if callable(consume):
            consume()
        self.update()

    def refresh_from_environment(self) -> None:
        """Repaint only cells changed since the previous GUI refresh."""
        consume = getattr(self.environment, "consume_render_changes", None)
        if not callable(consume):
            self.update()
            return
        dirty = consume()
        if dirty is None:
            self.invalidate_static_cache()
            self.update()
            return
        if not dirty:
            return
        if self._cell_size <= 0 or self._grid_rect.isNull():
            self.update()
            return
        padding = 3
        for row, column in dirty:
            rect = QRectF(
                self._grid_rect.left() + column * self._cell_size,
                self._grid_rect.top() + row * self._cell_size,
                self._cell_size,
                self._cell_size,
            ).adjusted(-padding, -padding, padding, padding)
            self.update(rect.toAlignedRect())

    def invalidate_static_cache(self) -> None:
        self._static_cache = None
        self._static_cache_key = None

    def resizeEvent(self, event) -> None:  # noqa: N802
        self.invalidate_static_cache()
        super().resizeEvent(event)

    def _geometry(self, rows: int, columns: int) -> tuple[float, float, float]:
        margin = 18.0
        available_width = max(1.0, self.width() - 2 * margin)
        available_height = max(1.0, self.height() - 2 * margin)
        cell_size = min(available_width / columns, available_height / rows)
        grid_width = cell_size * columns
        grid_height = cell_size * rows
        origin_x = (self.width() - grid_width) / 2.0
        origin_y = (self.height() - grid_height) / 2.0
        self._cell_size = cell_size
        self._grid_rect = QRectF(origin_x, origin_y, grid_width, grid_height)
        return origin_x, origin_y, cell_size

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        matrix = getattr(self.environment, "level_matrix", None)
        if not matrix or not matrix[0]:
            painter.fillRect(self.rect(), QColor("#0b0f16"))
            painter.setPen(QColor("#8d98aa"))
            painter.setFont(QFont(self.font().family(), 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Environment is initializing…")
            return

        rows, columns = len(matrix), len(matrix[0])
        origin_x, origin_y, cell_size = self._geometry(rows, columns)
        cache_key = (
            self.width(), self.height(), rows, columns,
            int(getattr(self.environment, "static_revision", 0)),
        )
        if self._static_cache is None or self._static_cache_key != cache_key:
            self._static_cache = self._build_static_cache(
                matrix, rows, columns, origin_x, origin_y, cell_size
            )
            self._static_cache_key = cache_key
        painter.drawPixmap(0, 0, self._static_cache)

        occupants_by_cell: dict[tuple[int, int], list[Any]] = {}
        indexed = getattr(self.environment, "positioned_agents", None)
        if callable(indexed):
            for occupant, position in indexed():
                occupants_by_cell.setdefault(position, []).append(occupant)
        else:
            for row in range(rows):
                for column in range(columns):
                    occupants = [
                        obj for obj in matrix[row][column]
                        if getattr(obj, "name", None)
                    ]
                    if occupants:
                        occupants_by_cell[(row, column)] = occupants

        for (row, column), occupants in occupants_by_cell.items():
            rect = QRectF(
                origin_x + column * cell_size,
                origin_y + row * cell_size,
                cell_size,
                cell_size,
            )
            self._draw_occupants(painter, rect, occupants)

    def _build_static_cache(
        self,
        matrix: list[list[list[Any]]],
        rows: int,
        columns: int,
        origin_x: float,
        origin_y: float,
        cell_size: float,
    ) -> QPixmap:
        pixmap = QPixmap(self.size())
        pixmap.fill(QColor("#0b0f16"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#2a3342"), 1.0))
        for row in range(rows):
            for column in range(columns):
                rect = QRectF(
                    origin_x + column * cell_size,
                    origin_y + row * cell_size,
                    cell_size,
                    cell_size,
                )
                painter.fillRect(
                    rect,
                    QColor("#141a24") if (row + column) % 2 == 0 else QColor("#111720"),
                )
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)
                terrain = [
                    obj for obj in matrix[row][column]
                    if not getattr(obj, "name", None) and getattr(obj, "symbol", None)
                ]
                self._draw_terrain(painter, rect, terrain)
        painter.setPen(QPen(QColor("#46536a"), 1.4))
        painter.drawRect(self._grid_rect)
        painter.end()
        return pixmap

    def _draw_terrain(self, painter: QPainter, rect: QRectF, terrain: list[Any]) -> None:
        if not terrain:
            return
        item = sorted(terrain, key=lambda value: (
            0 if bool(getattr(value, "is_target", False)) else
            1 if bool(getattr(value, "blocks_movement", False)) else
            2 if type(value).__name__ == "Checkpoint" else 3
        ))[0]
        draw_environment_symbol(
            painter,
            rect,
            symbol_for_terrain(item),
            label=str(getattr(item, "symbol", "?"))[:2].upper(),
        )

    def _draw_occupants(self, painter: QPainter, rect: QRectF, occupants: list[Any]) -> None:
        painter.save()
        visible = occupants[:4]
        radius = max(
            4.0,
            min(rect.width(), rect.height()) * (0.29 if len(visible) == 1 else 0.19),
        )
        offsets = [(0.5, 0.5), (0.33, 0.33), (0.67, 0.33), (0.50, 0.68)]
        for index, occupant in enumerate(visible):
            name = str(getattr(occupant, "name", "A"))
            ox, oy = offsets[index]
            center = QPointF(
                rect.left() + rect.width() * ox,
                rect.top() + rect.height() * oy,
            )
            label = name if rect.width() >= 75 else name[:3]
            draw_environment_symbol(
                painter,
                QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2),
                symbol_for_agent(occupant),
                label=label,
            )
        if len(occupants) > len(visible):
            painter.setPen(QColor("#d9e0ea"))
            painter.drawText(
                rect.adjusted(3, 3, -3, -3),
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
                f"+{len(occupants) - len(visible)}",
            )
        painter.restore()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        matrix = getattr(self.environment, "level_matrix", None)
        if not matrix or self._cell_size <= 0 or not self._grid_rect.contains(event.position()):
            QToolTip.hideText()
            return
        column = int((event.position().x() - self._grid_rect.left()) / self._cell_size)
        row = int((event.position().y() - self._grid_rect.top()) / self._cell_size)
        if 0 <= row < len(matrix) and 0 <= column < len(matrix[0]):
            names = [
                str(getattr(obj, "name")) for obj in matrix[row][column]
                if getattr(obj, "name", None)
            ]
            terrain = [
                str(getattr(obj, "display_name", type(obj).__name__))
                for obj in matrix[row][column] if not getattr(obj, "name", None)
            ]
            text = f"Row {row + 1}, column {column + 1}"
            details = names + terrain
            if details:
                text += "\n" + "\n".join(details)
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
