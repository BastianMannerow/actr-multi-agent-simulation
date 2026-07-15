"""Single source of truth for demo-matrix and legend symbols."""

from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class EnvironmentSymbol(str, Enum):
    WALL = "wall"
    CHECKPOINT = "checkpoint"
    GOAL = "goal"
    ACTR_AGENT = "actr_agent"
    HUMAN_AGENT = "human_agent"
    UNKNOWN_TERRAIN = "unknown_terrain"


ACTR_AGENT_COLOR = QColor("#2563eb")
HUMAN_AGENT_COLOR = QColor("#f59e0b")


def symbol_for_terrain(entity) -> EnvironmentSymbol:
    if bool(getattr(entity, "is_target", False)):
        return EnvironmentSymbol.GOAL
    name = type(entity).__name__
    if name == "Wall":
        return EnvironmentSymbol.WALL
    if name == "Checkpoint":
        return EnvironmentSymbol.CHECKPOINT
    return EnvironmentSymbol.UNKNOWN_TERRAIN


def symbol_for_agent(entity) -> EnvironmentSymbol:
    return (
        EnvironmentSymbol.HUMAN_AGENT
        if bool(getattr(entity, "is_human_controlled", False))
        else EnvironmentSymbol.ACTR_AGENT
    )


def draw_environment_symbol(
    painter: QPainter,
    rect: QRectF,
    symbol: EnvironmentSymbol,
    *,
    label: str | None = None,
) -> None:
    """Draw one symbol identically in the canvas and its legend."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    text = label

    if symbol == EnvironmentSymbol.GOAL:
        painter.setBrush(QColor("#16a34a"))
        painter.setPen(QPen(QColor("#bbf7d0"), max(1.0, rect.width() * 0.055)))
        center = rect.center()
        radius = min(rect.width(), rect.height()) * 0.34
        painter.drawEllipse(center, radius, radius)
        painter.setBrush(QColor("#dcfce7"))
        painter.drawEllipse(center, radius * 0.45, radius * 0.45)
        text = text or "G"
    elif symbol == EnvironmentSymbol.WALL:
        inner = rect.adjusted(2, 2, -2, -2)
        painter.fillRect(inner, QColor("#334155"))
        painter.setPen(QPen(QColor("#94a3b8"), max(1.0, rect.width() * 0.045)))
        inset = max(2.0, rect.width() * 0.12)
        painter.drawLine(
            inner.topLeft() + QPointF(inset, inset),
            inner.bottomRight() - QPointF(inset, inset),
        )
        painter.drawLine(
            inner.topRight() + QPointF(-inset, inset),
            inner.bottomLeft() + QPointF(inset, -inset),
        )
        text = text or "X"
    elif symbol == EnvironmentSymbol.CHECKPOINT:
        inner = rect.adjusted(4, 4, -4, -4)
        painter.setBrush(QColor("#7c3aed"))
        painter.setPen(QPen(QColor("#ddd6fe"), max(1.0, rect.width() * 0.05)))
        painter.drawRoundedRect(inner, 3.0, 3.0)
        text = text or "C"
    elif symbol in {EnvironmentSymbol.ACTR_AGENT, EnvironmentSymbol.HUMAN_AGENT}:
        color = ACTR_AGENT_COLOR if symbol == EnvironmentSymbol.ACTR_AGENT else HUMAN_AGENT_COLOR
        painter.setBrush(color)
        painter.setPen(QPen(QColor("#f1f5f9"), max(1.0, rect.width() * 0.045)))
        painter.drawEllipse(rect)
        text = text or ("A" if symbol == EnvironmentSymbol.ACTR_AGENT else "H")
    else:
        painter.fillRect(rect.adjusted(3, 3, -3, -3), QColor("#334155"))
        text = text or "?"

    if text and rect.width() >= 18 and rect.height() >= 18:
        font = QFont("Sans Serif")
        font.setBold(True)
        font.setPointSizeF(max(5.5, min(10.0, rect.width() * 0.24)))
        painter.setFont(font)
        painter.setPen(
            QColor("#081018")
            if symbol in {
                EnvironmentSymbol.ACTR_AGENT,
                EnvironmentSymbol.HUMAN_AGENT,
                EnvironmentSymbol.GOAL,
            }
            else QColor("#f8fafc")
        )
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
    painter.restore()


class EnvironmentLegendMarker(QWidget):
    """Legend marker rendered by the exact same painter as the environment."""

    def __init__(self, symbol: EnvironmentSymbol, parent=None) -> None:
        super().__init__(parent)
        self.symbol = symbol
        self.setFixedSize(28, 28)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        draw_environment_symbol(
            painter,
            QRectF(2.0, 2.0, self.width() - 4.0, self.height() - 4.0),
            self.symbol,
        )
