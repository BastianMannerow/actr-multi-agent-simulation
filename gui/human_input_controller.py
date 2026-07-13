"""Keyboard mapping for the optional human-controlled grid agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QPlainTextEdit,
    QSlider,
    QTabBar,
    QTextEdit,
)


class HumanInputController(QObject):
    """Translate WASD and arrow keys into human-agent movement commands."""

    _DIRECTIONS = {
        Qt.Key.Key_W: "up",
        Qt.Key.Key_Up: "up",
        Qt.Key.Key_A: "left",
        Qt.Key.Key_Left: "left",
        Qt.Key.Key_S: "down",
        Qt.Key.Key_Down: "down",
        Qt.Key.Key_D: "right",
        Qt.Key.Key_Right: "right",
    }

    _INTERACTIVE_CONTROLS = (
        QLineEdit,
        QPlainTextEdit,
        QTextEdit,
        QComboBox,
        QAbstractSpinBox,
        QAbstractButton,
        QAbstractItemView,
        QSlider,
        QTabBar,
    )

    def __init__(
        self,
        simulation: Any,
        *,
        enabled_predicate: Callable[[], bool] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.simulation = simulation
        self.enabled_predicate = enabled_predicate

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        del watched
        if event.type() != QEvent.Type.KeyPress:
            return False
        if self.enabled_predicate is not None and not self.enabled_predicate():
            return False
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, self._INTERACTIVE_CONTROLS):
            return False
        direction = self._DIRECTIONS.get(event.key())
        if direction is None:
            return False
        if not getattr(self.simulation, "initialized", False) or getattr(
            self.simulation, "human_agent", None
        ) is None:
            return False
        self.simulation.move_human_agent(direction)
        return True
