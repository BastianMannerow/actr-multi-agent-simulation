"""Segmented Step/Automatic execution-mode toggle."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QPushButton


class ExecutionModeToggle(QFrame):
    """Two-sided toggle where the inactive side is visibly subdued."""

    mode_changed = pyqtSignal(str)

    def __init__(self, mode: str = "single", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("modeToggle")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.step_button = QPushButton("Step", self)
        self.auto_button = QPushButton("Automatic", self)
        for button, value in (
            (self.step_button, "single"),
            (self.auto_button, "automatic"),
        ):
            button.setCheckable(True)
            button.setProperty("modeValue", value)
            button.setObjectName("modeSegment")
            self.group.addButton(button)
            layout.addWidget(button)
        self.group.buttonClicked.connect(self._clicked)
        self.set_mode(mode, emit=False)

    def mode(self) -> str:
        checked = self.group.checkedButton()
        return str(checked.property("modeValue")) if checked is not None else "single"

    def set_mode(self, mode: str, *, emit: bool = False) -> None:
        target = self.auto_button if mode == "automatic" else self.step_button
        changed = not target.isChecked()
        target.setChecked(True)
        if emit and changed:
            self.mode_changed.emit(self.mode())

    def _clicked(self, _button: QPushButton) -> None:
        self.mode_changed.emit(self.mode())
