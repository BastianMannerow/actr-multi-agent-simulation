"""Panel-level presentation of the simulation environment."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

from gui.environment_canvas import GridCanvas


class EnvironmentView(QFrame):
    """Panel containing grid metadata and the reusable renderer."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.environment: Any | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        heading = QHBoxLayout()
        title = QLabel("Environment")
        title.setObjectName("sectionTitle")
        self.info_label = QLabel("Not initialized")
        self.info_label.setObjectName("muted")
        self.info_label.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignVCenter
        )
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(self.info_label)
        layout.addLayout(heading)

        self.canvas = GridCanvas(self)
        layout.addWidget(self.canvas, 1)

    def set_environment(self, environment: Any) -> None:
        self.environment = environment
        self.canvas.set_environment(environment)
        self.refresh()

    def refresh(self) -> None:
        matrix = getattr(self.environment, "level_matrix", None)
        if matrix and matrix[0]:
            agent_count = sum(
                1
                for row in matrix
                for cell in row
                for obj in cell
                if getattr(obj, "name", None)
            )
            self.info_label.setText(
                f"{len(matrix[0])} × {len(matrix)} · "
                f"{agent_count} agent"
                f"{'s' if agent_count != 1 else ''}"
            )
        else:
            self.info_label.setText("Not initialized")
        self.canvas.update()
