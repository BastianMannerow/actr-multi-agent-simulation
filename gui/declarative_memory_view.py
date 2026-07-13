"""Declarative-memory graph tab for the running agent inspector."""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.analysis_graphs import (
    ZoomableGraphicsView,
    build_declarative_memory_scene,
)
from simulation.inspection.declarative_memory import DeclarativeMemoryInspector


class DeclarativeMemoryInspectorTab(QWidget):
    """Visualize the current contents of every pyactr declarative memory."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._agent_name: str | None = None
        self._signature: tuple[Any, ...] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        toolbar = QHBoxLayout()
        self.info_label = QLabel("No runtime agent selected")
        self.info_label.setObjectName("muted")
        fit_button = QPushButton("Fit View")
        png_button = QPushButton("Export PNG")
        svg_button = QPushButton("Export SVG")
        toolbar.addWidget(self.info_label)
        toolbar.addStretch(1)
        toolbar.addWidget(fit_button)
        toolbar.addWidget(png_button)
        toolbar.addWidget(svg_button)
        layout.addLayout(toolbar)

        self.graph = ZoomableGraphicsView(self)
        layout.addWidget(self.graph, 1)

        fit_button.clicked.connect(self.graph.reset_zoom)
        png_button.clicked.connect(lambda: self.graph.export_dialog("png"))
        svg_button.clicked.connect(lambda: self.graph.export_dialog("svg"))

    def update_agent(self, agent: Any | None) -> None:
        if agent is None:
            self._agent_name = None
            self._signature = None
            self.info_label.setText("No runtime agent selected")
            self.graph.setScene(
                build_declarative_memory_scene(
                    DeclarativeMemoryInspector.inspect_agent(None),
                    title="Declarative Memory",
                )
            )
            return
        snapshot = DeclarativeMemoryInspector.inspect_agent(agent)
        signature = (
            tuple(snapshot.memories),
            tuple(
                (
                    chunk.chunk_id,
                    chunk.label,
                    tuple(sorted(chunk.slots.items())),
                    tuple(chunk.traces),
                    chunk.activation,
                )
                for chunk in snapshot.chunks
            ),
        )
        agent_name = str(getattr(agent, "name", "Agent"))
        if signature == self._signature and agent_name == self._agent_name:
            return
        reset_zoom = agent_name != self._agent_name
        self._signature = signature
        self._agent_name = agent_name
        self.info_label.setText(
            f"{agent_name} · {len(snapshot.memories)} memories · "
            f"{len(snapshot.chunks)} chunks"
        )
        self.graph.setScene(
            build_declarative_memory_scene(
                snapshot,
                title=f"Current Declarative Memory — {agent_name}",
            )
        )
        if reset_zoom:
            self.graph.reset_zoom()
