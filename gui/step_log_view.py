"""Agent selection, event timeline, buffers, and declarative memory."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSplitter,
    QTabWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from gui.agent_tree import AgentTreeSelection, AgentTreeWidget
from gui.buffer_view import BufferInspectorTab
from gui.declarative_memory_view import DeclarativeMemoryInspectorTab
from gui.timeline_model import TimelineTableModel


class StepLogView(QFrame):
    """Agent navigator exposing events, all runtime buffers, and memory."""

    FIXED_TAB_COUNT = 2

    def __init__(self, tracer: Any, simulation: Any, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.tracer = tracer
        self.simulation = simulation
        self._buffer_tabs: dict[str, BufferInspectorTab] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        heading = QHBoxLayout()
        title = QLabel("Agent Inspector")
        title.setObjectName("sectionTitle")
        self.summary_label = QLabel("Simulation not started")
        self.summary_label.setObjectName("muted")
        self.summary_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(self.summary_label)
        outer.addLayout(heading)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        agent_panel = QFrame(splitter)
        agent_layout = QVBoxLayout(agent_panel)
        agent_layout.setContentsMargins(0, 0, 4, 0)
        agent_layout.setSpacing(7)
        agent_label = QLabel("Agents by type")
        agent_label.setObjectName("muted")
        self.agent_tree = AgentTreeWidget(agent_panel)
        self.agent_tree.selection_changed.connect(self._select_agent)
        agent_layout.addWidget(agent_label)
        agent_layout.addWidget(self.agent_tree, 1)

        details_panel = QFrame(splitter)
        details_layout = QVBoxLayout(details_panel)
        details_layout.setContentsMargins(4, 0, 0, 0)
        details_layout.setSpacing(7)
        self.details_label = QLabel(
            "Expand an agent type and select a runtime agent"
        )
        self.details_label.setObjectName("muted")
        details_layout.addWidget(self.details_label)

        self.tabs = QTabWidget(details_panel)
        self.timeline_page = self._build_timeline_page()
        self.memory_page = DeclarativeMemoryInspectorTab(self.tabs)
        self.tabs.addTab(self.timeline_page, "Step Timeline")
        self.tabs.addTab(self.memory_page, "Declarative Memory")
        details_layout.addWidget(self.tabs, 1)

        splitter.addWidget(agent_panel)
        splitter.addWidget(details_panel)
        splitter.setSizes([220, 1000])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)
        self.refresh()

    def _build_timeline_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        self.model = TimelineTableModel(self)
        self.table = QTableView(page)
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectItems)
        self.table.setCornerButtonEnabled(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.table.horizontalHeader().setDefaultSectionSize(185)
        self.table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.verticalHeader().setMinimumSectionSize(52)
        self.table.verticalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self.table, 1)
        return page

    @property
    def current_agent(self) -> str | None:
        selection = self.agent_tree.current_selection()
        return selection.runtime_name if selection is not None else None

    def refresh(self) -> None:
        selected = self.current_agent
        runtime_agents = list(getattr(self.simulation, "agent_list", []))
        self.agent_tree.set_agents(
            runtime_agents,
            preserve_runtime_name=selected,
        )
        selected = self.current_agent
        self.model.sync_records(
            list(getattr(self.tracer, "records", [])), selected
        )
        self._sync_buffer_tabs(selected)
        self.memory_page.update_agent(
            self.simulation.get_agent_by_name(selected) if selected else None
        )
        self._update_summary(selected)

    def _select_agent(
        self, selection: AgentTreeSelection | None
    ) -> None:
        selected = selection.runtime_name if selection is not None else None
        self.model.replace_records(
            list(getattr(self.tracer, "records", [])), selected
        )
        self._sync_buffer_tabs(selected, rebuild=True)
        self.memory_page.update_agent(
            self.simulation.get_agent_by_name(selected) if selected else None
        )
        self._update_summary(selected)

    def _sync_buffer_tabs(
        self, selected: str | None, rebuild: bool = False
    ) -> None:
        recorder = getattr(self.simulation, "buffer_history", None)
        names = (
            recorder.buffer_names(selected)
            if recorder is not None and selected
            else []
        )
        if rebuild or tuple(names) != tuple(self._buffer_tabs):
            while self.tabs.count() > self.FIXED_TAB_COUNT:
                widget = self.tabs.widget(self.FIXED_TAB_COUNT)
                self.tabs.removeTab(self.FIXED_TAB_COUNT)
                widget.deleteLater()
            self._buffer_tabs.clear()
            for buffer_name in names:
                tab = BufferInspectorTab(buffer_name, self.tabs)
                self._buffer_tabs[buffer_name] = tab
                self.tabs.addTab(tab, buffer_name)

        if recorder is None or selected is None:
            return
        for buffer_name, tab in self._buffer_tabs.items():
            tab.update_data(
                recorder.latest(selected, buffer_name),
                recorder.history(selected, buffer_name),
            )

    def _update_summary(self, selected: str | None) -> None:
        if selected:
            buffer_count = len(self._buffer_tabs)
            self.summary_label.setText(
                f"{selected} · {buffer_count} dynamically detected buffers"
            )
            self.details_label.setText(
                f"{selected}: timeline, declarative memory, and buffer histories"
            )
        elif getattr(self.simulation, "agent_list", []):
            self.summary_label.setText("Select a runtime agent")
            self.details_label.setText(
                "Expand an agent type and select a runtime agent"
            )
        else:
            self.summary_label.setText("Simulation not started")
            self.details_label.setText("Timeline, memory, and buffers")
