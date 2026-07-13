"""Static explainability views for discovered ACT-R agent types."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.agent_tree import AgentTreeSelection, AgentTreeWidget
from gui.analysis_graphs import (
    ZoomableGraphicsView,
    build_declarative_memory_scene,
    build_interaction_scene,
    build_state_transition_scene,
)
from simulation.discovery.agent_discovery import AgentDiscovery
from simulation.inspection.source_analysis import (
    AgentSourceAnalyzer,
    AgentStaticAnalysis,
)


class AgentAnalysisView(QFrame):
    """Visualize production flow, buffer access, and declarative memory."""

    def __init__(self, simulation: Any, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.simulation = simulation
        self.discovery = AgentDiscovery()
        self.analyzer = AgentSourceAnalyzer()
        self._analysis_cache: dict[str, AgentStaticAnalysis] = {}
        self._rendered_type: str | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        title = QLabel("Agent Analysis")
        title.setObjectName("sectionTitle")
        outer.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        self.agent_tree = AgentTreeWidget(splitter)
        self.agent_tree.setMinimumWidth(210)
        self.agent_tree.selection_changed.connect(self._selection_changed)

        details = QWidget(splitter)
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(8, 0, 0, 0)
        details_layout.setSpacing(8)

        self.tabs = QTabWidget(details)
        self.tabs.addTab(self._build_state_graph_tab(), "State Graph")
        self.tabs.addTab(self._build_interaction_tab(), "Buffer Interactions")
        self.tabs.addTab(
            self._build_declarative_memory_tab(),
            "Declarative Memory",
        )
        details_layout.addWidget(self.tabs, 1)

        splitter.addWidget(self.agent_tree)
        splitter.addWidget(details)
        splitter.setSizes([230, 980])
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)
        self.refresh()

    def analysis_for_agent(self, agent_name: str) -> AgentStaticAnalysis | None:
        agent = self.simulation.get_agent_by_name(agent_name)
        if agent is None:
            return None
        return self.analysis_for_type(
            str(getattr(agent, "actr_agent_type_name", ""))
        )

    def analysis_for_type(self, agent_type: str) -> AgentStaticAnalysis | None:
        cached = self._analysis_cache.get(agent_type)
        if cached is not None:
            return cached
        info = next(
            (
                item
                for item in self.discovery.discover()
                if item.name == agent_type
            ),
            None,
        )
        if info is None:
            return None
        analysis = self.analyzer.analyze(info)
        self._analysis_cache[agent_type] = analysis
        return analysis

    def _build_state_graph_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        self.state_graph_png = QPushButton("Export PNG")
        self.state_graph_svg = QPushButton("Export SVG")
        self.state_graph_fit = QPushButton("Fit View")
        toolbar.addWidget(self.state_graph_png)
        toolbar.addWidget(self.state_graph_svg)
        toolbar.addWidget(self.state_graph_fit)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)
        self.state_graph_view = ZoomableGraphicsView(page)
        self.state_graph_png.clicked.connect(
            lambda: self.state_graph_view.export_dialog("png")
        )
        self.state_graph_svg.clicked.connect(
            lambda: self.state_graph_view.export_dialog("svg")
        )
        self.state_graph_fit.clicked.connect(
            self.state_graph_view.reset_zoom
        )
        self.state_findings = QTextEdit(page)
        self.state_findings.setReadOnly(True)
        self.state_findings.setMaximumHeight(135)
        layout.addWidget(self.state_graph_view, 1)
        layout.addWidget(self.state_findings)
        return page

    def _build_interaction_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        interaction_tabs = QTabWidget(page)

        production_page = QWidget(page)
        production_layout = QVBoxLayout(production_page)
        production_toolbar = QHBoxLayout()
        self.production_png = QPushButton("Export PNG")
        self.production_svg = QPushButton("Export SVG")
        production_toolbar.addWidget(self.production_png)
        production_toolbar.addWidget(self.production_svg)
        production_toolbar.addStretch(1)
        production_layout.addLayout(production_toolbar)
        self.production_graph_view = ZoomableGraphicsView(page)
        self.production_png.clicked.connect(
            lambda: self.production_graph_view.export_dialog("png")
        )
        self.production_svg.clicked.connect(
            lambda: self.production_graph_view.export_dialog("svg")
        )
        production_layout.addWidget(self.production_graph_view, 1)
        interaction_tabs.addTab(
            production_page,
            "Productions → Buffers",
        )

        adapter_page = QWidget(page)
        adapter_layout = QVBoxLayout(adapter_page)
        adapter_toolbar = QHBoxLayout()
        self.adapter_png = QPushButton("Export PNG")
        self.adapter_svg = QPushButton("Export SVG")
        adapter_toolbar.addWidget(self.adapter_png)
        adapter_toolbar.addWidget(self.adapter_svg)
        adapter_toolbar.addStretch(1)
        adapter_layout.addLayout(adapter_toolbar)
        self.adapter_graph_view = ZoomableGraphicsView(page)
        self.adapter_png.clicked.connect(
            lambda: self.adapter_graph_view.export_dialog("png")
        )
        self.adapter_svg.clicked.connect(
            lambda: self.adapter_graph_view.export_dialog("svg")
        )
        adapter_layout.addWidget(self.adapter_graph_view, 1)
        interaction_tabs.addTab(
            adapter_page,
            "Adapter Methods → Buffers",
        )

        layout.addWidget(interaction_tabs, 1)
        return page

    def _build_declarative_memory_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        toolbar = QHBoxLayout()
        self.memory_png = QPushButton("Export PNG")
        self.memory_svg = QPushButton("Export SVG")
        self.memory_fit = QPushButton("Fit View")
        toolbar.addWidget(self.memory_png)
        toolbar.addWidget(self.memory_svg)
        toolbar.addWidget(self.memory_fit)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)
        self.memory_graph_view = ZoomableGraphicsView(page)
        self.memory_png.clicked.connect(
            lambda: self.memory_graph_view.export_dialog("png")
        )
        self.memory_svg.clicked.connect(
            lambda: self.memory_graph_view.export_dialog("svg")
        )
        self.memory_fit.clicked.connect(self.memory_graph_view.reset_zoom)
        layout.addWidget(self.memory_graph_view, 1)
        return page

    def refresh(self) -> None:
        infos = self.discovery.discover()
        selected = self.agent_tree.current_selection()
        self.agent_tree.set_agents(
            list(getattr(self.simulation, "agent_list", [])),
            template_types=[info.name for info in infos],
            preserve_runtime_name=(
                selected.runtime_name if selected is not None else None
            ),
        )
        if self.agent_tree.current_selection() is None:
            for index in range(self.agent_tree.topLevelItemCount()):
                item = self.agent_tree.topLevelItem(index)
                if item is not None:
                    self.agent_tree.setCurrentItem(item)
                    break
        self._render_current_selection()

    def _selection_changed(
        self, selection: AgentTreeSelection | None
    ) -> None:
        if selection is None:
            return
        self._render_current_selection(force=True)

    def _render_current_selection(self, *, force: bool = False) -> None:
        selection = self.agent_tree.current_selection()
        if selection is None:
            return
        if not force and self._rendered_type == selection.agent_type:
            return
        analysis = self.analysis_for_type(selection.agent_type)
        if analysis is None:
            return
        self._render_state_graph(analysis)
        self._render_interactions(analysis)
        self._render_declarative_memory(analysis)
        self._rendered_type = selection.agent_type

    def _render_state_graph(self, analysis: AgentStaticAnalysis) -> None:
        self.state_graph_view.setScene(build_state_transition_scene(analysis))
        self.state_graph_view.reset_zoom()
        findings = [
            f"Initial control state: {analysis.states.get(analysis.initial_state_id).label if analysis.states.get(analysis.initial_state_id) else analysis.initial_state_label}",
            f"Expanded productions: {len(analysis.productions)}",
            f"Reachable productions: {sum(1 for item in analysis.productions if item.reachable)}",
            "Unreachable productions: "
            + (
                ", ".join(analysis.unreachable_productions)
                if analysis.unreachable_productions
                else "none"
            ),
            "Dead-end states: "
            + (", ".join(analysis.dead_end_states) if analysis.dead_end_states else "none"),
            "Terminal states: "
            + (", ".join(analysis.terminal_states) if analysis.terminal_states else "none"),
            f"States participating in loops: {len(analysis.loop_states)}",
        ]
        if analysis.analysis_warnings:
            findings.extend(["", *analysis.analysis_warnings])
        self.state_findings.setPlainText("\n".join(findings))

    def _render_interactions(self, analysis: AgentStaticAnalysis) -> None:
        self.production_graph_view.setScene(
            build_interaction_scene(
                "Which productions read or overwrite which buffers",
                analysis.production_interactions,
            )
        )
        self.production_graph_view.reset_zoom()
        self.adapter_graph_view.setScene(
            build_interaction_scene(
                "Which adapter handlers read or overwrite which buffers",
                analysis.adapter_interactions,
            )
        )
        self.adapter_graph_view.reset_zoom()

    def _render_declarative_memory(
        self, analysis: AgentStaticAnalysis
    ) -> None:
        self.memory_graph_view.setScene(
            build_declarative_memory_scene(
                analysis.declarative_memory,
                title=(
                    f"Declarative Memory from Agent and Adapter Code — "
                    f"{analysis.agent_type}"
                ),
            )
        )
        self.memory_graph_view.reset_zoom()
