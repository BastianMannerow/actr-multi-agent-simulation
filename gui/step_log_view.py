"""Agent selection, virtualized event timeline, buffers, and memory."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any

from PyQt6.QtCore import QSignalBlocker, QTimer, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
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
    """Visibility-aware runtime inspector with bounded, searchable histories."""

    TIMELINE_TAB = 0
    MEMORY_TAB = 1
    BUFFER_TAB = 2
    LIVE_MIN_INTERVAL_SECONDS = 0.15
    SEARCH_CHUNK_SIZE = 2_000

    def __init__(
        self, tracer: Any, simulation: Any, parent=None, *, task_manager=None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.tracer = tracer
        self.simulation = simulation
        self.task_manager = task_manager
        self._selected_agent_name: str | None = None
        self._tab_load_generation = 0
        self._next_live_refresh_at = 0.0
        self._buffer_names: list[str] = []
        self._search_generation = 0
        self._search_query = ""
        self._search_last_index: int | None = None

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
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setElideMode(Qt.TextElideMode.ElideRight)
        self.timeline_page = self._build_timeline_page()
        self.memory_page = DeclarativeMemoryInspectorTab(
            self.tabs, task_manager=self.task_manager
        )
        self.buffer_page = self._build_buffer_page()
        self.tabs.addTab(self.timeline_page, "Step Timeline")
        self.tabs.addTab(self.memory_page, "Declarative Memory")
        self.tabs.addTab(self.buffer_page, "Buffer History")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        details_layout.addWidget(self.tabs, 1)

        splitter.addWidget(agent_panel)
        splitter.addWidget(details_panel)
        splitter.setSizes([220, 1000])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)
        self.refresh(force=True)

    def _build_timeline_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(7)

        navigation = QHBoxLayout()
        self.timeline_previous_button = QPushButton("Previous")
        self.timeline_next_button = QPushButton("Next")
        self.timeline_latest_button = QPushButton("Latest")
        self.timeline_page_label = QLabel("No agent selected")
        self.timeline_page_label.setObjectName("muted")
        navigation.addWidget(self.timeline_previous_button)
        navigation.addWidget(self.timeline_next_button)
        navigation.addWidget(self.timeline_latest_button)
        navigation.addStretch(1)
        navigation.addWidget(self.timeline_page_label)
        layout.addLayout(navigation)

        self.model = TimelineTableModel(self)
        self.model.page_info_changed.connect(self._update_timeline_navigation)
        self.timeline_previous_button.clicked.connect(self._timeline_previous)
        self.timeline_next_button.clicked.connect(self._timeline_next)
        self.timeline_latest_button.clicked.connect(self._timeline_latest)

        self.table = QTableView(page)
        self.table.setModel(self.model)
        self.table.selectionModel().currentChanged.connect(
            self._timeline_selection_changed
        )
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectItems)
        self.table.setCornerButtonEnabled(False)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.table.horizontalHeader().setDefaultSectionSize(185)
        self.table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed
        )
        self.table.verticalHeader().setMinimumSectionSize(52)
        self.table.verticalHeader().setDefaultSectionSize(64)
        self.table.verticalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(self.table, 1)

        search = QHBoxLayout()
        self.timeline_search = QLineEdit(page)
        self.timeline_search.setClearButtonEnabled(True)
        self.timeline_search.setPlaceholderText(
            "Search the complete timeline (event, type, timestamp)…"
        )
        self.timeline_previous_match = QPushButton("Previous Match")
        self.timeline_next_match = QPushButton("Next Match")
        self.timeline_search_status = QLabel("")
        self.timeline_search_status.setObjectName("muted")
        search.addWidget(QLabel("Search:"))
        search.addWidget(self.timeline_search, 1)
        search.addWidget(self.timeline_previous_match)
        search.addWidget(self.timeline_next_match)
        search.addWidget(self.timeline_search_status)
        layout.addLayout(search)
        self.timeline_search.textChanged.connect(self._timeline_query_changed)
        self.timeline_search.returnPressed.connect(
            lambda: self._start_timeline_search(1)
        )
        self.timeline_previous_match.clicked.connect(
            lambda: self._start_timeline_search(-1)
        )
        self.timeline_next_match.clicked.connect(
            lambda: self._start_timeline_search(1)
        )
        self._timeline_query_changed("")
        self._update_timeline_navigation()
        return page

    def _build_buffer_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)
        selector = QHBoxLayout()
        selector.addWidget(QLabel("Buffer:"))
        self.buffer_selector = QComboBox(page)
        self.buffer_selector.setEditable(False)
        self.buffer_selector.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.buffer_selector.setMaxVisibleItems(18)
        self.buffer_selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.buffer_selector.setMinimumContentsLength(24)
        self.buffer_selector.currentIndexChanged.connect(
            lambda _index: self._buffer_changed(self.buffer_selector.currentText())
        )
        selector.addWidget(self.buffer_selector, 1)
        selector.addStretch(1)
        layout.addLayout(selector)
        self.buffer_inspector = BufferInspectorTab("", page)
        layout.addWidget(self.buffer_inspector, 1)
        return page

    @property
    def current_agent(self) -> str | None:
        selection = self.agent_tree.current_selection()
        return selection.runtime_name if selection is not None else None

    def refresh(self, *, force: bool = False) -> None:
        now = perf_counter()
        automatic_running = self._automatic_running()
        if automatic_running and not force and now < self._next_live_refresh_at:
            return
        selected_before = self.current_agent or self._selected_agent_name
        runtime_agents = list(getattr(self.simulation, "agent_list", []))
        self.agent_tree.set_agents(runtime_agents, preserve_runtime_name=selected_before)
        selected = self.current_agent
        changed_agent = selected != self._selected_agent_name
        self._selected_agent_name = selected
        self._sync_buffer_selector(selected, rebuild=changed_agent)

        if self.tabs.currentIndex() == self.TIMELINE_TAB:
            records = self._timeline_records(selected)
            if changed_agent or force:
                self.model.replace_records(records, selected, already_filtered=True)
                self._select_timeline_source_index(self.model.source_count - 1)
                self._reset_timeline_search()
            else:
                self.model.sync_records(records, selected, already_filtered=True)
        else:
            self._update_active_tab(selected, force=force)
        self._update_summary(selected)
        self._next_live_refresh_at = (
            now + self.LIVE_MIN_INTERVAL_SECONDS if automatic_running else 0.0
        )

    def _select_agent(self, selection: AgentTreeSelection | None) -> None:
        selected = selection.runtime_name if selection is not None else None
        changed_agent = selected != self._selected_agent_name
        self._selected_agent_name = selected
        self._sync_buffer_selector(selected, rebuild=changed_agent)
        if self.tabs.currentIndex() == self.TIMELINE_TAB:
            self.model.replace_records(
                self._timeline_records(selected), selected, already_filtered=True
            )
            self._select_timeline_source_index(self.model.source_count - 1)
            self._reset_timeline_search()
        else:
            self._schedule_active_tab_update(force=True)
        self._update_summary(selected)

    def _timeline_records(self, selected: str | None) -> Sequence[dict[str, Any]]:
        indexed = getattr(self.tracer, "records_for_agent", None)
        if callable(indexed):
            return indexed(selected)
        if selected is None:
            return ()
        return tuple(
            record for record in getattr(self.tracer, "records", ())
            if record.get("agent_name") == selected
        )

    def _sync_buffer_selector(self, selected: str | None, rebuild: bool = False) -> None:
        recorder = getattr(self.simulation, "buffer_history", None)
        names = list(recorder.buffer_names(selected)) if recorder is not None and selected else []
        names = sorted(dict.fromkeys(names), key=str.casefold)
        if not rebuild and names == self._buffer_names:
            return
        previous = self.buffer_selector.currentText().strip()
        self._buffer_names = names
        blocker = QSignalBlocker(self.buffer_selector)
        self.buffer_selector.clear()
        self.buffer_selector.addItems(names)
        if previous in names:
            self.buffer_selector.setCurrentText(previous)
        elif names:
            self.buffer_selector.setCurrentIndex(0)
        del blocker
        current = self.buffer_selector.currentText().strip() if names else ""
        self.buffer_inspector.set_buffer_name(current)
        self.buffer_inspector.clear_data()
        self.buffer_selector.setEnabled(bool(names))
        self.tabs.setTabEnabled(self.BUFFER_TAB, bool(names))
        if self.tabs.currentIndex() == self.BUFFER_TAB and names:
            self._schedule_active_tab_update(force=True)

    def _buffer_changed(self, name: str) -> None:
        name = name.strip()
        if name not in self._buffer_names:
            return
        self.buffer_inspector.set_buffer_name(name)
        # Clear synchronously so a buffer with no history can never display the
        # previously selected buffer while the deferred lookup is pending.
        self.buffer_inspector.clear_data()
        if self.tabs.currentIndex() == self.BUFFER_TAB:
            self._schedule_active_tab_update(force=True)

    def _on_tab_changed(self, _index: int) -> None:
        self._schedule_active_tab_update(force=True)

    def _schedule_active_tab_update(self, *, force: bool) -> None:
        self._tab_load_generation += 1
        generation = self._tab_load_generation

        def load() -> None:
            if generation != self._tab_load_generation:
                return
            selected = self.current_agent or self._selected_agent_name
            if self.tabs.currentIndex() == self.TIMELINE_TAB:
                self.model.replace_records(
                    self._timeline_records(selected), selected, already_filtered=True
                )
                self._select_timeline_source_index(self.model.source_count - 1)
            else:
                self._update_active_tab(selected, force=force)

        QTimer.singleShot(0, load)

    def _automatic_running(self) -> bool:
        return (
            str(getattr(self.simulation, "execution_mode", "single")) == "automatic"
            and str(getattr(self.simulation, "run_state", "not_started")) == "running"
        )

    def _update_active_tab(self, selected: str | None, *, force: bool = False) -> None:
        index = self.tabs.currentIndex()
        if index == self.TIMELINE_TAB:
            return
        automatic_running = self._automatic_running()
        if index == self.MEMORY_TAB:
            self.memory_page.update_agent(
                self.simulation.get_agent_by_name(selected) if selected else None,
                automatic_running=automatic_running,
                force=force,
            )
            return
        if index != self.BUFFER_TAB:
            return
        recorder = getattr(self.simulation, "buffer_history", None)
        buffer_name = self.buffer_selector.currentText().strip()
        if recorder is None or selected is None or not buffer_name:
            self.buffer_inspector.clear_data()
            return
        history_view = getattr(recorder, "history_view", None)
        entries = (
            history_view(selected, buffer_name)
            if callable(history_view)
            else recorder.history(selected, buffer_name)
        )
        latest = recorder.latest(selected, buffer_name)
        if not entries and latest is None:
            self.buffer_inspector.clear_data()
            return
        self.buffer_inspector.update_data(
            latest, entries, automatic_running=automatic_running, force=force
        )

    def _timeline_selection_changed(self, current, _previous) -> None:
        source_index = (
            self.model.source_index_for_cell(current.row(), current.column())
            if current.isValid()
            else None
        )
        self._update_timeline_navigation(source_index)

    def _select_timeline_source_index(self, source_index: int) -> None:
        if self.model.source_count == 0 or source_index < 0:
            self.table.clearSelection()
            self._update_timeline_navigation(None)
            return
        source_index = max(0, min(source_index, self.model.source_count - 1))
        self.model.show_source_index(source_index)
        cell = self.model.cell_for_source_index(source_index)
        if cell is None:
            self._update_timeline_navigation(source_index)
            return
        index = self.model.index(*cell)
        self.table.setCurrentIndex(index)
        self.table.scrollTo(index, QTableView.ScrollHint.PositionAtCenter)
        self._update_timeline_navigation(source_index)

    def _timeline_previous(self) -> None:
        current = self._selected_timeline_source_index()
        if current is None:
            current = self.model.source_count
        self._select_timeline_source_index(current - 1)

    def _timeline_next(self) -> None:
        current = self._selected_timeline_source_index()
        if current is None:
            current = -1
        self._select_timeline_source_index(current + 1)

    def _timeline_latest(self) -> None:
        self._select_timeline_source_index(self.model.source_count - 1)

    def _update_timeline_navigation(self, source_index: int | None = None) -> None:
        if source_index is None:
            source_index = self._selected_timeline_source_index()
        count = self.model.source_count
        self.timeline_previous_button.setEnabled(
            count > 0 and source_index is not None and source_index > 0
        )
        self.timeline_next_button.setEnabled(
            count > 0 and source_index is not None and source_index < count - 1
        )
        self.timeline_latest_button.setEnabled(
            count > 0 and source_index is not None and source_index < count - 1
        )
        description = self.model.page_description()
        if source_index is not None and count:
            description += f" · selected event {source_index + 1:,} of {count:,}"
        self.timeline_page_label.setText(description)

    def _timeline_query_changed(self, text: str) -> None:
        self._search_generation += 1
        self._search_query = text.strip().casefold()
        self._search_last_index = None
        enabled = bool(self._search_query and self.model.source_count)
        self.timeline_previous_match.setEnabled(enabled)
        self.timeline_next_match.setEnabled(enabled)
        self.timeline_search_status.setText("" if enabled or not text else "No timeline events")

    def _reset_timeline_search(self) -> None:
        self._search_generation += 1
        self._search_last_index = None
        self.timeline_search_status.setText("")
        enabled = bool(self._search_query and self.model.source_count)
        self.timeline_previous_match.setEnabled(enabled)
        self.timeline_next_match.setEnabled(enabled)

    def _selected_timeline_source_index(self) -> int | None:
        index = self.table.currentIndex()
        if not index.isValid():
            return None
        return self.model.source_index_for_cell(index.row(), index.column())

    def _start_timeline_search(self, direction: int) -> None:
        query = self.timeline_search.text().strip().casefold()
        if not query or self.model.source_count == 0:
            return
        self._search_generation += 1
        generation = self._search_generation
        if self._search_last_index is not None:
            start = self._search_last_index + direction
        else:
            start = 0 if direction > 0 else self.model.source_count - 1
        self.timeline_search_status.setText("Searching…")
        self.timeline_previous_match.setEnabled(False)
        self.timeline_next_match.setEnabled(False)

        def scan(position: int) -> None:
            if generation != self._search_generation:
                return
            records = self.model.records
            count = len(records)
            if position < 0 or position >= count:
                self.timeline_search_status.setText("No further matches")
                self.timeline_previous_match.setEnabled(True)
                self.timeline_next_match.setEnabled(True)
                return
            stop = min(count, position + self.SEARCH_CHUNK_SIZE) if direction > 0 else max(-1, position - self.SEARCH_CHUNK_SIZE)
            indices = range(position, stop, direction)
            for record_index in indices:
                record = records[record_index]
                haystack = " ".join(
                    str(record.get(key, ""))
                    for key in ("type", "event", "timestamp", "sequence")
                ).casefold()
                if query in haystack:
                    self._search_last_index = record_index
                    self.model.show_source_index(record_index)
                    cell = self.model.cell_for_source_index(record_index)
                    if cell is not None:
                        qt_index = self.model.index(*cell)
                        self.table.setCurrentIndex(qt_index)
                        self.table.scrollTo(
                            qt_index, QTableView.ScrollHint.PositionAtCenter
                        )
                    self.timeline_search_status.setText(
                        f"Match {record_index + 1:,} of {count:,}"
                    )
                    self.timeline_previous_match.setEnabled(True)
                    self.timeline_next_match.setEnabled(True)
                    return
            QTimer.singleShot(0, lambda: scan(stop))

        QTimer.singleShot(0, lambda: scan(start))

    def _update_summary(self, selected: str | None) -> None:
        if selected:
            buffer_count = len(self._buffer_names)
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
