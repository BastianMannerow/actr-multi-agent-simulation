"""Per-buffer current-state and bounded change-history widgets."""

from __future__ import annotations

import json
from collections.abc import Sequence
from time import perf_counter
from typing import Any

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
)

from simulation.inspection.buffer_history import BufferHistoryRecorder


class BufferHistoryTableModel(QAbstractTableModel):
    """Bounded window over an append-only buffer history.

    The recorder retains the complete history.  The Qt model only flattens the
    currently visible window, preventing a buffer tab from allocating one row
    per historical change after a long simulation.
    """

    MAX_VISIBLE_ROWS = 500
    page_info_changed = pyqtSignal()

    COLUMNS = (
        ("Time", "timestamp"),
        ("Change", "change"),
        ("State", "buffer_state"),
        ("Trigger", "event_type"),
        ("Event", "event"),
        ("Content", "content"),
    )

    CHANGE_LABELS = {
        "initial": "Initial",
        "filled": "Filled",
        "cleared": "Cleared",
        "state_changed": "State changed",
        "content_changed": "Content changed",
        "module_changed": "Module state changed",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._entries: Sequence[dict[str, Any]] = ()
        self._source_count = 0
        self._page_start = 0
        self._page_end = 0
        self._rows: list[dict[str, Any]] = []

    @property
    def can_go_previous(self) -> bool:
        return self._page_start > 0

    @property
    def can_go_next(self) -> bool:
        return self._page_end < self._source_count

    @property
    def is_latest_page(self) -> bool:
        return self._page_end >= self._source_count

    @property
    def source_count(self) -> int:
        return self._source_count

    def replace_entries(self, entries: Sequence[dict[str, Any]]) -> None:
        self._entries = entries
        self._source_count = len(entries)
        self._show_latest_page()

    def sync_entries(self, entries: Sequence[dict[str, Any]]) -> None:
        new_count = len(entries)
        if entries is not self._entries or new_count < self._source_count:
            self._entries = entries
            self._source_count = new_count
            self._show_latest_page()
            return
        if new_count == self._source_count:
            return
        was_latest = self.is_latest_page
        self._source_count = new_count
        if was_latest:
            self._show_latest_page()
        else:
            self.page_info_changed.emit()

    def show_previous_page(self) -> None:
        if not self.can_go_previous:
            return
        end = self._page_start
        start = max(0, end - self.MAX_VISIBLE_ROWS)
        self._set_page(start, end)

    def show_next_page(self) -> None:
        if not self.can_go_next:
            return
        start = self._page_end
        end = min(self._source_count, start + self.MAX_VISIBLE_ROWS)
        self._set_page(start, end)

    def show_latest_page(self) -> None:
        self._show_latest_page()

    def ensure_source_index(self, source_index: int) -> None:
        if self._source_count == 0:
            self._set_page(0, 0)
            return
        source_index = max(0, min(source_index, self._source_count - 1))
        if self._page_start <= source_index < self._page_end:
            return
        start = (source_index // self.MAX_VISIBLE_ROWS) * self.MAX_VISIBLE_ROWS
        end = min(self._source_count, start + self.MAX_VISIBLE_ROWS)
        self._set_page(start, end)

    def source_index_for_row(self, row: int) -> int | None:
        if 0 <= row < len(self._rows):
            return self._page_start + row
        return None

    def row_for_source_index(self, source_index: int) -> int | None:
        if self._page_start <= source_index < self._page_end:
            return source_index - self._page_start
        return None

    def entry_at(self, source_index: int) -> dict[str, Any] | None:
        if 0 <= source_index < self._source_count:
            return self._entries[source_index]
        return None

    def page_description(self) -> str:
        if self._source_count == 0:
            return "No recorded changes"
        return (
            f"Changes {self._page_start + 1:,}–{self._page_end:,} "
            f"of {self._source_count:,}"
        )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.COLUMNS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        key = self.COLUMNS[index.column()][1]
        value = self._rows[index.row()].get(key, "")
        if role == Qt.ItemDataRole.DisplayRole:
            if key == "timestamp":
                try:
                    return f"{float(value):.4f}"
                except (TypeError, ValueError):
                    return str(value)
            return str(value or "")
        if role == Qt.ItemDataRole.ToolTipRole:
            return str(value or "")
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() == 0:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self.COLUMNS[section][0]
        return self._page_start + section + 1

    def _show_latest_page(self) -> None:
        end = self._source_count
        start = max(0, end - self.MAX_VISIBLE_ROWS)
        self._set_page(start, end)

    def _set_page(self, start: int, end: int) -> None:
        start = max(0, min(start, self._source_count))
        end = max(start, min(end, self._source_count))
        page = self._entries[start:end]
        rows = [self._flatten(entry) for entry in page]
        self.beginResetModel()
        self._page_start = start
        self._page_end = end
        self._rows = rows
        self.endResetModel()
        self.page_info_changed.emit()

    @classmethod
    def _flatten(cls, entry: dict[str, Any]) -> dict[str, Any]:
        snapshot = entry.get("snapshot") or {}
        chunks = snapshot.get("chunks", [])
        content = "<empty>" if not chunks else " | ".join(
            str(
                chunk.get("text")
                or json.dumps(chunk.get("slots", {}), ensure_ascii=False)
            )
            for chunk in chunks
        )
        raw_change = str(entry.get("change") or "")
        return {
            "timestamp": entry.get("timestamp"),
            "change": cls.CHANGE_LABELS.get(raw_change, raw_change),
            "buffer_state": snapshot.get("state"),
            "event_type": entry.get("event_type"),
            "event": entry.get("event"),
            "content": content,
        }


class BufferInspectorTab(QFrame):
    """Show current buffer state and a navigable bounded history window."""

    LIVE_MIN_INTERVAL_SECONDS = 0.20

    def __init__(self, buffer_name: str, parent=None) -> None:
        super().__init__(parent)
        self.buffer_name = buffer_name
        self._latest_signature: str | None = None
        self._next_live_update_at = 0.0
        self._selected_source_index: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.summary = QLabel("No snapshot recorded")
        self.summary.setObjectName("muted")
        layout.addWidget(self.summary)

        navigation = QHBoxLayout()
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.latest_button = QPushButton("Latest")
        self.page_label = QLabel("No recorded changes")
        self.page_label.setObjectName("muted")
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.next_button)
        navigation.addWidget(self.latest_button)
        navigation.addStretch(1)
        navigation.addWidget(self.page_label)
        layout.addLayout(navigation)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setChildrenCollapsible(False)

        self.current_text = QPlainTextEdit(splitter)
        self.current_text.setReadOnly(True)
        self.current_text.setObjectName("bufferCurrent")
        self.current_text.setPlaceholderText(
            "This buffer has not been captured yet."
        )

        self.model = BufferHistoryTableModel(self)
        self.model.page_info_changed.connect(self._update_navigation)
        self.previous_button.clicked.connect(self._select_previous_change)
        self.next_button.clicked.connect(self._select_next_change)
        self.latest_button.clicked.connect(self._select_latest_change)

        self.history_table = QTableView(splitter)
        self.history_table.setModel(self.model)
        self.history_table.selectionModel().currentRowChanged.connect(
            self._history_row_changed
        )
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setWordWrap(False)
        self.history_table.setSelectionBehavior(
            QTableView.SelectionBehavior.SelectRows
        )
        self.history_table.setSelectionMode(
            QTableView.SelectionMode.SingleSelection
        )
        self.history_table.verticalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Fixed
        )
        self.history_table.verticalHeader().setDefaultSectionSize(34)
        header = self.history_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.history_table.setColumnWidth(0, 90)
        self.history_table.setColumnWidth(1, 135)
        self.history_table.setColumnWidth(2, 100)
        self.history_table.setColumnWidth(3, 130)
        self.history_table.setColumnWidth(4, 260)

        splitter.addWidget(self.current_text)
        splitter.addWidget(self.history_table)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([220, 520])
        layout.addWidget(splitter, 1)
        self._update_navigation()

    def set_buffer_name(self, buffer_name: str) -> None:
        self.buffer_name = buffer_name

    def update_data(
        self,
        latest: dict[str, Any] | None,
        entries: Sequence[dict[str, Any]],
        *,
        automatic_running: bool = False,
        force: bool = False,
    ) -> None:
        now = perf_counter()
        if automatic_running and not force and now < self._next_live_update_at:
            return

        previous_count = self.model.source_count
        was_latest = (
            self._selected_source_index is None
            or self._selected_source_index >= max(0, previous_count - 1)
        )
        self.model.sync_entries(entries)
        if not entries:
            self._selected_source_index = None
            self.current_text.clear()
            self.summary.setText(
                f"{self.buffer_name or 'Buffer'}: no recorded changes"
            )
            self.history_table.clearSelection()
            self._latest_signature = None
            self._update_navigation()
            return

        if force or was_latest or self._selected_source_index is None:
            self._select_source_index(len(entries) - 1)
        else:
            self._select_source_index(
                min(self._selected_source_index, len(entries) - 1)
            )
        self.summary.setText(
            f"{self.buffer_name} · {len(entries):,} recorded change"
            f"{'s' if len(entries) != 1 else ''}"
        )
        if automatic_running:
            self._next_live_update_at = now + self.LIVE_MIN_INTERVAL_SECONDS
        else:
            self._next_live_update_at = 0.0

    def clear_data(self) -> None:
        self._latest_signature = None
        self._next_live_update_at = 0.0
        self._selected_source_index = None
        self.model.replace_entries(())
        self.current_text.clear()
        self.history_table.clearSelection()
        self.summary.setText(
            f"{self.buffer_name or 'Buffer'}: no recorded changes"
        )
        self._update_navigation()

    def _history_row_changed(self, current, _previous) -> None:
        source_index = self.model.source_index_for_row(current.row())
        if source_index is not None:
            self._selected_source_index = source_index
            self._show_entry(source_index)
        self._update_navigation()

    def _show_entry(self, source_index: int) -> None:
        entry = self.model.entry_at(source_index)
        if entry is None:
            self.current_text.clear()
            return
        snapshot = entry.get("snapshot")
        self.current_text.setPlainText(
            BufferHistoryRecorder.format_snapshot(snapshot)
        )
        self._latest_signature = json.dumps(
            snapshot, ensure_ascii=False, sort_keys=True, default=str
        )

    def _select_source_index(self, source_index: int) -> None:
        if self.model.source_count == 0:
            self._selected_source_index = None
            self._update_navigation()
            return
        source_index = max(0, min(source_index, self.model.source_count - 1))
        self.model.ensure_source_index(source_index)
        row = self.model.row_for_source_index(source_index)
        if row is None:
            return
        index = self.model.index(row, 0)
        self.history_table.setCurrentIndex(index)
        self.history_table.scrollTo(
            index, QTableView.ScrollHint.PositionAtCenter
        )
        self._selected_source_index = source_index
        self._show_entry(source_index)
        self._update_navigation()

    def _select_previous_change(self) -> None:
        if self.model.source_count == 0:
            return
        current = (
            self._selected_source_index
            if self._selected_source_index is not None
            else self.model.source_count
        )
        self._select_source_index(current - 1)

    def _select_next_change(self) -> None:
        if self.model.source_count == 0:
            return
        current = self._selected_source_index if self._selected_source_index is not None else -1
        self._select_source_index(current + 1)

    def _select_latest_change(self) -> None:
        if self.model.source_count:
            self._select_source_index(self.model.source_count - 1)

    def _update_navigation(self) -> None:
        count = self.model.source_count
        current = self._selected_source_index
        self.previous_button.setEnabled(count > 0 and current is not None and current > 0)
        self.next_button.setEnabled(
            count > 0 and current is not None and current < count - 1
        )
        self.latest_button.setEnabled(
            count > 0 and current is not None and current < count - 1
        )
        if count == 0:
            self.page_label.setText("No recorded changes")
        elif current is None:
            self.page_label.setText(self.model.page_description())
        else:
            self.page_label.setText(
                f"Change {current + 1:,} of {count:,} · "
                f"{self.model.page_description()}"
            )
