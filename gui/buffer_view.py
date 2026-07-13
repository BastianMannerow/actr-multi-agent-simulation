"""Per-buffer current-state and change-history widgets."""

from __future__ import annotations

import json
from typing import Any

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QTableView,
    QVBoxLayout,
)

from simulation.inspection.buffer_history import BufferHistoryRecorder


class BufferHistoryTableModel(QAbstractTableModel):
    """Virtualized table over the append-only changes of one buffer."""

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
        self._rows: list[dict[str, Any]] = []

    def replace_entries(self, entries: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._rows = [self._flatten(entry) for entry in entries]
        self.endResetModel()

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
        return section + 1

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
    """Show one buffer's latest state above its full change history."""

    def __init__(self, buffer_name: str, parent=None) -> None:
        super().__init__(parent)
        self.buffer_name = buffer_name

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.summary = QLabel("No snapshot recorded")
        self.summary.setObjectName("muted")
        layout.addWidget(self.summary)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setChildrenCollapsible(False)

        self.current_text = QPlainTextEdit(splitter)
        self.current_text.setReadOnly(True)
        self.current_text.setObjectName("bufferCurrent")
        self.current_text.setPlaceholderText(
            "This buffer has not been captured yet."
        )

        self.model = BufferHistoryTableModel(self)
        self.history_table = QTableView(splitter)
        self.history_table.setModel(self.model)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setWordWrap(False)
        self.history_table.setSelectionBehavior(
            QTableView.SelectionBehavior.SelectRows
        )
        self.history_table.setSelectionMode(
            QTableView.SelectionMode.SingleSelection
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

    def update_data(
        self,
        latest: dict[str, Any] | None,
        entries: list[dict[str, Any]],
    ) -> None:
        self.current_text.setPlainText(
            BufferHistoryRecorder.format_snapshot(latest)
        )
        self.model.replace_entries(entries)
        state = latest.get("state") if latest else "–"
        occupancy = "empty" if not latest or latest.get("empty") else "occupied"
        self.summary.setText(
            f"Current: {occupancy} · state {state} · "
            f"{len(entries)} recorded change{'s' if len(entries) != 1 else ''}"
        )
        if entries:
            self.history_table.scrollToBottom()
