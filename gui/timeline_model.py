"""Qt model for the per-agent event timeline."""

from __future__ import annotations

import bisect
import hashlib
from typing import Any

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QSize, Qt
from PyQt6.QtGui import QColor


class TimelineTableModel(QAbstractTableModel):
    """Sparse, incrementally updated event-type × timestamp table."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._agent_name: str | None = None
        self._types: list[str] = []
        self._times: list[Any] = []
        self._events: dict[tuple[str, Any], list[str]] = {}
        self._colors: dict[str, QColor] = {}
        self._processed_record_count = 0

    @property
    def agent_name(self) -> str | None:
        return self._agent_name

    def replace_records(self, records: list[dict[str, Any]], agent_name: str | None) -> None:
        """Rebuild the model when the selected agent or source history changes."""
        self.beginResetModel()
        self._agent_name = agent_name
        selected = [
            record
            for record in records
            if agent_name is not None and record.get("agent_name") == agent_name
        ]
        self._types = sorted({str(record.get("type", "")) for record in selected})
        self._times = self._sorted_times({record.get("timestamp", 0.0) for record in selected})
        self._events = {}
        for record in selected:
            event = record.get("event")
            if event is not None:
                key = (str(record.get("type", "")), record.get("timestamp", 0.0))
                self._events.setdefault(key, []).append(str(event))
        self._processed_record_count = len(records)
        self.endResetModel()

    def sync_records(self, records: list[dict[str, Any]], agent_name: str | None) -> None:
        """Consume only newly appended tracer records whenever possible."""
        if agent_name != self._agent_name or len(records) < self._processed_record_count:
            self.replace_records(records, agent_name)
            return
        if len(records) == self._processed_record_count:
            return

        new_records = records[self._processed_record_count :]
        self._processed_record_count = len(records)
        if agent_name is None:
            return

        for record in new_records:
            if record.get("agent_name") != agent_name:
                continue
            event_type = str(record.get("type", ""))
            timestamp = record.get("timestamp", 0.0)
            row = self._ensure_type(event_type)
            column = self._ensure_time(timestamp)
            event = record.get("event")
            if event is not None:
                self._events.setdefault((event_type, timestamp), []).append(str(event))
            index = self.index(row, column)
            self.dataChanged.emit(
                index,
                index,
                [
                    Qt.ItemDataRole.DisplayRole,
                    Qt.ItemDataRole.ToolTipRole,
                    Qt.ItemDataRole.BackgroundRole,
                    Qt.ItemDataRole.SizeHintRole,
                ],
            )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._types)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._times)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        event_type = self._types[index.row()]
        timestamp = self._times[index.column()]
        events = self._events.get((event_type, timestamp), [])
        if role == Qt.ItemDataRole.DisplayRole:
            return "\n".join(events)
        if role == Qt.ItemDataRole.ToolTipRole and events:
            return "\n\n".join(events)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.BackgroundRole and events:
            return self._color_for(event_type)
        if role == Qt.ItemDataRole.ForegroundRole and events:
            return QColor("#f7f9fc")
        if role == Qt.ItemDataRole.SizeHintRole:
            lines = max(1, sum(event.count("\n") + 1 for event in events))
            return QSize(180, max(52, min(180, 24 + lines * 17)))
        return None

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            value = self._times[section]
            try:
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return str(value)
        return self._types[section]

    def _ensure_type(self, event_type: str) -> int:
        try:
            return self._types.index(event_type)
        except ValueError:
            row = bisect.bisect_left(self._types, event_type)
            self.beginInsertRows(QModelIndex(), row, row)
            self._types.insert(row, event_type)
            self.endInsertRows()
            return row

    def _ensure_time(self, timestamp: Any) -> int:
        try:
            return self._times.index(timestamp)
        except ValueError:
            try:
                column = bisect.bisect_left(self._times, timestamp)
            except TypeError:
                # Mixed timestamp types are unusual; a stable string ordering keeps the UI usable.
                merged = self._times + [timestamp]
                ordered = self._sorted_times(set(merged))
                column = ordered.index(timestamp)
            self.beginInsertColumns(QModelIndex(), column, column)
            self._times.insert(column, timestamp)
            self.endInsertColumns()
            return column

    @staticmethod
    def _sorted_times(values: set[Any]) -> list[Any]:
        try:
            return sorted(values)
        except TypeError:
            return sorted(values, key=lambda value: (type(value).__name__, str(value)))

    def _color_for(self, event_type: str) -> QColor:
        cached = self._colors.get(event_type)
        if cached is not None:
            return cached
        digest = hashlib.sha256(event_type.encode("utf-8")).digest()
        hue = int.from_bytes(digest[:2], "big") % 360
        color = QColor.fromHsl(hue, 125, 88)
        self._colors[event_type] = color
        return color
