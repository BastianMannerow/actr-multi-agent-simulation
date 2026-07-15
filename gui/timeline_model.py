"""Virtualized Qt model for the per-agent event timeline."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor


class TimelineTableModel(QAbstractTableModel):
    """Bounded event-type × timestamp window over an append-only history.

    A complete simulation can contain tens or hundreds of thousands of ACT-R
    events. Creating one Qt header section for every timestamp makes agent
    selection increasingly expensive and can block the GUI thread. The model
    therefore keeps the source history by reference and materializes only a
    bounded timestamp window. Older windows remain available through explicit
    page navigation.
    """

    MAX_VISIBLE_TIMESTAMPS = 240
    MAX_VISIBLE_RECORDS = 1_200
    page_info_changed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._agent_name: str | None = None
        self._records: Sequence[dict[str, Any]] = ()
        self._source_count = 0
        self._page_start = 0
        self._page_end = 0
        self._types: list[str] = []
        self._times: list[Any] = []
        self._events: dict[tuple[str, Any], list[str]] = {}
        self._cell_record_indices: dict[tuple[str, Any], list[int]] = {}
        self._colors: dict[str, QColor] = {}

    @property
    def agent_name(self) -> str | None:
        return self._agent_name

    @property
    def source_count(self) -> int:
        return self._source_count

    @property
    def records(self) -> Sequence[dict[str, Any]]:
        return self._records

    @property
    def can_go_previous(self) -> bool:
        return self._page_start > 0

    @property
    def can_go_next(self) -> bool:
        return self._page_end < self._source_count

    @property
    def is_latest_page(self) -> bool:
        return self._page_end >= self._source_count

    def replace_records(
        self,
        records: Sequence[dict[str, Any]],
        agent_name: str | None,
        *,
        already_filtered: bool = False,
    ) -> None:
        """Select a history and display its newest bounded window.

        ``already_filtered`` is used by :class:`StepLogView`, which obtains an
        O(1) per-agent history view from ``Tracer.records_for_agent``. The
        compatibility path still accepts a global history, but should not be
        used for high-volume runtime refreshes.
        """
        if agent_name is None:
            selected: Sequence[dict[str, Any]] = ()
        elif already_filtered:
            selected = records
        else:
            selected = tuple(
                record
                for record in records
                if record.get("agent_name") == agent_name
            )
        self._agent_name = agent_name
        self._records = selected
        self._source_count = len(selected)
        self._show_latest_page()

    def sync_records(
        self,
        records: Sequence[dict[str, Any]],
        agent_name: str | None,
        *,
        already_filtered: bool = False,
    ) -> None:
        """Synchronize an append-only source without materializing all rows."""
        if agent_name != self._agent_name:
            self.replace_records(
                records,
                agent_name,
                already_filtered=already_filtered,
            )
            return

        if not already_filtered:
            # This path preserves the old API. Runtime code uses the indexed
            # per-agent path above and therefore never scans the global list.
            self.replace_records(records, agent_name, already_filtered=False)
            return

        new_count = len(records)
        if records is not self._records or new_count < self._source_count:
            self._records = records
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
        start = self._find_page_start(end)
        self._set_page(start, end)

    def show_next_page(self) -> None:
        if not self.can_go_next:
            return
        start = self._page_end
        end = self._find_page_end(start)
        self._set_page(start, end)

    def show_latest_page(self) -> None:
        if self._source_count == 0 and self._page_end == 0:
            return
        self._show_latest_page()

    def show_source_index(self, source_index: int) -> None:
        """Display the bounded page ending at ``source_index``."""
        if self._source_count == 0:
            return
        source_index = max(0, min(source_index, self._source_count - 1))
        end = source_index + 1
        start = self._find_page_start(end)
        self._set_page(start, end)

    def cell_for_source_index(self, source_index: int) -> tuple[int, int] | None:
        if not (self._page_start <= source_index < self._page_end):
            return None
        record = self._records[source_index]
        event_type = str(record.get("type", ""))
        timestamp = record.get("timestamp", 0.0)
        try:
            return self._types.index(event_type), self._times.index(timestamp)
        except ValueError:
            return None

    def source_index_for_cell(self, row: int, column: int) -> int | None:
        if not (0 <= row < len(self._types) and 0 <= column < len(self._times)):
            return None
        values = self._cell_record_indices.get(
            (self._types[row], self._times[column]), []
        )
        return values[-1] if values else None

    def page_description(self) -> str:
        if self._agent_name is None:
            return "No agent selected"
        if self._source_count == 0:
            return "No timeline events"
        return (
            f"Events {self._page_start + 1:,}–{self._page_end:,} "
            f"of {self._source_count:,} · {len(self._times)} timestamps"
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
            # Row heights are fixed in the view. Keeping a bounded hint avoids
            # expensive ResizeToContents scans over every visible cell.
            return QSize(180, 64)
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

    def _show_latest_page(self) -> None:
        end = self._source_count
        start = self._find_page_start(end)
        self._set_page(start, end)

    def _set_page(self, start: int, end: int) -> None:
        start = max(0, min(start, self._source_count))
        end = max(start, min(end, self._source_count))
        page_records = self._records[start:end]

        event_types: set[str] = set()
        timestamps: set[Any] = set()
        events: dict[tuple[str, Any], list[str]] = {}
        cell_record_indices: dict[tuple[str, Any], list[int]] = {}
        for offset, record in enumerate(page_records):
            event_type = str(record.get("type", ""))
            timestamp = record.get("timestamp", 0.0)
            event_types.add(event_type)
            timestamps.add(timestamp)
            event = record.get("event")
            key = (event_type, timestamp)
            cell_record_indices.setdefault(key, []).append(start + offset)
            if event is not None:
                events.setdefault(key, []).append(str(event))

        self.beginResetModel()
        self._page_start = start
        self._page_end = end
        self._types = sorted(event_types, key=str.lower)
        self._times = self._sorted_times(timestamps)
        self._events = events
        self._cell_record_indices = cell_record_indices
        self.endResetModel()
        self.page_info_changed.emit()

    def _find_page_start(self, end: int) -> int:
        if end <= 0:
            return 0
        timestamps: set[Any] = set()
        index = end - 1
        scanned = 0
        while index >= 0 and scanned < self.MAX_VISIBLE_RECORDS:
            timestamp = self._records[index].get("timestamp", 0.0)
            if timestamp not in timestamps and len(timestamps) >= self.MAX_VISIBLE_TIMESTAMPS:
                break
            timestamps.add(timestamp)
            scanned += 1
            index -= 1
        return index + 1

    def _find_page_end(self, start: int) -> int:
        if start >= self._source_count:
            return self._source_count
        timestamps: set[Any] = set()
        index = start
        scanned = 0
        while index < self._source_count and scanned < self.MAX_VISIBLE_RECORDS:
            timestamp = self._records[index].get("timestamp", 0.0)
            if timestamp not in timestamps and len(timestamps) >= self.MAX_VISIBLE_TIMESTAMPS:
                break
            timestamps.add(timestamp)
            scanned += 1
            index += 1
        return index

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
