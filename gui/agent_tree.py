"""Collapsible agent-type navigation shared by inspector and analysis views."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem


@dataclass(frozen=True, slots=True)
class AgentTreeSelection:
    display_name: str
    agent_type: str
    runtime_name: str | None
    template: bool = False


class AgentTreeWidget(QTreeWidget):
    """Group runtime agents under a collapsed top-level agent type."""

    selection_changed = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setIndentation(18)
        self.setRootIsDecorated(True)
        self.setAnimated(True)
        self.setMinimumWidth(180)
        self.setMaximumWidth(320)
        self._signature: tuple[tuple[str, tuple[str, ...]], ...] = ()
        self.itemSelectionChanged.connect(self._emit_selection)

    def set_agents(
        self,
        runtime_agents: list[Any],
        *,
        template_types: list[str] | None = None,
        preserve_runtime_name: str | None = None,
    ) -> None:
        grouped: dict[str, list[str]] = defaultdict(list)
        for agent in runtime_agents:
            agent_type = str(getattr(agent, "actr_agent_type_name", "Unknown"))
            runtime_name = str(getattr(agent, "name", ""))
            if runtime_name:
                grouped[agent_type].append(runtime_name)
        for agent_type in template_types or []:
            grouped.setdefault(str(agent_type), [])

        signature = tuple(
            (agent_type, tuple(sorted(names, key=str.lower)))
            for agent_type, names in sorted(grouped.items(), key=lambda item: item[0].lower())
        )
        if self._signature == signature:
            return
        self._signature = signature
        self.blockSignals(True)
        self.clear()
        selected_item: QTreeWidgetItem | None = None
        for agent_type, names in signature:
            parent = QTreeWidgetItem([agent_type])
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            parent.setToolTip(0, f"{len(names)} runtime agent(s)")
            parent.setData(0, Qt.ItemDataRole.UserRole, AgentTreeSelection(
                display_name=agent_type,
                agent_type=agent_type,
                runtime_name=None,
                template=True,
            ))
            parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsSelectable)
            self.addTopLevelItem(parent)
            for runtime_name in names:
                child = QTreeWidgetItem([runtime_name])
                child.setData(0, Qt.ItemDataRole.UserRole, AgentTreeSelection(
                    display_name=runtime_name,
                    agent_type=agent_type,
                    runtime_name=runtime_name,
                    template=False,
                ))
                parent.addChild(child)
                if runtime_name == preserve_runtime_name:
                    selected_item = child
            parent.setExpanded(False)
        if selected_item is not None:
            self.setCurrentItem(selected_item)
            selected_item.parent().setExpanded(True)
        self.blockSignals(False)
        self._emit_selection()

    def current_selection(self) -> AgentTreeSelection | None:
        item = self.currentItem()
        if item is None:
            return None
        value = item.data(0, Qt.ItemDataRole.UserRole)
        return value if isinstance(value, AgentTreeSelection) else None

    def select_first_runtime_agent(self) -> None:
        for index in range(self.topLevelItemCount()):
            parent = self.topLevelItem(index)
            if parent.childCount():
                self.setCurrentItem(parent.child(0))
                return

    def _emit_selection(self) -> None:
        self.selection_changed.emit(self.current_selection())
