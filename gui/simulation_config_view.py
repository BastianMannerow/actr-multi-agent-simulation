"""GUI editor for persistent simulation settings and discovered plug-ins."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from simulation.discovery.agent_discovery import AgentDiscovery, AgentTypeInfo
from simulation.config.models import AgentTypeConfig, SimulationConfig


@dataclass(slots=True)
class _AgentRow:
    info: AgentTypeInfo
    count: QSpinBox
    print_actions: QCheckBox
    status: QLabel


class SimulationConfigView(QFrame):
    """Scrollable editor for build-time simulation settings."""

    agents_changed = pyqtSignal()
    reset_requested = pyqtSignal()

    def __init__(self, initial_config: SimulationConfig, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.discovery = AgentDiscovery()
        self._agent_rows: dict[str, _AgentRow] = {}
        self._current_config = initial_config

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        heading = QHBoxLayout()
        title = QLabel("Simulation Configuration")
        title.setObjectName("sectionTitle")
        self.discovery_summary = QLabel("")
        self.discovery_summary.setObjectName("muted")
        self.refresh_agents_button = QPushButton("Rescan Agents")
        self.refresh_agents_button.clicked.connect(self.refresh_agent_types)
        self.reset_button = QPushButton("Reset to Default Settings")
        self.reset_button.clicked.connect(self.reset_requested.emit)
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(self.discovery_summary)
        heading.addWidget(self.refresh_agents_button)
        heading.addWidget(self.reset_button)
        outer.addLayout(heading)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget(scroll)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(2, 2, 8, 2)
        content_layout.setSpacing(12)

        content_layout.addWidget(self._build_world_group())
        content_layout.addWidget(self._build_human_agent_group())
        content_layout.addWidget(self._build_logging_group())
        self.agent_group = QGroupBox("Agent Types Discovered in the agents Folder")
        self.agent_grid = QGridLayout(self.agent_group)
        self.agent_grid.setColumnStretch(0, 1)
        self.agent_grid.setColumnStretch(3, 2)
        content_layout.addWidget(self.agent_group)
        content_layout.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)
        self.refresh_agent_types()

    def _build_world_group(self) -> QGroupBox:
        group = QGroupBox("Environment")
        form = QFormLayout(group)

        size_row = QWidget(group)
        size_layout = QHBoxLayout(size_row)
        size_layout.setContentsMargins(0, 0, 0, 0)
        self.width_spin = self._integer_spin(1, 500, self._current_config.width)
        self.height_spin = self._integer_spin(1, 500, self._current_config.height)
        size_layout.addWidget(QLabel("Width"))
        size_layout.addWidget(self.width_spin)
        size_layout.addSpacing(12)
        size_layout.addWidget(QLabel("Height"))
        size_layout.addWidget(self.height_spin)
        size_layout.addStretch(1)
        form.addRow("Grid size", size_row)

        focus_row = QWidget(group)
        focus_layout = QHBoxLayout(focus_row)
        focus_layout.setContentsMargins(0, 0, 0, 0)
        self.focus_x_spin = self._integer_spin(
            -10000, 10000, self._current_config.focus_position[0]
        )
        self.focus_y_spin = self._integer_spin(
            -10000, 10000, self._current_config.focus_position[1]
        )
        focus_layout.addWidget(QLabel("X"))
        focus_layout.addWidget(self.focus_x_spin)
        focus_layout.addSpacing(12)
        focus_layout.addWidget(QLabel("Y"))
        focus_layout.addWidget(self.focus_y_spin)
        focus_layout.addStretch(1)
        form.addRow("ACT-R focus position", focus_row)

        self.los_spin = self._integer_spin(0, 1000, self._current_config.los)
        form.addRow("Line of sight (LOS)", self.los_spin)
        return group


    def _build_human_agent_group(self) -> QGroupBox:
        group = QGroupBox("Human-Controlled Agent")
        form = QFormLayout(group)
        self.human_enabled_check = QCheckBox(
            "Add a human-controlled agent to normal simulations", group
        )
        self.human_enabled_check.setChecked(
            self._current_config.human_agent_enabled
        )
        self.human_name_edit = QLineEdit(
            self._current_config.human_agent_name, group
        )
        self.human_name_edit.setPlaceholderText("Human Player")
        self.human_name_edit.setEnabled(self.human_enabled_check.isChecked())
        self.human_enabled_check.toggled.connect(
            self.human_name_edit.setEnabled
        )
        form.addRow(self.human_enabled_check)
        form.addRow("Name", self.human_name_edit)
        controls = QLabel("Controls: WASD or arrow keys")
        controls.setObjectName("muted")
        form.addRow(controls)
        return group

    def _build_logging_group(self) -> QGroupBox:
        group = QGroupBox("Logging")
        form = QFormLayout(group)

        self.print_middleman_check = QCheckBox(
            "Print Middleman actions to stdout", group
        )
        self.print_middleman_check.setChecked(self._current_config.print_middleman)
        form.addRow(self.print_middleman_check)

        self.print_agent_actions_check = QCheckBox(
            "Print agent actions by default", group
        )
        self.print_agent_actions_check.setChecked(
            self._current_config.print_agent_actions
        )
        form.addRow(self.print_agent_actions_check)
        return group

    def refresh_agent_types(self) -> None:
        previous = {
            name: (row.count.value(), row.print_actions.isChecked())
            for name, row in self._agent_rows.items()
        }
        self._clear_grid()
        self._agent_rows.clear()

        infos = self.discovery.discover()
        headers = ["Agent model", "Count", "Log actions", "Adapter status"]
        for column, text in enumerate(headers):
            label = QLabel(text)
            label.setObjectName("muted")
            self.agent_grid.addWidget(label, 0, column)

        usable_count = 0
        for row_index, info in enumerate(infos, start=1):
            name_label = QLabel(info.name)
            if info.model_error:
                name_label.setToolTip(info.model_error)

            count = self._integer_spin(0, 999, 1)
            print_actions = QCheckBox()
            configured = self._current_config.agent_type_config.get(info.name)
            if info.name in previous:
                count.setValue(previous[info.name][0])
                print_actions.setChecked(previous[info.name][1])
            elif configured is not None:
                count.setValue(configured.count)
                print_actions.setChecked(configured.print_agent_actions)
            else:
                # Every newly discovered type starts with exactly one agent.
                count.setValue(1)
                print_actions.setChecked(self._current_config.print_agent_actions)

            status = QLabel()
            if not info.model_available:
                status.setText("Model could not be loaded")
                status.setProperty("status", "error")
                status.setToolTip(info.model_error or "Unknown model error")
                count.setEnabled(False)
                print_actions.setEnabled(False)
            elif info.adapter_error:
                status.setText("Adapter is invalid — start blocked")
                status.setProperty("status", "error")
                status.setToolTip(info.adapter_error)
                count.setEnabled(False)
                print_actions.setEnabled(False)
            elif info.adapter_available:
                usable_count += 1
                status.setText(f"Available: {info.adapter_class_name}")
                status.setProperty("status", "ok")
            else:
                usable_count += 1
                status.setText("Not present — no-op adapter will be used")
                status.setProperty("status", "warning")

            self.agent_grid.addWidget(name_label, row_index, 0)
            self.agent_grid.addWidget(count, row_index, 1)
            self.agent_grid.addWidget(print_actions, row_index, 2)
            self.agent_grid.addWidget(status, row_index, 3)
            self._agent_rows[info.name] = _AgentRow(
                info, count, print_actions, status
            )

        if not infos:
            self.agent_grid.addWidget(
                QLabel("No agent models were found."), 1, 0, 1, 4
            )
        self.discovery_summary.setText(
            f"{usable_count} usable model{'s' if usable_count != 1 else ''} detected"
        )
        self.agents_changed.emit()

    def collect_config(
        self,
        *,
        execution_mode: str,
        speed_factor: float,
    ) -> SimulationConfig:
        agent_configs = {
            name: AgentTypeConfig(
                count=row.count.value(),
                print_agent_actions=row.print_actions.isChecked(),
            )
            for name, row in self._agent_rows.items()
            if row.info.model_available
            and row.info.adapter_error is None
            and row.count.value() > 0
        }
        config = SimulationConfig(
            focus_position=(self.focus_x_spin.value(), self.focus_y_spin.value()),
            print_middleman=self.print_middleman_check.isChecked(),
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            speed_factor=float(speed_factor),
            print_agent_actions=self.print_agent_actions_check.isChecked(),
            los=self.los_spin.value(),
            execution_mode=execution_mode,
            human_agent_enabled=self.human_enabled_check.isChecked(),
            human_agent_name=self.human_name_edit.text().strip() or "Human Player",
            agent_type_config=agent_configs,
        )
        config.validate()
        return config

    def apply_config(self, config: SimulationConfig) -> None:
        self._current_config = config
        self.width_spin.setValue(config.width)
        self.height_spin.setValue(config.height)
        self.focus_x_spin.setValue(config.focus_position[0])
        self.focus_y_spin.setValue(config.focus_position[1])
        self.los_spin.setValue(config.los)
        self.human_enabled_check.setChecked(config.human_agent_enabled)
        self.human_name_edit.setText(config.human_agent_name)
        self.human_name_edit.setEnabled(config.human_agent_enabled)
        self.print_middleman_check.setChecked(config.print_middleman)
        self.print_agent_actions_check.setChecked(config.print_agent_actions)
        self.refresh_agent_types()

    def set_runtime_locked(self, locked: bool) -> None:
        for widget in (
            self.width_spin,
            self.height_spin,
            self.focus_x_spin,
            self.focus_y_spin,
            self.los_spin,
            self.human_enabled_check,
            self.human_name_edit,
            self.print_middleman_check,
            self.print_agent_actions_check,
            self.refresh_agents_button,
        ):
            widget.setEnabled(not locked)
        self.human_name_edit.setEnabled(
            not locked and self.human_enabled_check.isChecked()
        )
        for row in self._agent_rows.values():
            usable = row.info.model_available and row.info.adapter_error is None
            row.count.setEnabled(not locked and usable)
            row.print_actions.setEnabled(not locked and usable)

    @staticmethod
    def _integer_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _clear_grid(self) -> None:
        while self.agent_grid.count():
            item = self.agent_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
