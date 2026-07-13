"""Unified shell for simulation control and advanced agent analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QStringListModel, Qt
from PyQt6.QtGui import QCloseEvent, QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QCompleter,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
)

from gui.agent_analysis_view import AgentAnalysisView
from gui.environment_view import EnvironmentView
from gui.human_input_controller import HumanInputController
from gui.jump_progress_dialog import JumpProgressDialog
from gui.mode_toggle import ExecutionModeToggle
from gui.resources import application_icon_path
from gui.multi_simulation_view import MultiSimulationRunView
from gui.simulation_config_view import SimulationConfigView
from gui.step_log_view import StepLogView
from simulation.config.models import SPEED_PRESETS, SimulationConfig


class SimulationMainWindow(QMainWindow):
    """Single-window PyQt6 UI for simulation control and explainability."""

    def __init__(self, tracer: Any, simulation: Any, parent=None) -> None:
        super().__init__(parent)
        self.tracer = tracer
        self.simulation = simulation
        self._known_jump_agents: tuple[str, ...] = ()
        self._known_production_key: tuple[str | None, tuple[str, ...]] = (
            None,
            (),
        )
        self.jump_dialog: JumpProgressDialog | None = None
        self.human_input_controller = HumanInputController(
            self.simulation,
            enabled_predicate=self._human_controls_active,
            parent=self,
        )
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self.human_input_controller)
        self.setWindowTitle("ACT-R Multi-Agent Simulation")
        icon_path = application_icon_path()
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setMinimumSize(1280, 800)

        root = QFrame(self)
        root.setObjectName("appRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 8)
        root_layout.setSpacing(10)
        self.setCentralWidget(root)

        root_layout.addWidget(self._build_header())
        root_layout.addWidget(self._build_toolbar())

        self.content_stack = QStackedWidget(root)
        simulation_tab = QFrame(self.content_stack)
        simulation_layout = QVBoxLayout(simulation_tab)
        simulation_layout.setContentsMargins(0, 0, 0, 0)
        simulation_layout.setSpacing(0)
        simulation_layout.addWidget(self._build_simulation_splitter())
        self.content_stack.addWidget(simulation_tab)

        self.analysis_view = AgentAnalysisView(
            self.simulation,
            self.content_stack,
        )
        self.content_stack.addWidget(self.analysis_view)
        self.multi_run_view = MultiSimulationRunView(
            self.simulation,
            self.content_stack,
        )
        self.multi_run_view.set_config_provider(self._collect_current_config)
        self.content_stack.addWidget(self.multi_run_view)
        root_layout.addWidget(self.content_stack, 1)
        root_layout.addWidget(self._build_bottom_navigation())

        self.config_view.reset_requested.connect(
            self._reset_default_settings
        )

        self.statusBar().showMessage(
            "Review the configuration and start the simulation."
        )
        QShortcut(
            QKeySequence(Qt.Key.Key_Space),
            self,
            activated=self._trigger_step,
        )
        QShortcut(
            QKeySequence("F11"),
            self,
            activated=self.toggle_fullscreen,
        )
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)
        QShortcut(
            QKeySequence(Qt.Key.Key_Escape),
            self,
            activated=self._leave_fullscreen,
        )
        self.refresh()

    def _human_controls_active(self) -> bool:
        """Enable movement only while the interactive Environment page is visible."""
        return (
            hasattr(self, "content_stack")
            and self.content_stack.currentIndex() == 0
            and hasattr(self, "left_tabs")
            and self.left_tabs.currentWidget() is self.environment_view
        )

    def _build_bottom_navigation(self) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName("bottomNavigation")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.addStretch(1)
        self.navigation_group = QButtonGroup(self)
        self.navigation_group.setExclusive(True)
        self.simulation_nav_button = QPushButton("Simulation", frame)
        self.analysis_nav_button = QPushButton("Agent Analysis", frame)
        self.multi_run_nav_button = QPushButton("Multi Simulation Run", frame)
        for index, button in enumerate(
            (
                self.simulation_nav_button,
                self.analysis_nav_button,
                self.multi_run_nav_button,
            )
        ):
            button.setObjectName("navigationButton")
            button.setCheckable(True)
            button.setMinimumWidth(190)
            button.setMinimumHeight(42)
            self.navigation_group.addButton(button, index)
            layout.addWidget(button)
        layout.addStretch(1)
        self.simulation_nav_button.setChecked(True)
        self.navigation_group.idClicked.connect(
            self.content_stack.setCurrentIndex
        )
        return frame

    def _build_simulation_splitter(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        self.left_tabs = QTabWidget(splitter)
        self.environment_view = EnvironmentView(self.left_tabs)
        self.config_view = SimulationConfigView(
            self.simulation.config,
            self.left_tabs,
        )
        self.left_tabs.addTab(self.environment_view, "Environment")
        self.left_tabs.addTab(self.config_view, "Configuration")
        self.left_tabs.setCurrentWidget(self.config_view)

        self.step_log_view = StepLogView(
            self.tracer,
            self.simulation,
            splitter,
        )
        splitter.addWidget(self.left_tabs)
        splitter.addWidget(self.step_log_view)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 7)
        splitter.setSizes([650, 1050])
        return splitter

    def _build_header(self) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName("header")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 12, 16, 12)

        title = QLabel("ACT-R Multi-Agent Simulation")
        title.setObjectName("appTitle")
        layout.addWidget(title)
        layout.addStretch(1)

        self.state_value = QLabel("Not started")
        self.state_value.setObjectName("statusValue")
        self.time_value = QLabel("Time 0.00")
        self.time_value.setObjectName("statusValue")
        self.agent_value = QLabel("0 agents")
        self.agent_value.setObjectName("statusValue")
        layout.addWidget(self.state_value)
        layout.addWidget(self.time_value)
        layout.addWidget(self.agent_value)
        return frame

    def _build_toolbar(self) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName("toolbar")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)

        execution_group = QFrame(frame)
        execution_group.setObjectName("controlGroup")
        execution_layout = QVBoxLayout(execution_group)
        execution_layout.setContentsMargins(8, 6, 8, 6)
        execution_layout.setSpacing(6)

        execution_buttons = QHBoxLayout()
        execution_buttons.setSpacing(7)
        self.start_button = QPushButton("Start Simulation")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._start_simulation)
        execution_buttons.addWidget(self.start_button)
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self._toggle_pause)
        execution_buttons.addWidget(self.pause_button)
        self.step_button = QPushButton("Step [Space]")
        self.step_button.clicked.connect(self._trigger_step)
        execution_buttons.addWidget(self.step_button)
        execution_buttons.addStretch(1)
        execution_layout.addLayout(execution_buttons)

        runtime_controls = QHBoxLayout()
        runtime_controls.setSpacing(7)
        self.mode_toggle = ExecutionModeToggle(
            self.simulation.config.execution_mode, execution_group
        )
        self.mode_toggle.setMinimumWidth(190)
        self.mode_toggle.mode_changed.connect(self._mode_changed)
        runtime_controls.addWidget(self.mode_toggle)
        speed_caption = QLabel("Speed")
        speed_caption.setObjectName("muted")
        runtime_controls.addWidget(speed_caption)
        self.speed_slider = QSlider(Qt.Orientation.Horizontal, execution_group)
        self.speed_slider.setRange(0, len(SPEED_PRESETS) - 1)
        self.speed_slider.setSingleStep(1)
        self.speed_slider.setPageStep(1)
        self.speed_slider.setTickInterval(1)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setMinimumWidth(130)
        self.speed_slider.setMaximumWidth(190)
        self.speed_slider.setToolTip(
            "1/4 Realtime · 1/2 Realtime · Realtime · 2x Realtime · ASAP"
        )
        self.speed_value = QLabel("")
        self.speed_value.setObjectName("statusValue")
        self.speed_value.setMinimumWidth(94)
        self.speed_slider.setValue(
            self._speed_index(self.simulation.config.speed_factor)
        )
        self.speed_slider.valueChanged.connect(self._speed_changed)
        self._update_speed_label()
        runtime_controls.addWidget(self.speed_slider, 1)
        runtime_controls.addWidget(self.speed_value)
        execution_layout.addLayout(runtime_controls)
        layout.addWidget(execution_group, 1)

        jump_group = QFrame(frame)
        jump_group.setObjectName("controlGroup")
        jump_layout = QVBoxLayout(jump_group)
        jump_layout.setContentsMargins(8, 6, 8, 6)
        jump_layout.setSpacing(5)
        jump_title = QLabel("Production Jump")
        jump_title.setObjectName("groupTitle")
        jump_layout.addWidget(jump_title)
        jump_controls = QHBoxLayout()
        jump_controls.setSpacing(7)
        agent_label = QLabel("Agent")
        agent_label.setObjectName("muted")
        jump_controls.addWidget(agent_label)
        self.jump_agent_combo = QComboBox(jump_group)
        self.jump_agent_combo.setMinimumWidth(135)
        self.jump_agent_combo.currentIndexChanged.connect(
            self._jump_agent_changed
        )
        jump_controls.addWidget(self.jump_agent_combo)
        production_label = QLabel("Production")
        production_label.setObjectName("muted")
        jump_controls.addWidget(production_label)
        self.jump_input = QComboBox(jump_group)
        self.jump_input.setEditable(True)
        self.jump_input.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.jump_input.setMinimumWidth(180)
        self.jump_input.setMaximumWidth(280)
        self.production_completion_model = QStringListModel(self)
        self.production_completer = QCompleter(
            self.production_completion_model, self
        )
        self.production_completer.setCaseSensitivity(
            Qt.CaseSensitivity.CaseInsensitive
        )
        self.production_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.jump_input.setCompleter(self.production_completer)
        line_edit = self.jump_input.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText("Select an agent first")
            line_edit.returnPressed.connect(self._trigger_jump)
        self.jump_input.setEnabled(False)
        jump_controls.addWidget(self.jump_input, 1)
        self.jump_button = QPushButton("Jump")
        self.jump_button.clicked.connect(self._trigger_jump)
        self.jump_button.setEnabled(False)
        jump_controls.addWidget(self.jump_button)
        jump_layout.addLayout(jump_controls)
        layout.addWidget(jump_group, 1)

        self.export_button = QPushButton("Export History ZIP")
        self.export_button.setObjectName("exportButton")
        self.export_button.setMinimumHeight(72)
        self.export_button.clicked.connect(self._export_history)
        layout.addWidget(self.export_button)
        return frame

    def set_environment(self, environment: Any) -> None:
        self.environment_view.set_environment(environment)
        self.refresh()

    def refresh(self) -> None:
        self.environment_view.refresh()
        self.step_log_view.refresh()
        self.analysis_view.refresh()
        self.multi_run_view.refresh()
        initialized = bool(getattr(self.simulation, "initialized", False))
        run_state = str(
            getattr(self.simulation, "run_state", "not_started")
        )
        mode = (
            str(getattr(self.simulation, "execution_mode", "single"))
            if initialized
            else self.mode_toggle.mode()
        )
        jumping = bool(getattr(self.simulation, "jumping", False))

        self.time_value.setText(
            f"Time {float(getattr(self.simulation, 'global_sim_time', 0.0)):.2f}"
        )
        agent_count = len(getattr(self.simulation, "spatial_agents", []))
        self.agent_value.setText(
            f"{agent_count} agent{'s' if agent_count != 1 else ''}"
        )
        self.state_value.setText(self._state_label(run_state, mode))

        self.start_button.setText(
            "Restart Simulation" if initialized else "Start Simulation"
        )
        self.pause_button.setEnabled(
            initialized and run_state not in {"finished", "stopped"}
        )
        self.pause_button.setText(
            "Resume"
            if run_state == "paused"
            else ("Stop Jump" if jumping else "Pause")
        )
        self.step_button.setEnabled(
            initialized
            and run_state == "running"
            and mode == "single"
            and not jumping
        )
        self.export_button.setEnabled(initialized)

        if initialized and self.mode_toggle.mode() != mode:
            self.mode_toggle.set_mode(mode, emit=False)

        self._refresh_jump_options()
        self._update_jump_control_state()

        if self.jump_dialog is not None and self.jump_dialog.isVisible():
            self.jump_dialog.refresh()

        if jumping:
            target = getattr(self.simulation, "jump_target", "") or ""
            self.statusBar().showMessage(
                f"Running until production '{target}' fires."
            )
        elif getattr(self.simulation, "last_error", None):
            self.statusBar().showMessage(
                str(self.simulation.last_error),
                10000,
            )

    def _start_simulation(self) -> None:
        if self.simulation.initialized:
            answer = QMessageBox.question(
                self,
                "Restart Simulation",
                "Restarting discards the current in-memory history. "
                "Export it first if needed. Continue?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            config = self._collect_current_config()
            self.simulation.start_simulation(config)
            self.left_tabs.setCurrentWidget(self.environment_view)
            self.content_stack.setCurrentIndex(0)
            self.environment_view.canvas.setFocus()
            self.simulation_nav_button.setChecked(True)
            message = "Simulation started. Space executes one step in Step mode."
            if config.human_agent_enabled:
                message += " Move the human agent with WASD or the arrow keys."
            self.statusBar().showMessage(message, 8000)
        except Exception as exc:
            self.report_error(
                "Simulation could not be started",
                str(exc),
            )
        self.refresh()

    def _toggle_pause(self) -> None:
        if self.simulation.run_state == "paused":
            self.simulation.resume()
            self.statusBar().showMessage("Simulation resumed", 2500)
        else:
            self.simulation.pause()
            self.statusBar().showMessage("Simulation paused", 2500)
        self.refresh()

    def _trigger_step(self) -> None:
        focus = self.focusWidget()
        if isinstance(focus, (QLineEdit, QComboBox, QSlider)) or (
            focus is not None and self.config_view.isAncestorOf(focus)
        ):
            return
        try:
            if self.simulation.step_once():
                self.statusBar().showMessage("Step executed", 1800)
            elif self.simulation.initialized:
                self.statusBar().showMessage(
                    "Step is available only while Step mode is active "
                    "and running.",
                    3500,
                )
        except Exception as exc:
            self.report_error("Simulation error", str(exc))
        self.refresh()

    def _trigger_jump(self) -> None:
        agent_name = self.jump_agent_combo.currentData()
        if not agent_name:
            QMessageBox.warning(
                self,
                "Select an agent",
                "Select the runtime agent before choosing a production.",
            )
            return
        target = self.jump_input.currentText().strip()
        available = self.simulation.get_production_names(agent_name)
        canonical = next(
            (
                name
                for name in available
                if name.casefold() == target.casefold()
            ),
            None,
        )
        if canonical is None:
            QMessageBox.warning(
                self,
                "Unknown production",
                "Choose a production from the selected agent's "
                "autocomplete list.",
            )
            return
        target = canonical

        analysis = self.analysis_view.analysis_for_agent(agent_name)
        if analysis is None:
            answer = QMessageBox.warning(
                self,
                "Reachability unavailable",
                "The static agent analysis could not be loaded. The jump "
                "can still run, but no prerequisite path can be verified.",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        else:
            production = analysis.production(target)
            path = analysis.path_to_production(target)
            if production is None or not production.reachable or path is None:
                answer = QMessageBox.warning(
                    self,
                    "Production may be unreachable",
                    "Agent Analysis found no statically reachable path from "
                    "the initial buffer state to this production. Adapter "
                    "side effects may still make it reachable. Continue?",
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return

        if analysis is not None:
            if self.jump_dialog is not None:
                self.jump_dialog.close()
            self.jump_dialog = JumpProgressDialog(
                analysis=analysis,
                tracer=self.tracer,
                agent_name=agent_name,
                target_production=target,
                start_record_index=self._jump_progress_start_index(
                    agent_name,
                    target,
                ),
                parent=self,
            )
            self.jump_dialog.show()
            self.jump_dialog.raise_()

        self.simulation.start_jump(target, agent_name)
        self.refresh()

    def _jump_progress_start_index(
        self,
        agent_name: str,
        target_production: str,
    ) -> int:
        """Start after the last target firing, retaining the current cycle path."""
        records = list(getattr(self.tracer, "records", []))
        last_target_index = -1
        for index, record in enumerate(records):
            if str(record.get("agent_name", "")) != agent_name:
                continue
            if str(record.get("type", "")).upper() != "PROCEDURAL":
                continue
            event = str(record.get("event", "")).strip()
            prefix = "RULE FIRED:"
            fired = (
                event[len(prefix) :].strip()
                if event.upper().startswith(prefix)
                else event
            )
            if fired.casefold() == target_production.casefold():
                last_target_index = index
        return last_target_index + 1

    def _mode_changed(self, mode: str) -> None:
        if self.simulation.initialized:
            self.simulation.set_execution_mode(mode)
            self.statusBar().showMessage(
                "Mode: Automatic" if mode == "automatic" else "Mode: Step",
                2500,
            )
        self._persist_current_settings()
        self.refresh()

    def _speed_changed(self, _index: int) -> None:
        self._update_speed_label()
        speed_factor = self._selected_speed_factor()
        if self.simulation.initialized:
            self.simulation.set_speed_factor(speed_factor)
        self._persist_current_settings()

    def _update_speed_label(self) -> None:
        label, _ = SPEED_PRESETS[self.speed_slider.value()]
        self.speed_value.setText(label)

    def _jump_agent_changed(self) -> None:
        self._populate_productions_for_agent(
            self.jump_agent_combo.currentData()
        )
        self._update_jump_control_state()

    def _update_jump_control_state(self) -> None:
        initialized = bool(getattr(self.simulation, "initialized", False))
        jumping = bool(getattr(self.simulation, "jumping", False))
        selected_agent = self.jump_agent_combo.currentData()
        productions_available = bool(
            self.simulation.get_production_names(selected_agent)
            if selected_agent
            else []
        )
        jump_enabled = (
            initialized
            and bool(selected_agent)
            and productions_available
            and not jumping
        )
        self.jump_button.setEnabled(jump_enabled)
        self.jump_input.setEnabled(jump_enabled)
        self.jump_agent_combo.setEnabled(initialized and not jumping)

    def _export_history(self) -> None:
        suggested = Path.home() / "simulation_history.zip"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Complete Simulation History",
            str(suggested),
            "ZIP archive (*.zip)",
        )
        if not path:
            return
        try:
            destination = self.simulation.export_history(path)
            self.statusBar().showMessage(
                f"History exported to {destination}",
                8000,
            )
        except Exception as exc:
            self.report_error(
                "History could not be exported",
                str(exc),
            )

    def _refresh_jump_options(self) -> None:
        grouped: dict[str, list[str]] = {}
        for agent in self.simulation.agent_list:
            agent_type = str(
                getattr(agent, "actr_agent_type_name", "Unknown")
            )
            grouped.setdefault(agent_type, []).append(str(agent.name))
        signature = tuple(
            (agent_type, tuple(sorted(names, key=str.lower)))
            for agent_type, names in sorted(
                grouped.items(), key=lambda item: item[0].lower()
            )
        )
        flat_names = tuple(
            name for _, names in signature for name in names
        )
        if flat_names != self._known_jump_agents:
            current = self.jump_agent_combo.currentData()
            self.jump_agent_combo.blockSignals(True)
            self.jump_agent_combo.clear()
            self.jump_agent_combo.addItem("Select agent…", None)
            for agent_type, names in signature:
                self.jump_agent_combo.addItem(agent_type, None)
                model = self.jump_agent_combo.model()
                item = getattr(model, "item", lambda _index: None)(
                    self.jump_agent_combo.count() - 1
                )
                if item is not None:
                    item.setEnabled(False)
                for name in names:
                    self.jump_agent_combo.addItem(f"  {name}", name)
            index = self.jump_agent_combo.findData(current)
            self.jump_agent_combo.setCurrentIndex(
                index if index > 0 else 0
            )
            self.jump_agent_combo.blockSignals(False)
            self._known_jump_agents = flat_names
        self._populate_productions_for_agent(
            self.jump_agent_combo.currentData()
        )

    def _populate_productions_for_agent(
        self,
        agent_name: str | None,
    ) -> None:
        productions = tuple(
            self.simulation.get_production_names(agent_name)
            if agent_name
            else []
        )
        key = (agent_name, productions)
        if key == self._known_production_key:
            return
        current_text = self.jump_input.currentText().strip()
        self.jump_input.blockSignals(True)
        self.jump_input.clear()
        self.jump_input.addItems(list(productions))
        if current_text in productions:
            self.jump_input.setCurrentText(current_text)
        elif productions:
            self.jump_input.setCurrentIndex(0)
        self.jump_input.blockSignals(False)
        line_edit = self.jump_input.lineEdit()
        if line_edit is not None:
            line_edit.setPlaceholderText(
                "Select or type a production"
                if agent_name
                else "Select an agent first"
            )
        self.production_completion_model.setStringList(list(productions))
        self._known_production_key = key

    def _reset_default_settings(self) -> None:
        if self.simulation.initialized:
            answer = QMessageBox.question(
                self,
                "Reset Settings",
                "Reset the controls to their defaults? The running "
                "simulation is not restarted until you press Restart Simulation.",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        config = self.simulation.reset_settings()
        self.config_view.apply_config(config)
        self.mode_toggle.set_mode(config.execution_mode, emit=False)
        self.speed_slider.blockSignals(True)
        self.speed_slider.setValue(
            self._speed_index(config.speed_factor)
        )
        self.speed_slider.blockSignals(False)
        self._update_speed_label()
        if self.simulation.initialized:
            self.simulation.set_execution_mode(config.execution_mode)
            self.simulation.set_speed_factor(config.speed_factor)
        self.simulation.save_settings(config)
        self.statusBar().showMessage(
            "Default settings restored.",
            4000,
        )

    def _collect_current_config(self) -> SimulationConfig:
        return self.config_view.collect_config(
            execution_mode=self.mode_toggle.mode(),
            speed_factor=self._selected_speed_factor(),
        )

    def _persist_current_settings(self) -> None:
        if not hasattr(self, "config_view"):
            return
        try:
            config = self._collect_current_config()
        except Exception:
            return
        self.simulation.save_settings(config)

    def _selected_speed_factor(self) -> float:
        return float(SPEED_PRESETS[self.speed_slider.value()][1])

    @staticmethod
    def _speed_index(speed_factor: float) -> int:
        for index, (_, value) in enumerate(SPEED_PRESETS):
            if float(value) == float(speed_factor):
                return index
        return 2

    @staticmethod
    def _state_label(run_state: str, mode: str) -> str:
        mapping = {
            "not_started": "Not started",
            "running": "Automatic" if mode == "automatic" else "Step ready",
            "paused": "Paused",
            "jumping": "Production jump",
            "finished": "Finished",
            "stopped": "Stopped",
        }
        return mapping.get(run_state, run_state)

    def report_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(message, 10000)

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def _leave_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self.human_input_controller)
        self._persist_current_settings()
        thread = getattr(self.multi_run_view, "_thread", None)
        if thread is not None and thread.isRunning():
            thread.requestInterruption()
        self.simulation.stop_execution()
        event.accept()
