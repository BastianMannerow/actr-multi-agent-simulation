"""Configuration and asynchronous execution of multi-simulation batches."""

from __future__ import annotations

import multiprocessing
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import QSettings, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from simulation.discovery.agent_discovery import AgentDiscovery
from simulation.config.models import (
    AgentTypeConfig,
    ENVIRONMENT_MODES,
    SPEED_PRESETS,
    SimulationConfig,
    VIRTUAL_LEVELS,
)
from simulation.batch.multi_run import (
    MultiRunBatch,
    MultiRunScenario,
    bundle_multi_run_results,
    cleanup_batch_temp_directory,
    create_batch_temp_directory,
    execute_multi_run_task,
    recommended_worker_count,
)
from simulation.inspection.source_analysis import AgentSourceAnalyzer
from simulation.world.level_builder import level_dimensions


@dataclass(slots=True)
class _ScenarioWidgets:
    name: QLineEdit
    repetitions: QSpinBox
    scheduling: QComboBox
    environment: QComboBox
    speed: QComboBox
    end_condition: QComboBox
    end_value: QLineEdit
    settings_button: QPushButton
    config: SimulationConfig


class SimulationSettingsDialog(QDialog):
    """Edit one headless virtual-matrix scenario."""

    def __init__(self, config: SimulationConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Scenario Simulation Settings")
        self.resize(680, 760)
        self._source = config.without_human_agent()
        self._agent_rows: dict[str, tuple[QSpinBox, QCheckBox]] = {}

        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget(scroll)
        content_layout = QVBoxLayout(content)

        environment_group = QGroupBox("Environment", content)
        form = QFormLayout(environment_group)
        self.environment_mode_combo = QComboBox(environment_group)
        for label, value in ENVIRONMENT_MODES:
            self.environment_mode_combo.addItem(label, value)
        self.environment_mode_combo.setCurrentIndex(
            max(0, self.environment_mode_combo.findData(self._source.environment_mode))
        )
        self.virtual_level_combo = QComboBox(environment_group)
        for label, value in VIRTUAL_LEVELS:
            self.virtual_level_combo.addItem(label, value)
        self.virtual_level_combo.setCurrentIndex(
            max(0, self.virtual_level_combo.findData(self._source.virtual_level))
        )
        self.matrix_size_label = QLabel(environment_group)
        self.matrix_size_label.setObjectName("muted")
        self.focus_x_spin = self._spin(-10000, 10000, self._source.focus_position[0])
        self.focus_y_spin = self._spin(-10000, 10000, self._source.focus_position[1])
        self.los_spin = self._spin(0, 1000, self._source.los)
        form.addRow("Backend", self.environment_mode_combo)
        form.addRow("Level", self.virtual_level_combo)
        form.addRow("Matrix size", self.matrix_size_label)
        form.addRow("Focus X", self.focus_x_spin)
        form.addRow("Focus Y", self.focus_y_spin)
        form.addRow("Line of sight", self.los_spin)
        content_layout.addWidget(environment_group)


        logging_group = QGroupBox("Logging", content)
        logging_layout = QVBoxLayout(logging_group)
        self.middleman_check = QCheckBox("Print Middleman actions")
        self.middleman_check.setChecked(self._source.print_middleman)
        self.agent_actions_check = QCheckBox("Print agent actions by default")
        self.agent_actions_check.setChecked(self._source.print_agent_actions)
        self.experimental_pyactr_boost_check = QCheckBox(
            "Experimental pyactr performance boost"
        )
        self.experimental_pyactr_boost_check.setChecked(
            self._source.experimental_pyactr_performance_boost
        )
        self.experimental_pyactr_boost_check.setToolTip(
            "Applies the reversible pyactr_overrides package inside each "
            "worker process. Disable it to use unmodified pyactr 0.3.2."
        )
        logging_layout.addWidget(self.middleman_check)
        logging_layout.addWidget(self.agent_actions_check)
        logging_layout.addWidget(self.experimental_pyactr_boost_check)
        content_layout.addWidget(logging_group)

        agents_group = QGroupBox("Agents", content)
        agents_grid = QGridLayout(agents_group)
        agents_grid.addWidget(QLabel("Type"), 0, 0)
        agents_grid.addWidget(QLabel("Count"), 0, 1)
        agents_grid.addWidget(QLabel("Log actions"), 0, 2)
        for row, info in enumerate(AgentDiscovery().discover(), start=1):
            count = self._spin(0, 999, 0)
            print_actions = QCheckBox()
            current = self._source.agent_type_config.get(info.name)
            count.setValue(current.count if current is not None else 1)
            print_actions.setChecked(
                current.print_agent_actions if current is not None else self._source.print_agent_actions
            )
            usable = info.model_available and info.adapter_error is None
            count.setEnabled(usable)
            print_actions.setEnabled(usable)
            agents_grid.addWidget(QLabel(info.name), row, 0)
            agents_grid.addWidget(count, row, 1)
            agents_grid.addWidget(print_actions, row, 2)
            self._agent_rows[info.name] = (count, print_actions)
        content_layout.addWidget(agents_group)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.environment_mode_combo.currentIndexChanged.connect(self._update_environment_controls)
        self.virtual_level_combo.currentIndexChanged.connect(self._update_environment_controls)
        self._update_environment_controls()

    def _update_environment_controls(self) -> None:
        self.virtual_level_combo.setEnabled(True)
        level = str(self.virtual_level_combo.currentData())
        height, width = level_dimensions(level)
        self.matrix_size_label.setText(f"{width} × {height} (defined by Level Builder)")

    def configuration(self) -> SimulationConfig:
        config = SimulationConfig(
            focus_position=(self.focus_x_spin.value(), self.focus_y_spin.value()),
            print_middleman=self.middleman_check.isChecked(),
            speed_factor=self._source.speed_factor,
            print_agent_actions=self.agent_actions_check.isChecked(),
            experimental_pyactr_performance_boost=(
                self.experimental_pyactr_boost_check.isChecked()
            ),
            los=self.los_spin.value(),
            execution_mode="single",
            environment_mode=str(self.environment_mode_combo.currentData()),
            virtual_level=str(self.virtual_level_combo.currentData()),
            human_agent_enabled=False,
            agent_type_config={
                name: AgentTypeConfig(
                    count=count.value(),
                    print_agent_actions=print_actions.isChecked(),
                )
                for name, (count, print_actions) in self._agent_rows.items()
                if count.isEnabled() and count.value() > 0
            },
        )
        config.validate()
        return config

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin



class MultiRunThread(QThread):
    """Execute scheduling barriers and process-isolated simulations."""

    progress_changed = pyqtSignal(int, int, float, str)
    run_finished = pyqtSignal(object)
    batch_finished = pyqtSignal(str, object)
    batch_failed = pyqtSignal(str)

    def __init__(self, batch: MultiRunBatch, parent=None) -> None:
        super().__init__(parent)
        self.batch = batch

    def run(self) -> None:
        temp_root = create_batch_temp_directory()
        results: list[dict[str, Any]] = []
        started = time.perf_counter()
        durations: list[float] = []
        tasks = self.batch.expanded_tasks()
        total = len(tasks)
        configured_workers = self.batch.max_workers or recommended_worker_count(total)
        try:
            index = 0
            while index < total:
                if self.isInterruptionRequested():
                    break
                current = tasks[index]
                if current["scheduling"] == "sequential":
                    group = [current]
                    index += 1
                    workers = 1
                else:
                    group = []
                    while index < total and tasks[index]["scheduling"] == "parallel":
                        group.append(tasks[index])
                        index += 1
                    workers = min(configured_workers, len(group))
                group_results = self._execute_group(group, workers, temp_root)
                for result in group_results:
                    results.append(result)
                    durations.append(float(result.get("duration_seconds", 0.0)))
                    self.run_finished.emit(result)
                    completed = len(results)
                    eta = self._estimate_eta(
                        durations=durations,
                        elapsed=max(0.001, time.perf_counter() - started),
                        remaining=total - completed,
                        active_workers=workers,
                        maximum_workers=configured_workers,
                    )
                    self.progress_changed.emit(
                        completed,
                        total,
                        eta,
                        f"Completed run {completed} of {total}",
                    )

            if len(results) < total:
                for task in tasks[len(results) :]:
                    results.append(
                        {
                            **task,
                            "status": "cancelled",
                            "duration_seconds": 0.0,
                            "simulation_time": 0.0,
                            "event_count": 0,
                            "target_reached": False,
                            "error": "Batch cancellation requested.",
                            "history_path": None,
                        }
                    )
            destination = bundle_multi_run_results(self.batch, results)
            self.batch_finished.emit(str(destination), results)
        except BaseException as exc:
            self.batch_failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            cleanup_batch_temp_directory(temp_root)

    def _execute_group(
        self,
        tasks: list[dict[str, Any]],
        workers: int,
        temp_root: str,
    ) -> list[dict[str, Any]]:
        if not tasks:
            return []
        context = multiprocessing.get_context("spawn")
        kwargs: dict[str, Any] = {
            "max_workers": max(1, workers),
            "mp_context": context,
        }
        try:
            executor = ProcessPoolExecutor(max_tasks_per_child=1, **kwargs)
        except TypeError:  # Python versions before max_tasks_per_child
            executor = ProcessPoolExecutor(**kwargs)
        output: list[dict[str, Any]] = []
        with executor:
            future_to_task = {
                executor.submit(execute_multi_run_task, task, temp_root): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    output.append(future.result())
                except BaseException as exc:
                    output.append(
                        {
                            **task,
                            "status": "crashed",
                            "duration_seconds": 0.0,
                            "simulation_time": 0.0,
                            "event_count": 0,
                            "target_reached": False,
                            "error": f"Worker process failed: {type(exc).__name__}: {exc}",
                            "history_path": None,
                        }
                    )
        return sorted(output, key=lambda row: int(row.get("run_number", 0)))

    @staticmethod
    def _estimate_eta(
        *,
        durations: list[float],
        elapsed: float,
        remaining: int,
        active_workers: int,
        maximum_workers: int,
    ) -> float:
        if remaining <= 0 or not durations:
            return 0.0
        positive = [value for value in durations if value > 0]
        mean_duration = statistics.fmean(positive) if positive else 0.0
        if len(positive) == 1:
            effective_parallelism = max(1.0, float(active_workers))
        else:
            effective_parallelism = max(
                1.0,
                min(float(maximum_workers), sum(positive) / elapsed),
            )
        return max(0.0, mean_duration * remaining / effective_parallelism)


class MultiSimulationRunView(QFrame):
    """Batch scenario editor with progress, ETA, and aggregate export."""

    def __init__(self, simulation: Any, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("panel")
        self.simulation = simulation
        self.settings = QSettings()
        self._config_provider: Callable[[], SimulationConfig] | None = None
        self._rows: list[_ScenarioWidgets] = []
        self._thread: MultiRunThread | None = None
        self._known_productions = self._discover_productions()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Multi Simulation Run")
        title.setObjectName("sectionTitle")
        self.hardware_label = QLabel(self._hardware_text())
        self.hardware_label.setObjectName("muted")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.hardware_label)
        outer.addLayout(header)

        controls = QHBoxLayout()
        self.add_button = QPushButton("Add Scenario")
        self.duplicate_button = QPushButton("Duplicate Selected")
        self.remove_button = QPushButton("Remove Selected")
        self.current_settings_button = QPushButton("Use Current Settings")
        self.add_button.clicked.connect(self._add_default_scenario)
        self.duplicate_button.clicked.connect(self._duplicate_selected)
        self.remove_button.clicked.connect(self._remove_selected)
        self.current_settings_button.clicked.connect(self._apply_current_settings)
        controls.addWidget(self.add_button)
        controls.addWidget(self.duplicate_button)
        controls.addWidget(self.remove_button)
        controls.addWidget(self.current_settings_button)
        controls.addStretch(1)
        outer.addLayout(controls)

        self.scenario_table = QTableWidget(0, 8, self)
        self.scenario_table.setHorizontalHeaderLabels(
            [
                "Scenario",
                "Repetitions",
                "Scheduling",
                "Environment",
                "Speed",
                "End condition",
                "Time / production",
                "Simulation settings",
            ]
        )
        self.scenario_table.verticalHeader().setDefaultSectionSize(48)
        self.scenario_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.scenario_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        header_view = self.scenario_table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3, 4, 5, 6, 7):
            header_view.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        outer.addWidget(self.scenario_table, 2)

        note = QLabel(
            "Consecutive scenarios marked Parallel run together up to the hardware-sensitive worker limit. "
            "A Sequential scenario acts as a barrier and runs alone. Each child process is isolated, so a crash is recorded without stopping other runs."
        )
        note.setWordWrap(True)
        note.setObjectName("muted")
        outer.addWidget(note)

        batch_group = QGroupBox("Batch Output and Resource Limits", self)
        batch_layout = QGridLayout(batch_group)
        self.output_edit = QLineEdit(
            str(
                self.settings.value(
                    "multi_run/output_path",
                    str(Path.home() / "multi_simulation_history.zip"),
                )
            )
        )
        self.output_edit.editingFinished.connect(self._persist_output_path)
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_output)
        self.auto_workers_check = QCheckBox("Automatic hardware-sensitive workers")
        self.auto_workers_check.setChecked(True)
        self.worker_spin = QSpinBox()
        self.worker_spin.setRange(1, max(1, recommended_worker_count() * 2))
        self.worker_spin.setValue(recommended_worker_count())
        self.worker_spin.setEnabled(False)
        self.auto_workers_check.toggled.connect(
            lambda checked: self.worker_spin.setEnabled(not checked)
        )
        self.max_events_spin = QSpinBox()
        self.max_events_spin.setRange(100, 100_000_000)
        self.max_events_spin.setValue(100_000)
        batch_layout.addWidget(QLabel("Aggregate ZIP"), 0, 0)
        batch_layout.addWidget(self.output_edit, 0, 1)
        batch_layout.addWidget(browse_button, 0, 2)
        batch_layout.addWidget(self.auto_workers_check, 1, 0, 1, 2)
        batch_layout.addWidget(self.worker_spin, 1, 2)
        batch_layout.addWidget(QLabel("Safety limit per run (events)"), 2, 0)
        batch_layout.addWidget(self.max_events_spin, 2, 1)
        outer.addWidget(batch_group)

        run_controls = QHBoxLayout()
        self.start_button = QPushButton("Start Multi Simulation Run")
        self.start_button.setObjectName("primaryButton")
        self.start_button.clicked.connect(self._start_batch)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Not started")
        self.eta_label = QLabel("ETA —")
        self.eta_label.setObjectName("statusValue")
        run_controls.addWidget(self.start_button)
        run_controls.addWidget(self.progress, 1)
        run_controls.addWidget(self.eta_label)
        outer.addLayout(run_controls)

        self.result_table = QTableWidget(0, 6, self)
        self.result_table.setHorizontalHeaderLabels(
            ["Run", "Scenario", "Status", "Simulation time", "Wall time", "Error"]
        )
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        outer.addWidget(self.result_table, 1)

        self._add_default_scenario()

    def set_config_provider(self, provider: Callable[[], SimulationConfig]) -> None:
        self._config_provider = provider

    def refresh(self) -> None:
        self.hardware_label.setText(self._hardware_text())

    def _add_default_scenario(self) -> None:
        config = self._current_config()
        self._insert_scenario(
            MultiRunScenario(
                name=f"Scenario {len(self._rows) + 1}",
                repetitions=1,
                scheduling="parallel",
                speed_factor=-1.0,
                end_condition="simulation_time",
                end_value="10.0",
                config=config,
            )
        )

    def _insert_scenario(self, scenario: MultiRunScenario) -> None:
        row_index = self.scenario_table.rowCount()
        self.scenario_table.insertRow(row_index)
        name = QLineEdit(scenario.name)
        repetitions = QSpinBox()
        repetitions.setRange(1, 100_000)
        repetitions.setValue(scenario.repetitions)
        scheduling = QComboBox()
        scheduling.addItem("Parallel", "parallel")
        scheduling.addItem("Sequential", "sequential")
        scheduling.setCurrentIndex(max(0, scheduling.findData(scenario.scheduling)))
        environment = QComboBox()
        for label, value in ENVIRONMENT_MODES:
            environment.addItem(label, value)
        environment.setCurrentIndex(max(0, environment.findData(scenario.config.environment_mode)))
        speed = QComboBox()
        for label, value in SPEED_PRESETS:
            speed.addItem(label, value)
        speed.setCurrentIndex(max(0, speed.findData(scenario.speed_factor)))
        end_condition = QComboBox()
        end_condition.addItem("Simulation time", "simulation_time")
        end_condition.addItem("Production fired", "production")
        end_condition.setCurrentIndex(max(0, end_condition.findData(scenario.end_condition)))
        end_value = QLineEdit(scenario.end_value)
        settings_button = QPushButton(self._settings_summary(scenario.config))
        widgets = _ScenarioWidgets(
            name=name,
            repetitions=repetitions,
            scheduling=scheduling,
            environment=environment,
            speed=speed,
            end_condition=end_condition,
            end_value=end_value,
            settings_button=settings_button,
            config=scenario.config.without_human_agent(),
        )
        settings_button.clicked.connect(lambda _checked=False, row=widgets: self._edit_settings(row))
        environment.currentIndexChanged.connect(
            lambda _index, row=widgets: self._environment_changed(row)
        )
        self._environment_changed(widgets)
        end_condition.currentIndexChanged.connect(
            lambda _index, row=widgets: self._update_end_value_hint(row)
        )
        for column, widget in enumerate(
            (name, repetitions, scheduling, environment, speed, end_condition, end_value, settings_button)
        ):
            self.scenario_table.setCellWidget(row_index, column, widget)
        self._rows.append(widgets)
        self._update_end_value_hint(widgets)
        self.scenario_table.selectRow(row_index)

    def _duplicate_selected(self) -> None:
        index = self.scenario_table.currentRow()
        if index < 0 or index >= len(self._rows):
            return
        scenario = self._scenario_from_widgets(self._rows[index])
        scenario.name = f"{scenario.name} Copy"
        self._insert_scenario(scenario)

    def _remove_selected(self) -> None:
        index = self.scenario_table.currentRow()
        if index < 0 or index >= len(self._rows):
            return
        self.scenario_table.removeRow(index)
        self._rows.pop(index)

    def _apply_current_settings(self) -> None:
        index = self.scenario_table.currentRow()
        if index < 0 or index >= len(self._rows):
            return
        row = self._rows[index]
        row.config = self._current_config()
        row.environment.setCurrentIndex(max(0, row.environment.findData(row.config.environment_mode)))
        self._environment_changed(row)
        row.settings_button.setText(self._settings_summary(row.config))

    def _edit_settings(self, row: _ScenarioWidgets) -> None:
        dialog = SimulationSettingsDialog(row.config, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            row.config = dialog.configuration()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid simulation settings", str(exc))
            return
        row.environment.setCurrentIndex(max(0, row.environment.findData(row.config.environment_mode)))
        self._environment_changed(row)
        row.settings_button.setText(self._settings_summary(row.config))

    def _environment_changed(self, row: _ScenarioWidgets) -> None:
        mode = str(row.environment.currentData())
        row.config.environment_mode = mode
        row.scheduling.setEnabled(True)
        row.settings_button.setText(self._settings_summary(row.config))

    def _update_end_value_hint(self, row: _ScenarioWidgets) -> None:
        if row.end_condition.currentData() == "production":
            row.end_value.setPlaceholderText("Production name (any agent)")
            if row.end_value.text().strip() in {"", "10.0"} and self._known_productions:
                row.end_value.setText(self._known_productions[0])
        else:
            row.end_value.setPlaceholderText("Simulation time, e.g. 10.0")
            if not row.end_value.text().strip() or row.end_value.text() in self._known_productions:
                row.end_value.setText("10.0")

    def _start_batch(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        try:
            scenarios = [self._scenario_from_widgets(row) for row in self._rows]
            batch = MultiRunBatch(
                scenarios=scenarios,
                output_path=self.output_edit.text().strip(),
                max_workers=0 if self.auto_workers_check.isChecked() else self.worker_spin.value(),
                max_events_per_run=self.max_events_spin.value(),
            )
            batch.validate()
        except Exception as exc:
            QMessageBox.critical(self, "Multi-run configuration error", str(exc))
            return

        self._persist_output_path()
        self.result_table.setRowCount(0)
        self.progress.setValue(0)
        self.progress.setFormat("Starting…")
        self.eta_label.setText("ETA calculating…")
        self.start_button.setEnabled(False)
        self._set_editor_enabled(False)
        self._thread = MultiRunThread(batch, self)
        self._thread.progress_changed.connect(self._progress_changed)
        self._thread.run_finished.connect(self._append_result)
        self._thread.batch_finished.connect(self._batch_finished)
        self._thread.batch_failed.connect(self._batch_failed)
        self._thread.finished.connect(lambda: self.start_button.setEnabled(True))
        self._thread.start()

    def _progress_changed(self, completed: int, total: int, eta: float, text: str) -> None:
        percent = round(completed / max(1, total) * 100)
        self.progress.setValue(percent)
        self.progress.setFormat(f"{percent}% · {completed}/{total}")
        self.eta_label.setText(f"ETA {self._format_duration(eta)}")

    def _append_result(self, result: dict[str, Any]) -> None:
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)
        values = [
            result.get("run_number"),
            result.get("scenario_name"),
            result.get("status"),
            f"{float(result.get('simulation_time', 0.0)):.3f}",
            self._format_duration(float(result.get("duration_seconds", 0.0))),
            result.get("error") or "",
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            self.result_table.setItem(row, column, item)

    def _batch_finished(self, destination: str, _results: object) -> None:
        self.progress.setValue(100)
        self.progress.setFormat("100% · completed")
        self.eta_label.setText("ETA 0s")
        self._set_editor_enabled(True)
        QMessageBox.information(
            self,
            "Multi Simulation Run completed",
            f"All run histories and statuses were saved to:\n{destination}",
        )

    def _batch_failed(self, message: str) -> None:
        self.progress.setFormat("Batch failed")
        self.eta_label.setText("ETA —")
        self._set_editor_enabled(True)
        QMessageBox.critical(self, "Multi Simulation Run failed", message)

    def _set_editor_enabled(self, enabled: bool) -> None:
        for widget in (
            self.add_button,
            self.duplicate_button,
            self.remove_button,
            self.current_settings_button,
            self.scenario_table,
            self.output_edit,
            self.auto_workers_check,
            self.worker_spin,
            self.max_events_spin,
        ):
            widget.setEnabled(enabled)
        if enabled:
            self.worker_spin.setEnabled(not self.auto_workers_check.isChecked())

    def _scenario_from_widgets(self, row: _ScenarioWidgets) -> MultiRunScenario:
        config = SimulationConfig.from_dict(row.config.to_dict()).without_human_agent()
        config.environment_mode = str(row.environment.currentData())
        return MultiRunScenario(
            name=row.name.text().strip(),
            repetitions=row.repetitions.value(),
            scheduling=str(row.scheduling.currentData()),
            speed_factor=float(row.speed.currentData()),
            end_condition=str(row.end_condition.currentData()),
            end_value=row.end_value.text().strip(),
            config=config,
        )

    def _current_config(self) -> SimulationConfig:
        if self._config_provider is not None:
            try:
                return self._config_provider().without_human_agent()
            except Exception:
                pass
        return self.simulation.config.without_human_agent()

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Multi Simulation Archive",
            self.output_edit.text(),
            "ZIP archive (*.zip)",
        )
        if path:
            self.output_edit.setText(path)
            self._persist_output_path()

    def _persist_output_path(self) -> None:
        self.settings.setValue("multi_run/output_path", self.output_edit.text().strip())

    def _discover_productions(self) -> list[str]:
        analyzer = AgentSourceAnalyzer()
        names: set[str] = set()
        for info in AgentDiscovery().discover():
            try:
                names.update(production.name for production in analyzer.analyze(info).productions)
            except Exception:
                continue
        return sorted(names, key=str.lower)

    @staticmethod
    def _settings_summary(config: SimulationConfig) -> str:
        agent_count = sum(item.count for item in config.agent_type_config.values())
        return (
            f"Edit… · {config.environment_label} · "
            f"{config.width}×{config.height} · LOS {config.los} · {agent_count} agents"
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    @staticmethod
    def _hardware_text() -> str:
        return f"Recommended parallel workers: {recommended_worker_count()}"
