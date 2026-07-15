"""Runtime orchestration for the PyQt6 ACT-R simulation."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any

import pyactr as actr
import simpy
from PyQt6.QtCore import QTimer

from gui.application import create_application
from gui.main_window import SimulationMainWindow
from simulation.integrations import pyactr_extension
from simulation.runtime.agent_construct import AgentConstruct
from simulation.runtime.agent_type_factory import AgentTypeReturner
from simulation.world.environment import Environment
from simulation.world.factory import create_environment
from simulation.world.human_agent import HumanAgent
from simulation.runtime.middleman import Middleman
from simulation.inspection.buffer_history import BufferHistoryRecorder
from simulation.config.models import SPEED_PRESETS, SimulationConfig
from simulation.config.settings_store import SimulationSettingsStore
from simulation.export.history_export import SimulationHistoryExporter


class Simulation:
    """Build agents and coordinate pausable Step, Automatic, and Jump modes."""

    def __init__(self, interceptor: Any) -> None:
        self.interceptor = interceptor
        self.config = SimulationConfig()
        self.agent_type_returner = AgentTypeReturner()
        self.buffer_history = BufferHistoryRecorder()
        self.history_exporter = SimulationHistoryExporter()
        self.settings_store: SimulationSettingsStore | None = None

        self.global_sim_time = 0.0
        self.agent_list: list[AgentConstruct] = []
        self.human_agent: HumanAgent | None = None
        self.actr_environment: Any | None = None
        self.middleman: Middleman | None = None
        self.game_environment: Environment | None = None

        self.qt_app = None
        self.main_window: SimulationMainWindow | None = None
        self._auto_timer: QTimer | None = None
        self._jump_timer: QTimer | None = None

        self.initialized = False
        self.run_state = "not_started"
        self.execution_mode = self.config.execution_mode
        self.jumping = False
        self.jump_target: str | None = None
        self.jump_agent_name: str | None = None
        self.last_error: str | None = None
        self._mirror_config_attributes()

    def run_simulation(self) -> int:
        """Open the maximized GUI; the model is started from the GUI."""
        self.qt_app = create_application(sys.argv)
        self.settings_store = SimulationSettingsStore()
        self.config = self.settings_store.load()
        self.execution_mode = self.config.execution_mode
        self._mirror_config_attributes()
        self._auto_timer = QTimer()
        self._auto_timer.setSingleShot(True)
        self._auto_timer.timeout.connect(self._automatic_tick)
        self._jump_timer = QTimer()
        self._jump_timer.setSingleShot(True)
        self._jump_timer.timeout.connect(self._jump_tick)

        self.main_window = SimulationMainWindow(
            tracer=self.interceptor,
            simulation=self,
        )
        self.main_window.showMaximized()
        return self.qt_app.exec()

    def start_simulation(self, config: SimulationConfig) -> None:
        """Build or rebuild a simulation from the complete GUI configuration."""
        config.validate()
        self.stop_execution()
        if self.game_environment is not None:
            try:
                self.game_environment.close()
            except Exception:
                pass
        self.config = config
        if self.settings_store is not None:
            self.settings_store.save(config)
        self._mirror_config_attributes()
        self.execution_mode = config.execution_mode
        self.global_sim_time = 0.0
        self.agent_list.clear()
        self.human_agent = None
        self.interceptor.clear()
        self.buffer_history.clear()
        self.agent_type_returner.clear_cache()
        self.last_error = None
        self.jump_target = None
        self.jump_agent_name = None

        self.actr_environment = actr.Environment(
            focus_position=config.focus_position
        )
        self.middleman = Middleman(self, config.print_middleman)
        self.agent_builder()
        self.game_environment = create_environment(
            config,
            self.spatial_agents,
            self,
        )
        self.middleman.set_game_environment(self.game_environment)

        # Build each pyactr simulation with the real initial level frame rather
        # than a dummy stimulus. This prevents the first environment event from
        # overwriting current world data and makes later EmptySchedule resets
        # reuse a valid, up-to-date frame.
        for agent in self.agent_list:
            agent.update_stimulus(publish=False)
            agent.set_simulation()
            agent.update_stimulus(publish=True)
            self.buffer_history.capture_agent(
                agent, force=True, reason="initialization"
            )

        self.initialized = True
        self.run_state = "running"
        if self.main_window is not None:
            self.main_window.set_environment(self.game_environment)
        self.notify_gui()
        if self.execution_mode == "automatic":
            self._schedule_automatic_step()

    @property
    def spatial_agents(self) -> list[Any]:
        """Return all entities that occupy grid cells in normal simulations."""
        entities: list[Any] = list(self.agent_list)
        if self.human_agent is not None:
            entities.append(self.human_agent)
        return entities

    def agent_builder(self) -> None:
        """Build ACT-R agents and the optional human-controlled grid entity."""
        if self.actr_environment is None or self.middleman is None:
            raise RuntimeError("The ACT-R environment has not been initialized.")

        for agent_type, type_config in sorted(
            self.config.agent_type_config.items()
        ):
            print_actions = type_config.print_agent_actions
            for index in range(type_config.count):
                name = f"{agent_type} {index + 1}"
                agent = AgentConstruct(
                    agent_type,
                    self.actr_environment,
                    None,
                    self.middleman,
                    name,
                    name,
                    self.config.los,
                )
                agent.actr_time = 0.0
                agent.print_agent_actions = print_actions
                self.agent_list.append(agent)

        if self.config.human_agent_enabled:
            human_name = self.config.human_agent_name.strip()
            occupied_names = {agent.name.casefold() for agent in self.agent_list}
            if human_name.casefold() in occupied_names:
                raise ValueError(
                    "The human agent name must differ from every ACT-R agent name."
                )
            self.human_agent = HumanAgent(human_name)

        all_spatial_agents = self.spatial_agents
        for agent in self.agent_list:
            agent.set_agent_dictionary(all_spatial_agents)
            identifiers = list(agent.get_agent_dictionary())
            artifacts = self.agent_type_returner.return_agent_type(
                agent.actr_agent_type_name,
                self.actr_environment,
                identifiers,
            )
            if artifacts is None:
                raise ValueError(
                    f"Agent type '{agent.actr_agent_type_name}' is not an executable "
                    "ACT-R model."
                )
            actr_construct, actr_agent, actr_adapter = artifacts
            agent.set_actr_agent(actr_agent)
            agent.set_actr_adapter(actr_adapter)
            agent.set_actr_construct(actr_construct)

    def move_human_agent(self, direction: str) -> bool:
        """Move the optional human agent independently of ACT-R timing."""
        if (
            not self.initialized
            or self.human_agent is None
            or self.game_environment is None
        ):
            return False
        movements = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        delta = movements.get(direction)
        if delta is None:
            return False
        moved = self.game_environment.move_agent(
            self.human_agent, delta[0], delta[1]
        )
        if moved:
            self.interceptor.trace_external(
                timestamp=self.global_sim_time,
                agent_name=self.human_agent.name,
                event_type="HUMAN",
                event=f"MOVE {direction.upper()}",
            )
            self.notify_gui()
        return moved

    def set_execution_mode(self, mode: str) -> None:
        """Switch live between Step and Automatic execution."""
        if mode not in {"single", "automatic"}:
            raise ValueError(f"Unknown execution mode: {mode}")
        self.execution_mode = mode
        self.config.execution_mode = mode
        if self._auto_timer is not None:
            self._auto_timer.stop()
        if (
            self.initialized
            and self.run_state == "running"
            and mode == "automatic"
        ):
            self._schedule_automatic_step()
        self.notify_gui()

    def set_speed_factor(self, speed_factor: float) -> None:
        allowed = {value for _, value in SPEED_PRESETS}
        if float(speed_factor) not in allowed:
            raise ValueError("Unknown speed preset.")
        self.speed_factor = float(speed_factor)
        self.config.speed_factor = float(speed_factor)
        if (
            self.initialized
            and self.run_state == "running"
            and self.execution_mode == "automatic"
        ):
            if self._auto_timer is not None:
                self._auto_timer.stop()
            self._schedule_automatic_step()

    def pause(self) -> None:
        """Pause Automatic execution or cancel an active production jump."""
        if not self.initialized:
            return
        if self._auto_timer is not None:
            self._auto_timer.stop()
        if self._jump_timer is not None:
            self._jump_timer.stop()
        self.jumping = False
        self.run_state = "paused"
        self.notify_gui()

    def resume(self) -> None:
        """Resume using the currently selected execution mode."""
        if not self.initialized or not self.agent_list:
            return
        self.run_state = "running"
        self.jumping = False
        if self.execution_mode == "automatic":
            self._schedule_automatic_step()
        self.notify_gui()

    def stop_execution(self) -> None:
        if self._auto_timer is not None:
            self._auto_timer.stop()
        if self._jump_timer is not None:
            self._jump_timer.stop()
        self.jumping = False
        if self.initialized:
            self.run_state = "stopped"
        if self.game_environment is not None:
            try:
                self.game_environment.close()
            except Exception:
                pass

    def step_once(self, *, force: bool = False) -> bool:
        """Execute one visible cognitive event."""
        if not self.initialized or not self.agent_list:
            return False
        if not force and (
            self.run_state != "running" or self.execution_mode != "single"
        ):
            return False
        return self._execute_next_event() is not None

    def _schedule_automatic_step(self) -> None:
        if (
            not self.initialized
            or self.run_state != "running"
            or self.execution_mode != "automatic"
            or self.jumping
            or not self.agent_list
            or self._auto_timer is None
        ):
            return

        self.agent_list.sort(key=lambda agent: agent.actr_time)
        next_agent = self.agent_list[0]
        delay = max(
            0.0,
            float(next_agent.actr_time) - float(self.global_sim_time),
        )
        if float(self.speed_factor) == -1.0:
            self._auto_timer.start(0)
            return
        factor = 100.0 / float(self.speed_factor)
        milliseconds = max(1, round(delay * factor * 1000))
        self._auto_timer.start(milliseconds)

    def _automatic_tick(self) -> None:
        if self.run_state != "running" or self.execution_mode != "automatic":
            return
        self._execute_next_event()
        if not self.agent_list:
            self.run_state = "finished"
            self.notify_gui()
            return
        self._schedule_automatic_step()

    def _execute_next_event(self) -> tuple[AgentConstruct, Any] | None:
        if not self.agent_list:
            self.run_state = "finished"
            self.notify_gui()
            return None

        assert self.middleman is not None
        pyactr_extension.fix_pyactr()
        self.agent_list.sort(key=lambda agent: agent.actr_time)
        agent = self.agent_list[0]
        # The pyactr Environment is shared. Publish only the frame of the agent
        # that is about to step, so its visual requests and automatic buffers
        # cannot accidentally consume another agent's point of view.
        agent.update_stimulus(publish=True)

        try:
            with self.suppress_stdout():
                agent.simulation.step()
            event = agent.simulation.current_event
            if event is None:
                raise RuntimeError(
                    f"{agent.name} did not produce an ACT-R event."
                )

            event_time = float(getattr(event, "time", 0.0))
            if event_time > 0:
                agent.no_increase_count = 0
            else:
                agent.no_increase_count = (
                    getattr(agent, "no_increase_count", 0) + 1
                )

            if agent.no_increase_count >= 10:
                self.agent_list.remove(agent)
                if self.game_environment is not None:
                    self.game_environment.remove_agent_from_game(agent)
                if not self.agent_list:
                    self.run_state = "finished"
                self.notify_gui()
                return None

            agent.actr_time += event_time
            self.global_sim_time = agent.actr_time
            agent.actr_extension()
            if agent.print_agent_actions:
                print(f"{agent.name}, {agent.actr_time}, {event}")
            key = pyactr_extension.key_pressed(agent)
            if key:
                self.middleman.motor_input(key, agent)

            self.interceptor.trace(agent, event)
            self.buffer_history.capture_agent(
                agent, event=event, reason="event"
            )
            self.notify_gui()
            return agent, event

        except (
            simpy.core.EmptySchedule,
            AttributeError,
            IndexError,
            RuntimeError,
            TypeError,
        ) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            try:
                agent.handle_empty_schedule()
                self.buffer_history.capture_agent(
                    agent,
                    force=True,
                    reason="schedule_reset",
                )
            except Exception as reset_exc:
                self.last_error = (
                    f"{self.last_error}; reset failed: "
                    f"{type(reset_exc).__name__}: {reset_exc}"
                )
                if agent in self.agent_list:
                    self.agent_list.remove(agent)
                    if self.game_environment is not None:
                        self.game_environment.remove_agent_from_game(agent)
                if not self.agent_list:
                    self.run_state = "finished"
            self.notify_gui()
            return None

    def start_jump(
        self, production_name: str, agent_name: str | None = None
    ) -> None:
        """Run UI-friendly microsteps until the requested production fires."""
        target = production_name.strip()
        if not self.initialized or not target or not self.agent_list:
            return
        if self._auto_timer is not None:
            self._auto_timer.stop()
        self.jumping = True
        self.jump_target = target
        self.jump_agent_name = agent_name
        self.run_state = "jumping"
        self.last_error = None
        if self._jump_timer is not None:
            self._jump_timer.start(0)
        self.notify_gui()

    def _jump_tick(self) -> None:
        if not self.jumping:
            return
        if not self.agent_list:
            self._finish_jump(found=False, finished=True)
            return

        before = len(self.interceptor.records)
        self._execute_next_event()
        new_records = self.interceptor.records[before:]
        for record in new_records:
            if (
                self.jump_agent_name
                and record.get("agent_name") != self.jump_agent_name
            ):
                continue
            if self._record_matches_production(
                record, self.jump_target or ""
            ):
                self._finish_jump(found=True)
                return
        if not self.agent_list:
            self._finish_jump(found=False, finished=True)
            return
        if self.jumping and self._jump_timer is not None:
            self._jump_timer.start(0)

    def _finish_jump(
        self, *, found: bool, finished: bool = False
    ) -> None:
        self.jumping = False
        self.run_state = "finished" if finished else "paused"
        if found:
            self.last_error = None
        elif finished:
            self.last_error = (
                "The simulation ended before the requested production fired."
            )
        else:
            self.last_error = (
                "The production jump stopped before the target was reached."
            )
        self.notify_gui()

    @staticmethod
    def _record_matches_production(
        record: dict[str, Any], target: str
    ) -> bool:
        if str(record.get("type", "")).upper() != "PROCEDURAL":
            return False
        event = str(record.get("event", "")).strip()
        prefix = "RULE FIRED:"
        fired = (
            event[len(prefix) :].strip()
            if event.upper().startswith(prefix)
            else event
        )
        return fired.casefold() == target.strip().casefold()

    def get_production_names(self, agent_name: str | None = None) -> list[str]:
        """Return production names globally or for one selected runtime agent."""
        names: set[str] = set()
        for agent in self.agent_list:
            if agent_name and str(getattr(agent, "name", "")) != agent_name:
                continue
            productions = getattr(
                getattr(agent, "actr_agent", None), "productions", None
            )
            try:
                names.update(str(name) for name in productions.keys())
            except AttributeError:
                continue
        return sorted(names, key=str.lower)

    def save_settings(self, config: SimulationConfig | None = None) -> None:
        """Persist controls without rewriting a running simulation's build state."""
        payload = config or self.config
        if not self.initialized and config is not None:
            self.config = config
            self.execution_mode = config.execution_mode
            self._mirror_config_attributes()
        if self.settings_store is not None:
            self.settings_store.save(payload)

    def reset_settings(self) -> SimulationConfig:
        config = (
            self.settings_store.reset()
            if self.settings_store is not None
            else SimulationConfig()
        )
        if not self.initialized:
            self.config = config
            self.execution_mode = config.execution_mode
            self._mirror_config_attributes()
        return config

    def get_agent_by_name(self, agent_name: str) -> AgentConstruct | None:
        for agent in self.agent_list:
            if str(getattr(agent, "name", "")) == str(agent_name):
                return agent
        return None

    def replace_agent_buffer_from_string(
        self, agent_name: str, buffer_name: str, chunk_string: str
    ) -> None:
        agent = self.get_agent_by_name(agent_name)
        if agent is None:
            raise KeyError(f"Unknown agent: {agent_name}")
        chunk = pyactr_extension.chunk_from_string(chunk_string)
        pyactr_extension.replace_buffer(agent, buffer_name, chunk)
        self.buffer_history.capture_agent(
            agent, force=True, reason="manual_buffer_update"
        )
        self.last_error = None
        self.notify_gui()

    def export_history(self, path: str | Path) -> Path:
        if not self.initialized:
            raise RuntimeError("No simulation has been started yet.")
        for agent in self.agent_list:
            self.buffer_history.capture_agent(agent, reason="export")
        return self.history_exporter.export(path, self)

    def notify_gui(self) -> None:
        if self.main_window is not None:
            signal = getattr(self.main_window, "refresh_requested", None)
            if signal is not None:
                signal.emit()
            else:
                self.main_window.refresh()

    def _mirror_config_attributes(self) -> None:
        """Keep the public settings from the original Simulation API available."""
        self.focus_position = self.config.focus_position
        self.print_middleman = self.config.print_middleman
        self.width = self.config.width
        self.height = self.config.height
        self.speed_factor = self.config.speed_factor
        self.print_agent_actions = self.config.print_agent_actions
        self.los = self.config.los
        self.stepper = self.config.stepper
        self.human_agent_enabled = self.config.human_agent_enabled
        self.human_agent_name = self.config.human_agent_name
        self.environment_mode = self.config.environment_mode
        self.virtual_level = self.config.virtual_level
        self.agent_type_config = {
            name: value.to_dict()
            for name, value in self.config.agent_type_config.items()
        }

    @contextlib.contextmanager
    def suppress_stdout(self):
        """Suppress verbose pyactr traces while preserving framework logs."""
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
