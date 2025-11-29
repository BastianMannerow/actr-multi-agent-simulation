import contextlib
import os
import random
import sys

import simpy
import tkinter as tk
from gui.Stepper import StepLogWindow
from simulation import pyactrFunctionalityExtension
from simulation.Middleman import Middleman
from simulation.Environment import Environment
from simulation.AgentConstruct import AgentConstruct
from simulation.AgentTypeReturner import AgentTypeReturner
import simulation.LevelBuilder as levelbuilder
import pyactr as actr


class Simulation:
    """
    Orchestrates agents, environment, time, and optional stepper UI.

    Key concepts
    ------------
    - **ACT-R world:** pyactr Environment provides visual focus and time.
    - **Middleman:** bridges agent motor/input with the Environment.
    - **Stepper:** space-bar driven single-step execution and log view.

    Configuration
    -------------
    - `agent_type_config` maps an agent type to the number of instances and
      whether their internal logs should print.
    """

    def __init__(self, interceptor):
        # Sim configuration
        self.focus_position = (0, 2)
        self.print_middleman = False
        self.width = 16
        self.height = 16
        self.speed_factor = 50              # 100 = real-time cognition
        self.print_agent_actions = True
        self.los = 3
        self.stepper = True
        self.agent_type_config = {
            "Example": {"count": 3, "print_agent_actions": True}
        }

        # Runtime state
        self.global_sim_time = 0.0
        self.agent_list = []
        self.interceptor = interceptor

        # Agent & ACT-R setup
        self.agent_type_returner = AgentTypeReturner()
        self.actr_environment = actr.Environment(focus_position=self.focus_position)
        self.middleman = Middleman(self, self.print_middleman)

        # GUI (stepper) setup
        self.root = tk.Tk()
        if self.stepper:
            self.root.bind("<space>", lambda e: self.step_once())
            self.log_window = StepLogWindow(
                master=self.root,
                tracer=self.interceptor,
                simulation=self
            )

        # Jump to control (stepper-only)
        self.jumping = False
        self.jump_target = None

    def agent_builder(self):
        """
        Instantiate and wire agents.

        Naming
        ------
        - `names` is the source of display names.
        - `name_number` is set to the **entry from `names`** (string).
        """
        names = ["Example 1", "Example 2", "Example 3"]
        random.shuffle(names)

        for agent_type, config in self.agent_type_config.items():
            count = config["count"]
            print_actions = config.get("print_agent_actions", self.print_agent_actions)

            for _ in range(count):
                # Pick a unique display name from the shuffled pool
                name = names.pop()
                # Per requirement, `name_number` holds the *entry from names* (string)
                name_number = name

                agent = AgentConstruct(
                    agent_type,
                    self.actr_environment,
                    None,
                    self.middleman,
                    name,
                    name_number,
                    self.los,
                )
                agent.actr_time = 0.0
                agent.print_agent_actions = print_actions
                self.agent_list.append(agent)

        # Finalize ACT-R artifacts and back-references
        for agent in self.agent_list:
            agent.set_agent_dictionary(self.agent_list)
            ids = list(agent.get_agent_dictionary())
            actr_construct, actr_agent, actr_adapter = (
                AgentTypeReturner().return_agent_type(
                    agent.actr_agent_type_name,
                    self.actr_environment,
                    ids,
                )
            )
            agent.set_actr_agent(actr_agent)
            agent.set_actr_adapter(actr_adapter)
            agent.set_simulation()
            agent.set_actr_construct(actr_construct)

    def run_simulation(self):
        """
        Build agents, construct a minimal empty level, attach environment, and start UI loop.
        """
        self.agent_builder()
        level_matrix = levelbuilder.build_level(self.height, self.width, self.agent_list)
        self.game_environment = Environment(level_matrix, self.root)
        self.middleman.set_game_environment(self.game_environment)

        if not self.stepper:
            self.execute_step()

        self.root.mainloop()

    # -----------------------------
    # Non-stepper scheduling
    # -----------------------------
    def execute_step(self):
        """
        Schedule the next agent based on ACT-R time (non-stepper mode only).
        """
        if self.stepper:
            return

        for agent in self.agent_list:
            agent.update_stimulus()

        # Temporary upstream bug fix for pyactr visual updates (Oct 2025).
        pyactrFunctionalityExtension.fix_pyactr()

        self.agent_list.sort(key=lambda a: a.actr_time)
        next_agent = self.agent_list[0]
        delay = next_agent.actr_time - self.global_sim_time
        factor = 100 / self.speed_factor
        ms = max(1, round(delay * factor * 1000))
        self.root.after(ms, lambda: self.execute_agent_step(next_agent))

    def execute_agent_step(self, agent):
        """
        Execute one cognitive step for `agent`, handle inactivity timeouts,
        propagate motor inputs, and reschedule.
        """
        try:
            with self.suppress_stdout():
                agent.simulation.step()
            event = agent.simulation.current_event

            # Detect potential stalls
            if event.time > 0 or self.middleman is not None:
                agent.no_increase_count = 0
            else:
                agent.no_increase_count = getattr(agent, "no_increase_count", 0) + 1

            # Timeout guard
            if agent.no_increase_count >= 10:
                print(f"{agent.name} removed due to inactivity.")
                self.agent_list.remove(agent)
                self.game_environment.remove_agent_from_game(agent)
            else:
                agent.actr_time += event.time
                self.global_sim_time = agent.actr_time
                agent.actr_extension()
                if agent.print_agent_actions:
                    print(f"{agent.name}, {agent.actr_time}, {event}")
                key = pyactrFunctionalityExtension.key_pressed(agent)
                if key:
                    self.middleman.motor_input(key, agent)

            self.execute_step()

        except (simpy.core.EmptySchedule, AttributeError, IndexError, RuntimeError) as e:
            agent.handle_empty_schedule()
            self.root.after_idle(lambda: self.execute_step())

    # -----------------------------
    # Stepper mode
    # -----------------------------
    def step_once(self):
        """
        Execute exactly one cognitive step for the next agent (space-bar driven).
        """
        for agent in self.agent_list:
            agent.update_stimulus()

        pyactrFunctionalityExtension.fix_pyactr()
        self.agent_list.sort(key=lambda a: a.actr_time)
        na = self.agent_list[0]

        try:
            with self.suppress_stdout():
                na.simulation.step()
            event = na.simulation.current_event

            if event.time > 0:
                na.no_increase_count = 0
            else:
                na.no_increase_count = getattr(na, "no_increase_count", 0) + 1

            if na.no_increase_count >= 10:
                print(f"{na.name} removed due to inactivity.")
                self.agent_list.remove(na)
                self.game_environment.remove_agent_from_game(na)
            else:
                na.actr_time += event.time
                self.global_sim_time = na.actr_time
                na.actr_extension()
                if na.print_agent_actions:
                    print(f"{na.name}, {na.actr_time}, {event}")
                key = pyactrFunctionalityExtension.key_pressed(na)
                if key:
                    self.middleman.motor_input(key, na)
                self.interceptor.trace(na, event)
                self.log_window.log()

        except (simpy.core.EmptySchedule, AttributeError, IndexError, RuntimeError) as e:
            na.handle_empty_schedule()
        finally:
            self.notify_gui()

    # -----------------------------
    # Jump-to-production (stepper-only)
    # -----------------------------
    def start_jump(self, production_name: str):
        """
        Step until a given production fires. Stepper-only.
        """
        if not self.stepper:
            return
        self.jumping = True
        self.jump_target = f"RULE FIRED: {production_name}"
        self._jump_step()

    def _jump_step(self):
        if not getattr(self, "jumping", False):
            return
        before = len(self.interceptor.records)
        self.step_once()
        for r in self.interceptor.records[before:]:
            if r.get("type") == "PROCEDURAL" and str(r.get("event")) == self.jump_target:
                self.jumping = False
                print(f"Jump completed to {self.jump_target}")
                return
        self.root.after(1, self._jump_step)

    # -----------------------------
    # UI plumbing
    # -----------------------------
    def notify_gui(self):
        """Pump Tk events after a step to keep the stepper UI responsive."""
        if hasattr(self, "log_window"):
            self.log_window.window.update_idletasks()
            self.log_window.window.update()

    @contextlib.contextmanager
    def suppress_stdout(self):
        """Suppressing pyactr logs, which tend to confuse students."""
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout