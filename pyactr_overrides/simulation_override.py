"""Scheduler-oriented opt-in extensions for pyactr Simulation.

These overrides deliberately live outside the installed dependency and are
only activated through the GUI's experimental performance toggle.
"""

from __future__ import annotations

from typing import Any, Callable

from pyactr import simulation as pyactr_simulation
from simpy.events import Event, NORMAL


def peek_next_event_time(self) -> float:
    """Return the next internal SimPy event time without advancing the model."""
    environment = getattr(self, "_Simulation__simulation", None)
    if environment is None:
        return float("inf")
    return float(environment.peek())


def step_until(self, target_time: float, max_visible_events: int = 10000) -> int:
    """Advance through visible ACT-R events up to ``target_time``."""
    count = 0
    target = float(target_time)
    while (
        count < max(1, int(max_visible_events))
        and peek_next_event_time(self) <= target
    ):
        self.step()
        count += 1
    return count


def accelerated_procedural_generator(self):
    """pyactr 0.3.2 procedural scheduler with one zero-time handoff.

    Upstream yields five zero-duration SimPy timeouts after activating module
    processes. One handoff is sufficient for the event ordering used by the
    bundled models and avoids four internal scheduling operations per
    production cycle. Because this relies on pyactr internals it remains
    strictly opt-in and is covered by standard-versus-experimental trace tests.
    """
    simulation = self._Simulation__simulation
    procedural = self._Simulation__pr
    local_process = self.__localprocess__

    process = simulation.process(
        local_process(
            procedural._PROCEDURAL,
            procedural.procedural_process(simulation.now),
        )
    )
    self._Simulation__procs_started = yield process

    while True:
        try:
            self._Simulation__procs_started.remove(procedural._PROCEDURAL)
        except ValueError:
            yield self._Simulation__proc_activate
        else:
            for proc in self._Simulation__procs_started:
                name = proc[0]
                activate_event = self._Simulation__dict_extra_proc_activate[name]
                if not activate_event.triggered:
                    if proc[1].__name__ in procedural._INTERRUPTIBLE:
                        self._Simulation__interruptibles[name] = proc[1]
                    activate_event.succeed()
                elif (
                    name in self._Simulation__interruptibles
                    and proc[1] != self._Simulation__interruptibles[name]
                ):
                    self._Simulation__interruptibles[name] = proc[1]
                    self._Simulation__dict_extra_proc[name].interrupt()

        # Resume procedural matching after all normal-priority events already
        # scheduled for the same model time. This replaces pyactr's five
        # normal-priority zero-time queue rotations with one lower-priority
        # barrier.
        barrier = Event(simulation)
        barrier._ok = True
        barrier._value = None
        simulation.schedule(barrier, priority=NORMAL + 1, delay=0)
        yield barrier
        process = simulation.process(
            local_process(
                procedural._PROCEDURAL,
                procedural.procedural_process(simulation.now),
            )
        )
        self._Simulation__procs_started = yield process
        self._Simulation__proc_activate = simulation.event()


def apply(remember: Callable[[str, Any, str], None]) -> None:
    cls = pyactr_simulation.Simulation
    remember("Simulation.peek_next_event_time", cls, "peek_next_event_time")
    remember("Simulation.step_until", cls, "step_until")
    remember(
        "Simulation.__procprocessGenerator__",
        cls,
        "__procprocessGenerator__",
    )
    cls.peek_next_event_time = peek_next_event_time
    cls.step_until = step_until
    cls.__procprocessGenerator__ = accelerated_procedural_generator


def clear_caches() -> None:
    return None
