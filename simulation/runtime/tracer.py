"""Append-only simulation event tracer."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any


class Tracer:
    """Record global and per-agent ACT-R events in execution order."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._records_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.known_agents: set[str] = set()
        self._sequence = 0

    def clear(self) -> None:
        self.records.clear()
        self._records_by_agent.clear()
        self.known_agents.clear()
        self._sequence = 0

    def trace(self, agent: Any, event: Any) -> list[dict[str, Any]]:
        timestamp = float(getattr(agent, "actr_time", 0.0))
        name = str(getattr(agent, "name", "Unknown agent"))
        appended: list[dict[str, Any]] = []

        if name not in self.known_agents:
            self.known_agents.add(name)
            appended.append(
                self._append(
                    timestamp=timestamp,
                    event_type="agent_added",
                    event=None,
                    agent_name=name,
                )
            )

        try:
            event_type = str(event[1])
            description = event[2]
        except (IndexError, TypeError):
            event_type = type(event).__name__
            description = str(event)
        appended.append(
            self._append(
                timestamp=timestamp,
                event_type=event_type,
                event=description,
                agent_name=name,
            )
        )
        return appended

    def trace_external(
        self,
        *,
        timestamp: float,
        agent_name: str,
        event_type: str,
        event: Any,
    ) -> dict[str, Any]:
        """Record non-ACT-R activity such as human keyboard movement."""
        if agent_name not in self.known_agents:
            self.known_agents.add(agent_name)
            self._append(
                timestamp=timestamp,
                event_type="agent_added",
                event=None,
                agent_name=agent_name,
            )
        return self._append(
            timestamp=timestamp,
            event_type=event_type,
            event=event,
            agent_name=agent_name,
        )

    def _append(
        self,
        *,
        timestamp: float,
        event_type: str,
        event: Any,
        agent_name: str,
    ) -> dict[str, Any]:
        self._sequence += 1
        record = {
            "sequence": self._sequence,
            "timestamp": timestamp,
            "type": event_type,
            "event": event,
            "agent_name": agent_name,
        }
        self.records.append(record)
        self._records_by_agent[agent_name].append(record)
        return record

    def records_for_agent(self, agent_name: str | None) -> Sequence[dict[str, Any]]:
        """Return the append-only per-agent history without copying it."""
        if agent_name is None:
            return ()
        return self._records_by_agent.get(agent_name, ())

    def get_logs(self) -> list[dict[str, Any]]:
        return list(self.records)
