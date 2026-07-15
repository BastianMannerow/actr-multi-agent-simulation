"""Generic ACT-R buffer inspection and change-history recording."""

from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping


class BufferHistoryRecorder:
    """Capture a structured entry whenever an agent buffer changes."""

    _DIAGNOSTIC_ATTRIBUTES = (
        "activation",
        "attend_automatic",
        "current_focus",
        "delay",
        "execution",
        "finst",
        "last_key",
        "preparation",
        "processor",
        "recent",
    )

    def __init__(self) -> None:
        self._history: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._latest: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self._sequence = 0

    def clear(self) -> None:
        self._history.clear()
        self._latest.clear()
        self._sequence = 0

    def capture_agent(
        self,
        agent: Any,
        *,
        event: Any | None = None,
        force: bool = False,
        reason: str = "event",
    ) -> list[dict[str, Any]]:
        """Capture only buffers that can have changed since the previous event."""
        changed: list[dict[str, Any]] = []
        event_type, event_description = self._event_parts(event)
        timestamp = float(getattr(agent, "actr_time", 0.0))
        agent_name = str(getattr(agent, "name", "Unknown agent"))
        buffers = self.discover_buffers(agent)
        current_names = set(buffers)
        known_names = set(getattr(agent, "_known_buffer_names", set()))

        if force:
            candidate_names = current_names
            consumer = getattr(agent, "consume_dirty_buffers", None)
            if callable(consumer):
                consumer()
        else:
            candidate_names = current_names - known_names
            consumer = getattr(agent, "consume_dirty_buffers", None)
            if callable(consumer):
                candidate_names.update(consumer())
            candidate_names.update(
                self._buffers_implied_by_event(event, current_names)
            )
        agent._known_buffer_names = current_names
        if not candidate_names:
            return changed

        for buffer_name in sorted(candidate_names, key=str.lower):
            buffer = buffers.get(buffer_name)
            if buffer is None:
                continue
            snapshot = self.serialize_buffer(buffer)
            previous = self._latest.get(agent_name, {}).get(buffer_name)
            if not force and previous == snapshot:
                continue

            change_kind = self._change_kind(previous, snapshot)
            self._sequence += 1
            entry = {
                "sequence": self._sequence,
                "timestamp": timestamp,
                "agent_name": agent_name,
                "buffer_name": buffer_name,
                "reason": reason,
                "change": change_kind,
                "event_type": event_type,
                "event": event_description,
                "snapshot": snapshot,
            }
            self._history[agent_name][buffer_name].append(entry)
            # Snapshots are newly built JSON-safe structures and are never
            # mutated internally, so a second deep copy is unnecessary here.
            self._latest[agent_name][buffer_name] = snapshot
            changed.append(entry)
        return changed

    @staticmethod
    def _buffers_implied_by_event(
        event: Any | None, available: set[str]
    ) -> set[str]:
        if event is None:
            return set()
        module = getattr(event, "module", None)
        if module is None:
            try:
                module = event[1]
            except Exception:
                module = None
        if module is None:
            return set()
        normalized = str(module).strip()
        direct = next(
            (name for name in available if name.casefold() == normalized.casefold()),
            None,
        )
        return {direct} if direct is not None else set()

    def agent_names(self) -> list[str]:
        return sorted(self._history)

    def buffer_names(self, agent_name: str) -> list[str]:
        names = set(self._history.get(agent_name, {})) | set(
            self._latest.get(agent_name, {})
        )
        return sorted(names, key=str.lower)

    def history(
        self, agent_name: str, buffer_name: str
    ) -> list[dict[str, Any]]:
        """Return a copy for exports and external consumers."""
        return list(self._history.get(agent_name, {}).get(buffer_name, []))

    def history_view(
        self, agent_name: str, buffer_name: str
    ) -> list[dict[str, Any]]:
        """Return the append-only internal list for virtualized GUI models.

        Callers must treat the returned list as read-only.  Avoiding a full copy
        is essential when a buffer has accumulated hundreds of thousands of
        changes.
        """
        return self._history.get(agent_name, {}).get(buffer_name, [])

    def history_count(self, agent_name: str, buffer_name: str) -> int:
        return len(self._history.get(agent_name, {}).get(buffer_name, ()))

    def latest(
        self, agent_name: str, buffer_name: str
    ) -> dict[str, Any] | None:
        value = self._latest.get(agent_name, {}).get(buffer_name)
        return deepcopy(value) if value is not None else None

    def all_histories(
        self,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        return {
            agent: {
                buffer_name: list(entries)
                for buffer_name, entries in buffers.items()
            }
            for agent, buffers in self._history.items()
        }

    @staticmethod
    def discover_buffers(agent: Any) -> dict[str, Any]:
        """Return the authoritative pyactr buffer map for a running agent.

        pyactr stores the active buffers on the simulation object. This map
        includes buffers created on the model and the automatic ``manual``,
        ``visual``, and ``visual_location`` buffers added during
        ``ACTRModel.simulation()``. Model mappings are used only as a fallback
        before a running simulation exists.
        """
        simulation = getattr(agent, "simulation", None)
        simulation_buffers = getattr(
            simulation, "_Simulation__buffers", None
        )
        if isinstance(simulation_buffers, Mapping):
            return dict(
                sorted(
                    (
                        (str(name), value)
                        for name, value in simulation_buffers.items()
                        if value is not None
                    ),
                    key=lambda item: item[0].lower(),
                )
            )

        model = getattr(agent, "actr_agent", None)
        candidates: list[Any] = []
        if model is not None:
            candidates.extend(
                [
                    getattr(model, "_ACTRModel__buffers", None),
                    getattr(model, "goals", None),
                    getattr(model, "retrievals", None),
                    getattr(model, "visbuffers", None),
                ]
            )

        result: dict[str, Any] = {}
        for mapping in candidates:
            if isinstance(mapping, Mapping):
                for name, value in mapping.items():
                    if value is not None:
                        result[str(name)] = value
        return dict(sorted(result.items(), key=lambda item: item[0].lower()))

    @classmethod
    def serialize_buffer(cls, buffer: Any) -> dict[str, Any]:
        chunks: list[dict[str, Any]] = []
        try:
            chunks = [cls.serialize_chunk(chunk) for chunk in list(buffer)]
        except Exception:
            raw = getattr(buffer, "_data", None)
            if raw is not None:
                chunks = [
                    cls.serialize_chunk(chunk)
                    for chunk in cls._safe_iter(raw)
                ]

        diagnostics: dict[str, Any] = {}
        for attribute in cls._DIAGNOSTIC_ATTRIBUTES:
            try:
                value = getattr(buffer, attribute)
            except Exception:
                continue
            if callable(value):
                continue
            diagnostics[attribute] = cls._json_safe(value)

        state = getattr(buffer, "state", None)
        return {
            "buffer_class": (
                f"{type(buffer).__module__}.{type(buffer).__name__}"
            ),
            "capacity": 1,
            "state": cls._json_safe(state),
            "empty": len(chunks) == 0,
            "chunks": chunks,
            "module_attributes": diagnostics,
            "repr": repr(buffer),
        }

    @classmethod
    def serialize_chunk(cls, chunk: Any) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        try:
            for slot_name, slot_value in chunk:
                slots[str(slot_name)] = cls._unwrap_slot_value(slot_value)
        except Exception:
            slots = {}

        return {
            "chunk_class": f"{type(chunk).__module__}.{type(chunk).__name__}",
            "type": str(getattr(chunk, "typename", type(chunk).__name__)),
            "slots": slots,
            "text": str(chunk),
            "repr": repr(chunk),
        }

    @classmethod
    def format_snapshot(cls, snapshot: dict[str, Any] | None) -> str:
        if snapshot is None:
            return "No snapshot is available."
        lines = [
            f"Class: {snapshot.get('buffer_class', '')}",
            f"Capacity: {snapshot.get('capacity', 1)} chunk",
            f"State: {snapshot.get('state', '')}",
            f"Empty: {'yes' if snapshot.get('empty') else 'no'}",
        ]

        diagnostics = snapshot.get("module_attributes", {})
        if diagnostics:
            lines.append("\nModule attributes:")
            for name, value in diagnostics.items():
                lines.append(f"  {name}: {cls._display_value(value)}")

        chunks = snapshot.get("chunks", [])
        if not chunks:
            lines.append("\nContent: <empty>")
            return "\n".join(lines)
        for index, chunk in enumerate(chunks, start=1):
            lines.append(f"\nChunk {index}: {chunk.get('type', '')}")
            slots = chunk.get("slots", {})
            if slots:
                for name, value in slots.items():
                    lines.append(f"  {name}: {cls._display_value(value)}")
            else:
                lines.append(f"  {chunk.get('text', '')}")
        return "\n".join(lines)

    @classmethod
    def _unwrap_slot_value(cls, value: Any) -> Any:
        seen: set[int] = set()
        current = value
        for _ in range(5):
            if id(current) in seen:
                break
            seen.add(id(current))
            nested = getattr(current, "values", None)
            if nested is None or nested is current:
                break
            current = nested
        return cls._json_safe(current)

    @classmethod
    def _json_safe(cls, value: Any, depth: int = 0) -> Any:
        if depth > 5:
            return repr(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): cls._json_safe(item, depth + 1)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [cls._json_safe(item, depth + 1) for item in value]
        try:
            from collections import deque

            if isinstance(value, deque):
                return [cls._json_safe(item, depth + 1) for item in value]
        except ImportError:
            pass
        nested = getattr(value, "values", None)
        if nested is not None and nested is not value:
            return cls._json_safe(nested, depth + 1)
        return str(value)

    @staticmethod
    def _safe_iter(value: Any) -> Iterable[Any]:
        try:
            return list(value)
        except TypeError:
            return [value]

    @staticmethod
    def _event_parts(event: Any | None) -> tuple[str | None, str | None]:
        if event is None:
            return None, None
        try:
            return str(event[1]), str(event[2])
        except (IndexError, TypeError):
            return type(event).__name__, str(event)

    @staticmethod
    def _change_kind(
        previous: dict[str, Any] | None,
        current: dict[str, Any],
    ) -> str:
        if previous is None:
            return "initial"
        if previous.get("empty") and not current.get("empty"):
            return "filled"
        if not previous.get("empty") and current.get("empty"):
            return "cleared"
        if previous.get("chunks") != current.get("chunks"):
            return "content_changed"
        if previous.get("state") != current.get("state"):
            return "state_changed"
        if previous.get("module_attributes") != current.get(
            "module_attributes"
        ):
            return "module_changed"
        return "content_changed"

    @staticmethod
    def _display_value(value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)
