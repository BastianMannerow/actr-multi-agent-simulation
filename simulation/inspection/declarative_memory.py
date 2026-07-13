"""Declarative-memory inspection helpers for pyactr models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from simulation.inspection.buffer_history import BufferHistoryRecorder


@dataclass(slots=True)
class MemoryChunk:
    """Serializable representation of one pyactr declarative-memory chunk."""

    chunk_id: str
    memory_name: str
    chunk_type: str
    label: str
    slots: dict[str, Any]
    traces: list[float] = field(default_factory=list)
    activation: float | None = None
    source: str | None = None


@dataclass(slots=True)
class MemoryEdge:
    """A semantic or inferred relationship between two chunks."""

    source_id: str
    target_id: str
    label: str
    relation: str


@dataclass(slots=True)
class DeclarativeMemorySnapshot:
    """Complete runtime or static declarative-memory graph payload."""

    memories: list[str]
    chunks: list[MemoryChunk]
    edges: list[MemoryEdge]
    operations: list[dict[str, Any]] = field(default_factory=list)


class DeclarativeMemoryInspector:
    """Read the authoritative ``ACTRModel.decmems`` structures."""

    @classmethod
    def inspect_agent(cls, agent: Any) -> DeclarativeMemorySnapshot:
        model = getattr(agent, "actr_agent", None)
        decmems = getattr(model, "decmems", {}) if model is not None else {}
        if not isinstance(decmems, Mapping):
            decmems = {}

        chunks: list[MemoryChunk] = []
        memories: list[str] = []
        for memory_name, memory in sorted(
            decmems.items(), key=lambda item: str(item[0]).lower()
        ):
            name = str(memory_name)
            memories.append(name)
            try:
                items = list(memory.items())
            except Exception:
                items = []
            activations = getattr(memory, "activations", {})
            for index, (chunk, traces) in enumerate(items, start=1):
                serialized = BufferHistoryRecorder.serialize_chunk(chunk)
                trace_values = cls._trace_values(traces)
                activation = None
                try:
                    value = activations.get(chunk)
                    activation = float(value) if value is not None else None
                except Exception:
                    activation = None
                chunks.append(
                    MemoryChunk(
                        chunk_id=f"{name}:{index}",
                        memory_name=name,
                        chunk_type=str(serialized.get("type", "chunk")),
                        label=cls._chunk_label(serialized, index),
                        slots=dict(serialized.get("slots", {})),
                        traces=trace_values,
                        activation=activation,
                        source="runtime",
                    )
                )
        operations: list[dict[str, Any]] = []
        memory_by_identity = {
            id(memory): str(name) for name, memory in decmems.items()
        }
        simulation = getattr(agent, "simulation", None)
        buffers = getattr(simulation, "_Simulation__buffers", None)
        if not isinstance(buffers, Mapping):
            buffers = getattr(model, "_ACTRModel__buffers", {}) if model is not None else {}
        if isinstance(buffers, Mapping):
            for buffer_name, buffer in buffers.items():
                memory = getattr(buffer, "dm", None)
                memory_name = memory_by_identity.get(id(memory))
                if memory_name:
                    operations.append(
                        {
                            "actor": f"buffer:{buffer_name}",
                            "mode": "linked",
                            "memory_name": memory_name,
                            "detail": type(buffer).__name__,
                        }
                    )
        return DeclarativeMemorySnapshot(
            memories=memories,
            chunks=chunks,
            edges=cls.infer_edges(chunks),
            operations=operations,
        )

    @classmethod
    def infer_edges(cls, chunks: list[MemoryChunk]) -> list[MemoryEdge]:
        """Infer explicit references and weaker shared-value associations."""
        edges: list[MemoryEdge] = []
        seen: set[tuple[str, str, str, str]] = set()
        aliases: dict[str, list[MemoryChunk]] = {}
        for chunk in chunks:
            for alias in cls._aliases(chunk):
                aliases.setdefault(alias.casefold(), []).append(chunk)

        for chunk in chunks:
            for slot_name, raw_value in chunk.slots.items():
                value = cls._scalar(raw_value)
                if not value:
                    continue
                targets = aliases.get(value.casefold(), [])
                for target in targets:
                    if target.chunk_id == chunk.chunk_id:
                        continue
                    key = (chunk.chunk_id, target.chunk_id, str(slot_name), "reference")
                    if key not in seen:
                        seen.add(key)
                        edges.append(
                            MemoryEdge(
                                source_id=chunk.chunk_id,
                                target_id=target.chunk_id,
                                label=str(slot_name),
                                relation="reference",
                            )
                        )

        by_value: dict[str, list[tuple[MemoryChunk, str]]] = {}
        for chunk in chunks:
            for slot_name, raw_value in chunk.slots.items():
                value = cls._scalar(raw_value)
                if value and len(value) > 1:
                    by_value.setdefault(value.casefold(), []).append(
                        (chunk, str(slot_name))
                    )
        for value, occurrences in by_value.items():
            unique = []
            for item in occurrences:
                if all(existing[0].chunk_id != item[0].chunk_id for existing in unique):
                    unique.append(item)
            if len(unique) < 2:
                continue
            for index, (source, source_slot) in enumerate(unique):
                for target, target_slot in unique[index + 1 :]:
                    key = (source.chunk_id, target.chunk_id, value, "shared_value")
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append(
                        MemoryEdge(
                            source_id=source.chunk_id,
                            target_id=target.chunk_id,
                            label=f"{source_slot}/{target_slot}: {value}",
                            relation="shared_value",
                        )
                    )
        return edges

    @staticmethod
    def _trace_values(value: Any) -> list[float]:
        try:
            return [float(item) for item in list(value)]
        except Exception:
            try:
                return [float(value)]
            except Exception:
                return []

    @staticmethod
    def _chunk_label(serialized: dict[str, Any], index: int) -> str:
        chunk_type = str(serialized.get("type", "chunk"))
        slots = serialized.get("slots", {})
        concise = ", ".join(
            f"{name}={value}" for name, value in list(slots.items())[:3]
        )
        return f"{chunk_type} #{index}" + (f"\n{concise}" if concise else "")

    @staticmethod
    def _aliases(chunk: MemoryChunk) -> set[str]:
        aliases = {chunk.chunk_id, chunk.label.splitlines()[0]}
        for key in ("name", "id", "key", "state", "value"):
            value = chunk.slots.get(key)
            scalar = DeclarativeMemoryInspector._scalar(value)
            if scalar:
                aliases.add(scalar)
        return aliases

    @staticmethod
    def _scalar(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        nested = getattr(value, "values", None)
        if nested is not None and nested is not value:
            return DeclarativeMemoryInspector._scalar(nested)
        return str(value)
