"""Retrieval-centred declarative-memory inspection for pyactr models.

pyactr's :class:`DecMemBuffer` searches only its assigned ``dm`` mapping.  This
module mirrors that implementation and then applies the retrieval requests that
actually occur on production right-hand sides.  The runtime diagram therefore
contains chunks that are both:

* stored in a declarative memory bound to a retrieval buffer; and
* theoretically matchable by at least one ``+retrieval>`` request in the model.

Bound ACT-R variables are treated as wildcards for static/theoretical matching;
literal request values remain constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from simulation.inspection.buffer_history import BufferHistoryRecorder


@dataclass(slots=True)
class RetrievalQuery:
    """One production-side request sent to a pyactr retrieval buffer."""

    query_id: str
    production_name: str
    buffer_name: str
    chunk_type: str
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryChunk:
    """Serializable representation of one retrieval-relevant DM chunk."""

    chunk_id: str
    memory_name: str
    chunk_type: str
    label: str
    slots: dict[str, Any]
    traces: list[float] = field(default_factory=list)
    activation: float | None = None
    source: str | None = None
    retrieval_buffers: list[str] = field(default_factory=list)
    matched_queries: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryEdge:
    """An explicit slot reference from one chunk to another chunk."""

    source_id: str
    target_id: str
    label: str
    relation: str = "slot_reference"


@dataclass(slots=True)
class DeclarativeMemorySnapshot:
    """Runtime or static declarative-memory diagram payload."""

    memories: list[str]
    chunks: list[MemoryChunk]
    edges: list[MemoryEdge]
    operations: list[dict[str, Any]] = field(default_factory=list)
    retrieval_buffers: list[str] = field(default_factory=list)
    retrieval_memory_names: list[str] = field(default_factory=list)
    retrieval_queries: list[RetrievalQuery] = field(default_factory=list)
    scope: str = "retrieval-query-matching"


@dataclass(frozen=True, slots=True)
class RetrievalBinding:
    """One pyactr retrieval buffer and the declarative memory it can query."""

    buffer_name: str
    memory_name: str
    memory: Any


class DeclarativeMemoryInspector:
    """Read memories through the same binding and request path as pyactr."""

    @classmethod
    def estimate_agent_size(cls, agent: Any) -> tuple[int, int]:
        bindings = cls.retrieval_bindings(agent)
        queries = cls.retrieval_queries(agent)
        count = 0
        for memory_name, memory in cls._unique_bound_memories(bindings):
            buffers = {
                binding.buffer_name
                for binding in bindings
                if binding.memory_name == memory_name
            }
            applicable = [q for q in queries if q.buffer_name in buffers]
            for chunk in cls._memory_chunks(memory):
                if cls._matching_query_ids(chunk, applicable):
                    count += 1
        return len(cls._unique_bound_memories(bindings)), count

    @classmethod
    def estimate_agent_graph_complexity(
        cls,
        agent: Any,
        *,
        detailed_chunk_limit: int = 180,
    ) -> tuple[int, int, int]:
        """Estimate explicit reference complexity using the same filtered set."""
        snapshot = cls._inspect_agent(
            agent,
            chunk_limit=max(1, detailed_chunk_limit + 1),
            max_edges=0,
        )
        memory_count = len(snapshot.memories)
        chunk_count = cls.estimate_agent_size(agent)[1]
        if chunk_count == 0 or chunk_count > detailed_chunk_limit:
            return memory_count, chunk_count, 0
        return memory_count, chunk_count, len(cls.infer_edges(snapshot.chunks))

    @classmethod
    def inspect_agent(cls, agent: Any) -> DeclarativeMemorySnapshot:
        return cls._inspect_agent(agent)

    @classmethod
    def inspect_agent_window(
        cls,
        agent: Any,
        *,
        chunk_offset: int,
        chunk_limit: int,
        max_edges: int = 240,
    ) -> DeclarativeMemorySnapshot:
        return cls._inspect_agent(
            agent,
            chunk_offset=max(0, int(chunk_offset)),
            chunk_limit=max(1, int(chunk_limit)),
            max_edges=max(0, int(max_edges)),
        )

    @classmethod
    def retrieval_bindings(cls, agent: Any) -> list[RetrievalBinding]:
        """Resolve the concrete ``DecMemBuffer.dm`` bindings used by pyactr."""
        if agent is None:
            return []
        model = getattr(agent, "actr_agent", None)
        if model is None:
            return []
        decmems = getattr(model, "decmems", {})
        if not isinstance(decmems, Mapping):
            return []
        memory_by_identity = {
            id(memory): str(name) for name, memory in decmems.items()
        }

        candidates: dict[str, Any] = {}
        simulation = getattr(agent, "simulation", None)
        runtime_buffers = getattr(simulation, "_Simulation__buffers", None)
        if isinstance(runtime_buffers, Mapping):
            candidates.update(runtime_buffers)
        model_retrievals = getattr(model, "retrievals", None)
        if isinstance(model_retrievals, Mapping):
            for name, buffer in model_retrievals.items():
                candidates.setdefault(str(name), buffer)

        try:
            from pyactr.declarative import DecMemBuffer
        except Exception:  # pragma: no cover
            DecMemBuffer = ()  # type: ignore[assignment]

        only_memory = (
            next(iter(decmems.values()), None) if len(decmems) == 1 else None
        )
        result: list[RetrievalBinding] = []
        seen: set[tuple[str, int]] = set()
        for raw_name, buffer in sorted(
            candidates.items(), key=lambda item: str(item[0]).casefold()
        ):
            is_retrieval = (
                isinstance(buffer, DecMemBuffer)
                if DecMemBuffer
                else type(buffer).__name__ == "DecMemBuffer"
            )
            if not is_retrieval:
                continue
            memory = getattr(buffer, "dm", None)
            if memory is None:
                memory = only_memory
            if memory is None:
                continue
            memory_name = memory_by_identity.get(id(memory))
            if memory_name is None:
                continue
            key = (str(raw_name), id(memory))
            if key in seen:
                continue
            seen.add(key)
            result.append(
                RetrievalBinding(str(raw_name), memory_name, memory)
            )
        return result

    @classmethod
    def retrieval_queries(cls, agent: Any) -> list[RetrievalQuery]:
        """Extract executable ``+retrieval>`` patterns from built productions."""
        if agent is None:
            return []
        model = getattr(agent, "actr_agent", None)
        if model is None:
            return []
        retrieval_names = {
            str(name)
            for name in getattr(model, "retrievals", {})
        }
        productions = getattr(model, "productions", {})
        result: list[RetrievalQuery] = []
        for production_name, production in getattr(
            productions, "items", lambda: ()
        )():
            try:
                generator = production["rule"]()
                next(generator)  # left-hand side
                effects = next(generator)
            except Exception:
                continue
            if not isinstance(effects, Mapping):
                continue
            for raw_key, request_chunk in effects.items():
                key = str(raw_key)
                if not key.startswith("+"):
                    continue
                buffer_name = key[1:]
                if buffer_name not in retrieval_names:
                    continue
                chunk_type = str(getattr(request_chunk, "typename", ""))
                if not chunk_type:
                    continue
                constraints: dict[str, Any] = {}
                try:
                    pairs = request_chunk.removeunused()
                except Exception:
                    pairs = tuple(request_chunk) if hasattr(request_chunk, "__iter__") else ()
                for slot_name, value in pairs:
                    scalar = cls._scalar(value)
                    if cls._is_bound_variable(scalar):
                        continue
                    if scalar not in {None, "", "None"}:
                        constraints[str(slot_name)] = scalar
                result.append(
                    RetrievalQuery(
                        query_id=f"{production_name}:{buffer_name}:{len(result) + 1}",
                        production_name=str(production_name),
                        buffer_name=buffer_name,
                        chunk_type=chunk_type,
                        constraints=constraints,
                    )
                )
        return result

    @classmethod
    def _inspect_agent(
        cls,
        agent: Any,
        *,
        chunk_offset: int = 0,
        chunk_limit: int | None = None,
        max_edges: int | None = None,
    ) -> DeclarativeMemorySnapshot:
        bindings = cls.retrieval_bindings(agent)
        queries = cls.retrieval_queries(agent)
        memories_with_objects = cls._unique_bound_memories(bindings)
        memory_to_buffers: dict[str, list[str]] = {}
        for binding in bindings:
            memory_to_buffers.setdefault(binding.memory_name, []).append(
                binding.buffer_name
            )

        chunks: list[MemoryChunk] = []
        global_index = 0
        stop_at = (
            chunk_offset + chunk_limit if chunk_limit is not None else None
        )
        for memory_name, memory in memories_with_objects:
            buffers = sorted(
                memory_to_buffers.get(memory_name, []), key=str.casefold
            )
            applicable_queries = [
                query for query in queries if query.buffer_name in buffers
            ]
            activations = getattr(memory, "activations", {})
            for memory_index, (chunk, traces) in enumerate(
                cls._memory_items(memory), start=1
            ):
                matched = cls._matching_query_ids(chunk, applicable_queries)
                if not matched:
                    continue
                if global_index < chunk_offset:
                    global_index += 1
                    continue
                if stop_at is not None and global_index >= stop_at:
                    break
                serialized = BufferHistoryRecorder.serialize_chunk(chunk)
                activation = None
                try:
                    raw_activation = activations.get(chunk)
                    activation = (
                        float(raw_activation)
                        if raw_activation is not None
                        else None
                    )
                except Exception:
                    activation = None
                chunks.append(
                    MemoryChunk(
                        chunk_id=f"{memory_name}:{memory_index}",
                        memory_name=memory_name,
                        chunk_type=str(serialized.get("type", "chunk")),
                        label=cls._chunk_label(serialized, memory_index),
                        slots=dict(serialized.get("slots", {})),
                        traces=cls._trace_values(traces),
                        activation=activation,
                        source="runtime_retrieval_candidate",
                        retrieval_buffers=buffers,
                        matched_queries=matched,
                    )
                )
                global_index += 1
            if stop_at is not None and global_index >= stop_at:
                break

        edges = cls.infer_edges(chunks)
        if max_edges is not None and len(edges) > max_edges:
            edges = edges[:max_edges]
        operations = [
            {
                "actor": f"buffer:{binding.buffer_name}",
                "mode": "retrieval_link",
                "memory_name": binding.memory_name,
                "detail": "DecMemBuffer.retrieve iterates this declarative memory",
            }
            for binding in bindings
        ]
        operations.extend(
            {
                "actor": f"production:{query.production_name}",
                "mode": "retrieval_query",
                "buffer_name": query.buffer_name,
                "chunk_type": query.chunk_type,
                "constraints": dict(query.constraints),
            }
            for query in queries
        )
        return DeclarativeMemorySnapshot(
            memories=[name for name, _memory in memories_with_objects],
            chunks=chunks,
            edges=edges,
            operations=operations,
            retrieval_buffers=sorted(
                {binding.buffer_name for binding in bindings},
                key=str.casefold,
            ),
            retrieval_memory_names=[
                name for name, _memory in memories_with_objects
            ],
            retrieval_queries=queries,
            scope="retrieval-query-matching",
        )

    @classmethod
    def filter_static_chunks(
        cls,
        chunks: Iterable[MemoryChunk],
        queries: Iterable[RetrievalQuery],
    ) -> list[MemoryChunk]:
        """Apply the same theoretical query matching to source-derived chunks."""
        query_list = list(queries)
        result: list[MemoryChunk] = []
        for chunk in chunks:
            matched = cls.matching_query_ids_for_serialized(
                chunk.chunk_type, chunk.slots, query_list
            )
            if not matched:
                continue
            chunk.matched_queries = matched
            chunk.retrieval_buffers = sorted(
                {
                    query.buffer_name
                    for query in query_list
                    if query.query_id in matched
                },
                key=str.casefold,
            )
            result.append(chunk)
        return result

    @classmethod
    def matching_query_ids_for_serialized(
        cls,
        chunk_type: str,
        slots: Mapping[str, Any],
        queries: Iterable[RetrievalQuery],
    ) -> list[str]:
        matched: list[str] = []
        for query in queries:
            if str(chunk_type) != str(query.chunk_type):
                continue
            if all(
                cls._equivalent(slots.get(name), expected)
                for name, expected in query.constraints.items()
            ):
                matched.append(query.query_id)
        return matched

    @classmethod
    def infer_edges(cls, chunks: list[MemoryChunk]) -> list[MemoryEdge]:
        """Infer only explicit slot-to-chunk identity references."""
        edges: list[MemoryEdge] = []
        seen: set[tuple[str, str, str]] = set()
        aliases: dict[str, list[MemoryChunk]] = {}
        for chunk in chunks:
            for alias in cls._aliases(chunk):
                aliases.setdefault(alias.casefold(), []).append(chunk)
        for chunk in chunks:
            for slot_name, raw_value in chunk.slots.items():
                value = cls._scalar(raw_value)
                if not value:
                    continue
                for target in aliases.get(value.casefold(), []):
                    if target.chunk_id == chunk.chunk_id:
                        continue
                    key = (chunk.chunk_id, target.chunk_id, str(slot_name))
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append(
                        MemoryEdge(
                            source_id=chunk.chunk_id,
                            target_id=target.chunk_id,
                            label=str(slot_name),
                            relation="slot_reference",
                        )
                    )
        return edges

    @classmethod
    def normalize_snapshot(cls, value: Any) -> DeclarativeMemorySnapshot:
        """Normalize legacy/static payloads and prevent GUI AttributeErrors."""
        if isinstance(value, DeclarativeMemorySnapshot):
            # Defensively normalize nested values because older cached analyses
            # may contain dictionaries or earlier dataclass versions.
            chunks = [
                cls._normalize_chunk(item)
                for item in getattr(value, "chunks", [])
            ]
            edges = [
                cls._normalize_edge(item)
                for item in getattr(value, "edges", [])
            ]
            queries = [
                cls._normalize_query(item)
                for item in getattr(value, "retrieval_queries", [])
            ]
            return DeclarativeMemorySnapshot(
                memories=[str(item) for item in getattr(value, "memories", [])],
                chunks=chunks,
                edges=edges,
                operations=[
                    dict(item)
                    for item in getattr(value, "operations", [])
                    if isinstance(item, Mapping)
                ],
                retrieval_buffers=[
                    str(item) for item in getattr(value, "retrieval_buffers", [])
                ],
                retrieval_memory_names=[
                    str(item)
                    for item in getattr(value, "retrieval_memory_names", [])
                ],
                retrieval_queries=queries,
                scope=str(getattr(value, "scope", "retrieval-query-matching")),
            )
        mapping = value if isinstance(value, Mapping) else {
            "memories": getattr(value, "memories", []),
            "chunks": getattr(value, "chunks", []),
            "edges": getattr(value, "edges", []),
            "operations": getattr(value, "operations", []),
            "retrieval_buffers": getattr(value, "retrieval_buffers", []),
            "retrieval_memory_names": getattr(value, "retrieval_memory_names", []),
            "retrieval_queries": getattr(value, "retrieval_queries", []),
            "scope": getattr(value, "scope", "retrieval-query-matching"),
        }
        return DeclarativeMemorySnapshot(
            memories=[str(item) for item in mapping.get("memories", [])],
            chunks=[cls._normalize_chunk(item) for item in mapping.get("chunks", [])],
            edges=[cls._normalize_edge(item) for item in mapping.get("edges", [])],
            operations=[dict(item) for item in mapping.get("operations", []) if isinstance(item, Mapping)],
            retrieval_buffers=[str(item) for item in mapping.get("retrieval_buffers", [])],
            retrieval_memory_names=[str(item) for item in mapping.get("retrieval_memory_names", [])],
            retrieval_queries=[cls._normalize_query(item) for item in mapping.get("retrieval_queries", [])],
            scope=str(mapping.get("scope", "retrieval-query-matching")),
        )

    @staticmethod
    def _normalize_chunk(value: Any) -> MemoryChunk:
        if isinstance(value, MemoryChunk):
            return MemoryChunk(
                chunk_id=str(value.chunk_id),
                memory_name=str(value.memory_name),
                chunk_type=str(value.chunk_type),
                label=str(value.label),
                slots=dict(value.slots),
                traces=list(value.traces),
                activation=value.activation,
                source=value.source,
                retrieval_buffers=list(getattr(value, "retrieval_buffers", [])),
                matched_queries=list(getattr(value, "matched_queries", [])),
            )
        mapping = value if isinstance(value, Mapping) else {
            "chunk_id": getattr(value, "chunk_id", "chunk"),
            "memory_name": getattr(value, "memory_name", "decmem"),
            "chunk_type": getattr(value, "chunk_type", "chunk"),
            "label": getattr(value, "label", "chunk"),
            "slots": getattr(value, "slots", {}),
            "traces": getattr(value, "traces", []),
            "activation": getattr(value, "activation", None),
            "source": getattr(value, "source", None),
            "retrieval_buffers": getattr(value, "retrieval_buffers", []),
            "matched_queries": getattr(value, "matched_queries", []),
        }
        return MemoryChunk(
            chunk_id=str(mapping.get("chunk_id", mapping.get("id", "chunk"))),
            memory_name=str(mapping.get("memory_name", "decmem")),
            chunk_type=str(mapping.get("chunk_type", mapping.get("type", "chunk"))),
            label=str(mapping.get("label", mapping.get("chunk_id", "chunk"))),
            slots=dict(mapping.get("slots", {})),
            traces=list(mapping.get("traces", [])),
            activation=mapping.get("activation"),
            source=mapping.get("source"),
            retrieval_buffers=list(mapping.get("retrieval_buffers", [])),
            matched_queries=list(mapping.get("matched_queries", [])),
        )

    @staticmethod
    def _normalize_edge(value: Any) -> MemoryEdge:
        if isinstance(value, MemoryEdge):
            return MemoryEdge(
                source_id=str(value.source_id),
                target_id=str(value.target_id),
                label=str(value.label),
                relation=str(value.relation),
            )
        mapping = value if isinstance(value, Mapping) else {
            "source_id": getattr(value, "source_id", ""),
            "target_id": getattr(value, "target_id", ""),
            "label": getattr(value, "label", "reference"),
            "relation": getattr(value, "relation", "slot_reference"),
        }
        return MemoryEdge(
            source_id=str(mapping.get("source_id", mapping.get("source", ""))),
            target_id=str(mapping.get("target_id", mapping.get("target", ""))),
            label=str(mapping.get("label", "reference")),
            relation=str(mapping.get("relation", "slot_reference")),
        )

    @staticmethod
    def _normalize_query(value: Any) -> RetrievalQuery:
        if isinstance(value, RetrievalQuery):
            return RetrievalQuery(
                query_id=str(value.query_id),
                production_name=str(value.production_name),
                buffer_name=str(value.buffer_name),
                chunk_type=str(value.chunk_type),
                constraints=dict(value.constraints),
            )
        mapping = value if isinstance(value, Mapping) else {
            "query_id": getattr(value, "query_id", "query"),
            "production_name": getattr(value, "production_name", "unknown"),
            "buffer_name": getattr(value, "buffer_name", "retrieval"),
            "chunk_type": getattr(value, "chunk_type", "chunk"),
            "constraints": getattr(value, "constraints", {}),
        }
        return RetrievalQuery(
            query_id=str(mapping.get("query_id", mapping.get("id", "query"))),
            production_name=str(mapping.get("production_name", "unknown")),
            buffer_name=str(mapping.get("buffer_name", "retrieval")),
            chunk_type=str(mapping.get("chunk_type", "chunk")),
            constraints=dict(mapping.get("constraints", {})),
        )

    @classmethod
    def _matching_query_ids(
        cls, chunk: Any, queries: Iterable[RetrievalQuery]
    ) -> list[str]:
        chunk_type = str(getattr(chunk, "typename", ""))
        if not chunk_type:
            serialized = BufferHistoryRecorder.serialize_chunk(chunk)
            chunk_type = str(serialized.get("type", "chunk"))
            slots = serialized.get("slots", {})
        else:
            try:
                slots = {str(name): value for name, value in chunk}
            except Exception:
                slots = BufferHistoryRecorder.serialize_chunk(chunk).get("slots", {})
        return cls.matching_query_ids_for_serialized(chunk_type, slots, queries)

    @staticmethod
    def _memory_items(memory: Any):
        try:
            return list(memory.items())
        except Exception:
            return []

    @staticmethod
    def _memory_chunks(memory: Any):
        try:
            return list(memory)
        except Exception:
            return []

    @staticmethod
    def _unique_bound_memories(
        bindings: list[RetrievalBinding],
    ) -> list[tuple[str, Any]]:
        result: list[tuple[str, Any]] = []
        seen: set[int] = set()
        for binding in bindings:
            identity = id(binding.memory)
            if identity in seen:
                continue
            seen.add(identity)
            result.append((binding.memory_name, binding.memory))
        return result

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
        identity = DeclarativeMemoryInspector._identity_from_slots(
            serialized.get("slots", {})
        )
        return identity or f"{chunk_type}_{index}"

    @staticmethod
    def _identity_from_slots(slots: Mapping[str, Any]) -> str | None:
        for key in (
            "entity_id",
            "relation_id",
            "strategy_id",
            "cell_id",
            "target_id",
            "episode_id",
            "name",
            "id",
            "key",
        ):
            scalar = DeclarativeMemoryInspector._scalar(slots.get(key))
            if scalar:
                return scalar
        return None

    @staticmethod
    def _aliases(chunk: MemoryChunk) -> set[str]:
        aliases = {chunk.chunk_id, chunk.label, chunk.chunk_type}
        for key in (
            "name",
            "id",
            "key",
            "entity_id",
            "relation_id",
            "strategy_id",
            "cell_id",
            "target_id",
            "episode_id",
        ):
            scalar = DeclarativeMemoryInspector._scalar(chunk.slots.get(key))
            if scalar:
                aliases.add(scalar)
        return aliases

    @staticmethod
    def _is_bound_variable(value: str | None) -> bool:
        return bool(value and (value.startswith("=") or value.startswith("~=")))

    @classmethod
    def _equivalent(cls, actual: Any, expected: Any) -> bool:
        return (cls._scalar(actual) or "").casefold() == (
            cls._scalar(expected) or ""
        ).casefold()

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
