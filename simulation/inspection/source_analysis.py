"""Static and executable-source analysis for ACT-R agent models and adapters.

The analyzer deliberately separates three concepts that were previously merged:

* production rules, including non-goal buffer guards and actions;
* adapter transitions that overwrite the goal buffer after a production fires;
* declarative-memory contents versus chunks that merely live in a buffer.

Whenever possible, productions are read from an instantiated pyactr model. This is
more reliable than reconstructing rules from Python syntax because it includes
rules generated in loops, helper methods, and f-strings.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import json
import re
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from simulation.discovery.agent_discovery import AgentTypeInfo
from simulation.inspection.declarative_memory import (
    DeclarativeMemoryInspector,
    DeclarativeMemorySnapshot,
    MemoryChunk,
    RetrievalQuery,
)


@dataclass(slots=True)
class ProductionAnalysis:
    """One fully expanded pyactr production."""

    name: str
    raw_string: str
    source_label: str
    target_label: str
    conditions: dict[str, dict[str, Any]]
    effects: dict[str, dict[str, Any]]
    read_buffers: list[str] = field(default_factory=list)
    written_buffers: list[str] = field(default_factory=list)
    reachable: bool = False
    self_loop: bool = False
    source_state_id: str = ""
    target_state_id: str = ""
    guard_label: str = ""
    action_label: str = ""
    utility: float | None = None
    reward: float | None = None


@dataclass(slots=True)
class ControlStateAnalysis:
    """Canonical control state derived from the primary goal buffer."""

    state_id: str
    label: str
    chunk_type: str
    slots: dict[str, Any]
    phase: str
    state: str
    reachable: bool = False
    terminal: bool = False
    dead_end: bool = False
    adapter_handoff: bool = False
    loop_member: bool = False


@dataclass(slots=True)
class StateTransitionAnalysis:
    """Production or adapter transition between canonical control states."""

    transition_id: str
    source_state_id: str
    target_state_id: str
    label: str
    kind: str
    guard_label: str = ""
    action_label: str = ""
    reachable: bool = False
    production_name: str | None = None
    adapter_method: str | None = None
    trigger_production: str | None = None
    utility: float | None = None
    reward: float | None = None


@dataclass(slots=True)
class MethodBufferInteraction:
    """A read/write/delete interaction between code and an ACT-R buffer."""

    method_name: str
    function_name: str
    buffer_name: str
    mode: str
    detail: str | None = None
    triggered_by: tuple[str, ...] = ()


@dataclass(slots=True)
class _StaticCallSite:
    """One call reached by the bounded source interpreter."""

    call: ast.Call
    constants: dict[str, Any]
    method_name: str
    conditions: tuple[str, ...] = ()


@dataclass(slots=True)
class _StaticAssignmentSite:
    """One assignment reached by the bounded source interpreter."""

    target: ast.AST
    value_node: ast.AST
    value: Any
    constants: dict[str, Any]
    method_name: str
    conditions: tuple[str, ...] = ()


@dataclass(slots=True)
class _StaticIfSite:
    """One conditional reached with its local constant environment."""

    node: ast.If
    constants: dict[str, Any]
    method_name: str
    conditions: tuple[str, ...] = ()


@dataclass(slots=True)
class _StaticExecutionTrace:
    calls: list[_StaticCallSite] = field(default_factory=list)
    assignments: list[_StaticAssignmentSite] = field(default_factory=list)
    conditionals: list[_StaticIfSite] = field(default_factory=list)


@dataclass(slots=True)
class AgentStaticAnalysis:
    """Complete explainability payload for one agent type."""

    agent_type: str
    model_path: str | None
    adapter_path: str | None
    model_source: str
    adapter_source: str
    class_summary: str
    adapter_summary: str
    initial_state: dict[str, dict[str, Any]]
    initial_state_label: str
    productions: list[ProductionAnalysis]
    unreachable_productions: list[str]
    dead_end_states: list[str]
    terminal_states: list[str]
    loop_states: list[str]
    adapter_interactions: list[MethodBufferInteraction]
    production_interactions: list[MethodBufferInteraction]
    declared_buffers: list[str]
    declarative_memory: DeclarativeMemorySnapshot
    states: dict[str, ControlStateAnalysis]
    transitions: list[StateTransitionAnalysis]
    initial_state_id: str
    analysis_warnings: list[str] = field(default_factory=list)

    def production(self, name: str) -> ProductionAnalysis | None:
        target = name.strip().casefold()
        return next(
            (item for item in self.productions if item.name.casefold() == target),
            None,
        )

    def state_for_id(self, state_id: str) -> ControlStateAnalysis | None:
        return self.states.get(state_id)

    def transition_path_to_production(
        self, name: str
    ) -> list[StateTransitionAnalysis] | None:
        """Return the shortest path including adapter transitions."""
        target = self.production(name)
        if target is None or not self.initial_state_id:
            return None
        queue: deque[tuple[str, list[StateTransitionAnalysis]]] = deque(
            [(self.initial_state_id, [])]
        )
        visited: set[str] = set()
        outgoing: dict[str, list[StateTransitionAnalysis]] = defaultdict(list)
        for transition in self.transitions:
            outgoing[transition.source_state_id].append(transition)
        while queue:
            state_id, path = queue.popleft()
            if state_id in visited:
                continue
            visited.add(state_id)
            for transition in outgoing.get(state_id, []):
                candidate = path + [transition]
                if (
                    transition.kind == "production"
                    and transition.production_name
                    and transition.production_name.casefold() == target.name.casefold()
                ):
                    return candidate
                if transition.target_state_id not in visited:
                    queue.append((transition.target_state_id, candidate))
        return None

    def path_to_production(self, name: str) -> list[ProductionAnalysis] | None:
        """Compatibility view returning only production edges from the full path."""
        transition_path = self.transition_path_to_production(name)
        if transition_path is None:
            return None
        production_by_name = {item.name: item for item in self.productions}
        result: list[ProductionAnalysis] = []
        for transition in transition_path:
            if transition.kind != "production" or not transition.production_name:
                continue
            production = production_by_name.get(transition.production_name)
            if production is not None:
                result.append(production)
        return result

    def state_sequence_for_transition_path(
        self, path: list[StateTransitionAnalysis]
    ) -> list[str]:
        if not path:
            state = self.states.get(self.initial_state_id)
            return [state.label] if state is not None else []
        labels: list[str] = []
        first = self.states.get(path[0].source_state_id)
        if first is not None:
            labels.append(first.label)
        for transition in path:
            target = self.states.get(transition.target_state_id)
            labels.append(target.label if target is not None else transition.target_state_id)
        return labels

    def state_sequence_for_path(
        self, path: list[ProductionAnalysis]
    ) -> list[str]:
        """Compatibility helper used by older jump visualizations."""
        if not path:
            return []
        full = self.transition_path_to_production(path[-1].name)
        return self.state_sequence_for_transition_path(full or [])


class AgentSourceAnalyzer:
    """Inspect executable model rules and adapter source control flow."""

    _BUFFER_FUNCTIONS: dict[str, tuple[str, str]] = {
        "get_goal": ("g", "read"),
        "set_goal": ("g", "write"),
        "get_imaginal": ("*", "read"),
        "set_imaginal": ("*", "write"),
        "get_buffer": ("*", "read"),
        "set_buffer": ("*", "write"),
        "replace_buffer": ("*", "write"),
        "get_declarative_memory": ("decmem", "read"),
        "add_to_declarative_memory": ("decmem", "write"),
        "delete_declarative_chunk_type": ("decmem", "delete"),
        "get_declarative_chunk_type": ("decmem", "read"),
    }
    _TERMINAL_VALUES = {
        "finished",
        "complete",
        "completed",
        "done",
        "terminal",
        "stopped",
        "halted",
        "success",
        "failed",
    }
    _UNRESOLVED = object()
    _FUNCTIONS_KEY = "__agent_source_functions__"
    _INSTANCE_DEFAULTS_KEY = "__agent_source_instance_defaults__"
    _STATIC_CHUNK_KEY = "__agent_source_static_chunk__"
    _MAX_STATIC_PATHS = 256
    _MAX_STATIC_LOOP_ITEMS = 128
    _MAX_STATIC_CALLS = 5_000
    _MAX_STATIC_DEPTH = 16

    def analyze(self, info: AgentTypeInfo) -> AgentStaticAnalysis:
        model_source = self._safe_read(info.model_path)
        adapter_source = self._safe_read(info.adapter_path)
        warnings_list: list[str] = []

        model_constants = self._extract_known_constants(model_source)
        adapter_constants = dict(model_constants)
        adapter_constants.update(
            self._extract_known_constants(
                adapter_source,
                inherited=model_constants,
                adapter_mode=True,
            )
        )

        source_initial_state = self._extract_initial_state(
            model_source, model_constants
        )
        source_productions = self._extract_productions_from_source(
            model_source, model_constants
        )
        source_buffers = self._extract_declared_buffers(
            model_source, model_constants
        )

        runtime = self._inspect_runtime_model(info)
        if runtime is None:
            warnings_list.append(
                "The model could not be instantiated for exact production analysis; "
                "the bounded source interpreter was used instead."
            )
            initial_state = source_initial_state
            productions = source_productions
            declared_buffers = source_buffers
            memory_names = ["decmem"]
            preferred_control_slots = None
        else:
            initial_state = runtime["initial_state"] or source_initial_state
            productions = list(runtime["productions"])
            runtime_names = {item.name for item in productions}
            source_only = [
                item for item in source_productions
                if item.name not in runtime_names
            ]
            if source_only:
                productions.extend(source_only)
                warnings_list.append(
                    "Some productions were visible only in source analysis and were "
                    "merged with the instantiated pyactr model."
                )
            declared_buffers = sorted(
                set(runtime["declared_buffers"]) | set(source_buffers),
                key=str.lower,
            )
            memory_names = runtime["memory_names"]
            preferred_control_slots = runtime.get("control_slots")

        control_slots = (
            list(preferred_control_slots)
            if preferred_control_slots
            else self._control_slots(productions, initial_state)
        )
        states: dict[str, ControlStateAnalysis] = {}
        production_transitions = self._build_production_transitions(
            productions, initial_state, control_slots, states
        )
        initial_state_id = self._ensure_control_state(
            states,
            initial_state.get("g", {}),
            control_slots,
        )

        dispatch = self._extract_adapter_dispatch(
            adapter_source,
            [item.name for item in productions],
            adapter_constants,
        )
        adapter_interactions = self._extract_adapter_interactions(
            adapter_source,
            adapter_constants,
            dispatch,
        )
        adapter_transitions = self._extract_adapter_transitions(
            adapter_source,
            adapter_constants,
            dispatch,
            productions,
            control_slots,
            states,
            adapter_interactions,
        )
        if adapter_transitions:
            warnings_list.append(
                "Adapter branches are potential static paths. Their runtime selection "
                "still depends on world state, pathfinding results, and sensor data."
            )
        transitions = production_transitions + adapter_transitions

        self._mark_graph_reachability(
            initial_state_id,
            states,
            transitions,
            productions,
        )
        self._classify_states(states, transitions)

        unreachable = sorted(
            (item.name for item in productions if not item.reachable),
            key=str.lower,
        )
        dead_ends = sorted(
            (item.label for item in states.values() if item.dead_end),
            key=str.lower,
        )
        terminals = sorted(
            (item.label for item in states.values() if item.terminal),
            key=str.lower,
        )
        loops = sorted(
            (item.label for item in states.values() if item.loop_member),
            key=str.lower,
        )

        production_interactions = self._production_interactions(productions)
        declarative_memory = self._extract_declarative_memory(
            model_source,
            adapter_source,
            model_constants,
            adapter_constants,
            declared_buffers,
            memory_names,
            productions,
        )

        return AgentStaticAnalysis(
            agent_type=info.name,
            model_path=info.model_path,
            adapter_path=info.adapter_path,
            model_source=model_source,
            adapter_source=adapter_source,
            class_summary=self._class_summary(model_source, info.name),
            adapter_summary=self._adapter_summary(adapter_source, info.name),
            initial_state=initial_state,
            initial_state_label=self._full_state_label(initial_state),
            productions=productions,
            unreachable_productions=unreachable,
            dead_end_states=dead_ends,
            terminal_states=terminals,
            loop_states=loops,
            adapter_interactions=adapter_interactions,
            production_interactions=production_interactions,
            declared_buffers=declared_buffers,
            declarative_memory=declarative_memory,
            states=states,
            transitions=transitions,
            initial_state_id=initial_state_id,
            analysis_warnings=warnings_list,
        )

    # ------------------------------------------------------------------
    # Exact production extraction from the built pyactr model
    # ------------------------------------------------------------------
    def _inspect_runtime_model(
        self, info: AgentTypeInfo
    ) -> dict[str, Any] | None:
        try:
            import pyactr as actr

            module = importlib.import_module(info.model_module)
            model_class = getattr(module, info.model_class_name or info.name)
            environment = actr.Environment(focus_position=(0, 0))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                construct = model_class(environment)
                model = construct.build_agent(["A"])
            initial_chunk = getattr(construct, "initial_goal", None)
            initial_state = (
                {"g": self._serialize_chunk(initial_chunk, "initial")}
                if initial_chunk is not None
                else {}
            )
            productions: list[ProductionAnalysis] = []
            for name in model.productions.keys():
                production = model.productions[name]
                generator = production["rule"]()
                lhs = next(generator)
                rhs = next(generator)
                conditions = self._runtime_rule_side(lhs, left_side=True)
                effects = self._runtime_rule_side(rhs, left_side=False)
                productions.append(
                    self._make_production(
                        str(name),
                        repr(production),
                        conditions,
                        effects,
                        utility=float(production["utility"]),
                        reward=(
                            None
                            if getattr(production, "reward", None) is None
                            else float(production.reward)
                        ),
                    )
                )
            buffers = getattr(model, "_ACTRModel__buffers", {})
            declared_buffers = sorted(
                (str(name) for name in buffers.keys()), key=str.lower
            )
            memory_names = sorted(
                (str(name) for name in getattr(model, "decmems", {}).keys()),
                key=str.lower,
            ) or ["decmem"]
            preferred_control_slots = getattr(
                construct, "analysis_control_slots", None
            )
            return {
                "initial_state": initial_state,
                "productions": productions,
                "declared_buffers": declared_buffers,
                "memory_names": memory_names,
                "control_slots": (
                    list(preferred_control_slots)
                    if preferred_control_slots
                    else None
                ),
            }
        except Exception:
            return None

    def _runtime_rule_side(
        self, side: Mapping[str, Any], *, left_side: bool
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for raw_key, value in side.items():
            key = str(raw_key)
            if not key:
                continue
            marker = key[0]
            buffer_name = key[1:]
            mode = (
                {
                    "=": "read",
                    "?": "query",
                    "+": "request",
                    "~": "clear",
                    "-": "clear",
                    "!": "execute",
                }.get(marker, "read")
                if left_side
                else {
                    "=": "write",
                    "+": "request",
                    "~": "clear",
                    "-": "clear",
                    "!": "execute",
                    "?": "query",
                }.get(marker, "write")
            )
            if isinstance(value, Mapping):
                payload = {
                    "mode": mode,
                    "type": None,
                    "slots": {
                        str(slot): self._clean_scalar(slot_value)
                        for slot, slot_value in value.items()
                        if self._clean_scalar(slot_value) is not None
                    },
                }
            else:
                payload = self._serialize_chunk(value, mode)
            result[buffer_name] = payload
        return result

    def _serialize_chunk(self, chunk: Any, mode: str) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        try:
            iterable = list(chunk)
        except Exception:
            iterable = []
        for slot_name, raw_value in iterable:
            value = self._clean_scalar(raw_value)
            if value is not None:
                slots[str(slot_name)] = value
        return {
            "mode": mode,
            "type": str(getattr(chunk, "typename", type(chunk).__name__)),
            "slots": slots,
        }

    @classmethod
    def _clean_scalar(cls, value: Any) -> Any:
        seen: set[int] = set()
        current = value
        for _ in range(6):
            if id(current) in seen:
                break
            seen.add(id(current))
            nested = getattr(current, "values", None)
            if nested is None or nested is current:
                break
            current = nested
        if current is None:
            return None
        text = str(current)
        if text in {"", "None", "nil"}:
            return None
        if isinstance(current, (str, int, float, bool)):
            return current
        return text

    def _make_production(
        self,
        name: str,
        raw_string: str,
        conditions: dict[str, dict[str, Any]],
        effects: dict[str, dict[str, Any]],
        *,
        utility: float | None = None,
        reward: float | None = None,
    ) -> ProductionAnalysis:
        read_buffers = sorted(conditions, key=str.lower)
        written_buffers = sorted(
            (
                buffer_name
                for buffer_name, payload in effects.items()
                if payload.get("mode") in {"write", "request", "clear"}
            ),
            key=str.lower,
        )
        return ProductionAnalysis(
            name=name,
            raw_string=raw_string,
            source_label=self._full_state_label(conditions),
            target_label=self._full_state_label(effects),
            conditions=conditions,
            effects=effects,
            read_buffers=read_buffers,
            written_buffers=written_buffers,
            utility=utility,
            reward=reward,
        )

    # ------------------------------------------------------------------
    # Canonical control-state graph
    # ------------------------------------------------------------------
    @staticmethod
    def _control_slots(
        productions: list[ProductionAnalysis],
        initial_state: dict[str, dict[str, Any]],
    ) -> list[str]:
        slots: set[str] = set()
        for production in productions:
            goal = production.conditions.get("g", {})
            slots.update(str(name) for name in goal.get("slots", {}))
        if not slots:
            slots.update(initial_state.get("g", {}).get("slots", {}))
        return sorted(slots, key=str.lower)

    def _build_production_transitions(
        self,
        productions: list[ProductionAnalysis],
        initial_state: dict[str, dict[str, Any]],
        control_slots: list[str],
        states: dict[str, ControlStateAnalysis],
    ) -> list[StateTransitionAnalysis]:
        transitions: list[StateTransitionAnalysis] = []
        for index, production in enumerate(productions, start=1):
            source_goal = production.conditions.get("g", {})
            source_id = self._ensure_control_state(states, source_goal, control_slots)
            target_goal = self._merge_goal_payload(
                source_goal,
                production.effects.get("g", {}),
            )
            target_id = self._ensure_control_state(states, target_goal, control_slots)
            guard = self._guard_label(production.conditions, control_slots)
            action = self._action_label(production.effects, control_slots)
            production.source_state_id = source_id
            production.target_state_id = target_id
            production.guard_label = guard
            production.action_label = action
            production.self_loop = source_id == target_id
            transitions.append(
                StateTransitionAnalysis(
                    transition_id=f"production:{index}:{production.name}",
                    source_state_id=source_id,
                    target_state_id=target_id,
                    label=production.name,
                    kind="production",
                    guard_label=guard,
                    action_label=action,
                    production_name=production.name,
                    utility=production.utility,
                    reward=production.reward,
                )
            )
        return transitions

    @staticmethod
    def _merge_goal_payload(
        source: dict[str, Any], effect: dict[str, Any]
    ) -> dict[str, Any]:
        merged = {
            "type": source.get("type") or effect.get("type") or "goal",
            "mode": effect.get("mode", "write"),
            "slots": dict(source.get("slots", {})),
        }
        merged["slots"].update(effect.get("slots", {}))
        if effect.get("type"):
            merged["type"] = effect["type"]
        return merged

    def _ensure_control_state(
        self,
        states: dict[str, ControlStateAnalysis],
        goal_payload: dict[str, Any],
        control_slots: list[str],
    ) -> str:
        payload_slots = goal_payload.get("slots", {})
        slots = {
            name: payload_slots.get(name, "*")
            for name in control_slots
        }
        chunk_type = str(goal_payload.get("type") or "goal")
        state_id = json.dumps(
            {"type": chunk_type, "slots": slots},
            sort_keys=True,
            default=str,
        )
        if state_id not in states:
            phase = str(slots.get("phase", ""))
            state_value = str(slots.get("state", ""))
            details = [
                f"{name}={value}"
                for name, value in slots.items()
                if str(value) != "*"
            ]
            label = (
                f"{phase} / {state_value}"
                if phase and state_value
                else "\n".join(details)
                if details
                else chunk_type
            )
            states[state_id] = ControlStateAnalysis(
                state_id=state_id,
                label=label,
                chunk_type=chunk_type,
                slots=slots,
                phase=phase or chunk_type,
                state=state_value or label,
            )
        return state_id

    def _guard_label(
        self,
        conditions: dict[str, dict[str, Any]],
        control_slots: list[str],
    ) -> str:
        parts: list[str] = []
        for buffer_name, payload in sorted(conditions.items()):
            slots = dict(payload.get("slots", {}))
            if buffer_name == "g":
                slots = {
                    key: value
                    for key, value in slots.items()
                    if key not in control_slots
                }
            if not slots and buffer_name == "g":
                continue
            details = ", ".join(
                f"{name}={value}" for name, value in sorted(slots.items())
            )
            mode = payload.get("mode")
            if details:
                parts.append(f"{buffer_name}: {details}")
            elif mode == "query":
                parts.append(f"{buffer_name}: query")
        return "\n".join(parts)

    def _action_label(
        self,
        effects: dict[str, dict[str, Any]],
        control_slots: list[str],
    ) -> str:
        parts: list[str] = []
        for buffer_name, payload in sorted(effects.items()):
            slots = dict(payload.get("slots", {}))
            if buffer_name == "g":
                slots = {
                    key: value
                    for key, value in slots.items()
                    if key not in control_slots
                }
                if not slots:
                    continue
            mode = str(payload.get("mode", "write"))
            details = ", ".join(
                f"{name}={value}" for name, value in sorted(slots.items())
            )
            parts.append(
                f"{mode} {buffer_name}" + (f": {details}" if details else "")
            )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Adapter dispatch, state overrides, and buffer interactions
    # ------------------------------------------------------------------
    def _extract_adapter_dispatch(
        self,
        source: str,
        production_names: list[str],
        constants: dict[str, Any] | None = None,
    ) -> dict[str, list[str]]:
        """Map adapter handler methods to the productions that trigger them.

        This deliberately keeps the complete structural traversal used by the
        original analyzer.  Constant-aware evaluation is added only for the
        comparison values, so f-strings and indexed variables do not narrow the
        traversed adapter call graph.
        """
        if not source.strip():
            return {}
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        extending = methods.get("extending_actr")
        if extending is None:
            return {}
        dispatch: dict[str, set[str]] = defaultdict(set)
        known = list(production_names)
        local_constants = dict(constants or {})
        for node in ast.walk(extending):
            if not isinstance(node, ast.If):
                continue
            triggers = self._production_names_from_test(
                node.test, known, local_constants
            )
            if not triggers:
                continue
            for statement in node.body:
                for call in ast.walk(statement):
                    if not isinstance(call, ast.Call):
                        continue
                    method = self._self_method_name(call.func)
                    if method and method != "extending_actr":
                        dispatch[method].update(triggers)

        # Supplement the complete structural pass with path-local constants.
        # This resolves loop/indexed f-strings such as f"{test[0]}" while the
        # structural pass remains authoritative for graph completeness.
        trace = self._contextual_trace(
            source,
            local_constants,
            adapter_mode=True,
            roots=("extending_actr",),
        )
        for site in trace.conditionals:
            if site.method_name != "extending_actr":
                continue
            triggers = self._production_names_from_test(
                site.node.test, known, site.constants
            )
            if not triggers:
                continue
            for statement in site.node.body:
                for call in ast.walk(statement):
                    if not isinstance(call, ast.Call):
                        continue
                    method = self._self_method_name(call.func)
                    if method and method != "extending_actr":
                        dispatch[method].update(triggers)

        # Structural pattern matching is treated equivalently to if/elif
        # dispatch, without replacing the broad traversal above.
        for match in (
            node for node in ast.walk(extending) if isinstance(node, ast.Match)
        ):
            if "production" not in self._node_text(match.subject).casefold():
                continue
            for case in match.cases:
                triggers = self._production_names_from_pattern(case.pattern, known)
                if not triggers:
                    continue
                for statement in case.body:
                    for call in ast.walk(statement):
                        if not isinstance(call, ast.Call):
                            continue
                        method = self._self_method_name(call.func)
                        if method and method != "extending_actr":
                            dispatch[method].update(triggers)
        return {
            method: sorted(values, key=str.lower)
            for method, values in dispatch.items()
        }

    def _production_names_from_test(
        self,
        test: ast.AST,
        production_names: list[str],
        constants: dict[str, Any] | None = None,
    ) -> set[str]:
        constants = constants or {}
        if isinstance(test, ast.BoolOp):
            result: set[str] = set()
            for value in test.values:
                result.update(
                    self._production_names_from_test(
                        value, production_names, constants
                    )
                )
            return result
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return set()

        known = set(production_names)
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            left_text = self._node_text(test.left).casefold()
            right_node = test.comparators[0]
            right_text = self._node_text(right_node).casefold()
            mentions_production = (
                "production" in left_text or "production" in right_text
            )
            if not mentions_production:
                return set()
            left = self._safe_eval(test.left, dict(constants), adapter_mode=True)
            right = self._safe_eval(right_node, dict(constants), adapter_mode=True)
            operator = test.ops[0]
            if isinstance(operator, ast.Eq):
                candidates: set[str] = set()
                for value in (left, right):
                    if value is self._UNRESOLVED:
                        continue
                    if isinstance(value, (set, tuple, list, frozenset)):
                        candidates.update(str(item) for item in value)
                    else:
                        candidates.add(str(value))
                return candidates & known
            if isinstance(operator, ast.In):
                if right is self._UNRESOLVED:
                    return set()
                values = (
                    right
                    if isinstance(right, (set, tuple, list, frozenset))
                    else [right]
                )
                return {str(item) for item in values} & known

        if (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Attribute)
            and test.args
            and "production" in self._node_text(test.func.value).casefold()
        ):
            value = self._safe_eval(test.args[0], dict(constants), adapter_mode=True)
            patterns = (
                list(value)
                if isinstance(value, (set, tuple, list, frozenset))
                else ([] if value is self._UNRESOLVED else [value])
            )
            if test.func.attr == "startswith":
                return {
                    name for name in production_names
                    if any(name.startswith(str(pattern)) for pattern in patterns)
                }
            if test.func.attr == "endswith":
                return {
                    name for name in production_names
                    if any(name.endswith(str(pattern)) for pattern in patterns)
                }
        return set()

    @staticmethod
    def _value_collection(value: Any) -> list[Any]:
        if value is AgentSourceAnalyzer._UNRESOLVED:
            return []
        if isinstance(value, (set, tuple, list, frozenset)):
            return list(value)
        return [value]

    def _production_names_from_pattern(
        self, pattern: ast.pattern, production_names: list[str]
    ) -> set[str]:
        if isinstance(pattern, ast.MatchOr):
            result: set[str] = set()
            for child in pattern.patterns:
                result.update(
                    self._production_names_from_pattern(child, production_names)
                )
            return result
        value = self._literal_pattern_value(pattern)
        if value is self._UNRESOLVED:
            return set()
        text = str(value)
        return {text} if text in set(production_names) else set()

    def _extract_adapter_transitions(
        self,
        source: str,
        constants: dict[str, Any],
        dispatch: dict[str, list[str]],
        productions: list[ProductionAnalysis],
        control_slots: list[str],
        states: dict[str, ControlStateAnalysis],
        interactions: list[MethodBufferInteraction],
    ) -> list[StateTransitionAnalysis]:
        if not source.strip() or not dispatch:
            return []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        method_calls: dict[str, set[str]] = defaultdict(set)
        for method_name, method in methods.items():
            for node in ast.walk(method):
                if not isinstance(node, ast.Call):
                    continue
                called = self._self_method_name(node.func)
                if called and called in methods and called != method_name:
                    method_calls[method_name].add(called)

        production_by_name = {item.name: item for item in productions}
        interactions_by_method: dict[str, list[MethodBufferInteraction]] = defaultdict(list)
        for interaction in interactions:
            interactions_by_method[interaction.method_name].append(interaction)

        set_goal_method = methods.get("_set_goal")
        set_goal_parameters = (
            [argument.arg for argument in set_goal_method.args.args if argument.arg != "self"]
            if set_goal_method is not None
            else ["phase", "state", "previous"]
        )

        transitions: list[StateTransitionAnalysis] = []
        sequence = 0
        for root_method_name, trigger_names in dispatch.items():
            root_method = methods.get(root_method_name)
            if root_method is None:
                continue
            reachable_methods = self._transitive_methods(root_method_name, method_calls)
            goal_calls: list[tuple[str, ast.AST, ast.Call]] = []
            for method_name in reachable_methods:
                method = methods.get(method_name)
                if method is None:
                    continue
                for node in ast.walk(method):
                    if (
                        isinstance(node, ast.Call)
                        and self._called_name(node.func) == "_set_goal"
                    ):
                        goal_calls.append((method_name, method, node))

            for trigger_name in trigger_names:
                production = production_by_name.get(trigger_name)
                if production is None:
                    continue
                for call_index, (call_method_name, call_method, call) in enumerate(goal_calls):
                    resolved_arguments: dict[str, Any] = {}
                    for parameter_index, parameter_name in enumerate(set_goal_parameters):
                        value = self._call_argument(
                            call,
                            parameter_index,
                            {parameter_name},
                            constants,
                            adapter_mode=True,
                        )
                        if value is not None:
                            resolved_arguments[parameter_name] = value
                    phase = resolved_arguments.get("phase")
                    state_value = resolved_arguments.get("state")
                    if not phase or not state_value:
                        continue

                    source_state = states.get(production.target_state_id)
                    source_slots = dict(source_state.slots) if source_state is not None else {}
                    target_slots: dict[str, Any] = {}
                    for slot_name in control_slots:
                        aliases = [slot_name]
                        if slot_name == "prev_phase":
                            aliases.extend(["previous", "previous_phase"])
                        value = next(
                            (
                                resolved_arguments[alias]
                                for alias in aliases
                                if alias in resolved_arguments
                            ),
                            source_slots.get(slot_name),
                        )
                        if value is not None:
                            target_slots[slot_name] = value
                    target_type = (
                        source_state.chunk_type
                        if source_state is not None and source_state.chunk_type
                        else "goal"
                    )
                    target_payload = {
                        "type": target_type,
                        "mode": "write",
                        "slots": target_slots,
                    }
                    target_id = self._ensure_control_state(
                        states, target_payload, control_slots
                    )
                    source_id = production.target_state_id
                    parent_map = self._parent_map(call_method)
                    condition = self._branch_condition(
                        call,
                        call_method,
                        parent_map,
                        multiple=len(goal_calls) > 1,
                    )
                    if call_method_name != root_method_name:
                        condition = (
                            f"via {call_method_name}: {condition}"
                            if condition
                            else f"via {call_method_name}"
                        )
                    writes = sorted(
                        {
                            item.buffer_name
                            for item in interactions_by_method.get(root_method_name, [])
                            if item.mode in {"write", "request", "delete", "clear"}
                        },
                        key=str.lower,
                    )
                    action = (
                        "writes " + ", ".join(writes)
                        if writes
                        else "overwrites goal state"
                    )
                    sequence += 1
                    transitions.append(
                        StateTransitionAnalysis(
                            transition_id=(
                                f"adapter:{sequence}:{trigger_name}:{root_method_name}:"
                                f"{call_method_name}:{call_index}"
                            ),
                            source_state_id=source_id,
                            target_state_id=target_id,
                            label=root_method_name,
                            kind="adapter",
                            guard_label=condition,
                            action_label=action,
                            adapter_method=root_method_name,
                            trigger_production=trigger_name,
                        )
                    )

        # Direct pyactrFunctionalityExtension/alias goal writes are used by
        # adapters that do not wrap the operation in a local ``_set_goal``
        # helper.  Run this only for roots without legacy transitions so the
        # complete branch structure above remains authoritative.
        roots_with_transitions = {
            item.adapter_method for item in transitions if item.adapter_method
        }
        for root_method_name, trigger_names in dispatch.items():
            if root_method_name in roots_with_transitions:
                continue
            trace = self._contextual_trace(
                source,
                constants,
                adapter_mode=True,
                roots=(root_method_name,),
            )
            direct_sites: list[tuple[_StaticCallSite, dict[str, Any]]] = []
            for site in trace.calls:
                call = site.call
                if self._called_name(call.func) != "set_goal":
                    continue
                # A self._set_goal call belongs to the legacy path above.
                if self._self_method_name(call.func) == "_set_goal":
                    continue
                chunk_node: ast.AST | None = None
                if len(call.args) >= 2:
                    chunk_node = call.args[1]
                else:
                    for keyword in call.keywords:
                        if keyword.arg in {"chunk", "goal", "value"}:
                            chunk_node = keyword.value
                            break
                if chunk_node is None:
                    continue
                value = self._safe_eval(
                    chunk_node, dict(site.constants), adapter_mode=True
                )
                payload = self._static_chunk_payload(value)
                if payload is None and isinstance(chunk_node, ast.Call):
                    payload = self._static_chunk_from_call(
                        chunk_node,
                        dict(site.constants),
                        True,
                        0,
                    )
                if payload is not None:
                    direct_sites.append((site, payload))

            for trigger_name in trigger_names:
                production = production_by_name.get(trigger_name)
                if production is None:
                    continue
                source_state = states.get(production.target_state_id)
                for call_index, (site, payload) in enumerate(direct_sites):
                    payload_slots = dict(payload.get("slots", {}))
                    source_slots = (
                        dict(source_state.slots) if source_state is not None else {}
                    )
                    target_slots = {
                        slot_name: payload_slots.get(
                            slot_name, source_slots.get(slot_name)
                        )
                        for slot_name in control_slots
                        if payload_slots.get(slot_name, source_slots.get(slot_name))
                        is not None
                    }
                    if not target_slots:
                        continue
                    target_payload = {
                        "type": (
                            payload.get("type")
                            or (
                                source_state.chunk_type
                                if source_state is not None
                                else "goal"
                            )
                        ),
                        "mode": "write",
                        "slots": target_slots,
                    }
                    target_id = self._ensure_control_state(
                        states, target_payload, control_slots
                    )
                    writes = sorted(
                        {
                            item.buffer_name
                            for item in interactions_by_method.get(
                                root_method_name, []
                            )
                            if item.mode
                            in {"write", "request", "delete", "clear"}
                        },
                        key=str.lower,
                    )
                    action = (
                        "writes " + ", ".join(writes)
                        if writes
                        else "overwrites goal state"
                    )
                    condition = (
                        " and ".join(site.conditions)
                        if site.conditions
                        else "after production fires"
                    )
                    sequence += 1
                    transitions.append(
                        StateTransitionAnalysis(
                            transition_id=(
                                f"adapter:{sequence}:{trigger_name}:"
                                f"{root_method_name}:{site.method_name}:"
                                f"direct_set_goal_{call_index}"
                            ),
                            source_state_id=production.target_state_id,
                            target_state_id=target_id,
                            label=root_method_name,
                            kind="adapter",
                            guard_label=condition,
                            action_label=action,
                            adapter_method=root_method_name,
                            trigger_production=trigger_name,
                        )
                    )
        return transitions

    def _adapter_goal_payload_from_call(
        self,
        call: ast.Call,
        constants: dict[str, Any],
        methods: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
        control_slots: list[str],
    ) -> dict[str, Any] | None:
        function_name = self._called_name(call.func)
        self_method = self._self_method_name(call.func)
        chunk_node: ast.AST | None = None

        if function_name == "set_goal" and not (
            self_method and self_method in methods
        ):
            # pyactrFunctionalityExtension.set_goal(agent_construct, chunk).
            # For directly imported wrappers, prefer an explicit ``chunk``
            # keyword and otherwise use the final positional argument.
            chunk_node = next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg in {"chunk", "new_chunk", "goal"}
                ),
                None,
            )
            if chunk_node is None and call.args:
                chunk_node = call.args[-1]
        elif function_name in {"replace_buffer", "set_buffer"}:
            buffer_name = self._call_argument(
                call,
                1,
                {"name", "key", "buffer_name"},
                constants,
                adapter_mode=True,
            )
            if str(buffer_name).casefold() not in {"g", "goal"}:
                return None
            chunk_node = (
                call.args[2]
                if len(call.args) > 2
                else next(
                    (
                        keyword.value
                        for keyword in call.keywords
                        if keyword.arg in {"chunk", "new_chunk"}
                    ),
                    None,
                )
            )
        elif function_name == "add":
            function_text = self._node_text(call.func).casefold()
            if "goal" in function_text and call.args:
                chunk_node = call.args[0]

        if chunk_node is not None:
            value = self._safe_eval(
                chunk_node, constants, adapter_mode=True
            )
            payload = self._static_chunk_payload(value)
            if payload is None and isinstance(chunk_node, ast.Call):
                payload = self._static_chunk_from_call(
                    chunk_node, constants, True, 0
                )
            if payload is not None:
                payload["mode"] = "write"
                return payload

        # Conservative fallback for an inherited/custom ``self._set_goal``
        # helper whose implementation is not visible in the adapter file.
        if (
            self_method
            and self_method not in methods
            and "set_goal" in self_method.casefold()
        ):
            values: dict[str, Any] = {}
            keyword_nodes = {
                keyword.arg: keyword.value
                for keyword in call.keywords
                if keyword.arg is not None
            }
            aliases = list(control_slots)
            for common in ("phase", "state", "previous", "prev_phase", "policy", "outcome"):
                if common not in aliases:
                    aliases.append(common)
            for index, name in enumerate(aliases):
                node = keyword_nodes.get(name)
                if node is None and index < len(call.args):
                    node = call.args[index]
                if node is None:
                    continue
                value = self._safe_eval(
                    node, constants, adapter_mode=True
                )
                if value is not self._UNRESOLVED:
                    values[name] = value
            slots: dict[str, Any] = {}
            for slot_name in control_slots:
                slot_aliases = [slot_name]
                if slot_name == "prev_phase":
                    slot_aliases.extend(("previous", "previous_phase"))
                for alias in slot_aliases:
                    if alias in values:
                        slots[slot_name] = values[alias]
                        break
            if slots:
                return {"type": "goal", "mode": "write", "slots": slots}
        return None

    def _extract_adapter_interactions(
        self,
        source: str,
        constants: dict[str, Any],
        dispatch: dict[str, list[str]],
    ) -> list[MethodBufferInteraction]:
        if not source.strip():
            return []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        calls: dict[str, set[str]] = defaultdict(set)
        for method_name, method in methods.items():
            for call in (
                node for node in ast.walk(method) if isinstance(node, ast.Call)
            ):
                called = self._self_method_name(call.func)
                if called and called in methods and called != method_name:
                    calls[method_name].add(called)

        trace = self._contextual_trace(
            source,
            constants,
            adapter_mode=True,
            roots=tuple(methods),
        )
        direct: dict[str, list[MethodBufferInteraction]] = defaultdict(list)
        direct_seen: set[tuple[str, str, str, str]] = set()
        for site in trace.calls:
            function_name = self._called_name(site.call.func)
            if function_name not in self._BUFFER_FUNCTIONS:
                continue
            default_buffer, mode = self._BUFFER_FUNCTIONS[function_name]
            buffer_name = default_buffer
            if default_buffer == "*":
                buffer_name = (
                    self._resolve_buffer_argument(
                        site.call, function_name, site.constants
                    )
                    or "dynamic"
                )
            key = (site.method_name, function_name, buffer_name, mode)
            if key in direct_seen:
                continue
            direct_seen.add(key)
            detail = self._call_excerpt(source, site.call)
            if site.conditions:
                condition = " and ".join(site.conditions)
                detail = (f"{detail} · when {condition}" if detail else condition)
            direct[site.method_name].append(
                MethodBufferInteraction(
                    method_name=site.method_name,
                    function_name=function_name,
                    buffer_name=buffer_name,
                    mode=mode,
                    detail=detail,
                )
            )

        roots = sorted(dispatch, key=str.lower)
        if not roots:
            roots = sorted(direct, key=str.lower)
        result: list[MethodBufferInteraction] = []
        for root in roots:
            reachable_methods = self._transitive_methods(root, calls)
            seen: set[tuple[str, str]] = set()
            for method_name in reachable_methods:
                for interaction in direct.get(method_name, []):
                    key = (interaction.buffer_name, interaction.mode)
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append(
                        MethodBufferInteraction(
                            method_name=root,
                            function_name=interaction.function_name,
                            buffer_name=interaction.buffer_name,
                            mode=interaction.mode,
                            detail=(
                                f"via {method_name}: {interaction.detail}"
                                if method_name != root
                                else interaction.detail
                            ),
                            triggered_by=tuple(dispatch.get(root, [])),
                        )
                    )
        return result

    @staticmethod
    def _transitive_methods(
        root: str, calls: dict[str, set[str]]
    ) -> list[str]:
        result: list[str] = []
        stack = [root]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            result.append(current)
            stack.extend(sorted(calls.get(current, set()), reverse=True))
        return result

    def _branch_condition(
        self,
        call: ast.Call,
        method: ast.FunctionDef,
        parent_map: dict[ast.AST, ast.AST],
        *,
        multiple: bool,
    ) -> str:
        conditions: list[str] = []
        current: ast.AST = call
        while current is not method:
            parent = parent_map.get(current)
            if parent is None:
                break
            if isinstance(parent, ast.If):
                test = self._node_text(parent.test)
                if self._contains_node(parent.body, current):
                    conditions.append(test)
                elif self._contains_node(parent.orelse, current):
                    conditions.append(f"not ({test})")
            current = parent
        conditions.reverse()
        if conditions:
            return " and ".join(value for value in conditions if value)
        return "otherwise / fall-through" if multiple else "after production fires"

    @staticmethod
    def _contains_node(statements: list[ast.stmt], target: ast.AST) -> bool:
        return any(target is node for statement in statements for node in ast.walk(statement))

    @staticmethod
    def _parent_map(root: ast.AST) -> dict[ast.AST, ast.AST]:
        result: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(root):
            for child in ast.iter_child_nodes(parent):
                result[child] = parent
        return result

    # ------------------------------------------------------------------
    # Reachability and classification
    # ------------------------------------------------------------------
    def _mark_graph_reachability(
        self,
        initial_state_id: str,
        states: dict[str, ControlStateAnalysis],
        transitions: list[StateTransitionAnalysis],
        productions: list[ProductionAnalysis],
    ) -> None:
        outgoing: dict[str, list[StateTransitionAnalysis]] = defaultdict(list)
        for transition in transitions:
            outgoing[transition.source_state_id].append(transition)
        queue: deque[str] = deque([initial_state_id])
        visited: set[str] = set()
        while queue:
            state_id = queue.popleft()
            if state_id in visited:
                continue
            visited.add(state_id)
            state = states.get(state_id)
            if state is not None:
                state.reachable = True
            for transition in outgoing.get(state_id, []):
                transition.reachable = True
                if transition.target_state_id not in visited:
                    queue.append(transition.target_state_id)
        reachable_productions = {
            transition.production_name
            for transition in transitions
            if transition.kind == "production" and transition.reachable
        }
        for production in productions:
            production.reachable = production.name in reachable_productions

    def _classify_states(
        self,
        states: dict[str, ControlStateAnalysis],
        transitions: list[StateTransitionAnalysis],
    ) -> None:
        outgoing: dict[str, list[StateTransitionAnalysis]] = defaultdict(list)
        graph: dict[str, set[str]] = defaultdict(set)
        for transition in transitions:
            outgoing[transition.source_state_id].append(transition)
            graph[transition.source_state_id].add(transition.target_state_id)
            graph.setdefault(transition.target_state_id, set())
            if transition.kind == "adapter":
                source = states.get(transition.source_state_id)
                if source is not None:
                    source.adapter_handoff = True
        for state in states.values():
            state_value = str(state.slots.get("state", "")).casefold()
            state.terminal = state_value in self._TERMINAL_VALUES
            state.dead_end = (
                state.reachable
                and not state.terminal
                and not outgoing.get(state.state_id)
            )
        for component in self._strongly_connected_components(graph):
            if len(component) > 1:
                for state_id in component:
                    if state_id in states:
                        states[state_id].loop_member = True
            elif component:
                only = next(iter(component))
                if only in graph.get(only, set()) and only in states:
                    states[only].loop_member = True

    @staticmethod
    def _strongly_connected_components(
        graph: dict[str, set[str]]
    ) -> list[set[str]]:
        index = 0
        stack: list[str] = []
        on_stack: set[str] = set()
        indices: dict[str, int] = {}
        lowlinks: dict[str, int] = {}
        components: list[set[str]] = []

        def visit(node: str) -> None:
            nonlocal index
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)
            for target in graph.get(node, set()):
                if target not in indices:
                    visit(target)
                    lowlinks[node] = min(lowlinks[node], lowlinks[target])
                elif target in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[target])
            if lowlinks[node] == indices[node]:
                component: set[str] = set()
                while stack:
                    item = stack.pop()
                    on_stack.remove(item)
                    component.add(item)
                    if item == node:
                        break
                components.append(component)

        for node in list(graph):
            if node not in indices:
                visit(node)
        return components

    # ------------------------------------------------------------------
    # Buffer interaction matrix payloads
    # ------------------------------------------------------------------
    def _production_interactions(
        self, productions: list[ProductionAnalysis]
    ) -> list[MethodBufferInteraction]:
        result: list[MethodBufferInteraction] = []
        for production in productions:
            for buffer_name in production.read_buffers:
                result.append(
                    MethodBufferInteraction(
                        method_name=production.name,
                        function_name="production condition",
                        buffer_name=buffer_name,
                        mode="read",
                        detail=production.guard_label or production.source_label,
                    )
                )
            for buffer_name in production.written_buffers:
                mode = str(production.effects.get(buffer_name, {}).get("mode", "write"))
                result.append(
                    MethodBufferInteraction(
                        method_name=production.name,
                        function_name="production effect",
                        buffer_name=buffer_name,
                        mode=mode,
                        detail=production.action_label or production.target_label,
                    )
                )
        return result

    # ------------------------------------------------------------------
    # Declarative memory: explicit contents vs linked buffers
    # ------------------------------------------------------------------
    def _extract_declarative_memory(
        self,
        model_source: str,
        adapter_source: str,
        model_constants: dict[str, Any],
        adapter_constants: dict[str, Any],
        declared_buffers: list[str],
        memory_names: list[str],
        productions: list[ProductionAnalysis],
    ) -> DeclarativeMemorySnapshot:
        chunks: list[MemoryChunk] = []
        operations: list[dict[str, Any]] = []
        memories = set(memory_names or ["decmem"])

        for source_name, source, constants, adapter_mode in (
            ("agent", model_source, model_constants, False),
            ("adapter", adapter_source, adapter_constants, True),
        ):
            explicit_chunks, explicit_operations = self._explicit_memory_writes(
                source,
                source_name,
                constants,
                adapter_mode,
            )
            chunks.extend(explicit_chunks)
            operations.extend(explicit_operations)

        retrieval_queries: list[RetrievalQuery] = []
        for production in productions:
            for buffer_name, effect in production.effects.items():
                if str(effect.get("mode", "")).casefold() != "request":
                    continue
                if "retrieval" not in str(buffer_name).casefold():
                    continue
                retrieval_queries.append(
                    RetrievalQuery(
                        query_id=f"{production.name}:{buffer_name}:{len(retrieval_queries) + 1}",
                        production_name=production.name,
                        buffer_name=str(buffer_name),
                        chunk_type=str(effect.get("type") or "chunk"),
                        constraints={
                            str(name): value
                            for name, value in dict(effect.get("slots", {})).items()
                            if not str(value).startswith("=")
                        },
                    )
                )

        retrieval_buffers = sorted(
            {query.buffer_name for query in retrieval_queries}
            or {
                name
                for name in declared_buffers
                if "retrieval" in str(name).casefold()
            },
            key=str.casefold,
        )
        retrieval_memory_names = sorted(memories, key=str.lower)
        if len(memories) == 1:
            memory_name = next(iter(memories))
            for buffer_name in retrieval_buffers:
                operations.append(
                    {
                        "actor": f"buffer:{buffer_name}",
                        "mode": "retrieval_link",
                        "memory_name": memory_name,
                        "detail": (
                            "pyactr DecMemBuffer.retrieve matches a production "
                            "request against chunks in this memory"
                        ),
                    }
                )
        operations.extend(
            {
                "actor": f"production:{query.production_name}",
                "mode": "retrieval_query",
                "buffer_name": query.buffer_name,
                "chunk_type": query.chunk_type,
                "constraints": dict(query.constraints),
            }
            for query in retrieval_queries
        )

        memory_filtered = [
            chunk for chunk in chunks
            if chunk.memory_name in retrieval_memory_names
        ]
        filtered_chunks = DeclarativeMemoryInspector.filter_static_chunks(
            memory_filtered, retrieval_queries
        )

        return DeclarativeMemorySnapshot(
            memories=retrieval_memory_names,
            chunks=filtered_chunks,
            edges=DeclarativeMemoryInspector.infer_edges(filtered_chunks),
            operations=operations,
            retrieval_buffers=retrieval_buffers,
            retrieval_memory_names=retrieval_memory_names,
            retrieval_queries=retrieval_queries,
            scope="retrieval-query-matching-static",
        )

    def _explicit_memory_writes(
        self,
        source: str,
        source_name: str,
        constants: dict[str, Any],
        adapter_mode: bool,
    ) -> tuple[list[MemoryChunk], list[dict[str, Any]]]:
        if not source.strip():
            return [], []
        trace = self._contextual_trace(
            source,
            constants,
            adapter_mode=adapter_mode,
            include_all_functions=True,
        )
        chunks: list[MemoryChunk] = []
        operations: list[dict[str, Any]] = []
        seen_chunks: set[tuple[str, str, str, str]] = set()
        seen_operations: set[tuple[str, str, str]] = set()
        sequence = 0
        for site in trace.calls:
            call = site.call
            function_name = self._called_name(call.func)
            func_text = self._node_text(call.func)
            if function_name == "add_to_declarative_memory":
                argument_index = 1
                memory_name = "decmem"
            elif function_name == "add" and (
                "decmem" in func_text or ".dm" in func_text
            ):
                argument_index = 0
                match = re.search(r'decmems?\[[\'"]([^\'"]+)', func_text)
                memory_name = match.group(1) if match else "decmem"
            else:
                continue
            argument_node: ast.AST | None = (
                call.args[argument_index]
                if len(call.args) > argument_index
                else next(
                    (
                        keyword.value
                        for keyword in call.keywords
                        if keyword.arg in {"chunk", "new_chunk"}
                    ),
                    None,
                )
            )
            expression = (
                self._node_text(argument_node) if argument_node is not None else None
            )
            payload: dict[str, Any] | None = None
            if argument_node is not None:
                value = self._safe_eval(
                    argument_node,
                    site.constants,
                    adapter_mode=adapter_mode,
                )
                payload = self._static_chunk_payload(value)
                if payload is None and isinstance(argument_node, ast.Call):
                    payload = self._static_chunk_from_call(
                        argument_node,
                        site.constants,
                        adapter_mode,
                        0,
                    )
            operation_key = (site.method_name, memory_name, expression or "")
            if operation_key not in seen_operations:
                seen_operations.add(operation_key)
                operations.append(
                    {
                        "actor": f"{source_name}.{site.method_name}",
                        "mode": "write",
                        "memory_name": memory_name,
                        "detail": expression or "dynamic chunk",
                    }
                )
            if payload is None:
                continue
            label = self._chunk_label(payload)
            signature = (
                site.method_name,
                memory_name,
                str(payload.get("type") or "chunk"),
                json.dumps(payload.get("slots", {}), sort_keys=True, default=str),
            )
            if signature in seen_chunks:
                continue
            seen_chunks.add(signature)
            sequence += 1
            chunk_id = (
                f"{source_name}:{site.method_name}:"
                f"{expression or 'inline'}:{sequence}"
            )
            chunks.append(
                MemoryChunk(
                    chunk_id=chunk_id,
                    memory_name=memory_name,
                    chunk_type=str(payload.get("type") or "chunk"),
                    label=label,
                    slots=dict(payload.get("slots", {})),
                    source="explicit_static",
                )
            )
        return chunks, operations

    def _chunk_payload_from_call(
        self,
        call: ast.Call,
        constants: dict[str, Any],
        adapter_mode: bool,
    ) -> dict[str, Any]:
        function_name = self._called_name(call.func)
        if function_name in {"chunkstring", "chunk_from_string"}:
            raw = self._call_argument(
                call,
                0,
                {"string"},
                constants,
                adapter_mode=adapter_mode,
            )
            if raw:
                raw = self._resolve_placeholders(
                    raw, constants, adapter_mode=adapter_mode
                )
                return self._parse_chunk_definition(raw)
        typename = self._call_argument(
            call,
            1,
            {"typename"},
            constants,
            adapter_mode=adapter_mode,
        ) or "chunk"
        slots: dict[str, Any] = {}
        for keyword in call.keywords:
            if keyword.arg in {"nameofchunk", "typename", "string"}:
                continue
            value = self._safe_eval(keyword.value, constants, adapter_mode)
            slots[str(keyword.arg)] = (
                self._node_text(keyword.value)
                if value is self._UNRESOLVED
                else value
            )
        return {"type": typename, "slots": slots, "mode": "static"}

    @staticmethod
    def _chunk_label(payload: dict[str, Any]) -> str:
        slots = payload.get("slots", {})
        for key in (
            "entity_id", "relation_id", "strategy_id", "cell_id",
            "target_id", "episode_id", "name", "id", "key",
        ):
            value = slots.get(key)
            if value not in (None, "", "None"):
                return str(value)
        return str(payload.get("type") or "chunk")

    # ------------------------------------------------------------------
    # Source fallbacks and safe literal resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_read(path: str | None) -> str:
        if not path:
            return ""
        try:
            return Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            return f"# Could not read source: {type(exc).__name__}: {exc}\n"

    def _extract_known_constants(
        self,
        source: str,
        *,
        inherited: dict[str, Any] | None = None,
        adapter_mode: bool = False,
    ) -> dict[str, Any]:
        """Collect safely resolvable constants without executing user code.

        The returned mapping also carries the source function definitions used by
        the bounded evaluator.  Local variables are resolved later in their real
        lexical/loop context by :meth:`_contextual_trace`.
        """
        constants: dict[str, Any] = dict(inherited or {})
        try:
            tree = ast.parse(source or "\n")
        except SyntaxError:
            return constants

        inherited_functions = dict(constants.get(self._FUNCTIONS_KEY, {}))
        inherited_functions.update(
            {
                node.name: node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
        )
        constants[self._FUNCTIONS_KEY] = inherited_functions
        class_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]

        assignments: list[ast.Assign | ast.AnnAssign | ast.NamedExpr] = []
        instance_target_keys: set[str] = set()

        def target_is_instance_attribute(target: ast.AST) -> bool:
            if isinstance(target, (ast.Tuple, ast.List)):
                return all(target_is_instance_attribute(item) for item in target.elts)
            if isinstance(target, ast.Starred):
                return target_is_instance_attribute(target.value)
            return (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in {"self", "cls"}
            )

        def collect_scope(
            statements: list[ast.stmt],
            *,
            allow_name_targets: bool,
            allow_instance_targets: bool,
        ) -> None:
            for statement in statements:
                if isinstance(statement, ast.Assign):
                    accepted = all(
                        (allow_name_targets and isinstance(target, ast.Name))
                        or (allow_instance_targets and target_is_instance_attribute(target))
                        for target in statement.targets
                    )
                    if accepted:
                        assignments.append(statement)
                    continue
                if isinstance(statement, ast.AnnAssign):
                    accepted = (
                        allow_name_targets and isinstance(statement.target, ast.Name)
                    ) or (
                        allow_instance_targets
                        and target_is_instance_attribute(statement.target)
                    )
                    if accepted and statement.value is not None:
                        assignments.append(statement)
                    continue
                if isinstance(statement, ast.ClassDef):
                    collect_scope(
                        [
                            item
                            for item in statement.body
                            if not isinstance(
                                item,
                                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                            )
                        ],
                        allow_name_targets=True,
                        allow_instance_targets=True,
                    )
                    for item in statement.body:
                        if isinstance(
                            item, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ) and item.name == "__init__":
                            init_assignments = [
                                node
                                for node in ast.walk(item)
                                if isinstance(
                                    node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)
                                )
                            ]
                            for node in init_assignments:
                                targets = (
                                    list(node.targets)
                                    if isinstance(node, ast.Assign)
                                    else [node.target]
                                )
                                if all(
                                    target_is_instance_attribute(target)
                                    for target in targets
                                ):
                                    assignments.append(node)
                                    for target in targets:
                                        key = self._target_key(target)
                                        if key:
                                            instance_target_keys.add(key)
                    continue
                if isinstance(statement, ast.If):
                    collect_scope(
                        statement.body + statement.orelse,
                        allow_name_targets=allow_name_targets,
                        allow_instance_targets=allow_instance_targets,
                    )
                elif isinstance(statement, ast.Try):
                    nested = list(statement.body) + list(statement.orelse) + list(statement.finalbody)
                    for handler in statement.handlers:
                        nested.extend(handler.body)
                    collect_scope(
                        nested,
                        allow_name_targets=allow_name_targets,
                        allow_instance_targets=allow_instance_targets,
                    )

        collect_scope(
            list(tree.body),
            allow_name_targets=True,
            allow_instance_targets=False,
        )
        for _ in range(16):
            changed = False
            for node in assignments:
                value_node = node.value
                value = self._safe_eval(
                    value_node, constants, adapter_mode
                )
                if value is self._UNRESOLVED:
                    continue
                targets: list[ast.AST]
                if isinstance(node, ast.Assign):
                    targets = list(node.targets)
                else:
                    targets = [node.target]
                before = dict(constants)
                for target in targets:
                    self._bind_static_target(target, value, constants)
                if constants != before:
                    changed = True
            if not changed:
                break

        # Class constants are frequently referenced through ``self`` from both
        # model and adapter code.  Keep all non-private scalar/container names as
        # aliases; local execution environments override them when necessary.
        for key, value in list(constants.items()):
            if not isinstance(key, str) or key.startswith("__") or "." in key:
                continue
            constants.setdefault(f"self.{key}", value)
            for class_name in class_names:
                constants.setdefault(f"{class_name}.{key}", value)

        inherited_defaults = dict(
            constants.get(self._INSTANCE_DEFAULTS_KEY, {})
        )
        for key in instance_target_keys:
            if key in constants:
                inherited_defaults[key] = constants[key]
            if key.startswith("self.") and key[5:] in constants:
                inherited_defaults[key[5:]] = constants[key[5:]]
        constants[self._INSTANCE_DEFAULTS_KEY] = inherited_defaults
        return constants

    @staticmethod
    def _target_key(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            try:
                return ast.unparse(node)
            except Exception:
                return None
        if isinstance(node, ast.Subscript):
            try:
                return ast.unparse(node)
            except Exception:
                return None
        return None

    def _bind_static_target(
        self,
        target: ast.AST,
        value: Any,
        constants: dict[str, Any],
    ) -> None:
        """Bind names, attributes and destructuring targets in a local scope."""
        if isinstance(target, (ast.Tuple, ast.List)):
            try:
                values = list(value)
            except Exception:
                return
            starred_index = next(
                (
                    index
                    for index, item in enumerate(target.elts)
                    if isinstance(item, ast.Starred)
                ),
                None,
            )
            if starred_index is None:
                if len(values) != len(target.elts):
                    return
                for item, item_value in zip(target.elts, values):
                    self._bind_static_target(item, item_value, constants)
                return
            prefix = target.elts[:starred_index]
            suffix = target.elts[starred_index + 1 :]
            if len(values) < len(prefix) + len(suffix):
                return
            for item, item_value in zip(prefix, values[: len(prefix)]):
                self._bind_static_target(item, item_value, constants)
            starred = target.elts[starred_index]
            assert isinstance(starred, ast.Starred)
            middle_end = len(values) - len(suffix)
            self._bind_static_target(
                starred.value, values[len(prefix) : middle_end], constants
            )
            if suffix:
                for item, item_value in zip(suffix, values[middle_end:]):
                    self._bind_static_target(item, item_value, constants)
            return
        if isinstance(target, ast.Starred):
            self._bind_static_target(target.value, value, constants)
            return
        key = self._target_key(target)
        if not key:
            return
        constants[key] = value
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                constants[f"{key}[{index}]"] = item
        if key.startswith("self."):
            constants.setdefault(key[5:], value)
        elif "." not in key and not key.startswith("__"):
            constants.setdefault(f"self.{key}", value)

    def _lookup_constant(
        self,
        expression: str,
        constants: Mapping[str, Any],
        adapter_mode: bool,
    ) -> Any:
        candidates = [expression]
        prefixes = (
            "self.agent_construct.actr_construct.",
            "agent_construct.actr_construct.",
            "self.agent_construct.",
            "agent_construct.",
            "self.actr_construct.",
            "actr_construct.",
        )
        for prefix in prefixes:
            if expression.startswith(prefix):
                suffix = expression[len(prefix) :]
                candidates.extend((f"self.{suffix}", suffix))
        if expression.startswith("self."):
            candidates.append(expression[5:])
        elif "." not in expression:
            candidates.append(f"self.{expression}")
        else:
            head, _, tail = expression.partition(".")
            if head[:1].isupper() and tail:
                candidates.extend((tail, f"self.{tail}"))
        for candidate in dict.fromkeys(candidates):
            if candidate in constants:
                return constants[candidate]
        return self._UNRESOLVED

    def _safe_eval(
        self,
        node: ast.AST,
        constants: dict[str, Any],
        adapter_mode: bool = False,
        _depth: int = 0,
    ) -> Any:
        """Evaluate a bounded, side-effect-free subset of Python syntax."""
        if _depth > self._MAX_STATIC_DEPTH:
            return self._UNRESOLVED
        try:
            return ast.literal_eval(node)
        except Exception:
            pass

        expression = self._node_text(node)
        looked_up = self._lookup_constant(expression, constants, adapter_mode)
        if looked_up is not self._UNRESOLVED:
            return looked_up

        if isinstance(node, ast.Name):
            return self._lookup_constant(node.id, constants, adapter_mode)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.NamedExpr):
            value = self._safe_eval(
                node.value, constants, adapter_mode, _depth + 1
            )
            if value is not self._UNRESOLVED:
                self._bind_static_target(node.target, value, constants)
            return value
        if isinstance(node, ast.Attribute):
            base = self._safe_eval(
                node.value, constants, adapter_mode, _depth + 1
            )
            if isinstance(base, Mapping) and node.attr in base:
                return base[node.attr]
            return self._UNRESOLVED
        if isinstance(node, ast.Slice):
            lower = (
                self._safe_eval(node.lower, constants, adapter_mode, _depth + 1)
                if node.lower is not None
                else None
            )
            upper = (
                self._safe_eval(node.upper, constants, adapter_mode, _depth + 1)
                if node.upper is not None
                else None
            )
            step = (
                self._safe_eval(node.step, constants, adapter_mode, _depth + 1)
                if node.step is not None
                else None
            )
            if any(value is self._UNRESOLVED for value in (lower, upper, step)):
                return self._UNRESOLVED
            return slice(lower, upper, step)
        if isinstance(node, ast.Subscript):
            base = self._safe_eval(
                node.value, constants, adapter_mode, _depth + 1
            )
            index = self._safe_eval(
                node.slice, constants, adapter_mode, _depth + 1
            )
            if base is self._UNRESOLVED or index is self._UNRESOLVED:
                return self._UNRESOLVED
            try:
                return base[index]
            except Exception:
                return self._UNRESOLVED
        if isinstance(node, ast.Starred):
            return self._safe_eval(
                node.value, constants, adapter_mode, _depth + 1
            )
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values: list[Any] = []
            for item in node.elts:
                value = self._safe_eval(
                    item, constants, adapter_mode, _depth + 1
                )
                if value is self._UNRESOLVED:
                    return self._UNRESOLVED
                if isinstance(item, ast.Starred):
                    try:
                        values.extend(value)
                    except Exception:
                        return self._UNRESOLVED
                else:
                    values.append(value)
            if isinstance(node, ast.Tuple):
                return tuple(values)
            if isinstance(node, ast.Set):
                try:
                    return set(values)
                except Exception:
                    return self._UNRESOLVED
            return values
        if isinstance(node, ast.Dict):
            result: dict[Any, Any] = {}
            for key_node, value_node in zip(node.keys, node.values):
                value = self._safe_eval(
                    value_node, constants, adapter_mode, _depth + 1
                )
                if value is self._UNRESOLVED:
                    return self._UNRESOLVED
                if key_node is None:
                    if not isinstance(value, Mapping):
                        return self._UNRESOLVED
                    result.update(value)
                    continue
                key = self._safe_eval(
                    key_node, constants, adapter_mode, _depth + 1
                )
                if key is self._UNRESOLVED:
                    return self._UNRESOLVED
                result[key] = value
            return result
        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            return self._eval_static_comprehension(
                node, constants, adapter_mode, _depth + 1
            )
        if isinstance(node, ast.UnaryOp):
            operand = self._safe_eval(
                node.operand, constants, adapter_mode, _depth + 1
            )
            if operand is self._UNRESOLVED:
                return self._UNRESOLVED
            try:
                if isinstance(node.op, ast.Not):
                    return not operand
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.UAdd):
                    return +operand
                if isinstance(node.op, ast.Invert):
                    return ~operand
            except Exception:
                return self._UNRESOLVED
        if isinstance(node, ast.BinOp):
            left = self._safe_eval(
                node.left, constants, adapter_mode, _depth + 1
            )
            right = self._safe_eval(
                node.right, constants, adapter_mode, _depth + 1
            )
            if left is self._UNRESOLVED or right is self._UNRESOLVED:
                return self._UNRESOLVED
            try:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
                if isinstance(node.op, ast.Mod):
                    return left % right
                if isinstance(node.op, ast.Pow):
                    return left**right
                if isinstance(node.op, ast.BitOr):
                    return left | right
                if isinstance(node.op, ast.BitAnd):
                    return left & right
                if isinstance(node.op, ast.BitXor):
                    return left ^ right
                if isinstance(node.op, ast.LShift):
                    return left << right
                if isinstance(node.op, ast.RShift):
                    return left >> right
            except Exception:
                return self._UNRESOLVED
        if isinstance(node, ast.BoolOp):
            values = [
                self._safe_eval(value, constants, adapter_mode, _depth + 1)
                for value in node.values
            ]
            if any(value is self._UNRESOLVED for value in values):
                return self._UNRESOLVED
            if isinstance(node.op, ast.And):
                result = values[0]
                for value in values[1:]:
                    if not result:
                        return result
                    result = value
                return result
            result = values[0]
            for value in values[1:]:
                if result:
                    return result
                result = value
            return result
        if isinstance(node, ast.Compare):
            left = self._safe_eval(
                node.left, constants, adapter_mode, _depth + 1
            )
            if left is self._UNRESOLVED:
                return self._UNRESOLVED
            for operator, comparator in zip(node.ops, node.comparators):
                right = self._safe_eval(
                    comparator, constants, adapter_mode, _depth + 1
                )
                if right is self._UNRESOLVED:
                    return self._UNRESOLVED
                result = self._safe_compare(left, operator, right)
                if result is self._UNRESOLVED or not result:
                    return result
                left = right
            return True
        if isinstance(node, ast.IfExp):
            test = self._safe_eval(
                node.test, constants, adapter_mode, _depth + 1
            )
            if test is self._UNRESOLVED:
                body = self._safe_eval(
                    node.body, constants, adapter_mode, _depth + 1
                )
                alternate = self._safe_eval(
                    node.orelse, constants, adapter_mode, _depth + 1
                )
                return body if body == alternate else self._UNRESOLVED
            return self._safe_eval(
                node.body if bool(test) else node.orelse,
                constants,
                adapter_mode,
                _depth + 1,
            )
        if isinstance(node, ast.FormattedValue):
            value = self._safe_eval(
                node.value, constants, adapter_mode, _depth + 1
            )
            if value is self._UNRESOLVED:
                return self._UNRESOLVED
            return self._format_fstring_value(
                value, node, constants, adapter_mode, _depth + 1
            )
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value_node in node.values:
                if isinstance(value_node, ast.Constant):
                    parts.append(str(value_node.value))
                    continue
                if not isinstance(value_node, ast.FormattedValue):
                    return self._UNRESOLVED
                resolved = self._safe_eval(
                    value_node.value, constants, adapter_mode, _depth + 1
                )
                if resolved is self._UNRESOLVED:
                    conversion = (
                        f"!{chr(value_node.conversion)}"
                        if value_node.conversion != -1
                        else ""
                    )
                    spec = ""
                    if value_node.format_spec is not None:
                        spec_text = self._safe_eval(
                            value_node.format_spec,
                            constants,
                            adapter_mode,
                            _depth + 1,
                        )
                        if spec_text is not self._UNRESOLVED:
                            spec = f":{spec_text}"
                    parts.append(
                        "{" + self._node_text(value_node.value) + conversion + spec + "}"
                    )
                else:
                    formatted = self._format_fstring_value(
                        resolved,
                        value_node,
                        constants,
                        adapter_mode,
                        _depth + 1,
                    )
                    if formatted is self._UNRESOLVED:
                        return self._UNRESOLVED
                    parts.append(str(formatted))
            return "".join(parts)
        if isinstance(node, ast.Call):
            return self._safe_eval_call(
                node, constants, adapter_mode, _depth + 1
            )
        return self._UNRESOLVED

    @staticmethod
    def _safe_compare(left: Any, operator: ast.cmpop, right: Any) -> Any:
        try:
            if isinstance(operator, ast.Eq):
                return left == right
            if isinstance(operator, ast.NotEq):
                return left != right
            if isinstance(operator, ast.Lt):
                return left < right
            if isinstance(operator, ast.LtE):
                return left <= right
            if isinstance(operator, ast.Gt):
                return left > right
            if isinstance(operator, ast.GtE):
                return left >= right
            if isinstance(operator, ast.In):
                return left in right
            if isinstance(operator, ast.NotIn):
                return left not in right
            if isinstance(operator, ast.Is):
                return left is right
            if isinstance(operator, ast.IsNot):
                return left is not right
        except Exception:
            return AgentSourceAnalyzer._UNRESOLVED
        return AgentSourceAnalyzer._UNRESOLVED

    def _format_fstring_value(
        self,
        value: Any,
        node: ast.FormattedValue,
        constants: dict[str, Any],
        adapter_mode: bool,
        depth: int,
    ) -> Any:
        try:
            if node.conversion == ord("r"):
                value = repr(value)
            elif node.conversion == ord("s"):
                value = str(value)
            elif node.conversion == ord("a"):
                value = ascii(value)
            format_spec = ""
            if node.format_spec is not None:
                resolved_spec = self._safe_eval(
                    node.format_spec, constants, adapter_mode, depth + 1
                )
                if resolved_spec is self._UNRESOLVED:
                    return self._UNRESOLVED
                format_spec = str(resolved_spec)
            return format(value, format_spec)
        except Exception:
            return self._UNRESOLVED

    def _safe_eval_call(
        self,
        node: ast.Call,
        constants: dict[str, Any],
        adapter_mode: bool,
        depth: int,
    ) -> Any:
        function_name = self._called_name(node.func)
        args = [
            self._safe_eval(item, constants, adapter_mode, depth + 1)
            for item in node.args
        ]
        keywords = {
            keyword.arg: self._safe_eval(
                keyword.value, constants, adapter_mode, depth + 1
            )
            for keyword in node.keywords
            if keyword.arg is not None
        }

        if function_name in {
            "chunkstring",
            "chunk_from_string",
            "makechunk",
            "build_chunkstring_by_tuples",
        }:
            payload = self._static_chunk_from_call(
                node, constants, adapter_mode, depth + 1
            )
            if payload is not None:
                return {self._STATIC_CHUNK_KEY: payload}

        if any(value is self._UNRESOLVED for value in args) or any(
            value is self._UNRESOLVED for value in keywords.values()
        ):
            # A source helper may still resolve the call because defaults or an
            # unused argument can make the unresolved value irrelevant.
            helper = self._source_function_for_call(node, constants)
            if helper is not None:
                return self._evaluate_source_function(
                    helper, node, constants, adapter_mode, depth + 1
                )
            return self._UNRESOLVED

        safe_builtins: dict[str, Any] = {
            "str": str,
            "repr": repr,
            "ascii": ascii,
            "int": int,
            "float": float,
            "bool": bool,
            "len": len,
            "list": list,
            "tuple": tuple,
            "set": set,
            "frozenset": frozenset,
            "dict": dict,
            "sorted": sorted,
            "reversed": lambda value: list(reversed(value)),
            "min": min,
            "max": max,
            "sum": sum,
            "range": lambda *values: list(range(*values)),
            "enumerate": lambda value, start=0: list(enumerate(value, start)),
            "zip": lambda *values: list(zip(*values)),
            "format": format,
        }
        if isinstance(node.func, ast.Name) and function_name in safe_builtins:
            try:
                return safe_builtins[function_name](*args, **keywords)
            except Exception:
                return self._UNRESOLVED

        if isinstance(node.func, ast.Name) and function_name == "getattr":
            if len(node.args) < 2:
                return self._UNRESOLVED
            attribute = args[1]
            if not isinstance(attribute, str):
                return self._UNRESOLVED
            base_text = self._node_text(node.args[0])
            value = self._lookup_constant(
                f"{base_text}.{attribute}", constants, adapter_mode
            )
            if value is not self._UNRESOLVED:
                return value
            return args[2] if len(args) > 2 else self._UNRESOLVED

        if isinstance(node.func, ast.Attribute):
            base = self._safe_eval(
                node.func.value, constants, adapter_mode, depth + 1
            )
            if base is not self._UNRESOLVED:
                method = node.func.attr
                allowed_methods = {
                    "lower",
                    "upper",
                    "casefold",
                    "strip",
                    "lstrip",
                    "rstrip",
                    "removeprefix",
                    "removesuffix",
                    "replace",
                    "capitalize",
                    "title",
                    "swapcase",
                    "zfill",
                    "format",
                    "format_map",
                    "join",
                    "split",
                    "rsplit",
                    "partition",
                    "rpartition",
                    "startswith",
                    "endswith",
                    "get",
                    "keys",
                    "values",
                    "items",
                    "copy",
                    "index",
                    "count",
                }
                if method in allowed_methods:
                    try:
                        result = getattr(base, method)(*args, **keywords)
                        if method in {"keys", "values", "items"}:
                            return list(result)
                        return result
                    except Exception:
                        return self._UNRESOLVED

        helper = self._source_function_for_call(node, constants)
        if helper is not None:
            return self._evaluate_source_function(
                helper, node, constants, adapter_mode, depth + 1
            )
        return self._UNRESOLVED

    def _source_function_for_call(
        self, node: ast.Call, constants: Mapping[str, Any]
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        functions = constants.get(self._FUNCTIONS_KEY, {})
        if not isinstance(functions, Mapping):
            return None
        name = self._called_name(node.func)
        candidate = functions.get(name) if name else None
        return (
            candidate
            if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef))
            else None
        )

    def _evaluate_source_function(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        call: ast.Call,
        constants: dict[str, Any],
        adapter_mode: bool,
        depth: int,
    ) -> Any:
        if depth > self._MAX_STATIC_DEPTH:
            return self._UNRESOLVED
        environment = self._bind_function_call(
            function, call, constants, adapter_mode, depth + 1
        )
        active, returns = self._evaluate_function_block(
            list(function.body),
            [environment],
            adapter_mode,
            depth + 1,
        )
        _ = active
        resolved = [value for value in returns if value is not self._UNRESOLVED]
        if not resolved:
            return None if returns else self._UNRESOLVED
        first = resolved[0]
        return first if all(value == first for value in resolved[1:]) else self._UNRESOLVED

    def _evaluate_function_block(
        self,
        statements: list[ast.stmt],
        environments: list[dict[str, Any]],
        adapter_mode: bool,
        depth: int,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        active = environments[: self._MAX_STATIC_PATHS]
        returns: list[Any] = []
        for statement in statements:
            next_active: list[dict[str, Any]] = []
            for environment in active:
                if isinstance(statement, ast.Return):
                    value = (
                        None
                        if statement.value is None
                        else self._safe_eval(
                            statement.value,
                            environment,
                            adapter_mode,
                            depth + 1,
                        )
                    )
                    returns.append(value)
                    continue
                if isinstance(statement, ast.Assign):
                    value = self._safe_eval(
                        statement.value, environment, adapter_mode, depth + 1
                    )
                    updated = dict(environment)
                    if value is not self._UNRESOLVED:
                        for target in statement.targets:
                            self._bind_static_target(target, value, updated)
                    next_active.append(updated)
                    continue
                if isinstance(statement, ast.AnnAssign):
                    value = (
                        self._safe_eval(
                            statement.value,
                            environment,
                            adapter_mode,
                            depth + 1,
                        )
                        if statement.value is not None
                        else self._UNRESOLVED
                    )
                    updated = dict(environment)
                    if value is not self._UNRESOLVED:
                        self._bind_static_target(statement.target, value, updated)
                    next_active.append(updated)
                    continue
                if isinstance(statement, ast.If):
                    test = self._safe_eval(
                        statement.test, environment, adapter_mode, depth + 1
                    )
                    branches: list[list[ast.stmt]]
                    if test is self._UNRESOLVED:
                        branches = [statement.body, statement.orelse]
                    else:
                        branches = [statement.body if bool(test) else statement.orelse]
                    for branch in branches:
                        branch_active, branch_returns = self._evaluate_function_block(
                            branch,
                            [dict(environment)],
                            adapter_mode,
                            depth + 1,
                        )
                        next_active.extend(branch_active)
                        returns.extend(branch_returns)
                    continue
                if isinstance(statement, (ast.For, ast.AsyncFor)):
                    iterable = self._safe_eval(
                        statement.iter, environment, adapter_mode, depth + 1
                    )
                    items = self._bounded_static_iterable(iterable)
                    if items is None:
                        items = [self._UNRESOLVED]
                    for item in items:
                        loop_env = dict(environment)
                        if item is not self._UNRESOLVED:
                            self._bind_static_target(statement.target, item, loop_env)
                        branch_active, branch_returns = self._evaluate_function_block(
                            statement.body,
                            [loop_env],
                            adapter_mode,
                            depth + 1,
                        )
                        next_active.extend(branch_active)
                        returns.extend(branch_returns)
                    continue
                next_active.append(dict(environment))
            active = next_active[: self._MAX_STATIC_PATHS]
            if not active:
                break
        return active, returns

    def _bind_function_call(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        call: ast.Call | None,
        constants: dict[str, Any],
        adapter_mode: bool,
        depth: int = 0,
    ) -> dict[str, Any]:
        environment = dict(constants)
        if function.name != "__init__":
            instance_defaults = constants.get(
                self._INSTANCE_DEFAULTS_KEY, {}
            )
            if isinstance(instance_defaults, Mapping):
                for key, default in instance_defaults.items():
                    if environment.get(key, self._UNRESOLVED) == default:
                        environment.pop(key, None)
        positional = list(function.args.posonlyargs) + list(function.args.args)
        if positional and positional[0].arg in {"self", "cls"}:
            positional = positional[1:]
        defaults = list(function.args.defaults)
        default_offset = len(positional) - len(defaults)
        for index, parameter in enumerate(positional):
            if index >= default_offset:
                default = defaults[index - default_offset]
                value = self._safe_eval(
                    default, environment, adapter_mode, depth + 1
                )
                if value is not self._UNRESOLVED:
                    environment[parameter.arg] = value
        for parameter, default in zip(
            function.args.kwonlyargs, function.args.kw_defaults
        ):
            if default is None:
                continue
            value = self._safe_eval(
                default, environment, adapter_mode, depth + 1
            )
            if value is not self._UNRESOLVED:
                environment[parameter.arg] = value
        if call is None:
            if function.name == "build_agent":
                for parameter in positional:
                    if parameter.arg in {"agent_list", "agents", "agent_keys"}:
                        environment.setdefault(parameter.arg, ["A"])
            return environment

        for index, argument in enumerate(call.args):
            if index >= len(positional):
                break
            value = self._safe_eval(
                argument, constants, adapter_mode, depth + 1
            )
            if value is not self._UNRESOLVED:
                environment[positional[index].arg] = value
        keyword_targets = {parameter.arg for parameter in positional}
        keyword_targets.update(parameter.arg for parameter in function.args.kwonlyargs)
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg not in keyword_targets:
                continue
            value = self._safe_eval(
                keyword.value, constants, adapter_mode, depth + 1
            )
            if value is not self._UNRESOLVED:
                environment[keyword.arg] = value
        return environment

    def _eval_static_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        constants: dict[str, Any],
        adapter_mode: bool,
        depth: int,
    ) -> Any:
        environments = [dict(constants)]
        for generator in node.generators:
            expanded: list[dict[str, Any]] = []
            for environment in environments:
                iterable = self._safe_eval(
                    generator.iter, environment, adapter_mode, depth + 1
                )
                items = self._bounded_static_iterable(iterable)
                if items is None:
                    return self._UNRESOLVED
                for item in items:
                    candidate = dict(environment)
                    self._bind_static_target(generator.target, item, candidate)
                    accepted = True
                    for condition in generator.ifs:
                        value = self._safe_eval(
                            condition, candidate, adapter_mode, depth + 1
                        )
                        if value is self._UNRESOLVED or not bool(value):
                            accepted = False
                            break
                    if accepted:
                        expanded.append(candidate)
            environments = expanded[: self._MAX_STATIC_PATHS]
        if isinstance(node, ast.DictComp):
            result: dict[Any, Any] = {}
            for environment in environments:
                key = self._safe_eval(
                    node.key, environment, adapter_mode, depth + 1
                )
                value = self._safe_eval(
                    node.value, environment, adapter_mode, depth + 1
                )
                if key is self._UNRESOLVED or value is self._UNRESOLVED:
                    return self._UNRESOLVED
                result[key] = value
            return result
        values = [
            self._safe_eval(node.elt, environment, adapter_mode, depth + 1)
            for environment in environments
        ]
        if any(value is self._UNRESOLVED for value in values):
            return self._UNRESOLVED
        if isinstance(node, ast.SetComp):
            return set(values)
        if isinstance(node, ast.GeneratorExp):
            return tuple(values)
        return values

    def _static_chunk_from_call(
        self,
        call: ast.Call,
        constants: dict[str, Any],
        adapter_mode: bool,
        depth: int,
    ) -> dict[str, Any] | None:
        function_name = self._called_name(call.func)
        if function_name in {"chunkstring", "chunk_from_string"}:
            node = call.args[0] if call.args else next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == "string"
                ),
                None,
            )
            if node is None:
                return None
            raw = self._safe_eval(node, constants, adapter_mode, depth + 1)
            if not isinstance(raw, str):
                return None
            return self._parse_chunk_definition(
                self._resolve_placeholders(
                    raw, constants, adapter_mode=adapter_mode
                )
            )
        if function_name == "build_chunkstring_by_tuples":
            node = call.args[0] if call.args else next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == "pairs"
                ),
                None,
            )
            if node is None:
                return None
            pairs = self._safe_eval(node, constants, adapter_mode, depth + 1)
            if pairs is self._UNRESOLVED:
                return None
            try:
                raw = "\n".join(f"{slot} {value}" for slot, value in pairs)
            except Exception:
                return None
            return self._parse_chunk_definition(raw)
        if function_name == "makechunk":
            typename_node = (
                call.args[1]
                if len(call.args) > 1
                else next(
                    (
                        keyword.value
                        for keyword in call.keywords
                        if keyword.arg == "typename"
                    ),
                    None,
                )
            )
            chunk_type = (
                self._safe_eval(
                    typename_node, constants, adapter_mode, depth + 1
                )
                if typename_node is not None
                else "chunk"
            )
            if chunk_type is self._UNRESOLVED:
                chunk_type = "chunk"
            slots: dict[str, Any] = {}
            for keyword in call.keywords:
                if keyword.arg in {"nameofchunk", "typename", "string", None}:
                    continue
                value = self._safe_eval(
                    keyword.value, constants, adapter_mode, depth + 1
                )
                if value is self._UNRESOLVED:
                    value = self._node_text(keyword.value)
                slots[str(keyword.arg)] = value
            return {"type": str(chunk_type), "slots": slots, "mode": "static"}
        return None

    def _static_chunk_payload(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        payload = value.get(self._STATIC_CHUNK_KEY)
        return dict(payload) if isinstance(payload, Mapping) else None

    def _bounded_static_iterable(self, value: Any) -> list[Any] | None:
        if value is self._UNRESOLVED or isinstance(value, (str, bytes)):
            return None
        try:
            items = list(value)
        except Exception:
            return None
        return items[: self._MAX_STATIC_LOOP_ITEMS]

    def _contextual_trace(
        self,
        source: str,
        constants: dict[str, Any],
        *,
        adapter_mode: bool = False,
        roots: Iterable[str] | None = None,
        include_all_functions: bool = False,
    ) -> _StaticExecutionTrace:
        trace = _StaticExecutionTrace()
        try:
            tree = ast.parse(source or "\n")
        except SyntaxError:
            return trace
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        selected = list(roots or ())
        if not selected:
            selected = ["build_agent"] if "build_agent" in methods else list(methods)
        executed: set[str] = set()
        for name in selected:
            function = methods.get(name)
            if function is None:
                continue
            environment = self._bind_function_call(
                function, None, constants, adapter_mode
            )
            self._execute_function_trace(
                function,
                environment,
                methods,
                adapter_mode,
                trace,
                stack=(),
                conditions=(),
            )
            executed.add(name)
        if include_all_functions:
            for name, function in methods.items():
                if name in executed or len(trace.calls) >= self._MAX_STATIC_CALLS:
                    continue
                environment = self._bind_function_call(
                    function, None, constants, adapter_mode
                )
                self._execute_function_trace(
                    function,
                    environment,
                    methods,
                    adapter_mode,
                    trace,
                    stack=(),
                    conditions=(),
                )
        module_statements = [
            statement
            for statement in tree.body
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        self._execute_static_block(
            module_statements,
            [(dict(constants), ())],
            methods,
            adapter_mode,
            trace,
            method_name="<module>",
            stack=(),
        )
        return trace

    def _execute_function_trace(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        environment: dict[str, Any],
        methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
        adapter_mode: bool,
        trace: _StaticExecutionTrace,
        *,
        stack: tuple[str, ...],
        conditions: tuple[str, ...],
    ) -> None:
        if (
            function.name in stack
            or len(stack) >= self._MAX_STATIC_DEPTH
            or len(trace.calls) >= self._MAX_STATIC_CALLS
        ):
            return
        self._execute_static_block(
            list(function.body),
            [(environment, conditions)],
            methods,
            adapter_mode,
            trace,
            method_name=function.name,
            stack=stack + (function.name,),
        )

    def _execute_static_block(
        self,
        statements: list[ast.stmt],
        paths: list[tuple[dict[str, Any], tuple[str, ...]]],
        methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
        adapter_mode: bool,
        trace: _StaticExecutionTrace,
        *,
        method_name: str,
        stack: tuple[str, ...],
    ) -> list[tuple[dict[str, Any], tuple[str, ...]]]:
        active = paths[: self._MAX_STATIC_PATHS]
        for statement in statements:
            if len(trace.calls) >= self._MAX_STATIC_CALLS:
                break
            next_active: list[tuple[dict[str, Any], tuple[str, ...]]] = []
            for environment, conditions in active:
                if isinstance(statement, ast.Assign):
                    self._record_expression_calls(
                        statement.value,
                        environment,
                        methods,
                        adapter_mode,
                        trace,
                        method_name,
                        stack,
                        conditions,
                    )
                    value = self._safe_eval(
                        statement.value, environment, adapter_mode
                    )
                    updated = dict(environment)
                    if value is not self._UNRESOLVED:
                        for target in statement.targets:
                            self._bind_static_target(target, value, updated)
                    for target in statement.targets:
                        trace.assignments.append(
                            _StaticAssignmentSite(
                                target=target,
                                value_node=statement.value,
                                value=value,
                                constants=dict(updated),
                                method_name=method_name,
                                conditions=conditions,
                            )
                        )
                    next_active.append((updated, conditions))
                    continue
                if isinstance(statement, ast.AnnAssign):
                    value = self._UNRESOLVED
                    updated = dict(environment)
                    if statement.value is not None:
                        self._record_expression_calls(
                            statement.value,
                            environment,
                            methods,
                            adapter_mode,
                            trace,
                            method_name,
                            stack,
                            conditions,
                        )
                        value = self._safe_eval(
                            statement.value, environment, adapter_mode
                        )
                        if value is not self._UNRESOLVED:
                            self._bind_static_target(statement.target, value, updated)
                    trace.assignments.append(
                        _StaticAssignmentSite(
                            target=statement.target,
                            value_node=statement.value or statement.annotation,
                            value=value,
                            constants=dict(updated),
                            method_name=method_name,
                            conditions=conditions,
                        )
                    )
                    next_active.append((updated, conditions))
                    continue
                if isinstance(statement, ast.AugAssign):
                    self._record_expression_calls(
                        statement.value,
                        environment,
                        methods,
                        adapter_mode,
                        trace,
                        method_name,
                        stack,
                        conditions,
                    )
                    left = self._safe_eval(statement.target, environment, adapter_mode)
                    right = self._safe_eval(statement.value, environment, adapter_mode)
                    value = self._apply_static_binop(left, statement.op, right)
                    updated = dict(environment)
                    if value is not self._UNRESOLVED:
                        self._bind_static_target(statement.target, value, updated)
                    next_active.append((updated, conditions))
                    continue
                if isinstance(statement, ast.Expr):
                    self._record_expression_calls(
                        statement.value,
                        environment,
                        methods,
                        adapter_mode,
                        trace,
                        method_name,
                        stack,
                        conditions,
                    )
                    next_active.append((dict(environment), conditions))
                    continue
                if isinstance(statement, ast.If):
                    trace.conditionals.append(
                        _StaticIfSite(
                            node=statement,
                            constants=dict(environment),
                            method_name=method_name,
                            conditions=conditions,
                        )
                    )
                    self._record_expression_calls(
                        statement.test,
                        environment,
                        methods,
                        adapter_mode,
                        trace,
                        method_name,
                        stack,
                        conditions,
                    )
                    test = self._safe_eval(statement.test, environment, adapter_mode)
                    text = self._resolve_placeholders(
                        self._node_text(statement.test),
                        environment,
                        adapter_mode=adapter_mode,
                    )
                    branches: list[tuple[list[ast.stmt], tuple[str, ...]]]
                    if test is self._UNRESOLVED:
                        branches = [
                            (statement.body, conditions + (text,)),
                            (statement.orelse, conditions + (f"not ({text})",)),
                        ]
                    elif bool(test):
                        branches = [(statement.body, conditions + (text,))]
                    else:
                        branches = [
                            (statement.orelse, conditions + (f"not ({text})",))
                        ]
                    for branch, branch_conditions in branches:
                        next_active.extend(
                            self._execute_static_block(
                                branch,
                                [(dict(environment), branch_conditions)],
                                methods,
                                adapter_mode,
                                trace,
                                method_name=method_name,
                                stack=stack,
                            )
                        )
                    continue
                if isinstance(statement, (ast.For, ast.AsyncFor)):
                    self._record_expression_calls(
                        statement.iter,
                        environment,
                        methods,
                        adapter_mode,
                        trace,
                        method_name,
                        stack,
                        conditions,
                    )
                    iterable = self._safe_eval(
                        statement.iter, environment, adapter_mode
                    )
                    items = self._bounded_static_iterable(iterable)
                    if items is None:
                        items = [self._UNRESOLVED]
                    loop_paths: list[tuple[dict[str, Any], tuple[str, ...]]] = []
                    for item in items:
                        loop_environment = dict(environment)
                        if item is not self._UNRESOLVED:
                            self._bind_static_target(
                                statement.target, item, loop_environment
                            )
                        loop_paths.extend(
                            self._execute_static_block(
                                statement.body,
                                [(loop_environment, conditions)],
                                methods,
                                adapter_mode,
                                trace,
                                method_name=method_name,
                                stack=stack,
                            )
                        )
                    if statement.orelse:
                        loop_paths = self._execute_static_block(
                            statement.orelse,
                            loop_paths or [(dict(environment), conditions)],
                            methods,
                            adapter_mode,
                            trace,
                            method_name=method_name,
                            stack=stack,
                        )
                    next_active.extend(loop_paths)
                    continue
                if isinstance(statement, ast.While):
                    self._record_expression_calls(
                        statement.test,
                        environment,
                        methods,
                        adapter_mode,
                        trace,
                        method_name,
                        stack,
                        conditions,
                    )
                    test = self._safe_eval(statement.test, environment, adapter_mode)
                    if test is not self._UNRESOLVED and not bool(test):
                        branch = statement.orelse
                    else:
                        branch = statement.body
                    next_active.extend(
                        self._execute_static_block(
                            branch,
                            [(dict(environment), conditions)],
                            methods,
                            adapter_mode,
                            trace,
                            method_name=method_name,
                            stack=stack,
                        )
                    )
                    continue
                if isinstance(statement, (ast.With, ast.AsyncWith)):
                    updated = dict(environment)
                    for item in statement.items:
                        self._record_expression_calls(
                            item.context_expr,
                            updated,
                            methods,
                            adapter_mode,
                            trace,
                            method_name,
                            stack,
                            conditions,
                        )
                        value = self._safe_eval(
                            item.context_expr, updated, adapter_mode
                        )
                        if item.optional_vars is not None and value is not self._UNRESOLVED:
                            self._bind_static_target(item.optional_vars, value, updated)
                    next_active.extend(
                        self._execute_static_block(
                            statement.body,
                            [(updated, conditions)],
                            methods,
                            adapter_mode,
                            trace,
                            method_name=method_name,
                            stack=stack,
                        )
                    )
                    continue
                if isinstance(statement, ast.Try):
                    alternatives = [statement.body]
                    alternatives.extend(handler.body for handler in statement.handlers)
                    for branch in alternatives:
                        next_active.extend(
                            self._execute_static_block(
                                branch,
                                [(dict(environment), conditions)],
                                methods,
                                adapter_mode,
                                trace,
                                method_name=method_name,
                                stack=stack,
                            )
                        )
                    if statement.orelse:
                        next_active.extend(
                            self._execute_static_block(
                                statement.orelse,
                                [(dict(environment), conditions)],
                                methods,
                                adapter_mode,
                                trace,
                                method_name=method_name,
                                stack=stack,
                            )
                        )
                    if statement.finalbody:
                        next_active = self._execute_static_block(
                            statement.finalbody,
                            next_active or [(dict(environment), conditions)],
                            methods,
                            adapter_mode,
                            trace,
                            method_name=method_name,
                            stack=stack,
                        )
                    continue
                if isinstance(statement, ast.Match):
                    self._record_expression_calls(
                        statement.subject,
                        environment,
                        methods,
                        adapter_mode,
                        trace,
                        method_name,
                        stack,
                        conditions,
                    )
                    subject = self._safe_eval(
                        statement.subject, environment, adapter_mode
                    )
                    matched = False
                    for case in statement.cases:
                        pattern_value = self._literal_pattern_value(case.pattern)
                        if (
                            subject is not self._UNRESOLVED
                            and pattern_value is not self._UNRESOLVED
                            and subject != pattern_value
                        ):
                            continue
                        matched = True
                        next_active.extend(
                            self._execute_static_block(
                                case.body,
                                [(dict(environment), conditions)],
                                methods,
                                adapter_mode,
                                trace,
                                method_name=method_name,
                                stack=stack,
                            )
                        )
                        if subject is not self._UNRESOLVED:
                            break
                    if not matched:
                        next_active.append((dict(environment), conditions))
                    continue
                if isinstance(statement, ast.Return):
                    if statement.value is not None:
                        self._record_expression_calls(
                            statement.value,
                            environment,
                            methods,
                            adapter_mode,
                            trace,
                            method_name,
                            stack,
                            conditions,
                        )
                    continue
                if isinstance(statement, (ast.Raise, ast.Break, ast.Continue)):
                    continue
                if isinstance(
                    statement,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    next_active.append((dict(environment), conditions))
                    continue
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, ast.expr):
                        self._record_expression_calls(
                            child,
                            environment,
                            methods,
                            adapter_mode,
                            trace,
                            method_name,
                            stack,
                            conditions,
                        )
                next_active.append((dict(environment), conditions))
            active = next_active[: self._MAX_STATIC_PATHS]
            if not active:
                break
        return active

    def _record_expression_calls(
        self,
        expression: ast.AST,
        environment: dict[str, Any],
        methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
        adapter_mode: bool,
        trace: _StaticExecutionTrace,
        method_name: str,
        stack: tuple[str, ...],
        conditions: tuple[str, ...],
    ) -> None:
        calls = [node for node in ast.walk(expression) if isinstance(node, ast.Call)]
        for call in calls:
            if len(trace.calls) >= self._MAX_STATIC_CALLS:
                return
            trace.calls.append(
                _StaticCallSite(
                    call=call,
                    constants=dict(environment),
                    method_name=method_name,
                    conditions=conditions,
                )
            )
            called = self._called_name(call.func)
            function = methods.get(called) if called else None
            if function is None or called in stack:
                continue
            # Follow source-local helpers.  Module attributes are followed only
            # when the target name is actually defined in this source file.
            callee_environment = self._bind_function_call(
                function, call, environment, adapter_mode
            )
            self._execute_function_trace(
                function,
                callee_environment,
                methods,
                adapter_mode,
                trace,
                stack=stack,
                conditions=conditions,
            )

    def _apply_static_binop(
        self, left: Any, operator: ast.operator, right: Any
    ) -> Any:
        if left is self._UNRESOLVED or right is self._UNRESOLVED:
            return self._UNRESOLVED
        try:
            if isinstance(operator, ast.Add):
                return left + right
            if isinstance(operator, ast.Sub):
                return left - right
            if isinstance(operator, ast.Mult):
                return left * right
            if isinstance(operator, ast.Div):
                return left / right
            if isinstance(operator, ast.FloorDiv):
                return left // right
            if isinstance(operator, ast.Mod):
                return left % right
            if isinstance(operator, ast.Pow):
                return left**right
            if isinstance(operator, ast.BitOr):
                return left | right
            if isinstance(operator, ast.BitAnd):
                return left & right
        except Exception:
            return self._UNRESOLVED
        return self._UNRESOLVED

    @staticmethod
    def _literal_pattern_value(pattern: ast.pattern) -> Any:
        if isinstance(pattern, ast.MatchValue):
            try:
                return ast.literal_eval(pattern.value)
            except Exception:
                return AgentSourceAnalyzer._UNRESOLVED
        if isinstance(pattern, ast.MatchSingleton):
            return pattern.value
        if isinstance(pattern, ast.MatchAs) and pattern.name is None:
            return AgentSourceAnalyzer._UNRESOLVED
        return AgentSourceAnalyzer._UNRESOLVED

    def _resolve_placeholders(
        self,
        text: str,
        constants: dict[str, Any],
        *,
        adapter_mode: bool = False,
    ) -> str:
        def replace(match: re.Match[str]) -> str:
            try:
                node = ast.parse(match.group(1).strip(), mode="eval").body
            except SyntaxError:
                return match.group(0)
            value = self._safe_eval(node, constants, adapter_mode)
            return (
                match.group(0)
                if value is self._UNRESOLVED
                else self._display_constant(value)
            )

        return re.sub(r"\{([^{}]+)\}", replace, text)

    @staticmethod
    def _display_constant(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return str(value[0]) if len(value) == 1 else ", ".join(map(str, value))
        return str(value)

    def _extract_initial_state(
        self, source: str, constants: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        trace = self._contextual_trace(
            source,
            constants,
            roots=("build_agent",),
            include_all_functions=True,
        )
        for assignment in trace.assignments:
            key = self._target_key(assignment.target) or ""
            if not (key == "initial_goal" or key.endswith(".initial_goal")):
                continue
            payload = self._static_chunk_payload(assignment.value)
            if payload is not None:
                return {"g": payload}
            if isinstance(assignment.value_node, ast.Call):
                payload = self._static_chunk_from_call(
                    assignment.value_node,
                    assignment.constants,
                    False,
                    0,
                )
                if payload is not None:
                    return {"g": payload}
        return {}

    def _extract_productions_from_source(
        self, source: str, constants: dict[str, Any]
    ) -> list[ProductionAnalysis]:
        """Resolve production definitions with bounded lexical execution.

        This expands loops, helper-method calls, f-strings, indexed values,
        comprehensions and ``pyactrFunctionalityExtension.add_production`` while
        never executing model code.
        """
        trace = self._contextual_trace(
            source,
            constants,
            roots=("build_agent",),
            include_all_functions=True,
        )
        result: list[ProductionAnalysis] = []
        seen: set[tuple[str, str]] = set()
        for site in trace.calls:
            call = site.call
            function_name = self._called_name(call.func)
            if function_name == "productionstring":
                name_position, string_position = 0, 1
                utility_position, reward_position = 2, 3
            elif function_name == "add_production":
                # pyactrFunctionalityExtension.add_production(construct, name,
                # string, utility=...)
                name_position, string_position = 1, 2
                utility_position, reward_position = 3, 4
            else:
                continue
            local = site.constants
            name = self._call_argument(
                call, name_position, {"name", "production_name"}, local
            )
            raw = self._call_argument(
                call, string_position, {"string", "rule"}, local
            )
            if not name or not raw:
                continue
            raw = self._resolve_placeholders(raw, local)
            production_name = name
            key = (production_name, raw)
            if key in seen:
                continue
            seen.add(key)
            conditions, effects = self._parse_production(raw)

            def numeric_argument(
                position: int, keyword_names: set[str]
            ) -> float | None:
                argument: ast.AST | None = (
                    call.args[position] if len(call.args) > position else None
                )
                if argument is None:
                    argument = next(
                        (
                            keyword.value
                            for keyword in call.keywords
                            if keyword.arg in keyword_names
                        ),
                        None,
                    )
                if argument is None:
                    return None
                value = self._safe_eval(argument, local)
                if value is self._UNRESOLVED or value is None:
                    return None
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            result.append(
                self._make_production(
                    production_name,
                    raw,
                    conditions,
                    effects,
                    utility=numeric_argument(
                        utility_position, {"utility"}
                    ),
                    reward=numeric_argument(
                        reward_position, {"reward"}
                    ),
                )
            )
        return result

    def _extract_declared_buffers(
        self, source: str, constants: dict[str, Any]
    ) -> list[str]:
        buffers = {"g", "retrieval"}
        trace = self._contextual_trace(
            source,
            constants,
            roots=("build_agent",),
            include_all_functions=True,
        )
        for site in trace.calls:
            function = self._called_name(site.call.func)
            if function == "set_goal":
                value = self._call_argument(
                    site.call, 0, {"name"}, site.constants
                )
                if value:
                    buffers.add(value)
            elif function == "set_retrieval":
                value = self._call_argument(
                    site.call, 0, {"name"}, site.constants
                )
                if value:
                    buffers.add(value)
        return sorted(buffers, key=str.lower)

    def _call_argument(
        self,
        call: ast.Call,
        position: int,
        keyword_names: set[str],
        constants: dict[str, Any],
        *,
        adapter_mode: bool = False,
    ) -> str | None:
        node: ast.AST | None = (
            call.args[position] if len(call.args) > position else None
        )
        if node is None:
            node = next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg in keyword_names
                ),
                None,
            )
        if node is None:
            return None
        value = self._safe_eval(node, constants, adapter_mode)
        if value is self._UNRESOLVED:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return node.value
            return None
        return self._display_constant(value)

    @staticmethod
    def _argument_expression(call: ast.Call, position: int) -> str | None:
        if len(call.args) <= position:
            return None
        try:
            return ast.unparse(call.args[position]).strip()
        except Exception:
            return None

    def _resolve_buffer_argument(
        self,
        call: ast.Call,
        function_name: str,
        constants: dict[str, Any],
    ) -> str | None:
        if function_name in {"get_buffer", "set_buffer", "replace_buffer"}:
            target_index = 1
        elif function_name in {"get_imaginal", "set_imaginal"}:
            target_index = 2 if function_name == "set_imaginal" else 1
        else:
            return None
        return self._call_argument(
            call,
            target_index,
            {"name", "key", "buffer_name"},
            constants,
            adapter_mode=True,
        )

    @staticmethod
    def _called_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
        return None

    @staticmethod
    def _self_method_name(node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return node.attr
        return None

    @staticmethod
    def _node_text(node: ast.AST) -> str:
        try:
            return ast.unparse(node).strip()
        except Exception:
            return ""

    @staticmethod
    def _call_excerpt(source: str, node: ast.AST) -> str | None:
        try:
            return ast.get_source_segment(source, node)
        except Exception:
            return None

    @staticmethod
    def _class_summary(source: str, expected_name: str) -> str:
        return AgentSourceAnalyzer._summarize_class(source, expected_name)

    @staticmethod
    def _adapter_summary(source: str, expected_name: str) -> str:
        if not source.strip():
            return "No adapter source file is present."
        return AgentSourceAnalyzer._summarize_class(
            source, f"{expected_name}Adapter"
        )

    @staticmethod
    def _summarize_class(source: str, expected_name: str) -> str:
        try:
            tree = ast.parse(source or "\n")
        except SyntaxError as exc:
            return f"Source cannot be parsed: {exc}"
        class_node = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef)
                and node.name == expected_name
            ),
            next(
                (node for node in tree.body if isinstance(node, ast.ClassDef)),
                None,
            ),
        )
        if class_node is None:
            return "No class definition was found."
        methods = [
            node.name
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
        ]
        doc = ast.get_docstring(class_node) or ""
        headline = (
            doc.strip().splitlines()[0]
            if doc.strip()
            else "No class docstring"
        )
        return (
            f"Class {class_node.name}: {headline}\n"
            f"Methods ({len(methods)}): "
            f"{', '.join(methods) if methods else 'none'}"
        )

    def _parse_production(
        self, source: str
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        lhs, rhs = source.split("==>", 1) if "==>" in source else (source, "")
        return (
            self._parse_buffer_sections(lhs, left_side=True),
            self._parse_buffer_sections(rhs, left_side=False),
        )

    def _parse_buffer_sections(
        self, text: str, *, left_side: bool
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        current_name: str | None = None
        current_mode = "read" if left_side else "write"
        current_type: str | None = None
        current_slots: dict[str, Any] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if re.match(r"^[=+?~!-][A-Za-z0-9_]+>$", line):
                if current_name is not None:
                    result[current_name] = {
                        "mode": current_mode,
                        "type": current_type,
                        "slots": dict(current_slots),
                    }
                marker = line[0]
                current_name = line[1:-1]
                current_mode = {
                    "=": "read" if left_side else "write",
                    "+": "request",
                    "?": "query",
                    "~": "clear",
                    "!": "execute",
                    "-": "clear",
                }.get(marker, "read" if left_side else "write")
                current_type = None
                current_slots = {}
                continue
            if current_name is None:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
            if key == "isa":
                current_type = value
            else:
                current_slots[key] = value
        if current_name is not None:
            result[current_name] = {
                "mode": current_mode,
                "type": current_type,
                "slots": dict(current_slots),
            }
        return result

    @staticmethod
    def _parse_chunk_definition(text: str) -> dict[str, Any]:
        chunk_type: str | None = None
        slots: dict[str, Any] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
            if key == "isa":
                chunk_type = value
            else:
                slots[key] = value
        return {"type": chunk_type, "slots": slots, "mode": "initial"}

    @staticmethod
    def _full_state_label(buffers: dict[str, dict[str, Any]]) -> str:
        parts: list[str] = []
        for buffer_name in sorted(buffers):
            payload = buffers[buffer_name]
            details: list[str] = []
            if payload.get("type"):
                details.append(str(payload.get("type")))
            for slot_name, slot_value in sorted(
                payload.get("slots", {}).items()
            ):
                details.append(f"{slot_name}={slot_value}")
            mode = payload.get("mode")
            suffix = (
                f" [{mode}]"
                if mode and mode not in {"read", "initial", "write"}
                else ""
            )
            parts.append(
                f"{buffer_name}: "
                + (", ".join(details) if details else "<empty>")
                + suffix
            )
        return "\n".join(parts)
