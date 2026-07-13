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

        runtime = self._inspect_runtime_model(info)
        if runtime is None:
            warnings_list.append(
                "The model could not be instantiated for exact production analysis; "
                "the source fallback may omit dynamically generated rules."
            )
            initial_state = self._extract_initial_state(model_source, model_constants)
            productions = self._extract_productions_from_source(
                model_source, model_constants
            )
            declared_buffers = self._extract_declared_buffers(
                model_source, model_constants
            )
            memory_names = ["decmem"]
        else:
            initial_state = runtime["initial_state"]
            productions = runtime["productions"]
            declared_buffers = runtime["declared_buffers"]
            memory_names = runtime["memory_names"]

        control_slots = self._control_slots(productions, initial_state)
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
            adapter_source, [item.name for item in productions]
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
            return {
                "initial_state": initial_state,
                "productions": productions,
                "declared_buffers": declared_buffers,
                "memory_names": memory_names,
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
        self, source: str, production_names: list[str]
    ) -> dict[str, list[str]]:
        """Map adapter handler methods to the productions that trigger them."""
        if not source.strip():
            return {}
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        extending = methods.get("extending_actr")
        if extending is None:
            return {}
        dispatch: dict[str, set[str]] = defaultdict(set)
        known = list(production_names)
        for node in ast.walk(extending):
            if not isinstance(node, ast.If):
                continue
            triggers = self._production_names_from_test(node.test, known)
            if not triggers:
                continue
            for statement in node.body:
                for call in ast.walk(statement):
                    if not isinstance(call, ast.Call):
                        continue
                    method = self._self_method_name(call.func)
                    if method and method not in {"extending_actr"}:
                        dispatch[method].update(triggers)
        return {
            method: sorted(values, key=str.lower)
            for method, values in dispatch.items()
        }

    def _production_names_from_test(
        self, test: ast.AST, production_names: list[str]
    ) -> set[str]:
        if isinstance(test, ast.BoolOp):
            result: set[str] = set()
            for value in test.values:
                result.update(
                    self._production_names_from_test(value, production_names)
                )
            return result
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            left_text = self._node_text(test.left)
            comparator = test.comparators[0]
            if left_text == "production":
                if isinstance(test.ops[0], ast.Eq):
                    value = self._literal_collection(comparator)
                    return {str(item) for item in value}
                if isinstance(test.ops[0], ast.In):
                    return {str(item) for item in self._literal_collection(comparator)}
        if (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Attribute)
            and test.func.attr == "startswith"
            and self._node_text(test.func.value) == "production"
            and test.args
        ):
            prefixes = self._literal_collection(test.args[0])
            return {
                name
                for name in production_names
                if any(name.startswith(str(prefix)) for prefix in prefixes)
            }
        return set()

    @staticmethod
    def _literal_collection(node: ast.AST) -> list[Any]:
        try:
            value = ast.literal_eval(node)
        except Exception:
            return []
        if isinstance(value, (set, tuple, list, frozenset)):
            return list(value)
        return [value]

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
            if isinstance(node, ast.FunctionDef)
        }
        production_by_name = {item.name: item for item in productions}
        interactions_by_method: dict[str, list[MethodBufferInteraction]] = defaultdict(list)
        for interaction in interactions:
            interactions_by_method[interaction.method_name].append(interaction)

        transitions: list[StateTransitionAnalysis] = []
        sequence = 0
        for method_name, trigger_names in dispatch.items():
            method = methods.get(method_name)
            if method is None:
                continue
            parent_map = self._parent_map(method)
            goal_calls = [
                node
                for node in ast.walk(method)
                if isinstance(node, ast.Call)
                and self._called_name(node.func) == "_set_goal"
            ]
            for trigger_name in trigger_names:
                production = production_by_name.get(trigger_name)
                if production is None:
                    continue
                for call_index, call in enumerate(goal_calls):
                    phase = self._call_argument(
                        call, 0, {"phase"}, constants, adapter_mode=True
                    )
                    state_value = self._call_argument(
                        call, 1, {"state"}, constants, adapter_mode=True
                    )
                    previous = self._call_argument(
                        call, 2, {"previous"}, constants, adapter_mode=True
                    )
                    if not phase or not state_value:
                        continue
                    target_payload = {
                        "type": "goal",
                        "mode": "write",
                        "slots": {
                            "phase": phase,
                            "state": state_value,
                            "prev_phase": previous,
                        },
                    }
                    target_id = self._ensure_control_state(
                        states, target_payload, control_slots
                    )
                    source_id = production.target_state_id
                    condition = self._branch_condition(
                        call,
                        method,
                        parent_map,
                        multiple=len(goal_calls) > 1,
                    )
                    writes = sorted(
                        {
                            item.buffer_name
                            for item in interactions_by_method.get(method_name, [])
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
                                f"adapter:{sequence}:{trigger_name}:{method_name}:"
                                f"{call_index}"
                            ),
                            source_state_id=source_id,
                            target_state_id=target_id,
                            label=method_name,
                            kind="adapter",
                            guard_label=condition,
                            action_label=action,
                            adapter_method=method_name,
                            trigger_production=trigger_name,
                        )
                    )
        return transitions

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
            if isinstance(node, ast.FunctionDef)
        }
        direct: dict[str, list[MethodBufferInteraction]] = defaultdict(list)
        calls: dict[str, set[str]] = defaultdict(set)
        for method_name, method in methods.items():
            for call in (
                node for node in ast.walk(method) if isinstance(node, ast.Call)
            ):
                self_method = self._self_method_name(call.func)
                if self_method and self_method in methods and self_method != method_name:
                    calls[method_name].add(self_method)
                function_name = self._called_name(call.func)
                if function_name not in self._BUFFER_FUNCTIONS:
                    continue
                default_buffer, mode = self._BUFFER_FUNCTIONS[function_name]
                buffer_name = default_buffer
                if default_buffer == "*":
                    buffer_name = (
                        self._resolve_buffer_argument(
                            call, function_name, constants
                        )
                        or "dynamic"
                    )
                direct[method_name].append(
                    MethodBufferInteraction(
                        method_name=method_name,
                        function_name=function_name,
                        buffer_name=buffer_name,
                        mode=mode,
                        detail=self._call_excerpt(source, call),
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

        if len(memories) == 1:
            memory_name = next(iter(memories))
            for buffer_name in declared_buffers:
                operations.append(
                    {
                        "actor": f"buffer:{buffer_name}",
                        "mode": "buffer_link",
                        "memory_name": memory_name,
                        "detail": "harvest/retrieval link assigned by pyactr simulation()",
                    }
                )

        return DeclarativeMemorySnapshot(
            memories=sorted(memories, key=str.lower),
            chunks=chunks,
            edges=DeclarativeMemoryInspector.infer_edges(chunks),
            operations=operations,
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
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return [], []
        chunks: list[MemoryChunk] = []
        operations: list[dict[str, Any]] = []
        sequence = 0
        for method in (
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ):
            assignments: dict[str, dict[str, Any]] = {}
            for node in ast.walk(method):
                if not isinstance(node, ast.Assign) or not node.targets:
                    continue
                if not isinstance(node.value, ast.Call):
                    continue
                function_name = self._called_name(node.value.func)
                if function_name not in {"chunkstring", "chunk_from_string", "makechunk"}:
                    continue
                variable = self._target_key(node.targets[0])
                if not variable:
                    continue
                payload = self._chunk_payload_from_call(
                    node.value, constants, adapter_mode
                )
                assignments[variable] = payload
            for call in (
                node for node in ast.walk(method) if isinstance(node, ast.Call)
            ):
                function_name = self._called_name(call.func)
                func_text = self._node_text(call.func)
                if function_name == "add_to_declarative_memory":
                    argument_index = 1
                    memory_name = "decmem"
                elif function_name == "add" and (
                    "decmem" in func_text or ".dm" in func_text
                ):
                    argument_index = 0
                    match = re.search(r"decmems?\[['\"]([^'\"]+)", func_text)
                    memory_name = match.group(1) if match else "decmem"
                else:
                    continue
                expression = self._argument_expression(call, argument_index)
                payload = assignments.get(expression or "")
                sequence += 1
                if payload is not None:
                    label = self._chunk_label(payload)
                    chunk_id = f"{source_name}:{method.name}:{expression}:{sequence}"
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
                operations.append(
                    {
                        "actor": f"{source_name}.{method.name}",
                        "mode": "write",
                        "memory_name": memory_name,
                        "detail": expression or "dynamic chunk",
                    }
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
        label = str(payload.get("type") or "chunk")
        slots = payload.get("slots", {})
        if slots:
            label += "\n" + ", ".join(
                f"{name}={value}" for name, value in slots.items()
            )
        return label

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
        constants: dict[str, Any] = dict(inherited or {})
        try:
            tree = ast.parse(source or "\n")
        except SyntaxError:
            return constants
        assignments = [
            node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        ]
        for _ in range(8):
            changed = False
            for node in assignments:
                value = self._safe_eval(node.value, constants, adapter_mode)
                if value is self._UNRESOLVED:
                    continue
                for target in node.targets:
                    key = self._target_key(target)
                    if key and constants.get(key, self._UNRESOLVED) != value:
                        constants[key] = value
                        changed = True
                        if isinstance(value, (list, tuple)):
                            for index, item in enumerate(value):
                                constants[f"{key}[{index}]"] = item
            if not changed:
                break
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
        return None

    def _safe_eval(
        self,
        node: ast.AST,
        constants: dict[str, Any],
        adapter_mode: bool = False,
    ) -> Any:
        try:
            return ast.literal_eval(node)
        except Exception:
            pass
        expression = self._node_text(node)
        lookup = expression
        if adapter_mode:
            for prefix in (
                "self.agent_construct.actr_construct.",
                "agent_construct.actr_construct.",
            ):
                if lookup.startswith(prefix):
                    lookup = "self." + lookup[len(prefix) :]
                    break
        if lookup in constants:
            return constants[lookup]
        if isinstance(node, ast.Name):
            return constants.get(node.id, self._UNRESOLVED)
        if isinstance(node, ast.Attribute):
            return constants.get(lookup, self._UNRESOLVED)
        if isinstance(node, ast.Subscript):
            base = self._safe_eval(node.value, constants, adapter_mode)
            index = self._safe_eval(node.slice, constants, adapter_mode)
            if base is self._UNRESOLVED or index is self._UNRESOLVED:
                return self._UNRESOLVED
            try:
                return base[index]
            except Exception:
                return self._UNRESOLVED
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = [
                self._safe_eval(item, constants, adapter_mode)
                for item in node.elts
            ]
            if any(item is self._UNRESOLVED for item in values):
                return self._UNRESOLVED
            return tuple(values) if isinstance(node, ast.Tuple) else set(values) if isinstance(node, ast.Set) else values
        if isinstance(node, ast.Dict):
            keys = [
                self._safe_eval(item, constants, adapter_mode)
                for item in node.keys
            ]
            values = [
                self._safe_eval(item, constants, adapter_mode)
                for item in node.values
            ]
            if any(item is self._UNRESOLVED for item in keys + values):
                return self._UNRESOLVED
            return dict(zip(keys, values))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._safe_eval(node.left, constants, adapter_mode)
            right = self._safe_eval(node.right, constants, adapter_mode)
            if left is self._UNRESOLVED or right is self._UNRESOLVED:
                return self._UNRESOLVED
            try:
                return left + right
            except Exception:
                return self._UNRESOLVED
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            base = self._safe_eval(node.func.value, constants, adapter_mode)
            args = [self._safe_eval(item, constants, adapter_mode) for item in node.args]
            if base is self._UNRESOLVED or any(item is self._UNRESOLVED for item in args):
                return self._UNRESOLVED
            allowed = {
                "lower",
                "upper",
                "strip",
                "removeprefix",
                "removesuffix",
                "replace",
            }
            if node.func.attr in allowed:
                try:
                    return getattr(base, node.func.attr)(*args)
                except Exception:
                    return self._UNRESOLVED
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                elif isinstance(value, ast.FormattedValue):
                    resolved = self._safe_eval(value.value, constants, adapter_mode)
                    parts.append(
                        "{" + self._node_text(value.value) + "}"
                        if resolved is self._UNRESOLVED
                        else self._display_constant(resolved)
                    )
            return "".join(parts)
        return self._UNRESOLVED

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
        try:
            tree = ast.parse(source or "\n")
        except SyntaxError:
            return {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Attribute)
                and target.attr == "initial_goal"
                for target in node.targets
            ):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            raw = self._call_argument(node.value, 0, {"string"}, constants)
            if raw:
                return {
                    "g": self._parse_chunk_definition(
                        self._resolve_placeholders(raw, constants)
                    )
                }
        return {}

    def _extract_productions_from_source(
        self, source: str, constants: dict[str, Any]
    ) -> list[ProductionAnalysis]:
        """Conservative fallback for models that cannot be instantiated."""
        try:
            tree = ast.parse(source or "\n")
        except SyntaxError:
            return []
        result: list[ProductionAnalysis] = []
        for call in (
            node for node in ast.walk(tree) if isinstance(node, ast.Call)
        ):
            if self._called_name(call.func) != "productionstring":
                continue
            name = self._call_argument(call, 0, {"name"}, constants)
            raw = self._call_argument(call, 1, {"string"}, constants)
            if not raw:
                continue
            raw = self._resolve_placeholders(raw, constants)
            conditions, effects = self._parse_production(raw)
            result.append(
                self._make_production(
                    name or f"production_{len(result)+1}",
                    raw,
                    conditions,
                    effects,
                )
            )
        return result

    def _extract_declared_buffers(
        self, source: str, constants: dict[str, Any]
    ) -> list[str]:
        buffers = {"g", "retrieval"}
        try:
            tree = ast.parse(source or "\n")
        except SyntaxError:
            return sorted(buffers)
        for call in (
            node for node in ast.walk(tree) if isinstance(node, ast.Call)
        ):
            function = self._called_name(call.func)
            if function == "set_goal":
                value = self._call_argument(call, 0, {"name"}, constants)
                if value:
                    buffers.add(value)
            elif function == "set_retrieval":
                value = self._call_argument(call, 0, {"name"}, constants)
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
