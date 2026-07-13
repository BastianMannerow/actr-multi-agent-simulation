"""Static source analysis for ACT-R agent models and adapters."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from simulation.discovery.agent_discovery import AgentTypeInfo
from simulation.inspection.declarative_memory import (
    DeclarativeMemoryInspector,
    DeclarativeMemorySnapshot,
    MemoryChunk,
)


@dataclass(slots=True)
class ProductionAnalysis:
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


@dataclass(slots=True)
class MethodBufferInteraction:
    method_name: str
    function_name: str
    buffer_name: str
    mode: str
    detail: str | None = None


@dataclass(slots=True)
class AgentStaticAnalysis:
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
    loop_states: list[str]
    adapter_interactions: list[MethodBufferInteraction]
    production_interactions: list[MethodBufferInteraction]
    declared_buffers: list[str]
    declarative_memory: DeclarativeMemorySnapshot

    def production(self, name: str) -> ProductionAnalysis | None:
        target = name.strip().casefold()
        return next(
            (item for item in self.productions if item.name.casefold() == target),
            None,
        )

    def path_to_production(self, name: str) -> list[ProductionAnalysis] | None:
        """Return the shortest symbolic path from the initial buffer state."""
        target = self.production(name)
        if target is None:
            return None
        queue: list[
            tuple[dict[str, dict[str, Any]], list[ProductionAnalysis]]
        ] = [(self._copy_state(self.initial_state), [])]
        visited: set[str] = set()
        while queue:
            state, path = queue.pop(0)
            signature = self._state_signature(state)
            if signature in visited:
                continue
            visited.add(signature)
            for production in self.productions:
                if not self._state_matches(state, production.conditions):
                    continue
                candidate = path + [production]
                if production.name.casefold() == target.name.casefold():
                    return candidate
                next_state = self._apply_effects(state, production.effects)
                if self._state_signature(next_state) not in visited:
                    queue.append((next_state, candidate))
        return None

    def state_sequence_for_path(
        self,
        path: list[ProductionAnalysis],
    ) -> list[str]:
        state = self._copy_state(self.initial_state)
        labels = [self._state_label(state)]
        for production in path:
            state = self._apply_effects(state, production.effects)
            labels.append(self._state_label(state))
        return labels

    @staticmethod
    def _copy_state(
        state: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return {
            name: {
                **payload,
                "slots": dict(payload.get("slots", {})),
            }
            for name, payload in state.items()
        }

    @staticmethod
    def _apply_effects(
        state: dict[str, dict[str, Any]],
        effects: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        result = AgentStaticAnalysis._copy_state(state)
        for buffer_name, update in effects.items():
            mode = update.get("mode")
            if mode == "clear":
                result.pop(buffer_name, None)
                continue
            current = {
                **result.get(buffer_name, {}),
                "slots": dict(
                    result.get(buffer_name, {}).get("slots", {})
                ),
            }
            current["slots"].update(update.get("slots", {}))
            if update.get("type"):
                current["type"] = update.get("type")
            current["mode"] = mode
            result[buffer_name] = current
        return result

    @staticmethod
    def _state_matches(
        state: dict[str, dict[str, Any]],
        conditions: dict[str, dict[str, Any]],
    ) -> bool:
        if not conditions:
            return True
        for buffer_name, expected in conditions.items():
            current = state.get(buffer_name)
            if current is None:
                return False
            expected_type = expected.get("type")
            current_type = current.get("type")
            if expected_type and current_type and expected_type != current_type:
                return False
            for slot_name, slot_value in expected.get("slots", {}).items():
                if current.get("slots", {}).get(slot_name) != slot_value:
                    return False
        return True

    @staticmethod
    def _state_signature(state: dict[str, dict[str, Any]]) -> str:
        import json

        return json.dumps(state, sort_keys=True, default=str)

    @staticmethod
    def _state_label(state: dict[str, dict[str, Any]]) -> str:
        parts: list[str] = []
        for buffer_name in sorted(state):
            payload = state[buffer_name]
            details: list[str] = []
            if payload.get("type"):
                details.append(str(payload.get("type")))
            for slot_name, slot_value in sorted(
                payload.get("slots", {}).items()
            ):
                details.append(f"{slot_name}={slot_value}")
            parts.append(
                f"{buffer_name}: "
                + (", ".join(details) if details else "<empty>")
            )
        return "\n".join(parts)


class AgentSourceAnalyzer:
    """Inspect agent and adapter source files for explainable visualizations."""

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
        "delete_declarative_chunk_type": ("decmem", "write"),
        "get_declarative_chunk_type": ("decmem", "read"),
    }

    def analyze(self, info: AgentTypeInfo) -> AgentStaticAnalysis:
        model_source = self._safe_read(info.model_path)
        adapter_source = self._safe_read(info.adapter_path)

        model_constants = self._extract_known_constants(model_source)
        adapter_constants = dict(model_constants)
        adapter_constants.update(
            self._extract_known_constants(
                adapter_source,
                inherited=model_constants,
                adapter_mode=True,
            )
        )
        initial_state = self._extract_initial_state(
            model_source, model_constants
        )
        initial_label = (
            self._state_label(initial_state) or "<unknown initial state>"
        )
        productions = self._extract_productions(
            model_source, initial_state, model_constants
        )
        adapter_interactions = self._extract_adapter_interactions(
            adapter_source, adapter_constants
        )
        production_interactions = self._production_interactions(productions)
        declared_buffers = sorted(
            set(self._extract_declared_buffers(model_source, model_constants))
            | {name for production in productions for name in production.read_buffers}
            | {name for production in productions for name in production.written_buffers},
            key=str.lower,
        )
        declarative_memory = self._extract_declarative_memory(
            model_source,
            adapter_source,
            model_constants,
            adapter_constants,
        )

        state_graph = self._graph_from_productions(productions)
        dead_ends = [
            state for state, targets in state_graph.items() if not targets
        ]
        loops = sorted(
            {
                prod.source_label
                for prod in productions
                if prod.self_loop or prod.target_label == prod.source_label
            }
        )
        unreachable = sorted(
            prod.name for prod in productions if not prod.reachable
        )

        class_summary = self._class_summary(model_source, info.name)
        adapter_summary = self._adapter_summary(adapter_source, info.name)
        return AgentStaticAnalysis(
            agent_type=info.name,
            model_path=info.model_path,
            adapter_path=info.adapter_path,
            model_source=model_source,
            adapter_source=adapter_source,
            class_summary=class_summary,
            adapter_summary=adapter_summary,
            initial_state=initial_state,
            initial_state_label=initial_label,
            productions=productions,
            unreachable_productions=unreachable,
            dead_end_states=dead_ends,
            loop_states=loops,
            adapter_interactions=adapter_interactions,
            production_interactions=production_interactions,
            declared_buffers=declared_buffers,
            declarative_memory=declarative_memory,
        )

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
        """Resolve simple local/class attributes and adapter references safely."""
        constants: dict[str, Any] = dict(inherited or {})
        try:
            tree = ast.parse(source or "\n")
        except SyntaxError:
            return constants

        assignments = [
            node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        ]
        for _ in range(6):
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

        phases = constants.get("self.goal_phases")
        if isinstance(phases, (list, tuple)) and phases:
            constants.setdefault("phase", phases[0])
        return constants

    _UNRESOLVED = object()

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
        try:
            expression = ast.unparse(node).strip()
        except Exception:
            expression = ""
        lookup = expression
        if adapter_mode:
            for prefix in (
                "self.agent_construct.actr_construct.",
                "self.agent_construct.actr_adapter.agent_construct.actr_construct.",
                "agent_construct.actr_construct.",
            ):
                if lookup.startswith(prefix):
                    lookup = "self." + lookup[len(prefix):]
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
            if any(value is self._UNRESOLVED for value in values):
                return self._UNRESOLVED
            if isinstance(node, ast.Tuple):
                return tuple(values)
            if isinstance(node, ast.Set):
                return set(values)
            return values
        if isinstance(node, ast.Dict):
            keys = [self._safe_eval(item, constants, adapter_mode) for item in node.keys]
            values = [self._safe_eval(item, constants, adapter_mode) for item in node.values]
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
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for value in node.values:
                if isinstance(value, ast.Constant):
                    parts.append(str(value.value))
                elif isinstance(value, ast.FormattedValue):
                    resolved = self._safe_eval(value.value, constants, adapter_mode)
                    if resolved is self._UNRESOLVED:
                        parts.append("{" + ast.unparse(value.value).strip() + "}")
                    else:
                        parts.append(self._display_constant(resolved))
            return "".join(parts)
        return self._UNRESOLVED

    def _resolve_placeholders(
        self, text: str, constants: dict[str, Any], *, adapter_mode: bool = False
    ) -> str:
        def replace(match: re.Match[str]) -> str:
            expression = match.group(1).strip()
            try:
                node = ast.parse(expression, mode="eval").body
            except SyntaxError:
                return match.group(0)
            value = self._safe_eval(node, constants, adapter_mode)
            if value is self._UNRESOLVED:
                return match.group(0)
            return self._display_constant(value)

        return re.sub(r"\{([^{}]+)\}", replace, text)

    @staticmethod
    def _display_constant(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return str(value[0])
            return ", ".join(str(item) for item in value)
        return str(value)

    def _extract_initial_state(
        self,
        source: str,
        constants: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        tree = ast.parse(source or "\n")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Attribute) and target.attr == "initial_goal":
                    call = node.value
                    if isinstance(call, ast.Call):
                        for keyword in call.keywords:
                            if keyword.arg == "string":
                                raw = self._string_value(keyword.value, constants)
                                if raw:
                                    raw = self._resolve_placeholders(raw, constants)
                                    chunk = self._parse_chunk_definition(raw)
                                    if chunk:
                                        return {"g": chunk}
        return {}

    def _extract_productions(
        self,
        source: str,
        initial_state: dict[str, dict[str, Any]],
        constants: dict[str, Any],
    ) -> list[ProductionAnalysis]:
        tree = ast.parse(source or "\n")
        productions: list[ProductionAnalysis] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "productionstring":
                continue
            name = None
            raw_string = None
            for keyword in node.keywords:
                if keyword.arg == "name":
                    name = self._string_value(keyword.value, constants)
                elif keyword.arg == "string":
                    raw_string = self._string_value(keyword.value, constants)
            if not raw_string:
                continue
            raw_string = self._resolve_placeholders(raw_string, constants)
            if name:
                name = self._resolve_placeholders(name, constants)
            conditions, effects = self._parse_production(raw_string)
            source_label = self._state_label(conditions) or "<no conditions>"
            target_label = self._state_label(effects) or "<no effects>"
            production = ProductionAnalysis(
                name=name or f"production_{len(productions)+1}",
                raw_string=raw_string,
                source_label=source_label,
                target_label=target_label,
                conditions=conditions,
                effects=effects,
                read_buffers=sorted(conditions),
                written_buffers=sorted(
                    name
                    for name, payload in effects.items()
                    if payload.get("mode") in {"write", "request", "clear"}
                ),
            )
            production.self_loop = production.source_label == production.target_label
            productions.append(production)

        self._mark_reachability(productions, initial_state)
        return productions

    def _mark_reachability(
        self,
        productions: list[ProductionAnalysis],
        initial_state: dict[str, dict[str, Any]],
    ) -> None:
        if not productions:
            return
        state_to_prods: dict[str, list[ProductionAnalysis]] = {}
        for production in productions:
            state_to_prods.setdefault(production.source_label, []).append(production)

        queue: list[dict[str, dict[str, Any]]] = [initial_state or {}]
        seen_labels: set[str] = set()
        while queue:
            state = queue.pop(0)
            label = self._state_label(state)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            for production in productions:
                if self._state_matches(state, production.conditions):
                    if not production.reachable:
                        production.reachable = True
                    next_state = self._apply_effects(state, production.effects)
                    next_label = self._state_label(next_state)
                    if next_label not in seen_labels:
                        queue.append(next_state)

    @staticmethod
    def _apply_effects(
        state: dict[str, dict[str, Any]], effects: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        result = {
            name: {key: value for key, value in payload.items()}
            for name, payload in state.items()
        }
        for buffer_name, update in effects.items():
            mode = update.get("mode")
            if mode == "clear":
                result.pop(buffer_name, None)
                continue
            current = {**result.get(buffer_name, {})}
            slots = dict(current.get("slots", {}))
            slots.update(update.get("slots", {}))
            if update.get("type"):
                current["type"] = update.get("type")
            current["slots"] = slots
            current["mode"] = mode
            result[buffer_name] = current
        return result

    @staticmethod
    def _state_matches(
        state: dict[str, dict[str, Any]], conditions: dict[str, dict[str, Any]]
    ) -> bool:
        if not conditions:
            return True
        for buffer_name, expected in conditions.items():
            current = state.get(buffer_name)
            if current is None:
                return False
            expected_type = expected.get("type")
            current_type = current.get("type")
            if expected_type and current_type and expected_type != current_type:
                return False
            for slot_name, slot_value in expected.get("slots", {}).items():
                if current.get("slots", {}).get(slot_name) != slot_value:
                    return False
        return True

    def _extract_declared_buffers(
        self, source: str, constants: dict[str, Any]
    ) -> list[str]:
        buffers = {"g", "retrieval"}
        try:
            tree = ast.parse(source or "\n")
        except SyntaxError:
            return sorted(buffers)

        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        def values_for_argument(
            call: ast.Call, position: int, keyword_names: set[str]
        ) -> list[str]:
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
                return []
            direct = self._safe_eval(node, constants)
            if direct is not self._UNRESOLVED:
                if isinstance(direct, (list, tuple, set)):
                    return [str(value) for value in direct]
                return [str(direct)]

            # Resolve a loop variable such as
            # ``for name in self.workspaces: set_goal(name=name)``.
            if isinstance(node, ast.Name):
                current: ast.AST | None = call
                while current is not None:
                    current = parents.get(current)
                    if isinstance(current, ast.For):
                        target = self._target_key(current.target)
                        if target != node.id:
                            continue
                        iterable = self._safe_eval(current.iter, constants)
                        if isinstance(iterable, (list, tuple, set)):
                            return [str(value) for value in iterable]
                        break
                    if isinstance(current, (ast.FunctionDef, ast.ClassDef)):
                        break
            return []

        for call in (
            node for node in ast.walk(tree) if isinstance(node, ast.Call)
        ):
            function_name = self._called_name(call.func)
            if function_name == "set_goal":
                buffers.update(values_for_argument(call, 0, {"name"}))
            elif function_name == "set_retrieval":
                buffers.update(values_for_argument(call, 0, {"name"}))
            elif function_name in {"visualBuffer", "visual_buffer"}:
                buffers.update(
                    values_for_argument(call, 0, {"name_visual"})
                )
                buffers.update(
                    values_for_argument(
                        call, 1, {"name_visual_location"}
                    )
                )
        return sorted((name for name in buffers if name), key=str.lower)

    def _extract_declarative_memory(
        self,
        model_source: str,
        adapter_source: str,
        model_constants: dict[str, Any],
        adapter_constants: dict[str, Any],
    ) -> DeclarativeMemorySnapshot:
        chunks: list[MemoryChunk] = []
        operations: list[dict[str, Any]] = []
        memories: set[str] = {"decmem"}
        variable_chunks: dict[str, MemoryChunk] = {}

        for source_name, source, constants, adapter_mode in (
            ("agent", model_source, model_constants, False),
            ("adapter", adapter_source, adapter_constants, True),
        ):
            extracted, variable_map = self._extract_chunk_assignments(
                source, source_name, constants, adapter_mode
            )
            chunks.extend(extracted)
            variable_chunks.update(variable_map)
            source_operations, source_memories = self._memory_operations_from_source(
                source,
                source_name,
                constants,
                variable_chunks,
                adapter_mode,
            )
            operations.extend(source_operations)
            memories.update(source_memories)

        for chunk in chunks:
            memories.add(chunk.memory_name)
        return DeclarativeMemorySnapshot(
            memories=sorted(memories, key=str.lower),
            chunks=chunks,
            edges=DeclarativeMemoryInspector.infer_edges(chunks),
            operations=operations,
        )

    def _extract_chunk_assignments(
        self,
        source: str,
        source_name: str,
        constants: dict[str, Any],
        adapter_mode: bool,
    ) -> tuple[list[MemoryChunk], dict[str, MemoryChunk]]:
        if not source.strip():
            return [], {}
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return [], {}
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        chunks: list[MemoryChunk] = []
        variable_map: dict[str, MemoryChunk] = {}
        sequence = 0
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            function_name = self._called_name(call.func)
            if function_name not in {"chunkstring", "chunk_from_string", "makechunk"}:
                continue
            sequence += 1
            variable = f"inline_chunk_{sequence}"
            current: ast.AST | None = call
            while current is not None:
                current = parents.get(current)
                if isinstance(current, ast.Assign) and current.targets:
                    variable = self._target_key(current.targets[0]) or variable
                    break
                if isinstance(current, (ast.FunctionDef, ast.ClassDef)):
                    break

            payload: dict[str, Any] = {"type": None, "slots": {}}
            if function_name in {"chunkstring", "chunk_from_string"}:
                raw = self._call_argument(
                    call, 0, {"string"}, constants, adapter_mode=adapter_mode
                )
                if raw:
                    raw = self._resolve_placeholders(
                        raw, constants, adapter_mode=adapter_mode
                    )
                    payload = self._parse_chunk_definition(raw)
            else:
                typename = self._call_argument(
                    call, 1, {"typename"}, constants, adapter_mode=adapter_mode
                ) or "chunk"
                slots: dict[str, Any] = {}
                for keyword in call.keywords:
                    if keyword.arg in {"nameofchunk", "typename"}:
                        continue
                    value = self._safe_eval(
                        keyword.value, constants, adapter_mode
                    )
                    slots[str(keyword.arg)] = (
                        ast.unparse(keyword.value)
                        if value is self._UNRESOLVED
                        else value
                    )
                payload = {
                    "type": typename,
                    "slots": slots,
                    "mode": "static",
                }
            label = str(payload.get("type") or "chunk")
            slots = dict(payload.get("slots", {}))
            if slots:
                label += "\n" + ", ".join(
                    f"{name}={value}"
                    for name, value in list(slots.items())[:3]
                )
            chunk = MemoryChunk(
                chunk_id=f"{source_name}:{variable}:{sequence}",
                memory_name="decmem",
                chunk_type=str(payload.get("type") or "chunk"),
                label=label,
                slots=slots,
                source=source_name,
            )
            chunks.append(chunk)
            variable_map[variable] = chunk
            if variable.startswith("self."):
                variable_map[variable.removeprefix("self.")] = chunk
        return chunks, variable_map

    def _memory_operations_from_source(
        self,
        source: str,
        source_name: str,
        constants: dict[str, Any],
        variable_chunks: dict[str, MemoryChunk],
        adapter_mode: bool,
    ) -> tuple[list[dict[str, Any]], set[str]]:
        if not source.strip():
            return [], set()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return [], set()
        operations: list[dict[str, Any]] = []
        memories: set[str] = set()
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        def method_name(node: ast.AST) -> str:
            current: ast.AST | None = node
            while current is not None:
                if isinstance(current, ast.FunctionDef):
                    return current.name
                current = parents.get(current)
            return "module"

        decmem_counter = 0
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            function_name = self._called_name(call.func) or "call"
            func_text = ast.unparse(call.func) if hasattr(ast, "unparse") else function_name
            mode: str | None = None
            memory_name = "decmem"
            detail = ""
            if function_name == "set_decmem":
                decmem_counter += 1
                memory_name = "decmem" if decmem_counter == 1 else f"decmem{decmem_counter}"
                mode = "write"
                detail = "create/set declarative memory"
            elif function_name in {"set_goal", "set_retrieval"}:
                mode = "linked"
                detail = self._call_argument(
                    call, 0, {"name"}, constants, adapter_mode=adapter_mode
                ) or function_name
            elif function_name in {"visualBuffer", "visual_buffer"}:
                mode = "linked"
                first = self._call_argument(
                    call, 0, {"name_visual"}, constants, adapter_mode=adapter_mode
                )
                second = self._call_argument(
                    call, 1, {"name_visual_location"}, constants, adapter_mode=adapter_mode
                )
                detail = "/".join(value for value in (first, second) if value) or "visual buffers"
            elif function_name in {"add_to_declarative_memory"}:
                mode = "write"
                detail = self._argument_expression(call, 1) or "chunk"
            elif function_name in {"get_declarative_memory", "get_declarative_chunk_type"}:
                mode = "read"
                detail = self._argument_expression(call, 1) or "memory access"
            elif function_name == "delete_declarative_chunk_type":
                mode = "delete"
                detail = self._argument_expression(call, 1) or "chunk type"
            elif function_name == "add" and ("decmem" in func_text or ".dm" in func_text):
                mode = "write"
                detail = self._argument_expression(call, 0) or "chunk"
                match = re.search(r"decmems?\[['\"]([^'\"]+)", func_text)
                if match:
                    memory_name = match.group(1)
            if mode is None:
                continue
            memories.add(memory_name)
            operation = {
                "actor": f"{source_name}.{method_name(call)}",
                "mode": mode,
                "memory_name": memory_name,
                "detail": detail,
            }
            operations.append(operation)
            variable_key = detail.strip()
            chunk = variable_chunks.get(variable_key)
            if chunk is None and variable_key.startswith("self."):
                chunk = variable_chunks.get(variable_key.removeprefix("self."))
            if chunk is not None:
                chunk.memory_name = memory_name
        return operations, memories

    def _call_argument(
        self,
        call: ast.Call,
        position: int,
        keyword_names: set[str],
        constants: dict[str, Any],
        *,
        adapter_mode: bool = False,
    ) -> str | None:
        node: ast.AST | None = call.args[position] if len(call.args) > position else None
        if node is None:
            node = next(
                (keyword.value for keyword in call.keywords if keyword.arg in keyword_names),
                None,
            )
        if node is None:
            return None
        return self._string_value(node, constants, adapter_mode=adapter_mode)

    @staticmethod
    def _argument_expression(call: ast.Call, position: int) -> str | None:
        if len(call.args) <= position:
            return None
        try:
            return ast.unparse(call.args[position]).strip()
        except Exception:
            return None

    def _extract_adapter_interactions(
        self, source: str, constants: dict[str, Any]
    ) -> list[MethodBufferInteraction]:
        if not source.strip():
            return []
        tree = ast.parse(source)
        interactions: list[MethodBufferInteraction] = []
        class_node = next(
            (node for node in tree.body if isinstance(node, ast.ClassDef)),
            None,
        )
        if class_node is None:
            return interactions
        for method in [node for node in class_node.body if isinstance(node, ast.FunctionDef)]:
            for call in ast.walk(method):
                if not isinstance(call, ast.Call):
                    continue
                function_name = self._called_name(call.func)
                if function_name not in self._BUFFER_FUNCTIONS:
                    continue
                default_buffer, mode = self._BUFFER_FUNCTIONS[function_name]
                buffer_name = default_buffer
                if default_buffer == "*":
                    key_value = self._resolve_buffer_argument(
                        call, function_name, constants
                    )
                    buffer_name = key_value or "dynamic"
                elif function_name in {"get_imaginal", "set_imaginal"}:
                    key_value = self._resolve_buffer_argument(
                        call, function_name, constants
                    )
                    if key_value:
                        buffer_name = key_value
                interactions.append(
                    MethodBufferInteraction(
                        method_name=method.name,
                        function_name=function_name,
                        buffer_name=buffer_name,
                        mode=mode,
                        detail=self._call_excerpt(source, call),
                    )
                )
        return interactions

    def _production_interactions(
        self, productions: list[ProductionAnalysis]
    ) -> list[MethodBufferInteraction]:
        interactions: list[MethodBufferInteraction] = []
        for production in productions:
            for buffer_name in production.read_buffers:
                interactions.append(
                    MethodBufferInteraction(
                        method_name=production.name,
                        function_name="production condition",
                        buffer_name=buffer_name,
                        mode="read",
                        detail=production.source_label,
                    )
                )
            for buffer_name in production.written_buffers:
                interactions.append(
                    MethodBufferInteraction(
                        method_name=production.name,
                        function_name="production effect",
                        buffer_name=buffer_name,
                        mode="write",
                        detail=production.target_label,
                    )
                )
        return interactions

    @staticmethod
    def _called_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _resolve_buffer_argument(
        self, call: ast.Call, function_name: str, constants: dict[str, Any]
    ) -> str | None:
        if function_name in {"get_buffer", "set_buffer", "replace_buffer"}:
            target_index = 1
        elif function_name in {"get_imaginal", "set_imaginal"}:
            target_index = 2 if function_name == "set_imaginal" else 1
        else:
            return None
        if len(call.args) > target_index:
            return self._string_value(
                call.args[target_index], constants, adapter_mode=True
            )
        for keyword in call.keywords:
            if keyword.arg in {"name", "key", "buffer_name"}:
                return self._string_value(
                    keyword.value, constants, adapter_mode=True
                )
        return None

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
        return AgentSourceAnalyzer._summarize_class(source, f"{expected_name}Adapter")

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
                if isinstance(node, ast.ClassDef) and node.name == expected_name
            ),
            next((node for node in tree.body if isinstance(node, ast.ClassDef)), None),
        )
        if class_node is None:
            return "No class definition was found."
        methods = [node.name for node in class_node.body if isinstance(node, ast.FunctionDef)]
        doc = ast.get_docstring(class_node) or ""
        headline = doc.strip().splitlines()[0] if doc.strip() else "No class docstring"
        return (
            f"Class {class_node.name}: {headline}\n"
            f"Methods ({len(methods)}): {', '.join(methods) if methods else 'none'}"
        )

    def _string_value(
        self,
        node: ast.AST,
        constants: dict[str, Any] | None = None,
        *,
        adapter_mode: bool = False,
    ) -> str | None:
        value = self._safe_eval(node, constants or {}, adapter_mode)
        if value is self._UNRESOLVED:
            if isinstance(node, ast.JoinedStr):
                parts: list[str] = []
                for item in node.values:
                    if isinstance(item, ast.Constant):
                        parts.append(str(item.value))
                    elif isinstance(item, ast.FormattedValue):
                        parts.append("{" + ast.unparse(item.value).strip() + "}")
                return "".join(parts)
            return None
        return self._display_constant(value)

    def _parse_production(
        self, source: str
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        if "==>" in source:
            lhs, rhs = source.split("==>", 1)
        else:
            lhs, rhs = source, ""
        return self._parse_buffer_sections(lhs, left_side=True), self._parse_buffer_sections(rhs, left_side=False)

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
                    "!": "meta",
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

    def _parse_chunk_definition(self, text: str) -> dict[str, Any]:
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
    def _state_label(buffers: dict[str, dict[str, Any]]) -> str:
        parts: list[str] = []
        for buffer_name in sorted(buffers):
            payload = buffers[buffer_name]
            detail: list[str] = []
            if payload.get("type"):
                detail.append(str(payload.get("type")))
            for slot_name, slot_value in sorted(payload.get("slots", {}).items()):
                detail.append(f"{slot_name}={slot_value}")
            inner = ", ".join(detail) if detail else "<empty>"
            parts.append(f"{buffer_name}: {inner}")
        return "\n".join(parts)

    @staticmethod
    def _graph_from_productions(
        productions: Iterable[ProductionAnalysis],
    ) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = {}
        for production in productions:
            graph.setdefault(production.source_label, set()).add(production.target_label)
            graph.setdefault(production.target_label, set())
        return graph
