"""Explainable transition summaries for ACT-R state graphs.

The overview graph intentionally shows only compact module involvement.  Full
conditions, effects, utilities and adapter context are exposed through a
clickable detail payload instead of being painted into the scene.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from simulation.inspection.source_analysis import (
    AgentStaticAnalysis,
    MethodBufferInteraction,
    StateTransitionAnalysis,
)


@dataclass(frozen=True, slots=True)
class ModuleAccess:
    module: str
    buffer: str
    modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransitionExplanation:
    code: str
    kind: str
    title: str
    source_state: str
    target_state: str
    summary: str
    modules: tuple[ModuleAccess, ...]
    guard: str
    actions: str
    production_name: str | None = None
    adapter_method: str | None = None
    trigger_production: str | None = None
    utility: float | None = None
    reward: float | None = None

    def to_payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["payload_type"] = "state_transition_explanation"
        return value


_MODULE_ALIASES = {
    "g": "Goal",
    "goal": "Goal",
    "retrieval": "Retrieval",
    "decmem": "Declarative memory",
    "manual": "Manual",
    "visual": "Visual",
    "visual_location": "Visual location",
    "imaginal": "Imaginal",
}


def module_name(buffer_name: str) -> str:
    """Map concrete ACT-R buffer names to a compact cognitive module label."""
    normalized = str(buffer_name or "").strip().casefold()
    if normalized in _MODULE_ALIASES:
        return _MODULE_ALIASES[normalized]
    if "retrieval" in normalized:
        return "Retrieval"
    if "decmem" in normalized or "declarative" in normalized:
        return "Declarative memory"
    if "manual" in normalized or "motor" in normalized:
        return "Manual"
    if "visual_location" in normalized:
        return "Visual location"
    if "visual" in normalized:
        return "Visual"
    if "imaginal" in normalized:
        return "Imaginal"
    if normalized in {"g", "goal"}:
        return "Goal"
    return str(buffer_name or "Module")


def transition_explanation(
    analysis: AgentStaticAnalysis,
    transition: StateTransitionAnalysis,
    code: str,
) -> TransitionExplanation:
    """Create one user-facing explanation without putting long prose on edges."""
    source = analysis.states.get(transition.source_state_id)
    target = analysis.states.get(transition.target_state_id)
    source_label = source.label if source is not None else transition.source_state_id
    target_label = target.label if target is not None else transition.target_state_id

    accesses: dict[tuple[str, str], set[str]] = {}
    if transition.kind == "production":
        production = analysis.production(transition.production_name or transition.label)
        if production is not None:
            for buffer_name in production.read_buffers:
                key = (module_name(buffer_name), str(buffer_name))
                accesses.setdefault(key, set()).add("read")
            for buffer_name in production.written_buffers:
                effect = production.effects.get(buffer_name, {})
                mode = str(effect.get("mode", "write"))
                key = (module_name(buffer_name), str(buffer_name))
                accesses.setdefault(key, set()).add(mode)
    else:
        for item in _adapter_interactions(analysis, transition):
            key = (module_name(item.buffer_name), str(item.buffer_name))
            accesses.setdefault(key, set()).add(str(item.mode or "access"))

    # Every state transition necessarily involves the goal/control buffer even
    # when static adapter-source analysis cannot resolve a helper call.
    if not any(module == "Goal" for module, _buffer in accesses):
        accesses.setdefault(("Goal", "g"), set()).add("write")

    modules = tuple(
        ModuleAccess(module, buffer_name, tuple(sorted(modes)))
        for (module, buffer_name), modes in sorted(
            accesses.items(), key=lambda item: (item[0][0].casefold(), item[0][1].casefold())
        )
    )
    module_text = ", ".join(dict.fromkeys(item.module for item in modules)) or "Goal"
    if transition.kind == "production":
        title = transition.production_name or transition.label
        summary = (
            f"Production {code} can move the control state from {source_label} "
            f"to {target_label}. It involves {module_text}."
        )
    else:
        title = transition.adapter_method or transition.label
        trigger = transition.trigger_production or "the preceding production"
        summary = (
            f"Adapter transition {code} is evaluated after {trigger}. It can "
            f"replace the control state {source_label} with {target_label} and "
            f"involves {module_text}."
        )

    return TransitionExplanation(
        code=code,
        kind=transition.kind,
        title=title,
        source_state=source_label,
        target_state=target_label,
        summary=summary,
        modules=modules,
        guard=transition.guard_label or "No additional condition was recovered.",
        actions=transition.action_label or "Control-state update.",
        production_name=transition.production_name,
        adapter_method=transition.adapter_method,
        trigger_production=transition.trigger_production,
        utility=transition.utility,
        reward=transition.reward,
    )


def compact_module_caption(explanations: list[TransitionExplanation]) -> str:
    """Return a short, stable module caption for one routed edge bundle."""
    ordered: list[str] = []
    for explanation in explanations:
        for access in explanation.modules:
            if access.module not in ordered:
                ordered.append(access.module)
    # Four long names on an edge are no longer an overview.  Keep the first
    # three and communicate the complete set in the click dialog.
    display = {
        "Declarative memory": "DM",
        "Visual location": "Visual loc.",
    }
    compact = [display.get(name, name) for name in ordered]
    if len(compact) > 3:
        return " · ".join(compact[:3]) + f" · +{len(compact) - 3}"
    return " · ".join(compact) or "Goal"


def _adapter_interactions(
    analysis: AgentStaticAnalysis,
    transition: StateTransitionAnalysis,
) -> list[MethodBufferInteraction]:
    method = str(transition.adapter_method or transition.label).casefold()
    return [
        item
        for item in analysis.adapter_interactions
        if method
        and (
            str(item.method_name).casefold() == method
            or str(item.function_name).casefold() == method
        )
    ]
