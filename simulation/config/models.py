"""Typed configuration for interactive and multi-run ACT-R demo simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SPEED_PRESETS: tuple[tuple[str, float], ...] = (
    ("1/4 Realtime", 25.0),
    ("1/2 Realtime", 50.0),
    ("Realtime", 100.0),
    ("2x Realtime", 200.0),
    ("ASAP", -1.0),
)

ENVIRONMENT_MODES: tuple[tuple[str, str], ...] = (
    ("Virtual Matrix", "virtual"),
)

VIRTUAL_LEVELS: tuple[tuple[str, str], ...] = (
    ("Demo Matrix", "demo_matrix"),
)


@dataclass(slots=True)
class AgentTypeConfig:
    count: int = 1
    print_agent_actions: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AgentTypeConfig":
        payload = payload or {}
        return cls(
            count=max(0, int(payload.get("count", 1))),
            print_agent_actions=bool(payload.get("print_agent_actions", True)),
        )


@dataclass(slots=True)
class SimulationConfig:
    focus_position: tuple[int, int] = (0, 2)
    print_middleman: bool = False
    speed_factor: float = 100.0
    print_agent_actions: bool = True
    experimental_pyactr_performance_boost: bool = False
    los: int = 3
    execution_mode: str = "single"
    environment_mode: str = "virtual"
    virtual_level: str = "demo_matrix"
    human_agent_enabled: bool = False
    human_agent_name: str = "Human Player"
    agent_type_config: dict[str, AgentTypeConfig] = field(
        default_factory=lambda: {
            "CountingAgent": AgentTypeConfig(count=1, print_agent_actions=True),
            "Runner": AgentTypeConfig(count=1, print_agent_actions=True),
        }
    )

    @property
    def height(self) -> int:
        from simulation.world.level_builder import level_dimensions
        return level_dimensions(self.virtual_level)[0]

    @property
    def width(self) -> int:
        from simulation.world.level_builder import level_dimensions
        return level_dimensions(self.virtual_level)[1]

    @property
    def stepper(self) -> bool:
        return self.execution_mode == "single"

    @property
    def speed_label(self) -> str:
        for label, value in SPEED_PRESETS:
            if float(self.speed_factor) == float(value):
                return label
        return f"{self.speed_factor:g}%"

    @property
    def environment_label(self) -> str:
        labels = {value: label for label, value in ENVIRONMENT_MODES}
        return labels.get(self.environment_mode, self.environment_mode)

    def validate(self) -> None:
        if float(self.speed_factor) not in {value for _, value in SPEED_PRESETS}:
            raise ValueError("The speed must use one of the predefined presets.")
        if self.los < 0:
            raise ValueError("Line of sight cannot be negative.")
        if self.execution_mode not in {"single", "automatic"}:
            raise ValueError("Unknown execution mode.")
        if self.environment_mode != "virtual":
            raise ValueError("The Demo Simulation supports only the virtual matrix backend.")
        if self.virtual_level not in {value for _, value in VIRTUAL_LEVELS}:
            raise ValueError("Unknown virtual level.")
        if self.human_agent_enabled and not self.human_agent_name.strip():
            raise ValueError("The human agent needs a name.")
        if sum(max(0, item.count) for item in self.agent_type_config.values()) < 1:
            raise ValueError("At least one ACT-R agent must be enabled.")

    def without_human_agent(self) -> "SimulationConfig":
        payload = self.to_dict()
        payload["human_agent_enabled"] = False
        payload["human_agent_name"] = "Human Player"
        return type(self).from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus_position": list(self.focus_position),
            "print_middleman": self.print_middleman,
            "width": self.width,
            "height": self.height,
            "speed_factor": self.speed_factor,
            "speed_label": self.speed_label,
            "print_agent_actions": self.print_agent_actions,
            "experimental_pyactr_performance_boost": self.experimental_pyactr_performance_boost,
            "los": self.los,
            "execution_mode": self.execution_mode,
            "stepper": self.stepper,
            "environment_mode": self.environment_mode,
            "environment_label": self.environment_label,
            "virtual_level": self.virtual_level,
            "human_agent_enabled": self.human_agent_enabled,
            "human_agent_name": self.human_agent_name,
            "agent_type_config": {
                name: config.to_dict()
                for name, config in self.agent_type_config.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SimulationConfig":
        payload = payload or {}
        focus = payload.get("focus_position", [0, 2])
        try:
            focus_position = (int(focus[0]), int(focus[1]))
        except Exception:
            focus_position = (0, 2)

        raw_agents = payload.get("agent_type_config", {})
        agent_types = {
            str(name): AgentTypeConfig.from_dict(value)
            for name, value in raw_agents.items()
            if isinstance(value, dict) and str(name) in {"CountingAgent", "Runner"}
        }
        if not agent_types:
            agent_types = {
                "CountingAgent": AgentTypeConfig(count=1),
                "Runner": AgentTypeConfig(count=1),
            }

        speed = float(payload.get("speed_factor", 100.0))
        if speed not in {value for _, value in SPEED_PRESETS}:
            speed = 100.0
        execution_mode = str(payload.get("execution_mode", "single"))
        if execution_mode not in {"single", "automatic"}:
            execution_mode = "single"

        config = cls(
            focus_position=focus_position,
            print_middleman=bool(payload.get("print_middleman", False)),
            speed_factor=speed,
            print_agent_actions=bool(payload.get("print_agent_actions", True)),
            experimental_pyactr_performance_boost=bool(
                payload.get("experimental_pyactr_performance_boost", False)
            ),
            los=max(0, int(payload.get("los", 3))),
            execution_mode=execution_mode,
            environment_mode="virtual",
            virtual_level="demo_matrix",
            human_agent_enabled=bool(payload.get("human_agent_enabled", False)),
            human_agent_name=str(payload.get("human_agent_name", "Human Player")).strip() or "Human Player",
            agent_type_config=agent_types,
        )
        return config
