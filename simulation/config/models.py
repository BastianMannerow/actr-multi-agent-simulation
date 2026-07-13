"""Typed simulation configuration shared by the GUI and runtime."""

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


@dataclass(slots=True)
class AgentTypeConfig:
    """Configuration for one dynamically discovered agent model."""

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
    """All simulation settings that can be edited in the GUI."""

    focus_position: tuple[int, int] = (0, 2)
    print_middleman: bool = False
    width: int = 16
    height: int = 16
    speed_factor: float = 100.0
    print_agent_actions: bool = True
    los: int = 3
    execution_mode: str = "single"
    human_agent_enabled: bool = False
    human_agent_name: str = "Human Player"
    agent_type_config: dict[str, AgentTypeConfig] = field(
        default_factory=lambda: {
            "Example": AgentTypeConfig(count=1, print_agent_actions=True)
        }
    )

    @property
    def stepper(self) -> bool:
        return self.execution_mode == "single"

    @property
    def speed_label(self) -> str:
        for label, value in SPEED_PRESETS:
            if float(self.speed_factor) == float(value):
                return label
        return f"{self.speed_factor:g}%"

    def validate(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("Width and height must both be at least 1.")
        allowed_speeds = {value for _, value in SPEED_PRESETS}
        if float(self.speed_factor) not in allowed_speeds:
            raise ValueError("The speed must use one of the predefined presets.")
        if self.los < 0:
            raise ValueError("Line of sight cannot be negative.")
        if self.execution_mode not in {"single", "automatic"}:
            raise ValueError("Unknown execution mode.")
        if self.human_agent_enabled and not self.human_agent_name.strip():
            raise ValueError("The human agent needs a name.")
        if sum(max(0, item.count) for item in self.agent_type_config.values()) < 1:
            raise ValueError("At least one agent must be enabled.")

    def without_human_agent(self) -> "SimulationConfig":
        """Return a detached configuration suitable for headless multi-runs."""
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
            "los": self.los,
            "execution_mode": self.execution_mode,
            "stepper": self.stepper,
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
            if isinstance(value, dict)
        }
        if not agent_types:
            agent_types = {"Example": AgentTypeConfig(count=1)}

        speed = float(payload.get("speed_factor", 100.0))
        allowed_speeds = {value for _, value in SPEED_PRESETS}
        if speed not in allowed_speeds:
            speed = 100.0

        config = cls(
            focus_position=focus_position,
            print_middleman=bool(payload.get("print_middleman", False)),
            width=max(1, int(payload.get("width", 16))),
            height=max(1, int(payload.get("height", 16))),
            speed_factor=speed,
            print_agent_actions=bool(payload.get("print_agent_actions", True)),
            los=max(0, int(payload.get("los", 3))),
            execution_mode=str(payload.get("execution_mode", "single")),
            human_agent_enabled=bool(payload.get("human_agent_enabled", False)),
            human_agent_name=str(payload.get("human_agent_name", "Human Player")).strip() or "Human Player",
            agent_type_config=agent_types,
        )
        if config.execution_mode not in {"single", "automatic"}:
            config.execution_mode = "single"
        return config
