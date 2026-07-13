"""Persistent storage of the last simulation configuration."""

from __future__ import annotations

import json

from PyQt6.QtCore import QSettings

from simulation.config.models import SimulationConfig


class SimulationSettingsStore:
    """Persist the last GUI configuration between application launches."""

    KEY = "simulation/last_configuration"

    def __init__(self) -> None:
        self.settings = QSettings()

    def load(self) -> SimulationConfig:
        raw = self.settings.value(self.KEY, "")
        if not raw:
            return SimulationConfig()
        try:
            payload = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return SimulationConfig()
        return SimulationConfig.from_dict(payload)

    def save(self, config: SimulationConfig) -> None:
        self.settings.setValue(
            self.KEY,
            json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True),
        )
        self.settings.sync()

    def reset(self) -> SimulationConfig:
        self.settings.remove(self.KEY)
        self.settings.sync()
        return SimulationConfig()
