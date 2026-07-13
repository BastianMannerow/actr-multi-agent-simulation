"""Discovery of ACT-R models and optional Python adapters in ``agents``."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentTypeInfo:
    """Description of one model file and its optional adapter."""

    name: str
    model_module: str
    model_path: str
    model_class_name: str | None
    adapter_module: str | None
    adapter_path: str | None
    adapter_class_name: str | None
    model_error: str | None = None
    adapter_error: str | None = None

    @property
    def model_available(self) -> bool:
        return self.model_class_name is not None and self.model_error is None

    @property
    def adapter_available(self) -> bool:
        return self.adapter_class_name is not None and self.adapter_error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_module": self.model_module,
            "model_path": self.model_path,
            "model_class_name": self.model_class_name,
            "model_available": self.model_available,
            "model_error": self.model_error,
            "adapter_module": self.adapter_module,
            "adapter_path": self.adapter_path,
            "adapter_class_name": self.adapter_class_name,
            "adapter_available": self.adapter_available,
            "adapter_error": self.adapter_error,
        }


class AgentDiscovery:
    """Scan the agents package without maintaining a hard-coded type list."""

    def __init__(self, base_package: str = "agents") -> None:
        self.base_package = base_package

    def discover(self) -> list[AgentTypeInfo]:
        importlib.invalidate_caches()
        package = importlib.import_module(self.base_package)
        package_path = Path(package.__file__).resolve().parent
        result: list[AgentTypeInfo] = []

        for model_path in sorted(
            package_path.glob("*.py"), key=lambda path: path.name.lower()
        ):
            stem = model_path.stem
            if (
                stem == "__init__"
                or stem.startswith("_")
                or stem.endswith("Adapter")
            ):
                continue
            result.append(self._inspect_type(stem, model_path))
        return result

    def _inspect_type(self, name: str, model_path: Path) -> AgentTypeInfo:
        model_module_name = f"{self.base_package}.{name}"
        model_class_name: str | None = None
        model_error: str | None = None
        try:
            model_module = importlib.import_module(model_module_name)
            model_class = self._resolve_class(model_module, name)
            if model_class is None:
                model_error = f"No model class named '{name}' was found."
            else:
                model_class_name = model_class.__name__
        except Exception as exc:
            # Discovery reports broken plug-ins without crashing the GUI.
            model_error = f"{type(exc).__name__}: {exc}"

        adapter_path = model_path.with_name(f"{name}Adapter.py")
        adapter_module_name = (
            f"{self.base_package}.{name}Adapter"
            if adapter_path.exists()
            else None
        )
        adapter_class_name: str | None = None
        adapter_error: str | None = None
        if adapter_module_name is not None:
            try:
                adapter_module = importlib.import_module(adapter_module_name)
                adapter_class = self._resolve_class(
                    adapter_module, f"{name}Adapter"
                )
                if adapter_class is None:
                    adapter_error = (
                        f"No adapter class named '{name}Adapter' was found."
                    )
                else:
                    adapter_class_name = adapter_class.__name__
            except Exception as exc:
                adapter_error = f"{type(exc).__name__}: {exc}"

        return AgentTypeInfo(
            name=name,
            model_module=model_module_name,
            model_path=str(model_path),
            model_class_name=model_class_name,
            adapter_module=adapter_module_name,
            adapter_path=str(adapter_path) if adapter_path.exists() else None,
            adapter_class_name=adapter_class_name,
            model_error=model_error,
            adapter_error=adapter_error,
        )

    @staticmethod
    def _resolve_class(
        module: ModuleType, expected_name: str
    ) -> type[Any] | None:
        expected = getattr(module, expected_name, None)
        if inspect.isclass(expected) and expected.__module__ == module.__name__:
            return expected
        local_classes = [
            obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if obj.__module__ == module.__name__
        ]
        return local_classes[0] if len(local_classes) == 1 else None
