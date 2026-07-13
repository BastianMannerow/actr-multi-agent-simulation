"""Dynamic ACT-R model factory with optional adapter support."""

from __future__ import annotations

import importlib
import inspect
from typing import Any, Optional, Type


class NullAdapter:
    """No-op adapter used when a model intentionally has no adapter file."""

    def __init__(self, _environment: Any = None) -> None:
        self.agent_construct: Any | None = None

    def extending_actr(self) -> None:
        return None


class AgentTypeReturner:
    """Resolve ``agents/<Type>.py`` and optional ``<Type>Adapter.py`` plug-ins."""

    def __init__(self, base_package: str = "agents") -> None:
        self.base_package = base_package
        self._cache: dict[str, tuple[Type[Any], Type[Any] | None]] = {}

    def clear_cache(self) -> None:
        """Discard resolved classes before rebuilding after a GUI rescan."""
        self._cache.clear()
        importlib.invalidate_caches()

    @staticmethod
    def _first_local_class(module: Any) -> Optional[Type[Any]]:
        classes = [
            obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if obj.__module__ == module.__name__
        ]
        return classes[0] if len(classes) == 1 else None

    def _resolve_agent_classes(
        self, name: str
    ) -> tuple[Type[Any], Type[Any] | None]:
        if name in self._cache:
            return self._cache[name]

        runner_module_name = f"{self.base_package}.{name}"
        try:
            runner_module = importlib.import_module(runner_module_name)
        except ImportError as exc:
            raise ValueError(
                f"Module '{runner_module_name}' could not be imported. "
                f"Expected file: agents/{name}.py."
            ) from exc

        runner_cls = getattr(
            runner_module, name, None
        ) or self._first_local_class(runner_module)
        if runner_cls is None:
            raise ValueError(
                f"No unambiguous model class was found in "
                f"'{runner_module_name}'."
            )

        adapter_cls: Type[Any] | None = None
        adapter_module_name = f"{self.base_package}.{name}Adapter"
        try:
            adapter_module = importlib.import_module(adapter_module_name)
        except ModuleNotFoundError as exc:
            if exc.name != adapter_module_name:
                raise ValueError(
                    f"Adapter module '{adapter_module_name}' has a missing "
                    f"dependency: {exc.name}."
                ) from exc
        except ImportError as exc:
            raise ValueError(
                f"Adapter module '{adapter_module_name}' could not be "
                f"imported: {exc}."
            ) from exc
        else:
            adapter_cls = (
                getattr(adapter_module, f"{name}Adapter", None)
                or self._first_local_class(adapter_module)
            )
            if adapter_cls is None:
                raise ValueError(
                    f"No unambiguous adapter class was found in "
                    f"'{adapter_module_name}'."
                )

        self._cache[name] = (runner_cls, adapter_cls)
        return runner_cls, adapter_cls

    def return_agent_type(
        self,
        name: str,
        actr_environment: Any,
        agent_id_list: list[Any],
    ) -> Optional[tuple[Any, Any, Any]]:
        if name == "Human":
            return None

        runner_cls, adapter_cls = self._resolve_agent_classes(name)
        runner = runner_cls(actr_environment)
        actr_agent = runner.build_agent(agent_id_list)
        adapter = (adapter_cls or NullAdapter)(actr_environment)
        return runner, actr_agent, adapter
