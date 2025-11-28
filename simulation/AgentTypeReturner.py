from typing import Any, List, Optional, Tuple, Type
import importlib
import inspect


class AgentTypeReturner:
    """
    Dynamic factory for ACT-R agent instantiation.

    Responsibilities
    ----------------
    - Resolve logical agent type names to concrete runner and adapter classes.
    - Enforce a simple naming convention: for type ``T`` expect modules
      ``agents.T`` and ``agents.TAdapter`` containing classes ``T`` and
      ``TAdapter`` respectively.
    - Provide a single entry point that returns the tuple
      ``(runner_instance, actr_agent, adapter_instance)`` used by the
      simulation layer.

    Notes
    -----
    - The special type ``"Human"`` returns ``None`` because human
      participants are controlled externally.
    - Resolved classes are cached per type name to avoid repeated imports.
    """

    def __init__(self, base_package: str = "agents") -> None:
        """
        Parameters
        ----------
        base_package : str, optional
            Python package that contains all agent and adapter modules.
            By default, this is ``"agents"``.
        """
        self.base_package = base_package
        self._cache: dict[str, Tuple[Type[Any], Type[Any]]] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _import_module(self, fullname: str):
        """
        Import a module by fully qualified name and return it.

        Raises
        ------
        ImportError
            If the module cannot be imported.
        """
        return importlib.import_module(fullname)

    @staticmethod
    def _first_local_class(module) -> Optional[Type[Any]]:
        """
        Return the first class defined in ``module`` or ``None``.

        This is used as a fallback if the expected class name is not
        available but the module still defines a single relevant class.
        """
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:
                return obj
        return None

    def _resolve_agent_classes(self, name: str) -> Tuple[Type[Any], Type[Any]]:
        """
        Resolve runner and adapter classes for a logical agent type.

        Parameters
        ----------
        name : str
            Logical agent name, e.g. ``"Example"``.

        Returns
        -------
        (Type, Type)
            Tuple ``(runner_class, adapter_class)``.

        Raises
        ------
        ValueError
            If modules or classes cannot be found according to the
            naming convention.
        """
        if name in self._cache:
            return self._cache[name]

        runner_module_name = f"{self.base_package}.{name}"
        adapter_module_name = f"{self.base_package}.{name}Adapter"

        try:
            runner_module = self._import_module(runner_module_name)
        except ImportError as exc:
            raise ValueError(
                f"Could not import runner module '{runner_module_name}' "
                f"for agent type {name!r}. "
                f"Expected a file '{name}.py' in package '{self.base_package}'."
            ) from exc

        try:
            adapter_module = self._import_module(adapter_module_name)
        except ImportError as exc:
            raise ValueError(
                f"Could not import adapter module '{adapter_module_name}' "
                f"for agent type {name!r}. "
                f"Expected a file '{name}Adapter.py' in package '{self.base_package}'."
            ) from exc

        runner_cls = getattr(runner_module, name, None)
        adapter_cls = getattr(adapter_module, f"{name}Adapter", None)

        # Fallback: use first locally defined class if the expected
        # class name is not present.
        if runner_cls is None:
            runner_cls = self._first_local_class(runner_module)
        if adapter_cls is None:
            adapter_cls = self._first_local_class(adapter_module)

        if runner_cls is None or adapter_cls is None:
            raise ValueError(
                f"Could not resolve classes for agent type {name!r}. "
                f"Expected class '{name}' in '{runner_module_name}' and "
                f"'{name}Adapter' in '{adapter_module_name}'."
            )

        self._cache[name] = (runner_cls, adapter_cls)
        return runner_cls, adapter_cls

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def return_agent_type(
        self,
        name: str,
        actr_environment: Any,
        agent_id_list: List[Any],
    ) -> Optional[Tuple[Any, Any, Any]]:
        """
        Instantiate and return agent artifacts for the given logical type.

        Parameters
        ----------
        name : str
            Logical agent label. Examples: ``"Human"``, ``"Example"``.
        actr_environment : Any
            Environment handle passed to agent constructors.
        agent_id_list : list
            Identifiers consumed by the agent's ``build_agent`` routine.

        Returns
        -------
        Optional[Tuple[Any, Any, Any]]
            - ``None`` for human players (manual input elsewhere).
            - Tuple ``(runner, actr_agent, adapter)`` for modeled agents.

        Raises
        ------
        ValueError
            If ``name`` cannot be resolved as an agent type.
        """
        if name == "Human":
            # Human participants are controlled externally; no ACT-R instance.
            return None

        runner_cls, adapter_cls = self._resolve_agent_classes(name)

        # Runner encapsulates the ACT-R model; adapter bridges sim ↔ agent I/O.
        runner = runner_cls(actr_environment)
        actr_agent = runner.build_agent(agent_id_list)
        adapter = adapter_cls(actr_environment)

        return runner, actr_agent, adapter
