from typing import Any, List, Optional, Tuple
from agents.Example import Example
from agents.ExampleAdapter import ExampleAdapter


class AgentTypeReturner:
    """
    Thin factory for ACT-R agent instantiation.

    Scope
    -----
    - Centralizes name → constructor mapping for simulation agents.
    - Returns the trio required by the simulation loop:
      (runner_instance, actr_agent, adapter_instance).

    Status
    ------
    - Transitional utility. Prefer a pluggable registry (e.g., decorator-based)
      to avoid editing this file when adding new agents.

    Conventions
    -----------
    - For "Human": no programmatic agent is created → return None.
    - For modeled agents: build runner, then `runner.build_agent(ids)`,
      and construct the adapter.
    """

    def __init__(self) -> None:
        # Stateless by design; kept for API symmetry and future extension.
        pass

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
            Logical agent label. Examples: "Human", "Example".
        actr_environment : Any
            Environment handle passed to agent constructors.
        agent_id_list : List[Any]
            Identifiers consumed by the agent's `build_agent` routine.

        Returns
        -------
        Optional[Tuple[Any, Any, Any]]
            - None for human players (manual input elsewhere).
            - Tuple (runner, actr_agent, adapter) for modeled agents.

        Raises
        ------
        ValueError
            If `name` is not recognized.
        """
        if name == "Human":
            # Human participants are controlled externally; no ACT-R instance.
            return None

        elif name == "Example":
            # Runner encapsulates the ACT-R model; adapter bridges sim ↔ agent I/O.
            runner = Example(actr_environment)
            actr_agent = runner.build_agent(agent_id_list)
            adapter = ExampleAdapter(actr_environment)
            return runner, actr_agent, adapter

        # Keep the message explicit to aid configuration debugging.
        raise ValueError(f"Unknown agent type: {name!r}")
